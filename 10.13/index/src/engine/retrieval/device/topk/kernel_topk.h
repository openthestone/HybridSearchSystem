#pragma once

#include <float.h>
#include <cstdlib>
#include "kernel_operator.h"
#include "../device_common.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

namespace {
const static uint32_t DOC_NUMBER_LENGTH = 48;  // one slot per aggregator core; BLOCK_DIM_MAX is 40
const uint32_t DOC_NUMBER_BYTE_SIZE = DOC_NUMBER_LENGTH * 4;
const uint32_t EFFECTIVE_COUNT = 64;
const uint32_t COPY_DATA_ROUNDING = 8;
// NOTE: must match the aggregator block size; for multi-core it must align with MAX_BLOCK_SIZE in result_aggregator.cpp
const uint32_t MAX_BLOCK_SIZE_AGGREGATOR = 5888;
const uint32_t MAX_BLOCK_SIZE = 3840;
const uint32_t SAMPLE_SORT_MIN_LENGTH = 2;
const uint32_t BYTE_LENGTH = 8;
const uint32_t GATHER_MASK_PARAMETER = 8;
static constexpr uint32_t SAMPLE_SORT_MAX_LENGTH = 8;
static constexpr uint32_t SAMPLE_SORT_JUDGE_SIZE = 256;
}  // namespace

class KernelTOPK {
   public:
    __aicore__ inline KernelTOPK() {}
    __aicore__ inline void Init(GM_ADDR docVectorScores, GM_ADDR docLocationVectors, GM_ADDR docVectorScoresNext,
                                GM_ADDR docLocationVectorsNext, GM_ADDR docResultCountInDevice,
                                GM_ADDR docNumberInDevice, uint32_t loopCount, uint32_t blockDimAggregator,
                                uint32_t blockNumberShardAggregator, uint32_t topK, uint32_t topKEarlyQuit,
                                uint32_t docNumberStride) {
        // docNumberStride: 1 is the dense array the host uploads; RESULT_MESSAGE_SIZE_IN_DEVICE
        // (16) reads the Aggregator's own count block in place (NPUR_TOPK_COUNTS_IN_PLACE).
        docNumberGlobal.SetGlobalBuffer((__gm__ uint32_t*)docNumberInDevice, DOC_NUMBER_BYTE_SIZE * docNumberStride);
        uint32_t realDocNum = 0;
        for (uint32_t i = 0; i < blockDimAggregator; i++) {
            m_docNumberShard[i] = docNumberGlobal.GetValue(i * docNumberStride);
            realDocNum += m_docNumberShard[i];
        }
        m_realDocNum = realDocNum;
        m_blockNumber = (realDocNum + MAX_BLOCK_SIZE - 1) / MAX_BLOCK_SIZE;
        m_lastBlockLength = realDocNum - MAX_BLOCK_SIZE * (m_blockNumber - 1);
        m_topK = topK;
        m_topKEarlyQuit = topKEarlyQuit;
        m_blockDimAggregator = blockDimAggregator;
        m_blockNumberShardAggregator = blockNumberShardAggregator;
        m_loopCount = loopCount;
        docVectorScoreGlobal.SetGlobalBuffer((__gm__ float*)docVectorScores, realDocNum);
        docLocationGlobal.SetGlobalBuffer((__gm__ uint32_t*)docLocationVectors, realDocNum);
        docVectorScoreNextGlobal.SetGlobalBuffer((__gm__ float*)docVectorScoresNext, realDocNum);
        docLocationNextGlobal.SetGlobalBuffer((__gm__ uint32_t*)docLocationVectorsNext, realDocNum);
        effectiveCountGlobal.SetGlobalBuffer((__gm__ uint32_t*)docResultCountInDevice, EFFECTIVE_COUNT);

        uint32_t bufferSize = MAX_BLOCK_SIZE * sizeof(float);
        pipe.InitBuffer(InQueueDocScore, BUFFER_NUM, bufferSize);
        pipe.InitBuffer(InQueueDocLocation, BUFFER_NUM, bufferSize);
        pipe.InitBuffer(outQueueDocVectorScore, BUFFER_NUM, bufferSize);
        pipe.InitBuffer(outQueueDocLocation, BUFFER_NUM, bufferSize);
        pipe.InitBuffer(outQueueDocVectorScoreNext, BUFFER_NUM, bufferSize);
        pipe.InitBuffer(outQueueDocLocationNext, BUFFER_NUM, bufferSize);
        pipe.InitBuffer(maskQueue, BUFFER_NUM, MAX_BLOCK_SIZE / BYTE_LENGTH * sizeof(uint8_t));
        pipe.InitBuffer(maskSmallerQueue, BUFFER_NUM, MAX_BLOCK_SIZE / BYTE_LENGTH * sizeof(uint16_t));
    }

    __aicore__ inline void sort(float* arr, int n) {
        for (int i = 0; i < n - 1; ++i) {
            for (int j = 0; j < n - i - 1; ++j) {
                if (arr[j] < arr[j + 1]) {
                    float temp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = temp;
                }
            }
        }
    }

    __aicore__ inline void Process() {
        if (m_realDocNum == 0) {
            // Under NPUR_FUSE_AGG_TOPK a query that matched nothing reaches this kernel. Report
            // empty: at m_blockNumber 0 the loop runs anyway and GetPivot reads a zero-length global.
            effectiveCountGlobal.SetValue(0, 0u);
            effectiveCountGlobal.SetValue(1, 0u);
            effectiveCountGlobal.SetValue(2, 0u);
            return;
        }
        uint32_t targetDocNum = m_docNumberShard[0];
        uint32_t lastTargetDocNum = 0;
        uint32_t lastRandomLocation = 0;
        uint32_t blockNumber = m_blockNumber;
        uint32_t blockLength = MAX_BLOCK_SIZE;
        uint32_t lastBlockLength = m_lastBlockLength;
        uint32_t leftK = m_topK;
        // index into the topK result; also the actual final top count
        uint32_t srcResultGlobalIndex = 0;
        bool isUseDocVectorScore = true;
        uint32_t lookCountReal = 0;
        uint32_t srcSmallerSumAll = 0;
        uint32_t loopCount = m_loopCount;
        while (lookCountReal < loopCount) {
            lookCountReal++;
            float pivot = GetPivot(targetDocNum, leftK, isUseDocVectorScore, srcResultGlobalIndex, lastTargetDocNum,
                                   lastRandomLocation);
            lastTargetDocNum = targetDocNum;
            uint32_t srcGreaterSumAll = 0;
            srcSmallerSumAll = 0;
            uint32_t loopCountSubtractOne = blockNumber - 1;
            DEBUG_LOG(
                "TOPK Process isUseDocVectorScore: %d，leftK: %d ,srcResultGlobalIndex=%d, lookCountReal: "
                "%d,pivot: %f \n",
                isUseDocVectorScore, leftK, srcResultGlobalIndex, lookCountReal, pivot);
            DEBUG_LOG("TOPK Process blockNumber: %d, blockLength: %d,lastBlockLength: %d,targetDocNum : %d \n",
                      blockNumber, blockLength, lastBlockLength, targetDocNum);
            uint32_t realShardCount = 1;
            if (lookCountReal == 1) {
                realShardCount = m_blockDimAggregator;
            }
            for (uint32_t m = 0; m < realShardCount; m++) {
                uint32_t srcResultGlobalIndexCopyIn = srcResultGlobalIndex;
                if (realShardCount > 1 && m_blockDimAggregator > 1) {
                    srcResultGlobalIndexCopyIn = m * m_blockNumberShardAggregator * MAX_BLOCK_SIZE_AGGREGATOR;
                    uint32_t shardDocNum = m_docNumberShard[m];
                    blockNumber = (shardDocNum + blockLength - 1) / blockLength;
                    lastBlockLength = shardDocNum - blockLength * (blockNumber - 1);
                    loopCountSubtractOne = blockNumber - 1;
                }
                for (uint32_t i = 0; i < blockNumber; i++) {
                    uint32_t realDataLength = i == loopCountSubtractOne ? lastBlockLength : blockLength;
                    CopyIn(i, blockLength, realDataLength, isUseDocVectorScore, srcResultGlobalIndexCopyIn);
                    uint32_t countGreaterSumLoop = 0;
                    uint32_t countSmallerSumLoop = 0;
                    Compute(i, pivot, countGreaterSumLoop, countSmallerSumLoop, realDataLength, srcGreaterSumAll,
                            leftK);
                    CopyOut(i, countGreaterSumLoop, srcGreaterSumAll, countSmallerSumLoop, srcSmallerSumAll,
                            isUseDocVectorScore, srcResultGlobalIndex);
                    srcGreaterSumAll += countGreaterSumLoop;
                    srcSmallerSumAll += countSmallerSumLoop;
                    DEBUG_LOG(
                        "TOPK Process countGreaterSumLoop:%d, srcGreaterSumAll:%d, "
                        "countSmallerSumLoop:%d, srcSmallerSumAll:%d \n",
                        countGreaterSumLoop, srcGreaterSumAll, countSmallerSumLoop, srcSmallerSumAll);
                }
            }
            DEBUG_LOG("TOPK Process srcGreaterSumAll:%d, srcSmallerSumAll: %d \n", srcGreaterSumAll, srcSmallerSumAll);
            if (srcGreaterSumAll == leftK) {
                DEBUG_LOG("TOPK Process srcGreaterSumAll==leftK \n");
                srcResultGlobalIndex += srcGreaterSumAll;
                srcSmallerSumAll = 0;
                break;
            } else if (srcGreaterSumAll < leftK) {
                DEBUG_LOG("TOPK Process srcGreaterSumAll<leftK \n");
                srcResultGlobalIndex += srcGreaterSumAll;
                if (lookCountReal == loopCount) {
                    // fixed iteration count: on the last iteration, copy all remaining data into the result set
                    DEBUG_LOG("TOPK Process lookCountReal == LOOK_TIMES （srcGreaterSumAll < leftK）\n");
                    break;
                }
                targetDocNum = srcSmallerSumAll;
                leftK = leftK - srcGreaterSumAll;
                isUseDocVectorScore = false;
            } else {
                DEBUG_LOG("TOPK Process srcGreaterSumAll>leftK srcResultGlobalIndex: %d \n", srcResultGlobalIndex);
                targetDocNum = srcGreaterSumAll;
                srcSmallerSumAll = 0;
                if (srcResultGlobalIndex + srcGreaterSumAll <= m_topKEarlyQuit) {
                    DEBUG_LOG("TOPK Process srcResultGlobalIndex + srcGreaterSumAll < m_topKEarlyQuit , break \n");
                    srcResultGlobalIndex += srcGreaterSumAll;
                    break;
                }
                if (lookCountReal == loopCount) {
                    // fixed iteration count: on the last iteration, copy all remaining data into the result set
                    DEBUG_LOG("TOPK Process lookCountReal == LOOK_TIMES （srcGreaterSumAll > leftK）\n");
                    srcResultGlobalIndex += srcGreaterSumAll;
                    break;
                }
                isUseDocVectorScore = true;
            }
            DEBUG_LOG("TOPK Process srcResultGlobalIndex: %d\n", srcResultGlobalIndex);
            blockNumber = (targetDocNum + blockLength - 1) / blockLength;
            lastBlockLength = targetDocNum - blockLength * (blockNumber - 1);
        }
        effectiveCountGlobal.SetValue(0, srcResultGlobalIndex);
        effectiveCountGlobal.SetValue(1, srcSmallerSumAll);
        effectiveCountGlobal.SetValue(2, lookCountReal);  // value at index 2
    }

   private:
    __aicore__ inline float GetPivot(uint32_t targetDocNum, uint32_t leftK, bool isUseDocVectorScore,
                                     uint32_t srcResultGlobalIndex, uint32_t lastTargetDocNum,
                                     uint32_t& lastRandomLocation) {
        if (targetDocNum < leftK || targetDocNum < SAMPLE_SORT_JUDGE_SIZE || lastTargetDocNum == targetDocNum) {
            if (lastRandomLocation > targetDocNum - 1) {
                lastRandomLocation = 0;
            } else {
                lastRandomLocation++;
            }
            DEBUG_LOG("TOPK GetPivot random lastRandomLocation: %d\n", lastRandomLocation);
            return isUseDocVectorScore ? docVectorScoreGlobal.GetValue(srcResultGlobalIndex + lastRandomLocation)
                                       : docVectorScoreNextGlobal.GetValue(lastRandomLocation);
        } else {
            uint32_t numsSampleSize = targetDocNum / leftK;
            if (numsSampleSize > SAMPLE_SORT_MAX_LENGTH) {
                numsSampleSize = SAMPLE_SORT_MAX_LENGTH;
            }
            if (numsSampleSize < SAMPLE_SORT_MIN_LENGTH) {
                numsSampleSize = SAMPLE_SORT_MIN_LENGTH;
            }
            float numsSample[SAMPLE_SORT_MAX_LENGTH];
            uint32_t offsetAdd = targetDocNum / numsSampleSize;
            for (uint32_t i = 0; i < numsSampleSize; i++) {
                numsSample[i] = isUseDocVectorScore
                                    ? docVectorScoreGlobal.GetValue(srcResultGlobalIndex + i * offsetAdd)
                                    : docVectorScoreNextGlobal.GetValue(i * offsetAdd);
            }
            sort(numsSample, numsSampleSize);
            uint32_t location = numsSampleSize * leftK / targetDocNum;
            DEBUG_LOG("TOPK GetPivot sample location: %d\n", location);
            if (location == 0) {
                return numsSample[0];
            } else {
                if (location > numsSampleSize) {
                    location = numsSampleSize;
                }
                return numsSample[location - 1];
            }
        }
    }

    __aicore__ inline void CopyIn(uint32_t progress, uint32_t blockLength, uint32_t realDataLength,
                                  bool isUseDocVectorScore, uint32_t srcResultGlobalIndex) {
        DEBUG_LOG("TOPK CopyIn progress: %d ,srcResultGlobalIndex: %d \n", progress, srcResultGlobalIndex);
        AscendC::LocalTensor<float> docScoreLocal = InQueueDocScore.AllocTensor<float>();
        AscendC::LocalTensor<uint32_t> docLocationLocal = InQueueDocLocation.AllocTensor<uint32_t>();
        uint32_t offset = realDataLength % COPY_DATA_ROUNDING;
        if (offset != 0) {
            realDataLength = realDataLength + (COPY_DATA_ROUNDING - offset);
        }
        // CompareScalar walks the block length rounded up to 64 and GatherMask in groups of 8, so
        // both read past what DataCopy filled -- UNINITIALIZED local memory, and a stale value
        // above the pivot gets selected with a garbage location. Pin the tail to -FLT_MAX.
        if (realDataLength < MAX_BLOCK_SIZE) {
            AscendC::Duplicate(docScoreLocal[realDataLength], -FLT_MAX, MAX_BLOCK_SIZE - realDataLength);
        }
        if (isUseDocVectorScore) {
            uint32_t srcIndexStart = srcResultGlobalIndex + progress * blockLength;
            AscendC::DataCopy(docScoreLocal, docVectorScoreGlobal[srcIndexStart], realDataLength);
            AscendC::DataCopy(docLocationLocal, docLocationGlobal[srcIndexStart], realDataLength);
        } else {
            uint32_t srcIndexStart = progress * blockLength;
            AscendC::DataCopy(docScoreLocal, docVectorScoreNextGlobal[srcIndexStart], realDataLength);
            AscendC::DataCopy(docLocationLocal, docLocationNextGlobal[srcIndexStart], realDataLength);
        }
        InQueueDocScore.EnQue(docScoreLocal);
        InQueueDocLocation.EnQue(docLocationLocal);
    }
    __aicore__ inline void Compute(uint32_t progress, float pivot, uint32_t& countGreaterSumLoop,
                                   uint32_t& countSmallerSumLoop, uint32_t realDataLength, uint32_t srcGreaterSumAll,
                                   uint32_t leftK) {
        DEBUG_LOG("TOPK  Compute progress: %d \n", progress);
        AscendC::LocalTensor<float> greaterOutLocal = outQueueDocVectorScore.AllocTensor<float>();
        AscendC::LocalTensor<uint32_t> greaterLocationOutLocal = outQueueDocLocation.AllocTensor<uint32_t>();
        AscendC::LocalTensor<float> greaterInLocal = InQueueDocScore.DeQue<float>();
        AscendC::LocalTensor<uint32_t> greaterLocationInLocal = InQueueDocLocation.DeQue<uint32_t>();
        DEBUG_DUMP_TENSOR(greaterInLocal, 3200, DUMP_DEFAULT_LENGTH);  // 3200 is the log dump marker
        DEBUG_DUMP_TENSOR(greaterInLocal, 3201, DUMP_DEFAULT_LENGTH);  // 3201 is the log dump marker

        AscendC::LocalTensor<uint8_t> maskLocal = maskQueue.AllocTensor<uint8_t>();

        // CopyIn rounds the length up to 8, so up to 7 STALE elements sit at [realDataLength,
        // ceil8) -- the pin above starts at ceil8, leaving that hole open. Cover it, fenced.
        const uint32_t tailHole = realDataLength % COPY_DATA_ROUNDING;
        if (tailHole != 0) {
            AscendC::PipeBarrier<PIPE_ALL>();
            for (uint32_t t = realDataLength; t < realDataLength + (COPY_DATA_ROUNDING - tailHole); ++t) {
                greaterInLocal.SetValue(t, -FLT_MAX);
                greaterLocationInLocal.SetValue(t, 0u);
            }
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        uint32_t compareDataLength = realDataLength;
        uint32_t offset = realDataLength % 64;  // round up to 64
        if (offset != 0) {
            compareDataLength = realDataLength + (64 - offset);  // round up to 64
        }
        // CompareScalar requires the last argument to be 256-byte aligned.
        AscendC::CompareScalar(maskLocal, greaterInLocal, pivot, AscendC::CMPMODE::GT, compareDataLength);
        AscendC::LocalTensor<uint32_t> gatherMaskLocal = maskLocal.ReinterpretCast<uint32_t>();

        uint64_t countGreaterSum = 0;
        uint64_t countGreaterLocationSum = 0;

        AscendC::GatherMask(greaterOutLocal, greaterInLocal, gatherMaskLocal, true, realDataLength,
                            {1, 1, GATHER_MASK_PARAMETER, GATHER_MASK_PARAMETER}, countGreaterSum);
        AscendC::GatherMask(greaterLocationOutLocal, greaterLocationInLocal, gatherMaskLocal, true, realDataLength,
                            {1, 1, GATHER_MASK_PARAMETER, GATHER_MASK_PARAMETER}, countGreaterLocationSum);

        countGreaterSumLoop = (uint32_t)countGreaterSum;
        DEBUG_LOG("TOPK Compute progress:%d, countGreaterSum:%d ,countGreaterLocationSum:%d\n", progress,
                  countGreaterSum, countGreaterLocationSum);
        if (countGreaterSumLoop > 0) {
            outQueueDocVectorScore.EnQue<float>(greaterOutLocal);
            outQueueDocLocation.EnQue<uint32_t>(greaterLocationOutLocal);
        } else {
            outQueueDocVectorScore.FreeTensor<float>(greaterOutLocal);
            outQueueDocLocation.FreeTensor<uint32_t>(greaterLocationOutLocal);
        }
        // once there are enough values greater than the pivot, the smaller values need not be processed
        if (srcGreaterSumAll < leftK) {
            AscendC::LocalTensor<float> smallerOutLocal = outQueueDocVectorScoreNext.AllocTensor<float>();
            AscendC::LocalTensor<uint32_t> smallerLocationOutLocal = outQueueDocLocationNext.AllocTensor<uint32_t>();

            AscendC::LocalTensor<uint16_t> gatherMaskLocalTemp = maskLocal.ReinterpretCast<uint16_t>();
            AscendC::LocalTensor<uint16_t> maskSmallerLocal = maskSmallerQueue.AllocTensor<uint16_t>();
            AscendC::Not(maskSmallerLocal, gatherMaskLocalTemp, (int32_t)compareDataLength);
            AscendC::LocalTensor<uint32_t> gatherMaskSmallerLocal = maskSmallerLocal.ReinterpretCast<uint32_t>();

            uint64_t countSmallerSum = 0;
            uint64_t countSmallerLocationSum = 0;
            AscendC::GatherMask(smallerOutLocal, greaterInLocal, gatherMaskSmallerLocal, true, realDataLength,
                                {1, 1, GATHER_MASK_PARAMETER, GATHER_MASK_PARAMETER}, countSmallerSum);
            AscendC::GatherMask(smallerLocationOutLocal, greaterLocationInLocal, gatherMaskSmallerLocal, true,
                                realDataLength, {1, 1, GATHER_MASK_PARAMETER, GATHER_MASK_PARAMETER},
                                countSmallerLocationSum);

            countSmallerSumLoop = (uint32_t)countSmallerSum;

            DEBUG_LOG("TOPK Compute progress:%d, countSmallerSum:%d, countSmallerLocationSum:%d \n", progress,
                      countSmallerSum, countSmallerLocationSum);
            if (countSmallerSumLoop > 0) {
                outQueueDocVectorScoreNext.EnQue<float>(smallerOutLocal);
                outQueueDocLocationNext.EnQue<uint32_t>(smallerLocationOutLocal);
            } else {
                outQueueDocVectorScoreNext.FreeTensor<float>(smallerOutLocal);
                outQueueDocLocationNext.FreeTensor<uint32_t>(smallerLocationOutLocal);
            }
            maskSmallerQueue.FreeTensor<uint16_t>(maskSmallerLocal);
        }
        InQueueDocScore.FreeTensor<float>(greaterInLocal);
        InQueueDocLocation.FreeTensor<uint32_t>(greaterLocationInLocal);
        maskQueue.FreeTensor<uint8_t>(maskLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t countGreaterSumLoop, uint32_t srcGreaterSumAll,
                                   uint32_t countSmallerSumLoop, uint32_t srcSmallerSumAll, bool isUseDocVectorScore,
                                   uint32_t srcResultGlobalIndex) {
        DEBUG_LOG(
            "TOPK CopyOut progress:%d, countGreaterSumLoop:%d ,srcGreaterSumAll:%d, countSmallerSumLoop:%d, "
            "srcSmallerSumAll :%d ,isUseDocVectorScore :%d,srcResultGlobalIndex:%d \n",
            progress, countGreaterSumLoop, srcGreaterSumAll, countSmallerSumLoop, srcSmallerSumAll, isUseDocVectorScore,
            srcResultGlobalIndex);
        if (countGreaterSumLoop > 0) {
            AscendC::LocalTensor<float> greaterOutLocal = outQueueDocVectorScore.DeQue<float>();
            AscendC::LocalTensor<uint32_t> greaterLocationOutLocal = outQueueDocLocation.DeQue<uint32_t>();
            DEBUG_DUMP_TENSOR(greaterOutLocal, 3301, DUMP_DEFAULT_LENGTH);          // 3301 is the log dump marker
            DEBUG_DUMP_TENSOR(greaterLocationOutLocal, 3302, DUMP_DEFAULT_LENGTH);  // 3302 is the log dump marker
            uint32_t index = srcResultGlobalIndex + srcGreaterSumAll;
            // Exact-length writes: a rounded-up overhang of block i overlaps block i+1's write
            // with no ordering between them. Same defect and fix as KernelAggregator::CopyOut.
            DEBUG_LOG("TOPK CopyOut countGreaterSumLoop:%d \n", countGreaterSumLoop);
            AscendC::DataCopyExtParams gsCopy{1, static_cast<uint32_t>(countGreaterSumLoop * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(docVectorScoreGlobal[index], greaterOutLocal, gsCopy);
            AscendC::DataCopyExtParams glCopy{1, static_cast<uint32_t>(countGreaterSumLoop * sizeof(uint32_t)), 0, 0,
                                              0};
            AscendC::DataCopyPad(docLocationGlobal[index], greaterLocationOutLocal, glCopy);
            outQueueDocVectorScore.FreeTensor<float>(greaterOutLocal);
            outQueueDocLocation.FreeTensor<uint32_t>(greaterLocationOutLocal);
        }
        if (countSmallerSumLoop > 0) {
            AscendC::LocalTensor<float> smallerOutLocal = outQueueDocVectorScoreNext.DeQue<float>();
            AscendC::LocalTensor<uint32_t> smallerLocationOutLocal = outQueueDocLocationNext.DeQue<uint32_t>();
            // Same overlap on the smaller partition's packed writes.
            DEBUG_LOG("TOPK CopyOut countSmallerSumLoop:%d \n", countSmallerSumLoop);
            AscendC::DataCopyExtParams ssCopy{1, static_cast<uint32_t>(countSmallerSumLoop * sizeof(float)), 0, 0, 0};
            AscendC::DataCopyPad(docVectorScoreNextGlobal[srcSmallerSumAll], smallerOutLocal, ssCopy);
            AscendC::DataCopyExtParams slCopy{1, static_cast<uint32_t>(countSmallerSumLoop * sizeof(uint32_t)), 0, 0,
                                              0};
            AscendC::DataCopyPad(docLocationNextGlobal[srcSmallerSumAll], smallerLocationOutLocal, slCopy);
            DEBUG_DUMP_TENSOR(smallerOutLocal, 3303, DUMP_DEFAULT_LENGTH);          // 3303 is the log dump marker
            DEBUG_DUMP_TENSOR(smallerLocationOutLocal, 3304, DUMP_DEFAULT_LENGTH);  // 3304 is the log dump marker
            outQueueDocVectorScoreNext.FreeTensor<float>(smallerOutLocal);
            outQueueDocLocationNext.FreeTensor<uint32_t>(smallerLocationOutLocal);
        }
    }

   private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> InQueueDocScore;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> InQueueDocLocation;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueDocVectorScore;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueDocLocation;
    AscendC::GlobalTensor<float> docVectorScoreGlobal;
    AscendC::GlobalTensor<uint32_t> docLocationGlobal;
    AscendC::GlobalTensor<uint32_t> docNumberGlobal;

    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueDocVectorScoreNext;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueDocLocationNext;
    // holds the scores smaller than the pivot
    AscendC::GlobalTensor<float> docVectorScoreNextGlobal;
    // holds the indices of the scores smaller than the pivot
    AscendC::GlobalTensor<uint32_t> docLocationNextGlobal;

    uint32_t m_docNumberShard[DOC_NUMBER_LENGTH] = {0};
    uint32_t m_realDocNum = 0;
    uint32_t m_blockNumber = 0;
    uint32_t m_lastBlockLength = 0;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> maskQueue;
    AscendC::TQue<AscendC::TPosition::VECCALC, 1> maskSmallerQueue;

    AscendC::GlobalTensor<uint32_t> effectiveCountGlobal;
    uint32_t m_topK = 0;
    uint32_t m_topKEarlyQuit = 0;
    uint32_t m_blockDimAggregator = 0;
    // number of data blocks per shard when the aggregator runs multi-core
    uint32_t m_blockNumberShardAggregator = 0;
    uint32_t m_loopCount = 0;
};
