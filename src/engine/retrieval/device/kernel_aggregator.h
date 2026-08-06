#pragma once

#include <float.h>
#include "kernel_operator.h"
#include "device_common.h"
const uint16_t MASK_GLOBAL_BIT_COUNT = 16;
// GM has a 64B-overwrite issue during multi-core processing, so each shard's aggregate count is stored every 16 uint32
// slots. 20 = max 20 concurrent cores.
const uint32_t EFFECTIVE_OFFSET = 16;
const uint32_t EFFECTIVE_COUNT = EFFECTIVE_OFFSET * 20;
const uint32_t BUFFER_SIZE_CALCULATE = 32;
const uint32_t COPY_OFFSET = 16;
const uint32_t FOUR_BYTE_LENGTH = 4;
const uint32_t GATHER_MASK_PARAMETER = 8;

class KernelAggregator {
   public:
    __aicore__ inline KernelAggregator() {}

    __aicore__ inline void Init(AscendC::TPipe* pipe, GM_ADDR filterResults, GM_ADDR docVectorScores,
                                GM_ADDR docLocationInDeviceConst, GM_ADDR docLocationInDeviceResult,
                                GM_ADDR effectiveCountDevice, uint32_t docNumber, uint32_t blockNumber,
                                uint32_t blockLength, uint32_t lastBlockLength) {
        // blockLength must be a multiple of 256
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);  // set the default kernel type to pure AIV
        int64_t blockIdx = AscendC::GetBlockIdx();
        int64_t blockNum = AscendC::GetBlockNum();
        if (blockNum > 1) {
            // multi-core processing
            uint32_t blockNumberShard = blockNumber / blockNum;
            // tail core
            if (blockIdx == blockNum - 1) {
                blockNumber = blockNumber - blockNumberShard * blockIdx;
                docNumber = (blockNumber - 1) * blockLength + lastBlockLength;
            } else {
                blockNumber = blockNumberShard;
                docNumber = blockNumber * blockLength;
                lastBlockLength = blockLength;
            }
            uint32_t docNumberOffset = blockIdx * blockNumberShard * blockLength;
            docVectorScores = docVectorScores + docNumberOffset * FOUR_BYTE_LENGTH;
            docLocationInDeviceConst = docLocationInDeviceConst + docNumberOffset * FOUR_BYTE_LENGTH;
            docLocationInDeviceResult = docLocationInDeviceResult + docNumberOffset * FOUR_BYTE_LENGTH;
            filterResults = filterResults + docNumberOffset / MASK_GLOBAL_BIT_COUNT * sizeof(uint16_t);
            DEBUG_LOG(
                "Aggregator__i = blockIdx%d, blockNumber:%d\n, docNumber:%d\n,lastBlockLength:%d\n,"
                " docNumberOffset:%d\n",
                blockIdx, blockNumber, docNumber, lastBlockLength, docNumberOffset);
        }
        m_blockIdx = blockIdx;
        m_blockNumber = blockNumber;
        m_blockNumberSubtractOne = blockNumber - 1;
        if (blockNumber > 1) {
            m_blockLength = blockLength;
        } else {
            m_blockLength = lastBlockLength;
        }
        m_lastBlockLength = lastBlockLength;
        m_blockLengthMask = m_blockLength / MASK_GLOBAL_BIT_COUNT;
        docVectorScoreGlobal.SetGlobalBuffer((__gm__ float*)docVectorScores, docNumber);
        docLocationConstGlobal.SetGlobalBuffer((__gm__ uint32_t*)docLocationInDeviceConst, docNumber);
        docLocationResultGlobal.SetGlobalBuffer((__gm__ uint32_t*)docLocationInDeviceResult, docNumber);
        effectiveCountGlobal.SetGlobalBuffer((__gm__ uint32_t*)effectiveCountDevice, EFFECTIVE_COUNT);
        maskGlobal.SetGlobalBuffer((__gm__ uint16_t*)filterResults,
                                   (docNumber + MASK_GLOBAL_BIT_COUNT - 1) / MASK_GLOBAL_BIT_COUNT);
        m_pipe = pipe;
        uint32_t bufferSize = (m_blockLength + BUFFER_SIZE_CALCULATE - 1) / BUFFER_SIZE_CALCULATE *
                              BUFFER_SIZE_CALCULATE * FOUR_BYTE_LENGTH;
        DEBUG_LOG("Aggregator queBind bytesize:%d \n", bufferSize);
        m_pipe->InitBuffer(inQueueDocVectorScore, BUFFER_NUM, bufferSize);
        m_pipe->InitBuffer(outQueueDocVectorScore, BUFFER_NUM, bufferSize);
        m_pipe->InitBuffer(inQueueDocLocationConst, BUFFER_NUM, bufferSize);
        m_pipe->InitBuffer(outQueueDocLocationResult, BUFFER_NUM, bufferSize);
        m_pipe->InitBuffer(inQueueMask, BUFFER_NUM, bufferSize / MASK_GLOBAL_BIT_COUNT);
    }

    __aicore__ inline void Process() {
        if (!m_needWorking) {
            effectiveCountGlobal.SetValue(0, 0);
            return;
        }
        uint32_t effectiveCountAll = 0;
        for (uint32_t i = 0; i < m_blockNumber; i++) {
            uint32_t realDataLength;
            CopyIn(i, realDataLength);
            uint32_t effectiveCount = Compute(i, realDataLength);
            CopyOut(i, effectiveCount, effectiveCountAll);
            effectiveCountAll += effectiveCount;
            if (effectiveCount > 0) {
                DEBUG_LOG("Aggregator__i:%d,effectiveCount: %d, effectiveCountAll: %d  \n", i, effectiveCount,
                          effectiveCountAll);
            }
        }
        effectiveCountGlobal.SetValue(m_blockIdx * EFFECTIVE_OFFSET, effectiveCountAll);
    }

   private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t& realDataLength) {
        DEBUG_LOG("Aggregator______CopyIn, progress=%d \n", progress);
        AscendC::LocalTensor<float> srcLocal = inQueueDocVectorScore.AllocTensor<float>();
        AscendC::LocalTensor<uint32_t> docLocationConstLocal = inQueueDocLocationConst.AllocTensor<uint32_t>();
        AscendC::LocalTensor<uint16_t> srcMaskLocal = inQueueMask.AllocTensor<uint16_t>();
        uint32_t copyDataLength = m_blockLength;
        uint32_t copyMaskDataLength = m_blockLengthMask;
        if (progress == m_blockNumberSubtractOne) {
            // align the tail block
            realDataLength = m_lastBlockLength;
            uint32_t offset =
                realDataLength % COPY_OFFSET;  // a multiple of 8 floats is needed for 32B alignment; round up
            if (offset != 0) {
                copyDataLength = realDataLength + (COPY_OFFSET - offset);
            }
            copyMaskDataLength = (copyDataLength + MASK_GLOBAL_BIT_COUNT - 1) / MASK_GLOBAL_BIT_COUNT;
            offset = copyMaskDataLength % COPY_OFFSET;
            if (offset != 0) {
                copyMaskDataLength = copyMaskDataLength + (COPY_OFFSET - offset);
            }
        } else {
            realDataLength = m_blockLength;
        }
        DEBUG_LOG("Aggregator______CopyIn, copyDataLength=%d, copyMaskDataLength=%d \n", copyDataLength,
                  copyMaskDataLength);
        uint32_t index = progress * m_blockLength;
        AscendC::DataCopy(srcLocal, docVectorScoreGlobal[index], copyDataLength);
        AscendC::DataCopy(docLocationConstLocal, docLocationConstGlobal[index], copyDataLength);
        DEBUG_DUMP_TENSOR(srcLocal, 2000, DUMP_DEFAULT_LENGTH);               // 2000 is the log dump marker
        DEBUG_DUMP_TENSOR(docLocationConstLocal, 2001, DUMP_DEFAULT_LENGTH);  // 2001 is the log dump marker
        AscendC::DataCopy(srcMaskLocal, maskGlobal[index / MASK_GLOBAL_BIT_COUNT], copyMaskDataLength);
        DEBUG_DUMP_TENSOR(srcMaskLocal, 2002, DUMP_DEFAULT_LENGTH);  // 2002 is the log dump marker
        inQueueDocVectorScore.EnQue<float>(srcLocal);
        inQueueDocLocationConst.EnQue<uint32_t>(docLocationConstLocal);
        inQueueMask.EnQue<uint16_t>(srcMaskLocal);
    }

    __aicore__ inline uint32_t Compute(uint32_t progress, uint32_t realDataLength) {
        DEBUG_LOG("Aggregator______Compute: %d \n", progress);
        AscendC::LocalTensor<float> srcLocal = inQueueDocVectorScore.DeQue<float>();
        AscendC::LocalTensor<float> dstLocal = outQueueDocVectorScore.AllocTensor<float>();

        AscendC::LocalTensor<uint32_t> docLocationConstLocal = inQueueDocLocationConst.DeQue<uint32_t>();
        AscendC::LocalTensor<uint32_t> docLocationResultLocal = outQueueDocLocationResult.AllocTensor<uint32_t>();
        AscendC::LocalTensor<uint16_t> srcMaskLocal = inQueueMask.DeQue<uint16_t>();
        AscendC::LocalTensor<uint32_t> srcGatherMaskLocal = srcMaskLocal.ReinterpretCast<uint32_t>();
        uint64_t effectiveCount = 0;
        uint64_t effectiveLocationCount = 0;
        AscendC::GatherMask(dstLocal, srcLocal, srcGatherMaskLocal, true, realDataLength,
                            {1, 1, GATHER_MASK_PARAMETER, GATHER_MASK_PARAMETER}, effectiveCount);
        AscendC::GatherMask(docLocationResultLocal, docLocationConstLocal, srcGatherMaskLocal, true, realDataLength,
                            {1, 1, GATHER_MASK_PARAMETER, GATHER_MASK_PARAMETER}, effectiveLocationCount);
        if (effectiveCount > 0) {
            outQueueDocVectorScore.EnQue<float>(dstLocal);
            outQueueDocLocationResult.EnQue<uint32_t>(docLocationResultLocal);
        } else {
            outQueueDocVectorScore.FreeTensor(dstLocal);
            outQueueDocLocationResult.FreeTensor(docLocationResultLocal);
        }

        inQueueDocVectorScore.FreeTensor(srcLocal);
        inQueueDocLocationConst.FreeTensor(docLocationConstLocal);
        inQueueMask.FreeTensor(srcMaskLocal);
        return (uint32_t)effectiveCount;
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t effectiveCount, uint32_t effectiveCountIndex) {
        if (effectiveCount == 0) {
            return;
        }
        AscendC::LocalTensor<float> dstLocal = outQueueDocVectorScore.DeQue<float>();
        AscendC::LocalTensor<uint32_t> docLocationResultLocal = outQueueDocLocationResult.DeQue<uint32_t>();
        uint32_t offset = effectiveCount % COPY_OFFSET;
        if (offset != 0) {
            effectiveCount = effectiveCount + (COPY_OFFSET - offset);
        }
        AscendC::DataCopy(docVectorScoreGlobal[effectiveCountIndex], dstLocal, effectiveCount);
        AscendC::DataCopy(docLocationResultGlobal[effectiveCountIndex], docLocationResultLocal, effectiveCount);
        outQueueDocVectorScore.FreeTensor(dstLocal);
        outQueueDocLocationResult.FreeTensor(docLocationResultLocal);
    }

   private:
    AscendC::TPipe* m_pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueDocVectorScore;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueDocVectorScore;
    AscendC::GlobalTensor<float> docVectorScoreGlobal;

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueMask;
    AscendC::GlobalTensor<uint16_t> maskGlobal;

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueDocLocationConst;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueDocLocationResult;
    AscendC::GlobalTensor<uint32_t> docLocationConstGlobal;
    AscendC::GlobalTensor<uint32_t> docLocationResultGlobal;
    AscendC::GlobalTensor<uint32_t> effectiveCountGlobal;

    uint32_t m_blockNumber = 0;
    uint32_t m_blockNumberSubtractOne = 0;
    uint32_t m_blockLength = 0;
    uint32_t m_blockLengthMask = 0;
    uint32_t m_lastBlockLength = 0;
    uint32_t m_blockIdx = 0;
    bool m_needWorking = true;
};
