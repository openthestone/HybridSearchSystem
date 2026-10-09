#pragma once

#include "kernel_operator.h"
#include "../device_common.h"

class FilterConjunctionOp {
   public:
    __aicore__ inline FilterConjunctionOp() {}
    __aicore__ inline uint32_t Init(AscendC::TPipe* pipe, AscendC::GlobalTensor<uint64_t> postings,
                                    uint64_t& postingsGmOffset, uint32_t segmentLength,
                                    AscendC::GlobalTensor<uint32_t> params, uint32_t tileNum,
                                    AscendC::GlobalTensor<uint16_t> resultStack, uint32_t& stackIdx, bool isfinalOp,
                                    AscendC::GlobalTensor<uint16_t> result) {
        DEBUG_LOG("=================Conjunction===================\n");
        m_segmentLength = segmentLength;
        m_tileNum = tileNum;
        m_tileLength = m_segmentLength / m_tileNum;
        DEBUG_LOG("[INFO] m_segmentLength:%d, m_tileNum:%d, m_tileLength:%d\n", m_segmentLength, m_tileNum,
                  m_tileLength);
        m_pipe = pipe;
        m_pipe->InitBuffer(m_inQueueX, BUFFER_NUM, m_tileLength * sizeof(uint16_t));
        m_postings = postings[postingsGmOffset];
        uint32_t partNum = params.GetValue(0);
        m_pipe->InitBuffer(m_tmpBuffer, partNum * sizeof(uint32_t));
        m_postingNums = m_tmpBuffer.Get<uint32_t>();
        for (uint64_t i = 0; i < partNum; i++) {
            m_postingNums.SetValue(i, params.GetValue(i + 1));  // set the token count
            postingsGmOffset += params.GetValue(i + 1);
        }
        m_postingNums.SetSize(partNum);
        DEBUG_LOG("partNum:%d, postingNums.GetSize() %d\n", partNum, m_postingNums.GetSize());

        // No need for partNum blocks of LocalTensor; two are enough:
        // block 1 holds each part's OR result, block 2 holds the running AND result.
        m_pipe->InitBuffer(m_outQueue, BUFFER_NUM, 2 * m_tileLength * sizeof(uint16_t));
        m_repeatTimes = m_tileLength / m_mask;
        m_orIdx = 0;
        m_andIdx = m_tileLength;
        DEBUG_LOG("resultStack:%d, stackIdx: %d, \n", resultStack.GetSize() / m_segmentLength, stackIdx);
        // stack position for the result
        if (isfinalOp) {
            m_resultGm = result;
        } else {
            m_resultGm.SetGlobalBuffer((__gm__ uint16_t*)resultStack[stackIdx * segmentLength].GetPhyAddr(),
                                       segmentLength);
        }

        stackIdx += 1;

        return partNum + 1;
    }
    __aicore__ inline void Process() {
        DEBUG_LOG("loopCount: %d \n", m_tileNum);
        for (int32_t i = 0; i < m_tileNum; i++) {
            InnerProcess(i);
        }
    }

   private:
    __aicore__ inline void InnerProcess(int32_t progress) {
        DEBUG_LOG("InnerProcess: %d \n", progress);
        uint32_t postingsGmOffset = 0;
        AscendC::LocalTensor<uint16_t> result = m_outQueue.AllocTensor<uint16_t>();
        // the second block is filled with 1s, used for AND
        AscendC::Duplicate(result[m_andIdx], MAX_UINT16_VALUE, m_tileLength);
        m_outQueue.EnQue(result);
        for (uint64_t partIdx = 0; partIdx < m_postingNums.GetSize(); partIdx++) {
            uint32_t postingNum = m_postingNums.GetValue(partIdx);
            DEBUG_LOG("[INFO] or partIdx: %d , postingNum:%d, postingsGmOffset:%d \n", partIdx, postingNum,
                      postingsGmOffset);
            if (postingNum == 0) {
                ERROR_LOG("ERROR: InerOrProcess, postingNum is 0\n");
                return;
            }
            for (uint32_t i = 0; i < postingNum; i++) {
                CopyIn(progress, postingsGmOffset + i);
                ComputeOr(i);
            }
            postingsGmOffset += postingNum;
            // AND part_i with the previous result
            ComputeAnd();
        }
        CopyOut(progress);
    }

    __aicore__ inline void CopyIn(int32_t progress, uint32_t postingsGmOffset) {
        AscendC::GlobalTensor<uint16_t> tmpGm;
        tmpGm.SetGlobalBuffer((__gm__ uint16_t*)m_postings.GetValue(postingsGmOffset), m_segmentLength);
        AscendC::LocalTensor<uint16_t> xLocal = m_inQueueX.AllocTensor<uint16_t>();
        AscendC::DataCopy(xLocal, tmpGm[progress * m_tileLength], m_tileLength);
        m_inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void ComputeOr(uint32_t idx) {
        AscendC::LocalTensor<uint16_t> xLocal = m_inQueueX.DeQue<uint16_t>();
        AscendC::LocalTensor<uint16_t> result = m_outQueue.DeQue<uint16_t>();
        if (idx == 0) {
            AscendC::Copy(result[m_orIdx], xLocal, m_mask, m_repeatTimes, {1, 1, 8, 8});
        } else {
            // subsequent iterations: OR with the previous result
            AscendC::Or(result[m_orIdx], result[m_orIdx], xLocal, m_mask, m_repeatTimes, m_binaryParams);
        }
        m_outQueue.EnQue(result);
        m_inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void ComputeAnd() {
        uint32_t partNum = m_postingNums.GetSize();
        AscendC::LocalTensor<uint16_t> result = m_outQueue.DeQue<uint16_t>();
        AscendC::And(result[m_andIdx], result[m_andIdx], result[m_orIdx], m_mask, m_repeatTimes, m_binaryParams);
        m_outQueue.EnQue(result);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<uint16_t> result = m_outQueue.DeQue<uint16_t>();
        // finally copy out the AND result
        AscendC::DataCopy(m_resultGm[progress * m_tileLength], result[m_andIdx], m_tileLength);
        m_outQueue.FreeTensor(result);
    }

   private:
    AscendC::TPipe* m_pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> m_inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> m_outQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> m_tmpBuffer;
    AscendC::GlobalTensor<uint16_t> m_resultGm;
    AscendC::GlobalTensor<uint64_t> m_postings;
    AscendC::BinaryRepeatParams m_binaryParams{1, 1, 1, 8, 8, 8};
    uint32_t m_segmentLength{};  // length computed of each core
    uint32_t m_tileNum{};        // split data into xx tiles for each core
    uint32_t m_tileLength{};
    uint32_t m_orIdx{};     // index where the OR result is stored
    uint32_t m_andIdx{};    // index where the AND result is stored
    uint32_t m_mask = 128;  // 256/sizeof(uint16_t)
    uint32_t m_repeatTimes{};

    AscendC::LocalTensor<uint32_t> m_postingNums;
};
