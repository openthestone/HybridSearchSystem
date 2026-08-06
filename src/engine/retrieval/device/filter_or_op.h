#pragma once

#include "kernel_operator.h"
#include "device_common.h"

class FilterOrOp {
   public:
    __aicore__ inline FilterOrOp() {}
    // postingsGmOffset: index of the next posting to use
    // segmentLength: number of uint16_t elements per segment
    __aicore__ inline uint32_t Init(AscendC::TPipe* pipe, AscendC::GlobalTensor<uint64_t> postings,
                                    uint64_t& postingsGmOffset, uint32_t segmentLength,
                                    AscendC::GlobalTensor<uint32_t> params, uint32_t tileNum,
                                    AscendC::GlobalTensor<uint16_t> resultStack, uint32_t& stackIdx, bool isfinalOp,
                                    AscendC::GlobalTensor<uint16_t> result) {
        DEBUG_LOG("=================OR===================\n");
        DEBUG_LOG("kernel_%d \n", AscendC::GetBlockIdx());
        m_segmentLength = segmentLength;
        m_tileNum = tileNum;
        m_tileLength = m_segmentLength / m_tileNum;
        DEBUG_LOG("[INFO] m_segmentLength:%d, m_tileNum:%d, m_tileLength:%d\n", m_segmentLength, m_tileNum,
                  m_tileLength);
        m_pipe = pipe;
        m_pipe->InitBuffer(m_inQueueX, BUFFER_NUM, m_tileLength * sizeof(uint16_t));
        m_pipe->InitBuffer(m_outQueue, BUFFER_NUM, m_tileLength * sizeof(uint16_t));
        DEBUG_LOG("[INFO] params size: %d \n", params.GetSize());
        uint32_t postingNum = params.GetValue(0);
        uint32_t stackNum = params.GetValue(1);
        m_pipe->InitBuffer(m_targetsBuffer, (postingNum + stackNum) * sizeof(uint16_t*));
        // validate the number of postings
        DEBUG_LOG("[INFO] postingNum:%d, postings.GetSize():%d, postingsGmOffset:%d \n", postingNum, postings.GetSize(),
                  postingsGmOffset);
        // validate the number of stack entries
        DEBUG_LOG("[INFO] stackNum:%d, resultStack.GetSize():%d, stackIdx:%d, \n", stackNum,
                  resultStack.GetSize() / m_segmentLength, stackIdx);
        // gather the postings and stack entries to compute
        m_targets = m_targetsBuffer.Get<uint64_t>();
        uint32_t idx = 0;
        for (uint32_t i = 0; i < postingNum; i++) {
            m_targets.SetValue(idx, postings.GetValue(postingsGmOffset + i));
            idx++;
            DEBUG_LOG("[DEBUG] add postingGm %p\n", postings.GetValue(postingsGmOffset + i));
        }
        for (uint32_t i = 0; i < stackNum; i++) {
            m_targets.SetValue(idx, (uint64_t)(resultStack[m_segmentLength * (stackIdx - 1 - i)].GetPhyAddr()));
            idx++;
            DEBUG_LOG("[DEBUG] add stacksGm %p\n",
                      (uint64_t)(resultStack[m_segmentLength * (stackIdx - 1 - i)].GetPhyAddr()));
        }
        m_targets.SetSize(idx);
        // advance the posting and stack indices
        postingsGmOffset += postingNum;
        stackIdx -= stackNum;

        // stack position for the result
        if (isfinalOp) {
            m_resultGm = result;
        } else {
            m_resultGm.SetGlobalBuffer((__gm__ uint16_t*)resultStack[stackIdx * segmentLength].GetPhyAddr(),
                                       segmentLength);
        }
        stackIdx += 1;
        return FILTER_OR_PARM_NUM;
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
        AscendC::LocalTensor<uint16_t> result = m_outQueue.AllocTensor<uint16_t>();
        AscendC::Duplicate(result, (uint16_t)0, result.GetSize());
        m_outQueue.EnQue(result);
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<uint16_t, AscendC::MaskMode::COUNTER>(m_tileLength);
        for (uint32_t i = 0; i < m_targets.GetSize(); i++) {
            CopyIn(progress, i);
            ComputeOr(progress);
            DEBUG_DUMP_TENSOR(result, 31, m_tileLength);  // 31 is the log dump marker
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
        CopyOut(progress);
    }

    __aicore__ inline void CopyIn(int32_t progress, uint32_t targetsGmOffset) {
        AscendC::GlobalTensor<uint16_t> tmpGm;
        tmpGm.SetGlobalBuffer((__gm__ uint16_t*)m_targets.GetValue(targetsGmOffset), m_segmentLength);
        AscendC::LocalTensor<uint16_t> xLocal = m_inQueueX.AllocTensor<uint16_t>();
        AscendC::DataCopy(xLocal, tmpGm[progress * m_tileLength], m_tileLength);
        m_inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void ComputeOr(int32_t progress) {
        AscendC::LocalTensor<uint16_t> xLocal = m_inQueueX.DeQue<uint16_t>();
        AscendC::LocalTensor<uint16_t> result = m_outQueue.DeQue<uint16_t>();
        AscendC::Or<uint16_t, false>(result, result, xLocal, AscendC::MASK_PLACEHOLDER, 1, m_binaryParams);
        m_outQueue.EnQue(result);
        m_inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<uint16_t> result = m_outQueue.DeQue<uint16_t>();
        AscendC::DataCopy(m_resultGm[progress * m_tileLength], result, m_tileLength);
        m_outQueue.FreeTensor(result);
    }

   private:
    AscendC::TPipe* m_pipe{};
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> m_inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> m_outQueue;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> m_targetsBuffer;
    AscendC::GlobalTensor<uint16_t> m_resultGm;
    AscendC::LocalTensor<uint64_t> m_targets;  // actually the target GM addresses
    AscendC::BinaryRepeatParams m_binaryParams{1, 1, 1, 8, 8, 8};
    uint32_t m_segmentLength{};  // length computed of each core
    uint32_t m_tileNum{};        // split data into xx tiles for each core
    uint32_t m_tileLength{};
    uint32_t m_postingSize{};
};
