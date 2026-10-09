#pragma once

#include "kernel_operator.h"
#include "../device_common.h"

class FilterNotOp {
   public:
    __aicore__ inline FilterNotOp() {}
    __aicore__ inline uint32_t Init(AscendC::TPipe* pipe, AscendC::GlobalTensor<uint64_t> postings,
                                    uint64_t& postingsGmOffset, uint32_t segmentLength,
                                    AscendC::GlobalTensor<uint32_t> params, uint32_t tileNum,
                                    AscendC::GlobalTensor<uint16_t> resultStack, uint32_t& stackIdx, bool isfinalOp,
                                    AscendC::GlobalTensor<uint16_t> result) {
        DEBUG_LOG("=================NOT===================\n");
        m_segmentLength = segmentLength;
        m_tileNum = tileNum;
        m_tileLength = m_segmentLength / m_tileNum;
        m_pipe = pipe;
        m_pipe->InitBuffer(m_inQueueSrc, BUFFER_NUM, m_tileLength * sizeof(uint16_t));
        m_pipe->InitBuffer(m_outQueueDst, BUFFER_NUM, m_tileLength * sizeof(uint16_t));
        DEBUG_LOG("[INFO] m_segmentLength:%d, m_tileNum:%d, m_tileLength:%d\n", m_segmentLength, m_tileNum,
                  m_tileLength);
        uint32_t postingNum = params.GetValue(0);
        uint32_t stackNum = params.GetValue(1);
        // upstream guarantees postingNum + stackNum == 1
        if (postingNum == 1) {
            if (postings.GetSize() - postingsGmOffset < postingNum) {
                ERROR_LOG("ERROR: the number of remaining postings is not enough, remain:%d, needed: %d\n",
                          (postings.GetSize() - postingsGmOffset), postingNum);
            }
            m_srcGlobal.SetGlobalBuffer((__gm__ uint16_t*)postings.GetValue(postingsGmOffset), m_segmentLength);
            postingsGmOffset += postingNum;
        }
        if (stackNum == 1) {
            if (stackIdx < stackNum) {
                ERROR_LOG("ERROR: the number of remaining stackNum is not enough, remain:%d \n", stackIdx);
            }
            m_srcGlobal.SetGlobalBuffer((__gm__ uint16_t*)(resultStack[m_segmentLength * (stackIdx - 1)].GetPhyAddr()),
                                        m_segmentLength);
            stackIdx -= stackNum;
        }
        // stack position for the result
        if (isfinalOp) {
            m_dstGlobal = result;
        } else {
            m_dstGlobal.SetGlobalBuffer((__gm__ uint16_t*)resultStack[(stackIdx)*m_segmentLength].GetPhyAddr(),
                                        m_segmentLength);
        }
        stackIdx += 1;
        return FILTER_NOT_PARM_NUM;
    }
    __aicore__ inline void Process() {
        // loop count need to be doubled, due to double buffer
        DEBUG_LOG("loopCount: %d \n", m_tileNum);
        // tiling strategy, pipeline parallel
        AscendC::SetMaskCount();
        AscendC::SetVectorMask<uint16_t, AscendC::MaskMode::COUNTER>(m_tileLength);
        for (int32_t i = 0; i < m_tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
    }

   private:
    __aicore__ inline void CopyIn(int32_t progress) {
        AscendC::LocalTensor<uint16_t> srcLocal = m_inQueueSrc.AllocTensor<uint16_t>();
        AscendC::DataCopy(srcLocal, m_srcGlobal[progress * m_tileLength], m_tileLength);
        m_inQueueSrc.EnQue(srcLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        AscendC::LocalTensor<uint16_t> srcLocal = m_inQueueSrc.DeQue<uint16_t>();
        AscendC::LocalTensor<uint16_t> dstLocal = m_outQueueDst.AllocTensor<uint16_t>();
        AscendC::Not<uint16_t, false>(dstLocal, srcLocal, AscendC::MASK_PLACEHOLDER, 1, m_unaryRepeatParams);
        DEBUG_DUMP_TENSOR(srcLocal, 28, m_tileLength);  // 28 is the log dump marker
        DEBUG_DUMP_TENSOR(dstLocal, 29, m_tileLength);  // 29 is the log dump marker

        m_outQueueDst.EnQue<uint16_t>(dstLocal);
        m_inQueueSrc.FreeTensor(srcLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<uint16_t> dstLocal = m_outQueueDst.DeQue<uint16_t>();
        AscendC::DataCopy(m_dstGlobal[progress * m_tileLength], dstLocal, m_tileLength);
        m_outQueueDst.FreeTensor(dstLocal);
    }

   private:
    AscendC::TPipe* m_pipe{};
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> m_inQueueSrc;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> m_outQueueDst;
    AscendC::GlobalTensor<uint16_t> m_srcGlobal;
    AscendC::GlobalTensor<uint16_t> m_dstGlobal;
    AscendC::UnaryRepeatParams m_unaryRepeatParams{1, 1, 8, 8};
    uint32_t m_segmentLength{};  // length computed of each core
    uint32_t m_tileNum{};        // split data into xx tiles for each core
    uint32_t m_tileLength{};
};
