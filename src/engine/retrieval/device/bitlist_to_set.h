#pragma once

#include "kernel_operator.h"
#include "device_common.h"

class BitlistToSet {
   public:
    __aicore__ inline BitlistToSet() {}

    __aicore__ inline void Init(AscendC::TPipe* pipe, __gm__ uint8_t* postings, uint32_t byteSizePerSegment,
                                uint32_t maxPostingLength, __gm__ uint8_t* resultPosting, uint32_t beginIdx,
                                uint32_t endIdx) {
        m_resultElementCount = byteSizePerSegment / sizeof(uint16_t);
        m_repeatedCount = endIdx - beginIdx;
        m_postingAddrsGm.SetGlobalBuffer((__gm__ uint64_t*)postings + beginIdx, endIdx - beginIdx);
        m_dstPostingAddrsGm.SetGlobalBuffer((__gm__ uint64_t*)resultPosting + beginIdx, endIdx - beginIdx);

        m_pipe = pipe;
        m_pipe->InitBuffer(m_queIn, BUFFER_NUM, maxPostingLength);
        m_pipe->InitBuffer(m_queOut, BUFFER_NUM, byteSizePerSegment);
    }

    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < m_repeatedCount; i++) {
            // postingLength is always a multiple of 6
            uint32_t postingLength = *((__gm__ uint32_t*)(m_postingAddrsGm.GetValue(i) + 4));
            DEBUG_LOG("[INFO]] postingLength:%d\n", postingLength);
            uint32_t postingDataByteLength = postingLength / 3;
            uint32_t postingDataElementNum = postingDataByteLength / sizeof(uint16_t);
            const uint32_t& inElementCount = postingDataElementNum;
            DEBUG_LOG("[INFO] postingDataByteLength:%d, elementNum:%d\n", postingDataByteLength, inElementCount);
            AscendC::GlobalTensor<uint16_t> srcGm;
            AscendC::GlobalTensor<uint16_t> dstGm;
            srcGm.SetGlobalBuffer((__gm__ uint16_t*)(m_postingAddrsGm.GetValue(i) + HEADER_BYTE_SIZE),
                                  3 * inElementCount);
            dstGm.SetGlobalBuffer((__gm__ uint16_t*)(m_dstPostingAddrsGm.GetValue(i)), m_resultElementCount);

            CopyIn(srcGm, inElementCount);
            Compute(inElementCount);
            CopyOut(dstGm);
        }
    }

   private:
    __aicore__ inline void CopyIn(const AscendC::GlobalTensor<uint16_t>& srcGm, const uint32_t inElementCount) {
        AscendC::LocalTensor<uint16_t> srcLocal = m_queIn.AllocTensor<uint16_t>();
        DataCopyPadCustom_GM2UB(srcLocal, srcGm,
                                inElementCount * 3);  // 3x length: offset (u32) + data (u16) = 3x per element
        m_queIn.EnQue(srcLocal);
    }

    __aicore__ inline void Compute(const uint32_t inElementCount) {
        AscendC::LocalTensor<uint16_t> srcLocal = m_queIn.DeQue<uint16_t>();
        AscendC::LocalTensor<uint16_t> dstLocal = m_queOut.AllocTensor<uint16_t>();

        AscendC::LocalTensor<uint16_t> srcDataLocal =
            srcLocal[inElementCount * 2];  // offset is u32, so skip 2x elements
        AscendC::LocalTensor<uint32_t> dstOffsetLocal = srcLocal.ReinterpretCast<uint32_t>();
        dstLocal.SetSize(m_resultElementCount);
        // AllocTensor returns a tensor with random contents
        AscendC::Duplicate(dstLocal, (uint16_t)0, m_resultElementCount);
        // NOTE: AscendC::Scatter was a silent no-op on this CANN version (verified:
        // the converted bitset came back all-zero -> filter recall dropped). Replace
        // it with an explicit scalar scatter. offset is the BYTE offset of the u16
        // unit within the bitset (encoder writes unit_index * sizeof(uint16_t)), so
        // the destination element index is offset / sizeof(uint16_t).
        for (uint32_t i = 0; i < inElementCount; i++) {
            uint32_t elemIdx = dstOffsetLocal.GetValue(i) / sizeof(uint16_t);
            dstLocal.SetValue(elemIdx, srcDataLocal.GetValue(i));
        }
        m_queOut.EnQue(dstLocal);
        m_queIn.FreeTensor(srcLocal);
    }

    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<uint16_t>& dstGm) {
        AscendC::LocalTensor<uint16_t> dstLocal = m_queOut.DeQue<uint16_t>();
        DEBUG_DUMP_TENSOR(dstLocal, 146, m_resultElementCount);  // 146 is the log dump marker
        DataCopyPadCustom_UB2GM(dstGm, dstLocal, m_resultElementCount);
        m_queOut.FreeTensor(dstLocal);
    }

   private:
    AscendC::TPipe* m_pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> m_queIn;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> m_queOut;
    AscendC::GlobalTensor<uint64_t> m_postingAddrsGm;
    AscendC::GlobalTensor<uint64_t> m_dstPostingAddrsGm;

    uint32_t m_resultElementCount = 0;
    uint32_t m_repeatedCount = 0;
};
