#pragma once

#include "kernel_operator.h"
#include "../device_common.h"

class BitlistToSet {
   public:
    __aicore__ inline BitlistToSet() {}

    __aicore__ inline void Init(AscendC::TPipe* pipe, __gm__ uint8_t* postings, uint32_t byteSizePerSegment,
                                uint32_t maxPostingLength, __gm__ uint8_t* resultPosting, uint32_t beginIdx,
                                uint32_t endIdx, uint32_t clearSparse, uint32_t ablate, uint32_t useScatter) {
        m_clearSparse = clearSparse != 0;
        m_ablate = ablate;
        m_useScatter = useScatter;
        m_resultElementCount = byteSizePerSegment / sizeof(uint16_t);
        m_repeatedCount = endIdx - beginIdx;
        m_postingAddrsGm.SetGlobalBuffer((__gm__ uint64_t*)postings + beginIdx, endIdx - beginIdx);
        m_dstPostingAddrsGm.SetGlobalBuffer((__gm__ uint64_t*)resultPosting + beginIdx, endIdx - beginIdx);

        m_pipe = pipe;
        m_pipe->InitBuffer(m_queIn, BUFFER_NUM, maxPostingLength);
        m_pipe->InitBuffer(m_queOut, BUFFER_NUM, byteSizePerSegment);
    }

    __aicore__ inline void ProcessOne(uint64_t srcAddr, uint64_t dstAddr, uint32_t postingLength, uint32_t i,
                                      bool packed) {
        // SPARSE_PACKED is 4 bytes per unit ([unitIndex:16][mask:16]), SPARSE_BITMAP 6.
        const uint32_t u16PerUnit = packed ? 2 : 3;
        const uint32_t inElementCount = postingLength / u16PerUnit / sizeof(uint16_t);
        DEBUG_LOG("[INFO]] postingLength:%d\n", postingLength);
        AscendC::GlobalTensor<uint16_t> srcGm;
        AscendC::GlobalTensor<uint16_t> dstGm;
        srcGm.SetGlobalBuffer((__gm__ uint16_t*)(srcAddr + HEADER_BYTE_SIZE), u16PerUnit * inElementCount);
        dstGm.SetGlobalBuffer((__gm__ uint16_t*)dstAddr, m_resultElementCount);

        CopyIn(srcGm, inElementCount, u16PerUnit);
        Compute(inElementCount, i < BUFFER_NUM, packed);
        CopyOut(dstGm, inElementCount, packed);
    }

    __aicore__ inline void Process() {
        uint64_t readsAcc = 0;
        for (uint32_t i = 0; i < m_repeatedCount; i++) {
            const uint64_t srcAddr = m_postingAddrsGm.GetValue(i);
            const uint32_t postingLength = *((__gm__ uint32_t*)(srcAddr + 4));
            // Word 0 of the 8-byte header is [layout:4][hitCount:28].
            const bool packed = (*((__gm__ uint32_t*)srcAddr) >> 28) == SPARSE_PACKED_LAYOUT_CODE;
            const uint64_t dstAddr = m_dstPostingAddrsGm.GetValue(i);
            if (m_ablate == ABLATE_READS_ONLY) {
                readsAcc += srcAddr + dstAddr + postingLength;
                continue;
            }
            ProcessOne(srcAddr, dstAddr, postingLength, i, packed);
        }
        // Never taken -- a source address is never 0 -- but the compiler cannot prove that, so the
        // loads above cannot be folded away. Without this ABLATE=4 would measure an empty loop.
        if (m_ablate == ABLATE_READS_ONLY && readsAcc == 0 && m_repeatedCount > 0) {
            ProcessOne(m_postingAddrsGm.GetValue(0), m_dstPostingAddrsGm.GetValue(0), 6, 0, false);
        }
    }

   private:
    __aicore__ inline void CopyIn(const AscendC::GlobalTensor<uint16_t>& srcGm, const uint32_t inElementCount,
                                  const uint32_t u16PerUnit) {
        AscendC::LocalTensor<uint16_t> srcLocal = m_queIn.AllocTensor<uint16_t>();
        DataCopyPadCustom_GM2UB(srcLocal, srcGm, inElementCount * u16PerUnit);
        m_queIn.EnQue(srcLocal);
    }

    __aicore__ inline void Compute(const uint32_t inElementCount, bool needFullZero, bool packed) {
        AscendC::LocalTensor<uint16_t> srcLocal = m_queIn.DeQue<uint16_t>();
        AscendC::LocalTensor<uint16_t> dstLocal = m_queOut.AllocTensor<uint16_t>();

        AscendC::LocalTensor<uint16_t> srcDataLocal =
            srcLocal[inElementCount * 2];  // offset is u32, so skip 2x elements
        AscendC::LocalTensor<uint32_t> dstOffsetLocal = srcLocal.ReinterpretCast<uint32_t>();
        dstLocal.SetSize(m_resultElementCount);
        // AllocTensor returns random contents, so the first use of each buffer has to zero all of it.
        if ((needFullZero || !m_clearSparse) && m_ablate != ABLATE_DUPLICATE) {
            AscendC::Duplicate(dstLocal, (uint16_t)0, m_resultElementCount);
        }
        // offset is the BYTE offset of the u16 unit within the bitset, so the destination element
        // index is offset / 2.
        //
        // AscendC::Scatter DOES NOT EXIST ON THIS HARDWARE -- do not try it again. __NPU_ARCH__ 2201
        // (910B) selects dav_c220, whose ScatterImpl is ASCENDC_REPORT_NOT_SUPPORT(false, "Scatter").
        if (m_ablate != ABLATE_SCATTER) {
            if (m_useScatter != 0) {
                if (m_useScatter == 2) {
                    AscendC::ShiftRight(dstOffsetLocal, dstOffsetLocal, static_cast<uint32_t>(1), inElementCount);
                    AscendC::PipeBarrier<PIPE_V>();
                }
                AscendC::Scatter(dstLocal, srcDataLocal, dstOffsetLocal,
                                 static_cast<uint32_t>(reinterpret_cast<uint64_t>(dstLocal.GetPhyAddr())),
                                 inElementCount);
            } else if (packed) {
                for (uint32_t i = 0; i < inElementCount; i++) {
                    const uint32_t word = dstOffsetLocal.GetValue(i);
                    dstLocal.SetValue(word >> 16, (uint16_t)word);
                }
            } else {
                for (uint32_t i = 0; i < inElementCount; i++) {
                    uint32_t elemIdx = dstOffsetLocal.GetValue(i) / sizeof(uint16_t);
                    dstLocal.SetValue(elemIdx, srcDataLocal.GetValue(i));
                }
            }
        }
        m_queOut.EnQue(dstLocal);
        if (m_clearSparse) {
            m_pendingSrc = srcLocal;  // CopyOut needs the offsets to undo exactly what was written
        } else {
            m_queIn.FreeTensor(srcLocal);
        }
    }

    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<uint16_t>& dstGm, const uint32_t inElementCount,
                                   bool packed) {
        AscendC::LocalTensor<uint16_t> dstLocal = m_queOut.DeQue<uint16_t>();
        DEBUG_DUMP_TENSOR(dstLocal, 146, m_resultElementCount);  // 146 is the log dump marker
        if (m_ablate != ABLATE_COPYOUT) {
            DataCopyPadCustom_UB2GM(dstGm, dstLocal, m_resultElementCount);
        }
        // NPUR_BITLIST_CLEAR_SPARSE: clear only the units this posting wrote instead of a 16KB fill.
        if (m_clearSparse) {
            AscendC::LocalTensor<uint32_t> dstOffsetLocal = m_pendingSrc.ReinterpretCast<uint32_t>();
            for (uint32_t i = 0; i < inElementCount; i++) {
                const uint32_t word = dstOffsetLocal.GetValue(i);
                dstLocal.SetValue(packed ? (word >> 16) : (word / sizeof(uint16_t)), (uint16_t)0);
            }
            m_queIn.FreeTensor(m_pendingSrc);
        }
        m_queOut.FreeTensor(dstLocal);
    }

   private:
    AscendC::TPipe* m_pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> m_queIn;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> m_queOut;
    AscendC::GlobalTensor<uint64_t> m_postingAddrsGm;
    AscendC::GlobalTensor<uint64_t> m_dstPostingAddrsGm;

    // NPUR_BITLIST_ABLATE: skip one step to price it. Every value but 0 produces WRONG bitsets.
    static constexpr uint32_t ABLATE_SCATTER = 1;
    static constexpr uint32_t ABLATE_COPYOUT = 2;
    static constexpr uint32_t ABLATE_DUPLICATE = 3;
    static constexpr uint32_t ABLATE_READS_ONLY = 4;

    AscendC::LocalTensor<uint16_t> m_pendingSrc;
    uint32_t m_resultElementCount = 0;
    uint32_t m_repeatedCount = 0;
    uint32_t m_ablate = 0;
    uint32_t m_useScatter = 0;
    bool m_clearSparse = false;
};
