#pragma once

#include "kernel_operator.h"
#include "../device_common.h"

class FilterOrOp {
   public:
    __aicore__ inline FilterOrOp() {}
    // postingsGmOffset: index of the next posting to use
    // segmentLength: number of uint16_t elements per segment
    __aicore__ inline uint32_t Init(AscendC::TPipe* pipe, AscendC::GlobalTensor<uint64_t> postings,
                                    uint64_t& postingsGmOffset, uint32_t segmentLength,
                                    AscendC::GlobalTensor<uint32_t> params, uint32_t tileNum,
                                    AscendC::GlobalTensor<uint16_t> resultStack, uint32_t& stackIdx, bool isfinalOp,
                                    AscendC::GlobalTensor<uint16_t> result, uint32_t filterFlags) {
        m_flags = filterFlags;
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
        DEBUG_LOG("[INFO] postingNum:%d, postings.GetSize():%d, postingsGmOffset:%d \n", postingNum, postings.GetSize(),
                  postingsGmOffset);
        DEBUG_LOG("[INFO] stackNum:%d, resultStack.GetSize():%d, stackIdx:%d, \n", stackNum,
                  resultStack.GetSize() / m_segmentLength, stackIdx);
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
            const uint64_t target = m_targets.GetValue(i);
            // Bit 0 set: still the encoded (offset, mask) form. Every address here is
            // aclrtMalloc'd, so the bit is free, and stack entries are never tagged.
            if (target == EMPTY_OPERAND_TAG) {
                continue;  // all-zero posting; ORing it changes nothing
            }
            if ((target & SPARSE_OPERAND_TAG) != 0) {
                ApplySparse(progress, target);
                continue;
            }
            CopyIn(progress, i);
            ComputeOr(progress);
            DEBUG_DUMP_TENSOR(result, 31, m_tileLength);  // 31 is the log dump marker
        }
        AscendC::SetMaskNorm();
        AscendC::ResetMask();
        CopyOut(progress);
    }

    // ORing a zero is the identity, so only the operand's K set units are touched. Requires
    // tileNum == 1, which CreateContextData sets; the host only tags operands when that holds.
    __aicore__ inline void ApplySparse(int32_t progress, uint64_t taggedAddr) {
        const uint64_t addr = taggedAddr & ~SPARSE_OPERAND_TAG;
        // SPARSE_PACKED is one uint32 per unit; SPARSE_BITMAP a uint32 offset plus a uint16 mask.
        const bool packed = (m_flags & FILTER_FLAG_SPARSE_PACKED) != 0;
        const uint32_t bytesPerUnit = packed ? 4 : 6;
        // Same header the conversion kernel reads: word 1 is the posting's byte length.
        const uint32_t k = *((__gm__ uint32_t*)(addr + 4)) / bytesPerUnit;
        if (k == 0) {
            return;
        }
        // NPUR_OR_ABLATE: measurement only. FLOOR returns before either queue is touched, keeping
        // m_outQueue balanced against InnerProcess and CopyOut.
        const uint32_t ablate = (m_flags & FILTER_ABLATE_MASK) >> FILTER_ABLATE_SHIFT;
        if (ablate == OR_ABLATE_FLOOR) {
            return;
        }
        // In uint16 elements: 2 per unit packed, 3 per unit as separate offset and mask sections.
        const uint32_t inElementCount = k * (bytesPerUnit / sizeof(uint16_t));
        // FILTER_FLAG_OR_ALIGNED_COPY rounds the count up to a whole block so the copy never walks
        // a scalar tail. Safe: the placement loop stops at k, and FinishAdd pads the region by 32.
        const uint32_t blockElems = BLOCK_SIZE / sizeof(uint16_t);
        const uint32_t copyCount = (m_flags & FILTER_FLAG_OR_ALIGNED_COPY) != 0
                                       ? ((inElementCount + blockElems - 1) / blockElems) * blockElems
                                       : inElementCount;
        AscendC::GlobalTensor<uint16_t> srcGm;
        srcGm.SetGlobalBuffer((__gm__ uint16_t*)(addr + HEADER_BYTE_SIZE), copyCount);
        AscendC::LocalTensor<uint16_t> xLocal = m_inQueueX.AllocTensor<uint16_t>();
        if (ablate != OR_ABLATE_COPY) {
            if (copyCount != inElementCount || (inElementCount % blockElems) == 0) {
                AscendC::DataCopy(xLocal, srcGm, copyCount);
            } else {
                DataCopyPadCustom_GM2UB(xLocal, srcGm, inElementCount);
            }
        }
        m_inQueueX.EnQue(xLocal);

        AscendC::LocalTensor<uint16_t> srcLocal = m_inQueueX.DeQue<uint16_t>();
        AscendC::LocalTensor<uint16_t> result = m_outQueue.DeQue<uint16_t>();
        AscendC::LocalTensor<uint32_t> offsets = srcLocal.ReinterpretCast<uint32_t>();
        if (packed) {
            const uint32_t tileBase = static_cast<uint32_t>(progress) * m_tileLength;
            const bool noBound = m_tileNum == 1 && (m_flags & FILTER_FLAG_OR_NO_TILE_CHECK) != 0;
            if (ablate == 0) {
                for (uint32_t i = 0; i < k; i++) {
                    const uint32_t word = offsets.GetValue(i);
                    const uint32_t elemIdx = word >> 16;  // unit index, not a byte offset
                    if (!noBound && (elemIdx < tileBase || elemIdx >= tileBase + m_tileLength)) {
                        continue;
                    }
                    const uint32_t local = noBound ? elemIdx : elemIdx - tileBase;
                    result.SetValue(local, (uint16_t)(result.GetValue(local) | (uint16_t)word));
                }
            }
            m_outQueue.EnQue(result);
            m_inQueueX.FreeTensor(srcLocal);
            return;
        }
        AscendC::LocalTensor<uint16_t> values = srcLocal[k * 2];  // offsets are uint32, so 2x elements
        // Raising tileNum keeps working: the operand is re-read per tile, which is the `else` below.
        const uint32_t tileBase = static_cast<uint32_t>(progress) * m_tileLength;
        if (m_tileNum == 1 && (m_flags & FILTER_FLAG_OR_NO_TILE_CHECK) != 0) {
            // tileNum == 1, so the bound below is provably dead -- but it arrives at runtime in a
            // GM struct, so the compiler cannot fold it away.
            if (ablate == 0) {
                for (uint32_t i = 0; i < k; i++) {
                    // offset is the BYTE offset of the uint16 unit within the bitset, as the encoder writes it
                    const uint32_t elemIdx = offsets.GetValue(i) / sizeof(uint16_t);
                    result.SetValue(elemIdx, (uint16_t)(result.GetValue(elemIdx) | values.GetValue(i)));
                }
            }
        } else {
            if (ablate == 0) {
                for (uint32_t i = 0; i < k; i++) {
                    const uint32_t elemIdx = offsets.GetValue(i) / sizeof(uint16_t);
                    if (elemIdx < tileBase || elemIdx >= tileBase + m_tileLength) {
                        continue;
                    }
                    const uint32_t local = elemIdx - tileBase;
                    result.SetValue(local, (uint16_t)(result.GetValue(local) | values.GetValue(i)));
                }
            }
        }
        m_outQueue.EnQue(result);
        m_inQueueX.FreeTensor(srcLocal);
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
    uint32_t m_flags{};          // TextFilterContextData::flags, see FilterFlag
    uint32_t m_tileLength{};
    uint32_t m_postingSize{};
};
