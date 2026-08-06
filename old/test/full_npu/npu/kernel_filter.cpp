// npu/kernel_filter.cpp — AscendC AIV filter kernel: forward bitmap RPN.
//
// Each AIV block handles DOCS_PER_BLOCK docs (offset by block_offset_blocks).
// Bitmap layout matches dataset_HW.bin:
//   bitmap[doc_id][stride words], 64 bits/word, tag_id → word = tag_id>>6, bit = tag_id&63
//
// RPN tokens packed as uint32:
//   bits 31-28: opcode (0=TAG, 1=NOT_TAG, 2=AND_N, 3=OR_N)
//   bits 27-0:  arg (tag_id for TAG/NOT_TAG, child_count for AND_N/OR_N)
//
// Host passes `block_offset_blocks` so multiple launches can chunk large inputs.
//   doc_start = (block_offset_blocks + blockIdx) * DOCS_PER_BLOCK
//
// Output: result_u32[doc_id] = 1 if doc matches expr, else 0.

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kMaxStackDepth = 64;
constexpr uint32_t DOCS_PER_BLOCK_FILTER = 16;
constexpr uint32_t OP_TAG = 0;
constexpr uint32_t OP_NOT_TAG = 1;
constexpr uint32_t OP_AND_N = 2;
constexpr uint32_t OP_OR_N = 3;
}  // namespace

class KernelFilterOp {
    TPipe pipe;
    TBuf<TPosition::VECCALC> stack_buf_;

    GlobalTensor<uint32_t> bitmapGM_;  // 2 uint32 per bitmap word
    GlobalTensor<uint32_t> rpnGM_;
    GlobalTensor<uint32_t> resultGM_;  // 4 bytes per doc (uint32 for safe access)

    uint32_t stride_;  // 64-bit words per doc bitmap
    uint32_t rpn_len_;
    uint32_t total_docs_;
    uint32_t block_offset_blocks_;

   public:
    __aicore__ inline KernelFilterOp() {}

    __aicore__ inline void Init(GM_ADDR bitmap, GM_ADDR rpn, GM_ADDR result, uint32_t stride, uint32_t rpn_len,
                                uint32_t total_docs, uint32_t block_offset_blocks) {
        stride_ = stride;
        rpn_len_ = rpn_len;
        total_docs_ = total_docs;
        block_offset_blocks_ = block_offset_blocks;

        bitmapGM_.SetGlobalBuffer((__gm__ uint32_t*)bitmap);
        rpnGM_.SetGlobalBuffer((__gm__ uint32_t*)rpn);
        resultGM_.SetGlobalBuffer((__gm__ uint32_t*)result);

        pipe.InitBuffer(stack_buf_, kMaxStackDepth * sizeof(uint32_t));
    }

    __aicore__ inline void Process() {
        uint32_t global_block = GetBlockIdx() + block_offset_blocks_;
        uint32_t doc_start = global_block * DOCS_PER_BLOCK_FILTER;
        if (doc_start >= total_docs_)
            return;

        LocalTensor<uint32_t> stack = stack_buf_.Get<uint32_t>();

        for (uint32_t d_off = 0; d_off < DOCS_PER_BLOCK_FILTER; ++d_off) {
            uint32_t doc_id = doc_start + d_off;
            if (doc_id >= total_docs_)
                break;

            // 2 uint32 per uint64 bitmap word.
            uint64_t base_u32 = (uint64_t)doc_id * stride_ * 2u;
            int sp = 0;

            for (uint32_t i = 0; i < rpn_len_; ++i) {
                uint32_t slot = rpnGM_.GetValue(i);
                uint32_t op = slot >> 28;
                uint32_t arg = slot & 0x0FFFFFFFu;

                if (op == OP_TAG || op == OP_NOT_TAG) {
                    uint32_t word_idx = arg >> 6;
                    uint32_t bit_idx = arg & 63u;
                    uint32_t lo = bitmapGM_.GetValue(base_u32 + (uint64_t)word_idx * 2u);
                    uint32_t hi = bitmapGM_.GetValue(base_u32 + (uint64_t)word_idx * 2u + 1u);
                    uint32_t v;
                    if (bit_idx < 32u) {
                        v = (lo >> bit_idx) & 1u;
                    } else {
                        v = (hi >> (bit_idx - 32u)) & 1u;
                    }
                    if (op == OP_NOT_TAG)
                        v ^= 1u;
                    stack.SetValue(sp++, v);
                } else if (op == OP_AND_N) {
                    uint32_t acc = 1;
                    for (uint32_t c = 0; c < arg; ++c) {
                        uint32_t x = stack.GetValue(--sp);
                        acc &= x;
                        if (!acc) {
                            for (uint32_t j = c + 1; j < arg; ++j)
                                --sp;
                            break;
                        }
                    }
                    stack.SetValue(sp++, acc);
                } else if (op == OP_OR_N) {
                    uint32_t acc = 0;
                    for (uint32_t c = 0; c < arg; ++c) {
                        uint32_t x = stack.GetValue(--sp);
                        acc |= x;
                        if (acc) {
                            for (uint32_t j = c + 1; j < arg; ++j)
                                --sp;
                            break;
                        }
                    }
                    stack.SetValue(sp++, acc);
                }
            }

            uint32_t r = (sp == 1) ? stack.GetValue(0) : 0;
            resultGM_.SetValue(doc_id, r);
        }
    }
};

extern "C" __global__ __aicore__ void kernel_filter(GM_ADDR bitmap, GM_ADDR rpn, GM_ADDR result, uint32_t stride,
                                                    uint32_t rpn_len, uint32_t total_docs,
                                                    uint32_t block_offset_blocks) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelFilterOp op;
    op.Init(bitmap, rpn, result, stride, rpn_len, total_docs, block_offset_blocks);
    op.Process();
}
