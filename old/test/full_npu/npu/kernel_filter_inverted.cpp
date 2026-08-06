// npu/kernel_filter_inverted.cpp — AscendC AIV filter kernel: inverted
// (tag-major) postings, Tianji-style.
//
// Each AIV block handles one segment (16 docs).
// Postings layout (compact, per-query):
//   postings[tag_idx][seg]  uint16_t  (bit i = doc seg*16+i has this tag)
//   seg count = ceil(M / 16)
//
// RPN tokens packed as uint32:
//   bits 31-28: opcode (0=TAG, 1=NOT_TAG, 2=AND_N, 3=OR_N)
//   bits 27-0:  arg (tag_idx into compact postings array for TAG/NOT_TAG,
//                     child_count for AND_N/OR_N)
//
// Output: result_u32[doc_id] = 1 if doc matches expr, else 0.
//
// Per-block optimization: pre-fetch all n_tags segment u16s into UB once,
// then 16 doc RPN evals read from cached UB array (no repeated GM traffic).
// Without this cache, kernel re-reads same GM addr per doc — 16× amplification.

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kMaxStackDepth = 64;
constexpr uint32_t kMaxTagsCached = 256;
constexpr uint32_t OP_TAG = 0;
constexpr uint32_t OP_NOT_TAG = 1;
constexpr uint32_t OP_AND_N = 2;
constexpr uint32_t OP_OR_N = 3;
}  // namespace

class KernelFilterInvertedOp {
    TPipe pipe;
    TBuf<TPosition::VECCALC> stack_buf_;
    TBuf<TPosition::VECCALC> tagcache_buf_;

    GlobalTensor<uint16_t> postingsGM_;
    GlobalTensor<uint32_t> rpnGM_;
    GlobalTensor<uint32_t> resultGM_;

    uint32_t segments_;
    uint32_t rpn_len_;
    uint32_t total_docs_;
    uint32_t block_offset_blocks_;
    uint32_t n_tags_;

   public:
    __aicore__ inline KernelFilterInvertedOp() {}

    __aicore__ inline void Init(GM_ADDR postings, GM_ADDR rpn, GM_ADDR result, uint32_t segments, uint32_t rpn_len,
                                uint32_t total_docs, uint32_t block_offset_blocks, uint32_t n_tags) {
        segments_ = segments;
        rpn_len_ = rpn_len;
        total_docs_ = total_docs;
        block_offset_blocks_ = block_offset_blocks;
        n_tags_ = n_tags;

        postingsGM_.SetGlobalBuffer((__gm__ uint16_t*)postings);
        rpnGM_.SetGlobalBuffer((__gm__ uint32_t*)rpn);
        resultGM_.SetGlobalBuffer((__gm__ uint32_t*)result);

        pipe.InitBuffer(stack_buf_, kMaxStackDepth * sizeof(uint16_t));
        pipe.InitBuffer(tagcache_buf_, kMaxTagsCached * sizeof(uint16_t));
    }

    __aicore__ inline void Process() {
        uint32_t seg = GetBlockIdx() + block_offset_blocks_;
        if (seg >= segments_)
            return;

        uint32_t doc_base = seg * 16;

        LocalTensor<uint16_t> stack = stack_buf_.Get<uint16_t>();
        LocalTensor<uint16_t> cache = tagcache_buf_.Get<uint16_t>();

        // Pre-fetch all n_tags segment values into UB cache.
        uint32_t n = (n_tags_ < kMaxTagsCached) ? n_tags_ : kMaxTagsCached;
        for (uint32_t t = 0; t < n; ++t) {
            cache.SetValue(t, postingsGM_.GetValue((uint64_t)t * segments_ + seg));
        }

        for (uint32_t u = 0; u < 16; ++u) {
            uint32_t doc_id = doc_base + u;
            if (doc_id >= total_docs_)
                break;

            uint16_t bit_mask = (uint16_t)(1u << u);
            int sp = 0;

            for (uint32_t i = 0; i < rpn_len_; ++i) {
                uint32_t slot = rpnGM_.GetValue(i);
                uint32_t op = slot >> 28;
                uint32_t arg = slot & 0x0FFFFFFFu;

                if (op == OP_TAG || op == OP_NOT_TAG) {
                    uint16_t v = (arg < n) ? cache.GetValue(arg) : (uint16_t)0;
                    uint16_t bit = (v & bit_mask) ? 1 : 0;
                    if (op == OP_NOT_TAG)
                        bit ^= 1;
                    stack.SetValue(sp++, bit);
                } else if (op == OP_AND_N) {
                    uint16_t acc = 1;
                    for (uint32_t c = 0; c < arg; ++c) {
                        uint16_t x = stack.GetValue(--sp);
                        acc &= x;
                        if (!acc) {
                            for (uint32_t j = c + 1; j < arg; ++j)
                                --sp;
                            break;
                        }
                    }
                    stack.SetValue(sp++, acc);
                } else if (op == OP_OR_N) {
                    uint16_t acc = 0;
                    for (uint32_t c = 0; c < arg; ++c) {
                        uint16_t x = stack.GetValue(--sp);
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

            uint16_t r = (sp == 1) ? stack.GetValue(0) : 0;
            resultGM_.SetValue(doc_id, r ? 1u : 0u);
        }
    }
};

extern "C" __global__ __aicore__ void kernel_filter_inverted(GM_ADDR postings, GM_ADDR rpn, GM_ADDR result,
                                                             uint32_t segments, uint32_t rpn_len, uint32_t total_docs,
                                                             uint32_t block_offset_blocks, uint32_t n_tags) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelFilterInvertedOp op;
    op.Init(postings, rpn, result, segments, rpn_len, total_docs, block_offset_blocks, n_tags);
    op.Process();
}
