// Tianji-style AIV filter kernel: scalar RPN eval (multi-segment stride).
// Same eval logic as simple_filter but each AIV block iterates over multiple
// segments via blockNum stride (matches Tianji BitmapTextFilter structure).
//
// Simplified port of TianjiEngineRecall BitmapTextFilter:
//   - no CONJ op
//   - flat posting layout (no pointer indirection)
//   - BIT_SET only (no BIT_LIST preprocessing)
//   - scalar per-unit eval (vector op variant WIP)
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kMaxStackDepth = 64;
}

class KernelTianjiFilter {
    TPipe pipe;
    TBuf<TPosition::VECCALC> stack_buf_;

    GlobalTensor<uint16_t> postings_;
    GlobalTensor<uint32_t> rpn_;
    GlobalTensor<uint16_t> result_;

    uint32_t seg_u16_count_;
    uint32_t segments_num_;
    uint32_t rpn_len_;

   public:
    __aicore__ inline KernelTianjiFilter() {}

    __aicore__ inline void Init(GM_ADDR postings, GM_ADDR rpn, GM_ADDR result, uint32_t seg_u16_count, uint32_t rpn_len,
                                uint32_t segments_num) {
        seg_u16_count_ = seg_u16_count;
        segments_num_ = segments_num;
        rpn_len_ = rpn_len;

        postings_.SetGlobalBuffer((__gm__ uint16_t*)postings);
        rpn_.SetGlobalBuffer((__gm__ uint32_t*)rpn);
        result_.SetGlobalBuffer((__gm__ uint16_t*)result);

        pipe.InitBuffer(stack_buf_, kMaxStackDepth * sizeof(uint16_t));
    }

    __aicore__ inline void Process() {
        uint32_t block_id = GetBlockIdx();
        uint32_t block_num = GetBlockNum();
        LocalTensor<uint16_t> stack = stack_buf_.Get<uint16_t>();

        for (uint32_t seg = block_id; seg < segments_num_; seg += block_num) {
            for (uint32_t u = 0; u < seg_u16_count_; ++u) {
                int sp = 0;
                for (uint32_t i = 0; i < rpn_len_; ++i) {
                    uint32_t slot = rpn_.GetValue(i);
                    uint32_t op = slot >> 28;
                    uint32_t arg = slot & 0x0FFFFFFFu;
                    if (op == 0) {  // POSTING
                        uint64_t off = ((uint64_t)arg * segments_num_ + seg) * seg_u16_count_ + u;
                        uint16_t v = postings_.GetValue(off);
                        stack.SetValue(sp++, v);
                    } else if (op == 1) {  // AND
                        uint16_t b = stack.GetValue(--sp);
                        uint16_t a = stack.GetValue(--sp);
                        stack.SetValue(sp++, (uint16_t)(a & b));
                    } else if (op == 2) {  // OR
                        uint16_t b = stack.GetValue(--sp);
                        uint16_t a = stack.GetValue(--sp);
                        stack.SetValue(sp++, (uint16_t)(a | b));
                    } else if (op == 3) {  // NOT
                        uint16_t a = stack.GetValue(--sp);
                        stack.SetValue(sp++, (uint16_t)(~a));
                    }
                }
                uint16_t r = (sp == 1) ? stack.GetValue(0) : 0;
                result_.SetValue((uint64_t)seg * seg_u16_count_ + u, r);
            }
        }
    }
};

extern "C" __global__ __aicore__ void bitmap_text_filter(GM_ADDR postings, GM_ADDR rpn, GM_ADDR result,
                                                         uint32_t seg_u16_count, uint32_t rpn_len,
                                                         uint32_t segments_num) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelTianjiFilter op;
    op.Init(postings, rpn, result, seg_u16_count, rpn_len, segments_num);
    op.Process();
}
