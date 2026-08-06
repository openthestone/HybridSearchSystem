// npu/kernel_topk_merge.cpp — single-block merge of B*K partial candidates.
//
// Stage 2 of multi-block topk. Reads partial_scores[B*PARTIAL_STRIDE],
// partial_ids[B*PARTIAL_STRIDE], partial_counts[B*16], outputs top K via
// min-heap + heap-sort.

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t MAX_TOPK = 128;
constexpr uint32_t COUNT_STRIDE = 16;
constexpr uint32_t PARTIAL_STRIDE = ((MAX_TOPK + 15) / 16) * 16;  // = 128
}  // namespace

class KernelTopkMergeOp {
    TPipe pipe;
    TBuf<TPosition::VECCALC> heap_score_buf_;
    TBuf<TPosition::VECCALC> heap_id_buf_;

    GlobalTensor<float> partialScoresGM_;
    GlobalTensor<uint32_t> partialIdsGM_;
    GlobalTensor<uint32_t> partialCountsGM_;
    GlobalTensor<float> outScoresGM_;
    GlobalTensor<uint32_t> outIdsGM_;

    uint32_t k_;
    uint32_t n_blocks_;

   public:
    __aicore__ inline KernelTopkMergeOp() {}

    __aicore__ inline void Init(GM_ADDR partial_scores, GM_ADDR partial_ids, GM_ADDR partial_counts, GM_ADDR out_scores,
                                GM_ADDR out_ids, uint32_t k, uint32_t n_blocks) {
        partialScoresGM_.SetGlobalBuffer((__gm__ float*)partial_scores);
        partialIdsGM_.SetGlobalBuffer((__gm__ uint32_t*)partial_ids);
        partialCountsGM_.SetGlobalBuffer((__gm__ uint32_t*)partial_counts);
        outScoresGM_.SetGlobalBuffer((__gm__ float*)out_scores);
        outIdsGM_.SetGlobalBuffer((__gm__ uint32_t*)out_ids);

        k_ = (k <= MAX_TOPK) ? k : MAX_TOPK;
        n_blocks_ = n_blocks;

        pipe.InitBuffer(heap_score_buf_, MAX_TOPK * sizeof(float));
        pipe.InitBuffer(heap_id_buf_, MAX_TOPK * sizeof(uint32_t));
    }

    __aicore__ inline void SiftDown(LocalTensor<float>& hs, LocalTensor<uint32_t>& hi, uint32_t i, uint32_t heap_size) {
        while (true) {
            uint32_t l = 2 * i + 1;
            uint32_t r = 2 * i + 2;
            uint32_t smallest = i;
            if (l < heap_size && hs.GetValue(l) < hs.GetValue(smallest))
                smallest = l;
            if (r < heap_size && hs.GetValue(r) < hs.GetValue(smallest))
                smallest = r;
            if (smallest == i)
                break;
            float ts = hs.GetValue(i);
            uint32_t ti = hi.GetValue(i);
            hs.SetValue(i, hs.GetValue(smallest));
            hi.SetValue(i, hi.GetValue(smallest));
            hs.SetValue(smallest, ts);
            hi.SetValue(smallest, ti);
            i = smallest;
        }
    }

    __aicore__ inline void Process() {
        if (GetBlockIdx() != 0)
            return;

        LocalTensor<float> heap_s = heap_score_buf_.Get<float>();
        LocalTensor<uint32_t> heap_i = heap_id_buf_.Get<uint32_t>();

        uint32_t heap_size = 0;

        for (uint32_t b = 0; b < n_blocks_; ++b) {
            uint32_t cnt = partialCountsGM_.GetValue(b * COUNT_STRIDE);
            uint32_t base = b * PARTIAL_STRIDE;
            for (uint32_t i = 0; i < cnt; ++i) {
                float s = partialScoresGM_.GetValue(base + i);
                uint32_t id = partialIdsGM_.GetValue(base + i);

                if (heap_size < k_) {
                    uint32_t cur = heap_size;
                    heap_s.SetValue(cur, s);
                    heap_i.SetValue(cur, id);
                    ++heap_size;
                    while (cur > 0) {
                        uint32_t parent = (cur - 1) / 2;
                        if (heap_s.GetValue(cur) < heap_s.GetValue(parent)) {
                            float ts = heap_s.GetValue(cur);
                            uint32_t ti = heap_i.GetValue(cur);
                            heap_s.SetValue(cur, heap_s.GetValue(parent));
                            heap_i.SetValue(cur, heap_i.GetValue(parent));
                            heap_s.SetValue(parent, ts);
                            heap_i.SetValue(parent, ti);
                            cur = parent;
                        } else
                            break;
                    }
                } else if (s > heap_s.GetValue(0)) {
                    heap_s.SetValue(0, s);
                    heap_i.SetValue(0, id);
                    SiftDown(heap_s, heap_i, 0, heap_size);
                }
            }
        }

        uint32_t remaining = heap_size;
        while (remaining > 0) {
            uint32_t out_idx = remaining - 1;
            outScoresGM_.SetValue(out_idx, heap_s.GetValue(0));
            outIdsGM_.SetValue(out_idx, heap_i.GetValue(0));
            --remaining;
            if (remaining > 0) {
                heap_s.SetValue(0, heap_s.GetValue(remaining));
                heap_i.SetValue(0, heap_i.GetValue(remaining));
                SiftDown(heap_s, heap_i, 0, remaining);
            }
        }
        for (uint32_t i = heap_size; i < k_; ++i) {
            outScoresGM_.SetValue(i, -1e30f);
            outIdsGM_.SetValue(i, 0xFFFFFFFFu);
        }
    }
};

extern "C" __global__ __aicore__ void kernel_topk_merge(GM_ADDR partial_scores, GM_ADDR partial_ids,
                                                        GM_ADDR partial_counts, GM_ADDR out_scores, GM_ADDR out_ids,
                                                        uint32_t k, uint32_t n_blocks) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelTopkMergeOp op;
    op.Init(partial_scores, partial_ids, partial_counts, out_scores, out_ids, k, n_blocks);
    op.Process();
}
