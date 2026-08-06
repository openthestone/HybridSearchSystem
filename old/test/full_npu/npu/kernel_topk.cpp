// npu/kernel_topk.cpp — AscendC AIV scalar top-K kernel.
//
// Input:
//   scores_f32[N]   compacted matched scores (from kernel_aggregator)
//   ids_u32[N]      compacted matched doc ids
//   count_u32[1]    N = number of matched docs
//
// Output:
//   topk_scores[K]  top K scores, descending
//   topk_ids[K]     corresponding doc ids
//
// Single-block design: block 0 does serial partial sort over all N entries.
// Sufficient for N ≤ 100K, K ≤ 100 (Phase A.2 testbed). Production path
// would use Tianji's multi-block sample-sort (kernel_topk.h).
//
// Algorithm: min-heap of size K. For each input score, if score > heap_min,
// replace heap_min and sift-down. Final heap = top K (unsorted); host-side
// reverse to descending order.

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t MAX_TOPK = 128;  // cap K for static array sizing
}

class KernelTopkOp {
    TPipe pipe;
    TBuf<TPosition::VECCALC> heap_score_buf_;
    TBuf<TPosition::VECCALC> heap_id_buf_;

    GlobalTensor<float> scoresGM_;
    GlobalTensor<uint32_t> idsGM_;
    GlobalTensor<uint32_t> countGM_;
    GlobalTensor<float> outScoresGM_;
    GlobalTensor<uint32_t> outIdsGM_;

    uint32_t k_;
    uint32_t n_;  // actual matched count

   public:
    __aicore__ inline KernelTopkOp() {}

    __aicore__ inline void Init(GM_ADDR scores, GM_ADDR ids, GM_ADDR count, GM_ADDR out_scores, GM_ADDR out_ids,
                                uint32_t k) {
        scoresGM_.SetGlobalBuffer((__gm__ float*)scores);
        idsGM_.SetGlobalBuffer((__gm__ uint32_t*)ids);
        countGM_.SetGlobalBuffer((__gm__ uint32_t*)count);
        outScoresGM_.SetGlobalBuffer((__gm__ float*)out_scores);
        outIdsGM_.SetGlobalBuffer((__gm__ uint32_t*)out_ids);

        k_ = (k <= MAX_TOPK) ? k : MAX_TOPK;
        n_ = countGM_.GetValue(0);

        pipe.InitBuffer(heap_score_buf_, MAX_TOPK * sizeof(float));
        pipe.InitBuffer(heap_id_buf_, MAX_TOPK * sizeof(uint32_t));
    }

    // Min-heap of size heap_size, rooted at index 0.
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
            return;  // single-block

        LocalTensor<float> heap_s = heap_score_buf_.Get<float>();
        LocalTensor<uint32_t> heap_i = heap_id_buf_.Get<uint32_t>();

        // Initialize heap with first min(k_, n_) elements.
        uint32_t heap_size = 0;
        uint32_t init_count = (k_ < n_) ? k_ : n_;
        for (uint32_t i = 0; i < init_count; ++i) {
            heap_s.SetValue(i, scoresGM_.GetValue(i));
            heap_i.SetValue(i, idsGM_.GetValue(i));
            ++heap_size;
        }
        // Heapify (Floyd): sift-down from last parent to root.
        if (heap_size > 1) {
            int32_t p = (int32_t)((heap_size - 2) / 2);
            for (int32_t i = p; i >= 0; --i) {
                SiftDown(heap_s, heap_i, (uint32_t)i, heap_size);
            }
        }

        // For remaining elements: if score > heap_min, replace root + sift-down.
        for (uint32_t i = init_count; i < n_; ++i) {
            float s = scoresGM_.GetValue(i);
            if (s > heap_s.GetValue(0)) {
                heap_s.SetValue(0, s);
                heap_i.SetValue(0, idsGM_.GetValue(i));
                SiftDown(heap_s, heap_i, 0, heap_size);
            }
        }

        // Heap-sort: extract min K times, place from end (out[k-1] = smallest).
        // After this loop out[0..k-1] = descending (max first).
        uint32_t remaining = heap_size;
        while (remaining > 0) {
            uint32_t out_idx = remaining - 1;  // smallest goes to end
            outScoresGM_.SetValue(out_idx, heap_s.GetValue(0));
            outIdsGM_.SetValue(out_idx, heap_i.GetValue(0));
            // Move last heap element to root, shrink, sift-down.
            --remaining;
            if (remaining > 0) {
                heap_s.SetValue(0, heap_s.GetValue(remaining));
                heap_i.SetValue(0, heap_i.GetValue(remaining));
                SiftDown(heap_s, heap_i, 0, remaining);
            }
        }
        // Pad remaining slots with sentinel (-inf, 0).
        for (uint32_t i = heap_size; i < k_; ++i) {
            outScoresGM_.SetValue(i, -1e30f);
            outIdsGM_.SetValue(i, 0xFFFFFFFFu);
        }
    }
};

extern "C" __global__ __aicore__ void kernel_topk(GM_ADDR scores, GM_ADDR ids, GM_ADDR count, GM_ADDR out_scores,
                                                  GM_ADDR out_ids, uint32_t k) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelTopkOp op;
    op.Init(scores, ids, count, out_scores, out_ids, k);
    op.Process();
}
