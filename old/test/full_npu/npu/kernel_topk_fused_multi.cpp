// npu/kernel_topk_fused_multi.cpp — multi-block top-K, partial heaps + merge.
//
// Stage 1 (kernel_topk_fused_multi): B parallel blocks. Each block scans a
// chunk of aggregator segments, maintains local min-heap of size K. Writes K
// (score, id) pairs per block to partial_scores/partial_ids, count to
// partial_counts.
//
// Stage 2 (kernel_topk_merge): single block scans B*K candidates, picks
// final top K via min-heap, heap-sorts to descending.
//
// Layouts (same input as kernel_topk_fused):
//   scores_f32[M]        per-block compacted (slot [b*16, b*16+n))
//   ids_u32[M]           per-block compacted
//   count_pblk[B*16]     count[b*16] = n for aggregator block b
//
// Outputs (cache-line-strided to dodge 910B3 AIV race):
//   partial_scores[B * PARTIAL_STRIDE]   PARTIAL_STRIDE = next-16-multiple of K
//   partial_ids[B * PARTIAL_STRIDE]
//   partial_counts[B * 16]               count at bid*16
//
// Merge output:
//   out_scores[K]        descending
//   out_ids[K]
//
// Multi-block chunked launch (≤32 blocks per launch). Then single merge block.

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t MAX_TOPK = 128;
constexpr uint32_t COUNT_STRIDE = 16;
constexpr uint32_t DOCS_PER_BLOCK_AGG = 16;
// PARTIAL_STRIDE: K rounded up to a multiple of 16 (u32 = cache-line aligned).
// Guarantees each block's partial output occupies whole cache lines → no race.
constexpr uint32_t PARTIAL_STRIDE = ((MAX_TOPK + 15) / 16) * 16;  // = 128
}  // namespace

class KernelTopkFusedMultiOp {
    TPipe pipe;
    TBuf<TPosition::VECCALC> heap_score_buf_;
    TBuf<TPosition::VECCALC> heap_id_buf_;

    GlobalTensor<float> scoresGM_;
    GlobalTensor<uint32_t> idsGM_;
    GlobalTensor<uint32_t> countPblkGM_;
    GlobalTensor<float> partialScoresGM_;
    GlobalTensor<uint32_t> partialIdsGM_;
    GlobalTensor<uint32_t> partialCountsGM_;

    uint32_t k_;
    uint32_t total_blocks_;
    uint32_t blocks_per_block_;  // segments per topk block

   public:
    __aicore__ inline KernelTopkFusedMultiOp() {}

    __aicore__ inline void Init(GM_ADDR scores, GM_ADDR ids, GM_ADDR count_pblk, GM_ADDR partial_scores,
                                GM_ADDR partial_ids, GM_ADDR partial_counts, uint32_t k, uint32_t total_blocks,
                                uint32_t blocks_per_block) {
        scoresGM_.SetGlobalBuffer((__gm__ float*)scores);
        idsGM_.SetGlobalBuffer((__gm__ uint32_t*)ids);
        countPblkGM_.SetGlobalBuffer((__gm__ uint32_t*)count_pblk);
        partialScoresGM_.SetGlobalBuffer((__gm__ float*)partial_scores);
        partialIdsGM_.SetGlobalBuffer((__gm__ uint32_t*)partial_ids);
        partialCountsGM_.SetGlobalBuffer((__gm__ uint32_t*)partial_counts);

        k_ = (k <= MAX_TOPK) ? k : MAX_TOPK;
        total_blocks_ = total_blocks;
        blocks_per_block_ = blocks_per_block;

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
        LocalTensor<float> heap_s = heap_score_buf_.Get<float>();
        LocalTensor<uint32_t> heap_i = heap_id_buf_.Get<uint32_t>();

        uint32_t bid = GetBlockIdx();
        uint32_t seg_start = bid * blocks_per_block_;
        uint32_t seg_end = seg_start + blocks_per_block_;
        if (seg_end > total_blocks_)
            seg_end = total_blocks_;

        uint32_t heap_size = 0;

        for (uint32_t b = seg_start; b < seg_end; ++b) {
            uint32_t n = countPblkGM_.GetValue(b * COUNT_STRIDE);
            uint32_t slot_base = b * DOCS_PER_BLOCK_AGG;
            for (uint32_t i = 0; i < n; ++i) {
                float s = scoresGM_.GetValue(slot_base + i);
                uint32_t id = idsGM_.GetValue(slot_base + i);

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

        // Write heap (unordered) to partial output slots. Stride by PARTIAL_STRIDE
        // so each block's partial occupies whole cache lines (no AIV write race).
        uint32_t out_base = bid * PARTIAL_STRIDE;
        for (uint32_t i = 0; i < heap_size; ++i) {
            partialScoresGM_.SetValue(out_base + i, heap_s.GetValue(i));
            partialIdsGM_.SetValue(out_base + i, heap_i.GetValue(i));
        }
        for (uint32_t i = heap_size; i < k_; ++i) {
            partialScoresGM_.SetValue(out_base + i, -1e30f);
            partialIdsGM_.SetValue(out_base + i, 0xFFFFFFFFu);
        }
        // Count write also cache-line strided.
        partialCountsGM_.SetValue(bid * COUNT_STRIDE, heap_size);
    }
};

extern "C" __global__ __aicore__ void kernel_topk_fused_multi(GM_ADDR scores, GM_ADDR ids, GM_ADDR count_pblk,
                                                              GM_ADDR partial_scores, GM_ADDR partial_ids,
                                                              GM_ADDR partial_counts, uint32_t k, uint32_t total_blocks,
                                                              uint32_t blocks_per_block) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelTopkFusedMultiOp op;
    op.Init(scores, ids, count_pblk, partial_scores, partial_ids, partial_counts, k, total_blocks, blocks_per_block);
    op.Process();
}
