// npu/kernel_topk_fused.cpp — AscendC AIV scalar top-K fused variant.
//
// Reads aggregator's per-block layout directly:
//   scores_f32[M]        per-block compacted (slot [b*16, b*16+n))
//   ids_u32[M]           per-block compacted
//   count_pblk[B*16]     count[b*16] = n for block b (B = total_blocks)
//
// Single-block serial scan over all per-block slots, same min-heap.
// Avoids agg→topk D2H+reupload roundtrip in FusedPipeline.

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t MAX_TOPK = 128;
constexpr uint32_t COUNT_STRIDE = 16;
constexpr uint32_t DOCS_PER_BLOCK_AGG = 16;
}  // namespace

class KernelTopkFusedOp {
    TPipe pipe;
    TBuf<TPosition::VECCALC> heap_score_buf_;
    TBuf<TPosition::VECCALC> heap_id_buf_;

    GlobalTensor<float> scoresGM_;
    GlobalTensor<uint32_t> idsGM_;
    GlobalTensor<uint32_t> countPblkGM_;
    GlobalTensor<float> outScoresGM_;
    GlobalTensor<uint32_t> outIdsGM_;

    uint32_t k_;
    uint32_t total_blocks_;

   public:
    __aicore__ inline KernelTopkFusedOp() {}

    __aicore__ inline void Init(GM_ADDR scores, GM_ADDR ids, GM_ADDR count_pblk, GM_ADDR out_scores, GM_ADDR out_ids,
                                uint32_t k, uint32_t total_blocks) {
        scoresGM_.SetGlobalBuffer((__gm__ float*)scores);
        idsGM_.SetGlobalBuffer((__gm__ uint32_t*)ids);
        countPblkGM_.SetGlobalBuffer((__gm__ uint32_t*)count_pblk);
        outScoresGM_.SetGlobalBuffer((__gm__ float*)out_scores);
        outIdsGM_.SetGlobalBuffer((__gm__ uint32_t*)out_ids);

        k_ = (k <= MAX_TOPK) ? k : MAX_TOPK;
        total_blocks_ = total_blocks;

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

        for (uint32_t b = 0; b < total_blocks_; ++b) {
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

extern "C" __global__ __aicore__ void kernel_topk_fused(GM_ADDR scores, GM_ADDR ids, GM_ADDR count_pblk,
                                                        GM_ADDR out_scores, GM_ADDR out_ids, uint32_t k,
                                                        uint32_t total_blocks) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelTopkFusedOp op;
    op.Init(scores, ids, count_pblk, out_scores, out_ids, k, total_blocks);
    op.Process();
}
