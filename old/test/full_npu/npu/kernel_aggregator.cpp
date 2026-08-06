// npu/kernel_aggregator.cpp — AscendC AIV scalar compaction kernel.
//
// Input:
//   scores_f32[M]   dot(query, docs[i])
//   filter_u32[M]   1 if doc matches expr, else 0 (from kernel_filter)
//
// Output:
//   out_doc_ids[M]  original doc ids for matched docs (per-block compacted)
//   out_scores[M]   their scores (per-block compacted)
//   count_u32[B*S]  per-block matched count, stride S=16 to avoid cache-line
//                   race when concurrent AIV blocks write to consecutive slots
//                   (silent-drop bug observed at S=1).
//
// Each block handles DOCS_PER_BLOCK docs, writes compacted results into its
// own output slot [block_id * DOCS_PER_BLOCK, block_id * DOCS_PER_BLOCK + n).
// Host reduces per-block counts and concatenates compacted output.

#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t DOCS_PER_BLOCK_AGG = 16;
constexpr uint32_t COUNT_STRIDE = 16;  // pad to 64B (one cache line) per block
}  // namespace

class KernelAggregatorOp {
    TPipe pipe;
    TBuf<TPosition::VECCALC> local_doc_ids_buf_;
    TBuf<TPosition::VECCALC> local_scores_buf_;

    GlobalTensor<float> scoresGM_;
    GlobalTensor<uint32_t> filterGM_;
    GlobalTensor<uint32_t> outDocIdsGM_;
    GlobalTensor<float> outScoresGM_;
    GlobalTensor<uint32_t> countGM_;

    uint32_t total_docs_;
    uint32_t block_offset_blocks_;

   public:
    __aicore__ inline KernelAggregatorOp() {}

    __aicore__ inline void Init(GM_ADDR scores, GM_ADDR filter, GM_ADDR out_doc_ids, GM_ADDR out_scores, GM_ADDR count,
                                uint32_t total_docs, uint32_t block_offset_blocks) {
        total_docs_ = total_docs;
        block_offset_blocks_ = block_offset_blocks;

        scoresGM_.SetGlobalBuffer((__gm__ float*)scores);
        filterGM_.SetGlobalBuffer((__gm__ uint32_t*)filter);
        outDocIdsGM_.SetGlobalBuffer((__gm__ uint32_t*)out_doc_ids);
        outScoresGM_.SetGlobalBuffer((__gm__ float*)out_scores);
        countGM_.SetGlobalBuffer((__gm__ uint32_t*)count);

        pipe.InitBuffer(local_doc_ids_buf_, DOCS_PER_BLOCK_AGG * sizeof(uint32_t));
        pipe.InitBuffer(local_scores_buf_, DOCS_PER_BLOCK_AGG * sizeof(float));
    }

    __aicore__ inline void Process() {
        uint32_t global_block = GetBlockIdx() + block_offset_blocks_;
        uint32_t doc_start = global_block * DOCS_PER_BLOCK_AGG;
        if (doc_start >= total_docs_)
            return;

        LocalTensor<uint32_t> local_ids = local_doc_ids_buf_.Get<uint32_t>();
        LocalTensor<float> local_sc = local_scores_buf_.Get<float>();

        uint32_t local_count = 0;
        for (uint32_t d_off = 0; d_off < DOCS_PER_BLOCK_AGG; ++d_off) {
            uint32_t doc_id = doc_start + d_off;
            if (doc_id >= total_docs_)
                break;
            uint32_t f = filterGM_.GetValue(doc_id);
            if (f != 0u) {
                local_ids.SetValue(local_count, doc_id);
                local_sc.SetValue(local_count, scoresGM_.GetValue(doc_id));
                ++local_count;
            }
        }

        // Per-block output slot: [doc_start, doc_start + local_count).
        for (uint32_t i = 0; i < local_count; ++i) {
            outDocIdsGM_.SetValue(doc_start + i, local_ids.GetValue(i));
            outScoresGM_.SetValue(doc_start + i, local_sc.GetValue(i));
        }
        // Strided count slot to avoid cache-line races.
        countGM_.SetValue(global_block * COUNT_STRIDE, local_count);
    }
};

extern "C" __global__ __aicore__ void kernel_aggregator(GM_ADDR scores, GM_ADDR filter, GM_ADDR out_doc_ids,
                                                        GM_ADDR out_scores, GM_ADDR count, uint32_t total_docs,
                                                        uint32_t block_offset_blocks) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelAggregatorOp op;
    op.Init(scores, filter, out_doc_ids, out_scores, count, total_docs, block_offset_blocks);
    op.Process();
}
