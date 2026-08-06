// common/cpu_ref.h — CPU brute-force reference: full-corpus score + filter + top-K.
//
// Output is the ground truth for verifying NPU kernels (Phase A.2).
// Multi-threaded via OpenMP.

#pragma once

#include <algorithm>
#include <cstdint>
#include <queue>
#include <vector>

#include "dataset_hw.h"
#include "expr.h"
#include "timer.h"

namespace full_npu {

struct DocScore {
    float score;
    uint32_t doc_id;
};

// Comparator for min-heap (lowest score at top, so we can pop to keep top-K).
struct DocScoreMinHeapCmp {
    bool operator()(const DocScore& a, const DocScore& b) const {
        return a.score > b.score;  // smaller score = higher priority (top)
    }
};

struct CpuBruteResult {
    std::vector<DocScore> topk;  // sorted descending by score
    uint64_t match_count;        // total docs passing filter
    StageLatency timing;
};

// Run a single (query_vector, expr) over a subset of docs.
// Output: top-K matching docs by score.
inline CpuBruteResult CpuBruteForce(const DatasetHW& ds, uint64_t doc_subset, const float* query_vec,
                                    const BooleanExpr& expr, uint32_t topk, int threads) {
    CpuBruteResult r;
    r.match_count = 0;
    if (doc_subset == 0)
        doc_subset = ds.DocNum(0);

    const uint32_t vdim = ds.VectorDim();

    // Pass 1: filter + score, collect matches into a vector.
    // For top-K we use a min-heap bounded at topk.
    std::priority_queue<DocScore, std::vector<DocScore>, DocScoreMinHeapCmp> heap;

    Timer t_total;
    t_total.Start();

    // Filter stage timing.
    Timer t_filter;
    t_filter.Start();
    // (filter and score are fused in this loop, but we approximate filter
    //  as the bitmap eval portion by re-timing separately if needed.)
    double filter_acc = 0;

#pragma omp parallel num_threads(threads)
    {
        // Thread-local heap, merged at end.
        std::priority_queue<DocScore, std::vector<DocScore>, DocScoreMinHeapCmp> local;
        uint64_t local_match_count = 0;

#pragma omp for schedule(static) nowait
        for (uint64_t d = 0; d < doc_subset; ++d) {
            const uint64_t* bm = ds.Bitmap(d);
            if (EvalExpr(expr, bm)) {
                ++local_match_count;
                // Score: dot product FP32.
                const float* v = ds.Vector(d);
                float s = 0.0f;
                for (uint32_t k = 0; k < vdim; ++k)
                    s += v[k] * query_vec[k];

                if (local.size() < topk) {
                    local.push({s, (uint32_t)d});
                } else if (s > local.top().score) {
                    local.pop();
                    local.push({s, (uint32_t)d});
                }
            }
        }
// Merge thread-local into global.
#pragma omp critical
        {
            r.match_count += local_match_count;
            while (!local.empty()) {
                if (heap.size() < topk) {
                    heap.push(local.top());
                } else if (local.top().score > heap.top().score) {
                    heap.pop();
                    heap.push(local.top());
                }
                local.pop();
            }
        }
    }
    r.timing.filter_ms = t_filter.StopMs();

    // Drain heap (descending order).
    r.topk.resize(heap.size());
    for (size_t i = heap.size(); i > 0; --i) {
        r.topk[i - 1] = heap.top();
        heap.pop();
    }
    r.timing.total_ms = t_total.StopMs();
    r.timing.score_ms = 0;  // fused with filter
    r.timing.agg_ms = 0;
    r.timing.topk_ms = 0;
    return r;
}

// Recall@K vs reference (same K assumed).
inline double RecallVsRef(const std::vector<DocScore>& cand, const std::vector<DocScore>& ref) {
    if (ref.empty())
        return 1.0;
    std::vector<uint32_t> ref_ids;
    ref_ids.reserve(ref.size());
    for (auto& d : ref)
        ref_ids.push_back(d.doc_id);
    std::sort(ref_ids.begin(), ref_ids.end());
    uint32_t hit = 0;
    for (auto& d : cand) {
        if (std::binary_search(ref_ids.begin(), ref_ids.end(), d.doc_id))
            ++hit;
    }
    return static_cast<double>(hit) / static_cast<double>(ref.size());
}

}  // namespace full_npu
