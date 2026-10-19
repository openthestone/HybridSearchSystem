#include "schedule.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "../host/cpu_affinity.h"
#include "../query/filter_setup.h"
#include "acl/acl.h"
#include "flags.h"
#include "src/full_recall/retrieval/query/query_node_imp.h"
#include "src/full_recall/retrieval/searcher/full_recall_searcher.h"
#include "src/utils/logger.h"

namespace npur_harness {

bool RunPipeline(RunContext& rc) {
    // Only ONE batch is on the cards at a time and Extract touches no device, so nothing races.
    struct ShardCtx {
        std::unique_ptr<NpuRetrieval::FullRecallSearcher> searcher;
        std::vector<std::unique_ptr<NpuRetrieval::QueryNode>> owned;
        std::vector<NpuRetrieval::AggrDeviceResult> deviceResults;
        std::vector<std::shared_ptr<NpuRetrieval::FullRecallResult>> shardResults;
        char ok{1};
    };
    std::vector<ShardCtx> ctx[2];
    for (int b = 0; b < 2; ++b)
        ctx[b].resize(rc.nShards);
    std::atomic<bool> pipeFailed{false};

    auto deviceStage = [&](int buf, size_t start, size_t end) {
        auto body = [&](uint32_t s) {
            ShardCtx& c = ctx[buf][s];
            PinThreadToCard(s);
            aclrtSetDevice(rc.shardDevices[s]);
            c.owned.clear();
            c.searcher = std::make_unique<NpuRetrieval::FullRecallSearcher>(rc.shardDevices[s], rc.shardTables[s]);
            for (size_t q = start; q < end; ++q) {
                const FilterNode* ast = rc.haveFilters ? rc.filters.astFor(q) : nullptr;
                NpuRetrieval::QueryNode* tree = BuildQueryNode(ast, FLAGS_posting_field, c.owned);
                c.searcher->AddQuery(tree, rc.queries[q % rc.Q], rc.topks[q]);
            }
            c.ok = c.searcher->BatchSearchDevice(FLAGS_vec_field, /*isMultiShard=*/true, c.deviceResults) ? 1 : 0;
            if (!c.ok)
                pipeFailed = true;
        };
        std::vector<std::thread> ts;
        ts.reserve(rc.nShards - 1);
        for (uint32_t s = 0; s + 1 < rc.nShards; ++s)
            ts.emplace_back(body, s);
        body(rc.nShards - 1);  // last shard inline
        for (auto& t : ts)
            t.join();
    };
    auto extractStage = [&](int buf, size_t start, size_t end) {
        for (uint32_t s = 0; s < rc.nShards; ++s) {
            ShardCtx& c = ctx[buf][s];
            if (!c.ok || !c.searcher)
                continue;
            c.searcher->BatchSearchExtract(c.deviceResults, c.shardResults, /*isMultiShard=*/true);
        }
        std::vector<size_t> pos(rc.nShards), lim(rc.nShards);
        for (size_t q = start; q < end; ++q) {
            const size_t j = q - start;
            const size_t K = rc.topks[q];
            for (uint32_t s = 0; s < rc.nShards; ++s) {
                pos[s] = 0;
                const auto& rs = ctx[buf][s].shardResults;
                lim[s] = (j < rs.size() && rs[j]) ? std::min<size_t>(K, rs[j]->scoreWithIndexes.size()) : 0;
            }
            std::vector<int64_t>& out = rc.resultDocIds[q];
            out.clear();
            out.reserve(K);
            for (size_t o = 0; o < K; ++o) {
                int best = -1;
                float bestScore = 0.0f;
                for (uint32_t s = 0; s < rc.nShards; ++s) {
                    if (pos[s] < lim[s]) {
                        float sc = ctx[buf][s].shardResults[j]->scoreWithIndexes[pos[s]].score;
                        if (best < 0 || sc > bestScore) {
                            bestScore = sc;
                            best = static_cast<int>(s);
                        }
                    }
                }
                if (best < 0)
                    break;
                out.push_back(static_cast<int64_t>(ctx[buf][best].shardResults[j]->scoreWithIndexes[pos[best]].id));
                pos[best]++;
            }
        }
    };

    for (int w = 0; w < std::max(0, FLAGS_warmup) && rc.nq > 0; ++w)
        rc.RunBatch(0, std::min(rc.nq, rc.bs), /*record=*/false);

    const size_t numBatches = (rc.nq + rc.bs - 1) / rc.bs;
    std::vector<double> devMs(numBatches, 0.0), extMs(numBatches, 0.0);
    auto t0 = std::chrono::steady_clock::now();
    long prev = -1;
    for (size_t n = 0; n < numBatches; ++n) {
        const size_t start = n * rc.bs, end = std::min(rc.nq, start + rc.bs);
        auto d0 = std::chrono::steady_clock::now();
        std::thread dev([&, n, start, end] { deviceStage(static_cast<int>(n & 1), start, end); });
        if (prev >= 0) {  // Extract the previous batch on this thread, overlapping dev(n)
            const size_t ps = static_cast<size_t>(prev) * rc.bs, pe = std::min(rc.nq, ps + rc.bs);
            auto e0 = std::chrono::steady_clock::now();
            extractStage(static_cast<int>(static_cast<size_t>(prev) & 1), ps, pe);
            extMs[prev] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - e0).count();
        }
        dev.join();
        devMs[n] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - d0).count();
        if (pipeFailed) {
            LOG_ERROR("pipeline device stage failed at [" << start << "," << end << ")");
            return false;
        }
        prev = static_cast<long>(n);
    }
    if (prev >= 0) {  // final batch's Extract (nothing to overlap it with)
        const size_t ps = static_cast<size_t>(prev) * rc.bs, pe = std::min(rc.nq, ps + rc.bs);
        auto e0 = std::chrono::steady_clock::now();
        extractStage(static_cast<int>(static_cast<size_t>(prev) & 1), ps, pe);
        extMs[prev] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - e0).count();
    }
    rc.wallMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    for (size_t n = 0; n < numBatches; ++n) {
        const size_t start = n * rc.bs, end = std::min(rc.nq, start + rc.bs);
        const double lat = devMs[n] + extMs[n];
        for (size_t q = start; q < end; ++q) {
            rc.perQueryMs[q] = lat;
            rc.latencies_ms.push_back(lat);
        }
    }
    return true;
}

}  // namespace npur_harness
