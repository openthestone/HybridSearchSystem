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

bool RunRoundRobin(RunContext& rc) {
    // Round-robin baseline: isMultiShard=false, so a chunk's result is already the global top-K.
    std::atomic<bool> rrFailed{false};
    std::atomic<size_t> cursor{0};  // shared work queue: fetch_add(rc.bs) hands the next chunk to an idle card
    auto rrChunk = [&](uint32_t c, size_t start, size_t end, bool record,
                       std::chrono::steady_clock::time_point arrival) {
        NpuRetrieval::FullRecallSearcher searcher(rc.shardDevices[c], rc.shardTables[c]);
        std::vector<std::unique_ptr<NpuRetrieval::QueryNode>> owned;
        for (size_t q = start; q < end; ++q) {
            const FilterNode* ast = rc.haveFilters ? rc.filters.astFor(q) : nullptr;
            NpuRetrieval::QueryNode* tree = BuildQueryNode(ast, FLAGS_posting_field, owned);
            searcher.AddQuery(tree, rc.queries[q % rc.Q], rc.topks[q]);
        }
        auto ct0 = std::chrono::steady_clock::now();
        std::vector<std::shared_ptr<NpuRetrieval::FullRecallResult>> results;
        bool ok = searcher.BatchSearch(results, FLAGS_vec_field, /*isMultiShard=*/false);
        rc.ApplyFluctuation(rc.shardDevices[c],
                            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ct0).count());
        auto done = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(done - (rc.paced ? arrival : ct0)).count();
        if (!ok) {
            rrFailed = true;
            return;
        }
        if (record) {
            for (size_t k = 0; k < (end - start); ++k) {
                rc.perQueryMs[start + k] = ms;
                if (k < results.size() && results[k])
                    rc.resultDocIds[start + k] = results[k]->docIds;
            }
        }
    };
    auto rrWorker = [&](uint32_t c, bool record) {
        PinThreadToCard(c);
        aclrtSetDevice(rc.shardDevices[c]);
        // Static (--static_assign): chunk k belongs to card k%N, so a slowed card's chunks wait.
        size_t staticChunk = c;
        for (;;) {
            size_t start;
            if (FLAGS_static_assign) {
                start = staticChunk * rc.bs;
                staticChunk += rc.nShards;
            } else {
                start = cursor.fetch_add(rc.bs);
            }
            if (start >= rc.nq)
                break;
            auto arrival = std::chrono::steady_clock::now();
            if (rc.paced) {
                arrival = rc.arrivalOf(start);
                std::this_thread::sleep_until(arrival);  // no-op once we fall behind (saturated)
            }
            rrChunk(c, start, std::min(rc.nq, start + rc.bs), record, arrival);
            if (rrFailed)
                break;
        }
    };
    if (FLAGS_warmup > 0 && rc.nq > 0) {
        std::vector<std::thread> wu;
        for (uint32_t c = 0; c < rc.nShards; ++c)
            wu.emplace_back([&, c] {
                PinThreadToCard(c);
                aclrtSetDevice(rc.shardDevices[c]);
                for (int w = 0; w < FLAGS_warmup; ++w)
                    rrChunk(c, 0, std::min<size_t>(rc.nq, rc.bs), /*record=*/false,
                            std::chrono::steady_clock::now());  // unused: record=false
            });
        for (auto& t : wu)
            t.join();
    }
    auto rr_t0 = std::chrono::steady_clock::now();
    rc.paceT0 = rr_t0;
    std::vector<std::thread> workers;
    workers.reserve(rc.nShards);
    for (uint32_t c = 0; c < rc.nShards; ++c)
        workers.emplace_back(rrWorker, c, /*record=*/true);
    for (auto& t : workers)
        t.join();
    rc.wallMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - rr_t0).count();
    if (rrFailed) {
        LOG_ERROR("round-robin search failed");
        return false;
    }
    for (size_t q = 0; q < rc.nq; ++q)
        rc.latencies_ms.push_back(rc.perQueryMs[q]);  // single-card per-query proc time
    return true;
}

}  // namespace npur_harness
