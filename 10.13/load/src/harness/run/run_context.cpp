#include "run_context.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "../host/cpu_affinity.h"
#include "acl/acl.h"
#include "flags.h"
#include "src/full_recall/retrieval/query/query_node_imp.h"
#include "src/full_recall/retrieval/searcher/full_recall_searcher.h"
#include "src/utils/logger.h"

namespace npur_harness {

RunContext::RunContext(const ShardLayout& layout, std::vector<std::shared_ptr<NpuRetrieval::DataTable>>& tables,
                       const RunInputs& inputs)
    : shardDevices(layout.devices),
      shardTables(tables),
      queries(inputs.queries),
      filters(inputs.filters),
      haveFilters(inputs.haveFilters),
      topks(inputs.topks),
      nShards(layout.count()),
      nq(inputs.nq),
      Q(inputs.queries.size()),
      bs(static_cast<size_t>(std::max<int>(1, FLAGS_batch_size))),
      dumpShardLat(layout.count() > 1 && !FLAGS_shard_latency_dump.empty()),
      paceQps(FLAGS_target_qps),
      paced(FLAGS_target_qps > 0.0) {
    resultDocIds.resize(nq);
    latencies_ms.reserve(nq);
    perQueryMs.assign(nq, 0.0);
    if (dumpShardLat) {
        shardMs.assign(nShards, std::vector<double>(nq, 0.0));
        mergeMs.assign(nq, 0.0);
    }
    shardResults.resize(nShards);
    shardOk.assign(nShards, 1);
    acc.resize(nShards > 1 ? bs : 0);

    // --slow_cards: emulated by sleeping inside the measured latency, at the one place every
    // dispatch path goes through, so every scheme faces an identical disturbance.
    for (const std::string& d : SplitCsv(FLAGS_slow_cards))
        slowCards.push_back(static_cast<int32_t>(std::strtol(d.c_str(), nullptr, 10)));
    fluctuate = !slowCards.empty() && FLAGS_slow_factor > 1.0 && FLAGS_slow_period_ms > 0.0;
    fluctT0 = std::chrono::steady_clock::now();
}

// Stop + join on ANY exit path, so a joinable thread never outlives its mutex/cv.
RunContext::~RunContext() {
    {
        std::unique_lock<std::mutex> lk(poolMtx);
        poolStop = true;
        ++poolGen;
    }
    poolGo.notify_all();
    for (auto& t : poolThreads)
        if (t.joinable())
            t.join();
}

std::chrono::steady_clock::time_point RunContext::arrivalOf(size_t queryIdx) const {
    return paceT0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(static_cast<double>(queryIdx) / paceQps));
}

void RunContext::ApplyFluctuation(int32_t deviceId, double serviceMs) {
    if (!fluctuate)
        return;
    if (std::find(slowCards.begin(), slowCards.end(), deviceId) == slowCards.end())
        return;
    const double elapsed =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - fluctT0).count();
    double phase = elapsed - FLAGS_slow_period_ms * std::floor(elapsed / FLAGS_slow_period_ms);
    if (phase >= FLAGS_slow_duty * FLAGS_slow_period_ms)
        return;  // currently in the card's normal window
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>((FLAGS_slow_factor - 1.0) * serviceMs));
}

// --stream_merge: the deterministic tie-break keeps the result byte-identical to the barrier k-way
// merge whatever order shards finish in.
void RunContext::FoldShardIntoAcc(uint32_t s, size_t start, size_t end) {
    std::lock_guard<std::mutex> lk(accMtx);
    for (size_t q = start; q < end; ++q) {
        const size_t j = q - start;
        const size_t K = topks[q];
        const auto& rs = shardResults[s];
        if (j >= rs.size() || !rs[j])
            continue;
        const auto& sw = rs[j]->scoreWithIndexes;  // this shard's sorted top-K (ids valid in prefix)
        const size_t m = std::min(K, sw.size());
        std::vector<MergeEntry>& a = acc[j];
        mergeScratch.clear();
        size_t ia = 0, ib = 0;
        while (mergeScratch.size() < K && (ia < a.size() || ib < m)) {
            bool takeA;
            if (ia >= a.size())
                takeA = false;
            else if (ib >= m)
                takeA = true;
            else if (a[ia].score != sw[ib].score)
                takeA = a[ia].score > sw[ib].score;
            else
                takeA = a[ia].shard < s;  // equal score: lower shard first
            if (takeA) {
                mergeScratch.push_back(a[ia]);
                ++ia;
            } else {
                mergeScratch.push_back(MergeEntry{static_cast<int64_t>(sw[ib].id), sw[ib].score, s});
                ++ib;
            }
        }
        a.swap(mergeScratch);
    }
}

void RunContext::RunShardBody(uint32_t s, size_t start, size_t end, bool record) {
    auto st0 = std::chrono::steady_clock::now();
    PinThreadToCard(s);
    aclrtSetDevice(shardDevices[s]);  // thread-local device context for this shard
    NpuRetrieval::FullRecallSearcher searcher(shardDevices[s], shardTables[s]);
    std::vector<std::unique_ptr<NpuRetrieval::QueryNode>> owned;
    for (size_t q = start; q < end; ++q) {
        const FilterNode* ast = haveFilters ? filters.astFor(q) : nullptr;
        NpuRetrieval::QueryNode* tree = BuildQueryNode(ast, FLAGS_posting_field, owned);
        searcher.AddQuery(tree, queries[q % Q], topks[q]);
    }
    auto searchT0 = std::chrono::steady_clock::now();
    shardOk[s] = searcher.BatchSearch(shardResults[s], FLAGS_vec_field, /*isMultiShard=*/true) ? 1 : 0;
    ApplyFluctuation(shardDevices[s],
                     std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - searchT0).count());
    if (FLAGS_stream_merge && record && shardOk[s])
        FoldShardIntoAcc(s, start, end);
    if (dumpShardLat && record)
        shardMs[s][start] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - st0).count();
}

void RunContext::PoolWorker(uint32_t s) {
    uint64_t seen = 0;
    for (;;) {
        std::unique_lock<std::mutex> lk(poolMtx);
        poolGo.wait(lk, [&] { return poolStop || poolGen != seen; });
        if (poolStop)
            return;
        seen = poolGen;
        size_t st = jobStart, en = jobEnd;
        bool rec = jobRecord;
        lk.unlock();
        RunShardBody(s, st, en, rec);
        lk.lock();
        if (--poolBusy == 0)
            poolDone.notify_one();
    }
}

void RunContext::StartWorkerPool() {
    if (FLAGS_shard_worker_pool && nShards > 1)
        for (uint32_t s = 0; s + 1 < nShards; ++s)
            poolThreads.emplace_back(&RunContext::PoolWorker, this, s);
}

bool RunContext::RunBatch(size_t start, size_t end, bool record) {
    const size_t bn = end - start;
    if (nShards == 1) {
        NpuRetrieval::FullRecallSearcher searcher(shardDevices[0], shardTables[0]);
        std::vector<std::unique_ptr<NpuRetrieval::QueryNode>> owned;
        for (size_t q = start; q < end; ++q) {
            const FilterNode* ast = haveFilters ? filters.astFor(q) : nullptr;
            NpuRetrieval::QueryNode* tree = BuildQueryNode(ast, FLAGS_posting_field, owned);
            searcher.AddQuery(tree, queries[q % Q], topks[q]);
        }
        std::vector<std::shared_ptr<NpuRetrieval::FullRecallResult>> results;
        if (!searcher.BatchSearch(results, FLAGS_vec_field, /*isMultiShard=*/false))
            return false;
        if (record)
            for (size_t k = 0; k < bn; ++k)
                if (k < results.size() && results[k])
                    resultDocIds[start + k] = results[k]->docIds;
        return true;
    }
    for (uint32_t s = 0; s < nShards; ++s) {
        shardResults[s].clear();
        shardOk[s] = 1;
    }
    if (FLAGS_stream_merge && record)
        for (size_t j = 0; j < bn; ++j)
            acc[j].clear();
    if (FLAGS_shard_worker_pool && nShards > 1) {
        {
            std::unique_lock<std::mutex> lk(poolMtx);
            jobStart = start;
            jobEnd = end;
            jobRecord = record;
            poolBusy = nShards - 1;
            ++poolGen;
        }
        poolGo.notify_all();
        RunShardBody(nShards - 1, start, end, record);  // last shard inline
        std::unique_lock<std::mutex> lk(poolMtx);
        poolDone.wait(lk, [&] { return poolBusy == 0; });
    } else {
        std::vector<std::thread> threads;
        threads.reserve(nShards - 1);
        for (uint32_t s = 0; s + 1 < nShards; ++s)
            threads.emplace_back(&RunContext::RunShardBody, this, s, start, end, record);
        RunShardBody(nShards - 1, start, end, record);  // last shard inline
        for (auto& t : threads)
            t.join();
    }
    // --shard_allow_partial drops the failed shards and merges the survivors.
    bool allShardsOk = true;
    for (uint32_t s = 0; s < nShards; ++s)
        if (!shardOk[s])
            allShardsOk = false;
    if (!FLAGS_shard_allow_partial) {
        if (!allShardsOk)
            return false;
    } else if (!allShardsOk) {
        for (uint32_t s = 0; s < nShards; ++s)
            if (!shardOk[s])
                shardResults[s].clear();  // drop failed shards; the merge skips empty heads
    }
    if (record && FLAGS_stream_merge) {
        for (size_t q = start; q < end; ++q) {
            const size_t j = q - start;
            std::vector<int64_t>& out = resultDocIds[q];
            out.clear();
            out.reserve(acc[j].size());
            for (const auto& e : acc[j])
                out.push_back(e.id);
        }
    } else if (record) {
        auto m0 = std::chrono::steady_clock::now();
        // The global top-K is a subset of the union of the per-shard top-Ks, already sorted,
        // so merging the prefixes is exact.
        std::vector<size_t> pos(nShards), lim(nShards);
        for (size_t q = start; q < end; ++q) {
            const size_t j = q - start;
            const size_t K = topks[q];
            for (uint32_t s = 0; s < nShards; ++s) {
                pos[s] = 0;
                const auto& rs = shardResults[s];
                lim[s] = (j < rs.size() && rs[j]) ? std::min<size_t>(K, rs[j]->scoreWithIndexes.size()) : 0;
            }
            std::vector<int64_t>& out = resultDocIds[q];
            out.clear();
            out.reserve(K);
            for (size_t o = 0; o < K; ++o) {
                int best = -1;
                float bestScore = 0.0f;
                for (uint32_t s = 0; s < nShards; ++s) {
                    if (pos[s] < lim[s]) {
                        float sc = shardResults[s][j]->scoreWithIndexes[pos[s]].score;
                        if (best < 0 || sc > bestScore) {
                            bestScore = sc;
                            best = static_cast<int>(s);
                        }
                    }
                }
                if (best < 0)
                    break;  // all shard heads exhausted
                out.push_back(static_cast<int64_t>(shardResults[best][j]->scoreWithIndexes[pos[best]].id));
                pos[best]++;
            }
        }
        if (dumpShardLat)
            mergeMs[start] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - m0).count();
    }
    return true;
}

}  // namespace npur_harness
