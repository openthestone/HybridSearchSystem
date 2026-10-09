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

bool RunShardGroups(RunContext& rc) {
    // Shard groups: group g owns shards [g*gsize, (g+1)*gsize), so each group must hold a full
    // corpus split gsize ways -- the same dirs repeated per group, as run.sh's replication does.
    const uint32_t gsize = static_cast<uint32_t>(FLAGS_shard_group_size);
    const uint32_t nGroups = rc.nShards / gsize;
    const bool groupPipeline = FLAGS_pipeline;
    std::atomic<size_t> cursor{0};
    std::atomic<bool> grpFailed{false};
    struct ShardSlot {
        std::unique_ptr<NpuRetrieval::FullRecallSearcher> searcher;
        std::vector<std::unique_ptr<NpuRetrieval::QueryNode>> owned;
        std::vector<NpuRetrieval::AggrDeviceResult> deviceResults;
        std::vector<std::shared_ptr<NpuRetrieval::FullRecallResult>> res;
        char ok{1};
    };
    std::vector<std::vector<ShardSlot>> slot(static_cast<size_t>(nGroups) * 2);
    for (auto& sv : slot)
        sv.resize(gsize);
    const size_t numBatches = (rc.nq + rc.bs - 1) / rc.bs;
    std::vector<double> devMs(numBatches, 0.0), extMs(numBatches, 0.0);

    auto groupShardPrepare = [&](uint32_t grp, uint32_t j, int buf, size_t start, size_t end) {
        ShardSlot& k = slot[static_cast<size_t>(grp) * 2 + buf][j];
        const uint32_t s = grp * gsize + j;  // this group's j-th shard, globally
        k.owned.clear();
        k.searcher = std::make_unique<NpuRetrieval::FullRecallSearcher>(rc.shardDevices[s], rc.shardTables[s]);
        for (size_t q = start; q < end; ++q) {
            const FilterNode* ast = rc.haveFilters ? rc.filters.astFor(q) : nullptr;
            NpuRetrieval::QueryNode* tree = BuildQueryNode(ast, FLAGS_posting_field, k.owned);
            k.searcher->AddQuery(tree, rc.queries[q % rc.Q], rc.topks[q]);
        }
    };
    auto groupPrepare = [&](uint32_t grp, int buf, size_t start, size_t end) {
        for (uint32_t j = 0; j < gsize; ++j)
            groupShardPrepare(grp, j, buf, start, end);
    };

    auto groupShardLaunch = [&](uint32_t grp, uint32_t j, int buf, size_t start, size_t end) {
        ShardSlot& k = slot[static_cast<size_t>(grp) * 2 + buf][j];
        const uint32_t s = grp * gsize + j;
        PinThreadToCard(s);
        aclrtSetDevice(rc.shardDevices[s]);
        if (k.searcher == nullptr) {  // prepare must have run; never expected
            LOG_ERROR("group " << grp << " shard " << j << " launched without a prepared searcher");
            k.ok = 0;
            grpFailed = true;
            return;
        }
        auto shardT0 = std::chrono::steady_clock::now();
        k.ok = k.searcher->BatchSearchDevice(FLAGS_vec_field, /*isMultiShard=*/true, k.deviceResults) ? 1 : 0;
        rc.ApplyFluctuation(
            rc.shardDevices[s],
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - shardT0).count());
        if (!k.ok)
            grpFailed = true;
    };

    // poolShards: pipelined the workers take all of a group's shards, otherwise the caller runs
    // the last inline. 0 = pool off.
    const uint32_t poolShards = !FLAGS_shard_worker_pool ? 0u : (groupPipeline ? gsize : gsize - 1);
    const bool groupPrefetch = FLAGS_group_prefetch && groupPipeline && poolShards > 0;

    auto groupShardBody = [&](uint32_t grp, uint32_t j, int buf, size_t start, size_t end) {
        if (!groupPrefetch)
            groupShardPrepare(grp, j, buf, start, end);
        groupShardLaunch(grp, j, buf, start, end);
    };

    struct GroupPool {
        std::mutex mtx;
        std::condition_variable go, done;
        uint64_t gen{0};   // bumped once per chunk; each worker runs each generation once
        uint32_t busy{0};  // workers still running this chunk
        bool stop{false};
        int buf{0};
        size_t start{0}, end{0};
    };
    std::vector<std::unique_ptr<GroupPool>> gpool;  // unique_ptr: GroupPool holds a mutex, not movable
    std::vector<std::thread> gpoolThreads;
    if (poolShards > 0) {
        for (uint32_t g = 0; g < nGroups; ++g)
            gpool.emplace_back(std::make_unique<GroupPool>());
        for (uint32_t g = 0; g < nGroups; ++g)
            for (uint32_t j = 0; j < poolShards; ++j)
                gpoolThreads.emplace_back([&, g, j] {
                    GroupPool& p = *gpool[g];
                    uint64_t seen = 0;
                    for (;;) {
                        std::unique_lock<std::mutex> lk(p.mtx);
                        p.go.wait(lk, [&] { return p.stop || p.gen != seen; });
                        if (p.stop)
                            return;
                        seen = p.gen;
                        const int b = p.buf;
                        const size_t st = p.start, en = p.end;
                        lk.unlock();
                        groupShardBody(g, j, b, st, en);
                        lk.lock();
                        if (--p.busy == 0)
                            p.done.notify_one();
                    }
                });
    }
    struct GroupPoolJoiner {
        std::vector<std::unique_ptr<GroupPool>>& pools;
        std::vector<std::thread>& threads;
        ~GroupPoolJoiner() {
            for (auto& p : pools) {
                {
                    std::unique_lock<std::mutex> lk(p->mtx);
                    p->stop = true;
                    ++p->gen;
                }
                p->go.notify_all();
            }
            for (auto& t : threads)
                if (t.joinable())
                    t.join();
        }
    } gpoolJoiner{gpool, gpoolThreads};

    const uint32_t extShards = (FLAGS_group_extract_parallel && gsize > 1) ? gsize - 1 : 0;
    std::vector<std::unique_ptr<GroupPool>> epool;
    std::vector<std::thread> epoolThreads;
    if (extShards > 0) {
        for (uint32_t g = 0; g < nGroups; ++g)
            epool.emplace_back(std::make_unique<GroupPool>());
        for (uint32_t g = 0; g < nGroups; ++g)
            for (uint32_t j = 0; j < extShards; ++j)
                epoolThreads.emplace_back([&, g, j] {
                    // Bind the card once, so the aclrtFreeHost inside Extract has a context.
                    PinThreadToCard(static_cast<size_t>(g) * gsize + j);
                    aclrtSetDevice(rc.shardDevices[g * gsize + j]);
                    GroupPool& p = *epool[g];
                    uint64_t seen = 0;
                    for (;;) {
                        std::unique_lock<std::mutex> lk(p.mtx);
                        p.go.wait(lk, [&] { return p.stop || p.gen != seen; });
                        if (p.stop)
                            return;
                        seen = p.gen;
                        const int b = p.buf;
                        lk.unlock();
                        ShardSlot& k = slot[static_cast<size_t>(g) * 2 + b][j];
                        if (k.ok && k.searcher)
                            k.searcher->BatchSearchExtract(k.deviceResults, k.res, /*isMultiShard=*/true);
                        lk.lock();
                        if (--p.busy == 0)
                            p.done.notify_one();
                    }
                });
    }
    GroupPoolJoiner epoolJoiner{epool, epoolThreads};

    auto groupDeviceSpawn = [&](uint32_t grp, int buf, size_t start, size_t end) {
        std::vector<std::thread> ts;
        ts.reserve(gsize - 1);
        for (uint32_t j = 0; j + 1 < gsize; ++j)
            ts.emplace_back(groupShardBody, grp, j, buf, start, end);
        groupShardBody(grp, gsize - 1, buf, start, end);  // last shard inline
        for (auto& t : ts)
            t.join();
    };
    auto groupDeviceStart = [&](uint32_t grp, int buf, size_t start, size_t end) {
        GroupPool& p = *gpool[grp];
        {
            std::unique_lock<std::mutex> lk(p.mtx);
            p.buf = buf;
            p.start = start;
            p.end = end;
            p.busy = poolShards;
            ++p.gen;
        }
        p.go.notify_all();
    };
    auto groupDeviceFinish = [&](uint32_t grp, int buf, size_t start, size_t end) {
        for (uint32_t j = poolShards; j < gsize; ++j)
            groupShardBody(grp, j, buf, start, end);
        GroupPool& p = *gpool[grp];
        std::unique_lock<std::mutex> lk(p.mtx);
        p.done.wait(lk, [&] { return p.busy == 0; });
    };
    auto groupDevice = [&](uint32_t grp, int buf, size_t start, size_t end) {
        if (poolShards > 0) {
            groupDeviceStart(grp, buf, start, end);
            groupDeviceFinish(grp, buf, start, end);
        } else {
            groupDeviceSpawn(grp, buf, start, end);
        }
    };

    auto groupExtract = [&](uint32_t grp, int buf, size_t start, size_t end, bool record) {
        std::vector<ShardSlot>& sv = slot[static_cast<size_t>(grp) * 2 + buf];
        auto extractOne = [&](uint32_t j) {
            if (sv[j].ok && sv[j].searcher)
                sv[j].searcher->BatchSearchExtract(sv[j].deviceResults, sv[j].res, /*isMultiShard=*/true);
        };
        if (extShards > 0) {
            GroupPool& p = *epool[grp];
            {
                std::unique_lock<std::mutex> lk(p.mtx);
                p.buf = buf;
                p.busy = extShards;
                ++p.gen;
            }
            p.go.notify_all();
            for (uint32_t j = extShards; j < gsize; ++j)
                extractOne(j);  // this thread's share, so it is not idle while the workers run
            std::unique_lock<std::mutex> lk(p.mtx);
            p.done.wait(lk, [&] { return p.busy == 0; });
        } else {
            for (uint32_t j = 0; j < gsize; ++j)
                extractOne(j);
        }
        if (!record)
            return;
        std::vector<size_t> pos(gsize), lim(gsize);
        for (size_t q = start; q < end; ++q) {
            const size_t j0 = q - start;
            const size_t K = rc.topks[q];
            for (uint32_t j = 0; j < gsize; ++j) {
                pos[j] = 0;
                const auto& rs = sv[j].res;
                lim[j] = (j0 < rs.size() && rs[j0]) ? std::min<size_t>(K, rs[j0]->scoreWithIndexes.size()) : 0;
            }
            std::vector<int64_t>& out = rc.resultDocIds[q];
            out.clear();
            out.reserve(K);
            for (size_t o = 0; o < K; ++o) {
                int best = -1;
                float bestScore = 0.0f;
                for (uint32_t j = 0; j < gsize; ++j) {
                    if (pos[j] < lim[j]) {
                        const float sc = sv[j].res[j0]->scoreWithIndexes[pos[j]].score;
                        if (best < 0 || sc > bestScore) {
                            bestScore = sc;
                            best = static_cast<int>(j);
                        }
                    }
                }
                if (best < 0)
                    break;  // all heads exhausted
                out.push_back(static_cast<int64_t>(sv[best].res[j0]->scoreWithIndexes[pos[best]].id));
                pos[best]++;
            }
        }
    };

    auto groupWorker = [&](uint32_t grp, bool record) {
        PinThreadToCards(static_cast<size_t>(grp) * gsize, gsize);
        // Only one chunk is ever on those cards, so the per-device pools are never shared.
        long prevStart = -1;
        size_t prevEnd = 0;
        int prevBuf = 0, buf = 0;
        // Static (--static_assign) binds chunk k to group k % G, making SCHEDULING the only
        // difference between the two grouped runs.
        size_t staticChunk = grp;
        auto nextChunk = [&](size_t& outStart) -> bool {
            size_t st;
            if (FLAGS_static_assign) {
                st = staticChunk * rc.bs;
                staticChunk += nGroups;
            } else {
                st = cursor.fetch_add(rc.bs);
            }
            if (st >= rc.nq)
                return false;
            outStart = st;
            return true;
        };

        if (groupPrefetch) {
            size_t curStart = 0, curEnd = 0;
            if (!nextChunk(curStart))
                return;
            curEnd = std::min(rc.nq, curStart + rc.bs);
            groupPrepare(grp, 0, curStart, curEnd);
            int buf = 0, prevBuf = 0;
            long prevStart = -1;
            size_t prevEnd = 0;
            for (;;) {
                const auto d0 = std::chrono::steady_clock::now();
                groupDeviceStart(grp, buf, curStart, curEnd);
                if (prevStart >= 0) {
                    const auto e0 = std::chrono::steady_clock::now();
                    groupExtract(grp, prevBuf, static_cast<size_t>(prevStart), prevEnd, record);
                    if (record)
                        extMs[static_cast<size_t>(prevStart) / rc.bs] =
                            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - e0).count();
                }
                size_t nextStart = 0, nextEnd = 0;
                const bool haveNext = nextChunk(nextStart);
                if (haveNext) {
                    nextEnd = std::min(rc.nq, nextStart + rc.bs);
                    groupPrepare(grp, buf ^ 1, nextStart, nextEnd);
                }
                groupDeviceFinish(grp, buf, curStart, curEnd);
                if (record)
                    devMs[curStart / rc.bs] =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - d0).count();
                if (grpFailed)
                    return;
                prevStart = static_cast<long>(curStart);
                prevEnd = curEnd;
                prevBuf = buf;
                buf ^= 1;
                if (!haveNext)
                    break;
                curStart = nextStart;
                curEnd = nextEnd;
            }
            if (prevStart >= 0 && !grpFailed) {
                const auto e0 = std::chrono::steady_clock::now();  // last chunk: nothing overlaps it
                groupExtract(grp, prevBuf, static_cast<size_t>(prevStart), prevEnd, record);
                if (record)
                    extMs[static_cast<size_t>(prevStart) / rc.bs] =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - e0).count();
            }
            return;
        }

        for (;;) {
            size_t start;
            if (!nextChunk(start))
                break;
            const size_t end = std::min(rc.nq, start + rc.bs);
            const auto d0 = std::chrono::steady_clock::now();
            if (groupPipeline) {
                std::thread dev;
                if (poolShards > 0)
                    groupDeviceStart(grp, buf, start, end);
                else
                    dev = std::thread([&] { groupDeviceSpawn(grp, buf, start, end); });
                if (prevStart >= 0) {
                    const auto e0 = std::chrono::steady_clock::now();
                    groupExtract(grp, prevBuf, static_cast<size_t>(prevStart), prevEnd, record);
                    if (record)
                        extMs[static_cast<size_t>(prevStart) / rc.bs] =
                            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - e0).count();
                }
                if (poolShards > 0)
                    groupDeviceFinish(grp, buf, start, end);
                else
                    dev.join();
                if (record)
                    devMs[start / rc.bs] =
                        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - d0).count();
                if (grpFailed)
                    return;
                prevStart = static_cast<long>(start);
                prevEnd = end;
                prevBuf = buf;
                buf ^= 1;
                continue;
            }
            groupDevice(grp, 0, start, end);
            if (grpFailed)
                return;
            const auto e0 = std::chrono::steady_clock::now();
            if (record)
                devMs[start / rc.bs] = std::chrono::duration<double, std::milli>(e0 - d0).count();
            groupExtract(grp, 0, start, end, record);
            if (record)
                extMs[start / rc.bs] =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - e0).count();
        }
        if (groupPipeline && prevStart >= 0 && !grpFailed) {
            const auto e0 = std::chrono::steady_clock::now();  // last chunk: nothing left to overlap it
            groupExtract(grp, prevBuf, static_cast<size_t>(prevStart), prevEnd, record);
            if (record)
                extMs[static_cast<size_t>(prevStart) / rc.bs] =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - e0).count();
        }
    };
    if (FLAGS_warmup > 0 && rc.nq > 0) {
        std::vector<std::thread> wu;
        for (uint32_t g = 0; g < nGroups; ++g)
            wu.emplace_back([&, g] {
                PinThreadToCards(static_cast<size_t>(g) * gsize, gsize);
                for (int w = 0; w < FLAGS_warmup; ++w) {
                    if (groupPrefetch)  // groupShardBody skips the build in that mode
                        groupPrepare(g, 0, 0, std::min<size_t>(rc.nq, rc.bs));
                    groupDevice(g, 0, 0, std::min<size_t>(rc.nq, rc.bs));
                    groupExtract(g, 0, 0, std::min<size_t>(rc.nq, rc.bs), /*record=*/false);
                }
            });
        for (auto& t : wu)
            t.join();
    }
    std::printf("[Groups] %u group(s) x %u shard(s)%s | fan-out: %s | extract: %s | build: %s | dispatch: %s\n",
                nGroups, gsize, groupPipeline ? " | pipelined" : "",
                poolShards > 0 ? "persistent workers" : "spawned per chunk", extShards > 0 ? "parallel" : "serial",
                groupPrefetch ? "prefetched" : "inline",
                FLAGS_static_assign ? "FIXED (chunk k -> group k%G)" : "work-steal");
    auto grp_t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> workers;
    workers.reserve(nGroups);
    for (uint32_t g = 0; g < nGroups; ++g)
        workers.emplace_back(groupWorker, g, /*record=*/true);
    for (auto& t : workers)
        t.join();
    rc.wallMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - grp_t0).count();
    if (grpFailed) {
        LOG_ERROR("shard-group search failed");
        return false;
    }
    for (size_t n = 0; n < numBatches; ++n) {
        const size_t start = n * rc.bs, end = std::min(rc.nq, start + rc.bs);
        const double lat = devMs[n] + extMs[n];
        for (size_t q = start; q < end; ++q)
            rc.perQueryMs[q] = lat;
    }
    for (size_t q = 0; q < rc.nq; ++q)
        rc.latencies_ms.push_back(rc.perQueryMs[q]);
    // Groups subsume --pipeline: the group path pipelines inside each group.
    return true;
}

}  // namespace npur_harness
