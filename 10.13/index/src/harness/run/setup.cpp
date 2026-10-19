#include "setup.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>

#include "../host/cpu_affinity.h"
#include "../query/query_io.h"
#include "configuration/develop_configuration.h"
#include "flags.h"
#include "src/full_recall/retrieval/searcher/runtime/stream_manager.h"
#include "src/utils/env_switch.h"
#include "src/utils/logger.h"

namespace npur_harness {

bool ResolveShards(ShardLayout& out) {
    if (!FLAGS_shard_index_dirs.empty()) {
        out.dirs = SplitCsv(FLAGS_shard_index_dirs);
        for (const std::string& d : SplitCsv(FLAGS_device_ids))
            out.devices.push_back(static_cast<int32_t>(std::strtol(d.c_str(), nullptr, 10)));
        if (out.dirs.empty() || out.dirs.size() != out.devices.size()) {
            LOG_ERROR("--shard_index_dirs (" << out.dirs.size() << ") and --device_ids (" << out.devices.size()
                                             << ") must be non-empty and equal in count");
            return false;
        }
    } else {
        out.dirs = {FLAGS_index_dir};
        out.devices = {FLAGS_device_id};
    }
    std::string affinityErr;
    if (!SetCardCpus(FLAGS_card_cpus, out.devices, &affinityErr)) {
        LOG_ERROR(affinityErr);
        return false;
    }
    // Round-robin searches each query on ONE card, so every card must hold the WHOLE corpus.
    // Sharded (disjoint) index dirs here make recall land near 1/N; distinct dirs are the tell.
    if (FLAGS_round_robin && out.dirs.size() > 1) {
        bool allSame = true;
        for (size_t i = 1; i < out.dirs.size(); ++i)
            if (out.dirs[i] != out.dirs[0])
                allSame = false;
        if (!allSame) {
            LOG_ERROR("--round_robin needs the FULL corpus on every card, but --shard_index_dirs lists "
                      << out.dirs.size()
                      << " DIFFERENT dirs (these look like disjoint shards). Every query would "
                         "search only its card's slice and recall would collapse to ~1/N. Pass ONE "
                         "full-corpus index dir (run.sh replicates it across --device_ids).");
            return false;
        }
    }
    return true;
}

bool LoadShards(const ShardLayout& layout, std::vector<std::shared_ptr<NpuRetrieval::DataTable>>& tables,
                uint64_t& totalDocNum) {
    const uint32_t nShards = layout.count();
    tables.assign(nShards, nullptr);
    totalDocNum = 0;
    auto loadT0 = std::chrono::steady_clock::now();
    // char, not bool: distinct elements of a vector<bool> share a word, so concurrent writes race.
    std::vector<long long> shardLoadMs(nShards, 0);
    std::vector<char> shardLoadOk(nShards, 0);
    auto loadOne = [&](uint32_t s) {
        tables[s] = std::make_shared<NpuRetrieval::DataTable>();
        auto t0 = std::chrono::steady_clock::now();
        if (!tables[s]->LoadData(layout.devices[s], layout.dirs[s])) {
            LOG_ERROR("DataTable::LoadData failed for shard " << s << " dir=" << layout.dirs[s]
                                                              << " device=" << layout.devices[s]);
            return;  // shardLoadOk[s] stays 0; the caller reports it after every thread has joined
        }
        shardLoadMs[s] =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        shardLoadOk[s] = 1;
    };
    if (FLAGS_parallel_load && nShards > 1) {
        std::vector<std::thread> loaders;
        loaders.reserve(nShards);
        for (uint32_t s = 0; s < nShards; ++s)
            loaders.emplace_back(loadOne, s);
        for (auto& loader : loaders)
            loader.join();
    } else {
        for (uint32_t s = 0; s < nShards; ++s)
            loadOne(s);
    }
    for (uint32_t s = 0; s < nShards; ++s) {
        if (shardLoadOk[s] == 0)
            return false;
        totalDocNum += tables[s]->GetDocNum();
        std::printf("[Load] shard %u on device %d: %.1fM docs, %u segments in %.1fs\n", s, layout.devices[s],
                    static_cast<double>(tables[s]->GetDocNum()) / 1e6, tables[s]->GetSegmentNum(),
                    static_cast<double>(shardLoadMs[s]) / 1000.0);
    }
    auto totalMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - loadT0).count();
    std::printf("[Load] %u shard(s), %.1fM docs total, loaded in %.1fs\n", nShards,
                static_cast<double>(totalDocNum) / 1e6, static_cast<double>(totalMs) / 1000.0);
    std::fflush(stdout);
    return true;
}

bool InitStreamPool(const ShardLayout& layout) {
    if (!npur_env::On("NPUR_STREAM_POOL"))
        return true;
    std::vector<int32_t> pooled;
    for (uint32_t s = 0; s < layout.count(); ++s) {
        const int32_t dev = layout.devices[s];
        if (std::find(pooled.begin(), pooled.end(), dev) != pooled.end())
            continue;
        if (!NpuRetrieval::StreamManager::GetInstance()->Init(dev, FLAGS_full_recall_stream_init_size)) {
            LOG_ERROR("StreamManager::Init failed for device " << dev);
            return false;
        }
        pooled.push_back(dev);
    }
    std::printf("[StreamPool] %zu card(s), %d streams each\n", pooled.size(),
                static_cast<int>(FLAGS_full_recall_stream_init_size));
    std::fflush(stdout);
    return true;
}

uint64_t RecallCorpusDocNum(const ShardLayout& layout,
                            const std::vector<std::shared_ptr<NpuRetrieval::DataTable>>& tables, uint64_t totalDocNum) {
    const uint32_t nShards = layout.count();
    if (FLAGS_round_robin)
        return tables[0]->GetDocNum();
    if (FLAGS_shard_group_size > 0 && nShards >= static_cast<uint32_t>(FLAGS_shard_group_size) &&
        nShards % static_cast<uint32_t>(FLAGS_shard_group_size) == 0) {
        uint64_t docNum = 0;
        for (int j = 0; j < FLAGS_shard_group_size; ++j)
            docNum += tables[j]->GetDocNum();
        return docNum;
    }
    return totalDocNum;  // plain sharding: the shards are disjoint, so their sum
}

bool LoadRunInputs(RunInputs& out) {
    std::string qerr;
    if (!npur_port::LoadQueries(FLAGS_query_file, out.queries, &qerr)) {
        LOG_ERROR("failed to load queries: " << qerr);
        return false;
    }
    out.nq = (FLAGS_num_queries > 0) ? static_cast<size_t>(FLAGS_num_queries) : out.queries.size();

    if (!FLAGS_filter_file.empty()) {
        if (!LoadFilters(FLAGS_filter_file, out.nq, FLAGS_filter_seed, out.filters))
            return false;
        out.haveFilters = true;
    }
    out.topks.assign(out.nq, static_cast<uint32_t>(FLAGS_topk));
    if (!FLAGS_topk_file.empty() && !LoadTopKFile(FLAGS_topk_file, out.nq, out.topks))
        return false;
    return true;
}

}  // namespace npur_harness
