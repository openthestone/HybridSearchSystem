/*
 * setup.h -- everything fr_search does before the first query: work out which index dirs go on
 * which cards, load them, bring up the stream pool, and read the queries/filters/topks.
 *
 * Each of these returns false having already logged why. The caller owns aclFinalize().
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../query/filter_setup.h"
#include "src/full_recall/index/data_table.h"

namespace npur_harness {

// One index dir per card, in --device_ids order. A single card is one entry.
struct ShardLayout {
    std::vector<std::string> dirs;
    std::vector<int32_t> devices;

    uint32_t count() const {
        return static_cast<uint32_t>(dirs.size());
    }
};

// Resolves --shard_index_dirs/--device_ids (or --index_dir/--device_id), installs --card_cpus, and
// rejects --round_robin against disjoint shard dirs -- which would silently collapse recall to ~1/N.
bool ResolveShards(ShardLayout& out);

// One DataTable per shard, on its own thread under --parallel_load. Prints the [Load] lines.
bool LoadShards(const ShardLayout& layout, std::vector<std::shared_ptr<NpuRetrieval::DataTable>>& tables,
                uint64_t& totalDocNum);

// NPUR_STREAM_POOL=1 only. Without it GetStream() returns nullptr, every kernel lands on the
// default stream, and TOPK_CONCURRENT / AGG_CONCURRENT / DEFER_CONV_SYNC quietly become no-ops.
bool InitStreamPool(const ShardLayout& layout);

// The corpus the recall reference must be built against, which is NOT the shard sum under
// --round_robin (one card holds it all) or --shard_group_size (one group, not G times over).
uint64_t RecallCorpusDocNum(const ShardLayout& layout,
                            const std::vector<std::shared_ptr<NpuRetrieval::DataTable>>& tables, uint64_t totalDocNum);

struct RunInputs {
    std::vector<std::vector<float>> queries;
    size_t nq = 0;  // queries to run; --num_queries may exceed the file, and then queries cycle
    FilterSet filters;
    bool haveFilters = false;
    std::vector<uint32_t> topks;  // per query, --topk unless --topk_file overrides it
};

bool LoadRunInputs(RunInputs& out);

}  // namespace npur_harness
