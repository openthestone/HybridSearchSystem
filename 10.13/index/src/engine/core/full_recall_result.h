#pragma once
#include <cstdint>
#include <vector>
#include <atomic>
#include <memory>

namespace NpuRetrieval {

enum class GmPoolName {
    VECTOR_SCORE_RESULT_POOL = 0,
    TEXT_FILTER_RESULT_POOL = 1,
    TEXT_FILTER_STACK_POOL = 2,
    TEXT_FILTER_BITLIST2SET_POOL = 3,
    AGGREGATOR_POOL = 4,
    AGGREGATOR_SCRATCH_POOL = 5,
    TEXT_FILTER_POSTINGS_POOL = 6,
    SCORER_QUERY_POOL = 7,
    TEXT_FILTER_EXPR_POOL = 8,
    VALID_POOL_NUM,
};

struct GmBlock {
    GmPoolName poolName;
    char* data;     // pointer to the actual device memory
    uint32_t size;  // size of the chunk in bytes
    // aggregator read-count; when it reaches 0 all data has been consumed
    std::shared_ptr<std::atomic<uint32_t>> readCount = nullptr;
};

struct ScoreWithIndex {
    float score;
    uint32_t index;
    uint64_t id{0};

    ScoreWithIndex(float s, uint32_t i) : score(s), index(i) {}
};

struct FullRecallResult {
    // true = mode 1 (multi-shard search); false = mode 2 (single-shard index).
    bool isMultiShard;
    // Two response modes (saves one extra pass for performance):
    // mode 1 (multi-shard): results live in scoreWithIndexes, ready to sort after merging shards.
    std::vector<ScoreWithIndex> scoreWithIndexes;
    // mode 2 (single-shard): results live in scores + docIds; scoreWithIndexes is scratch only.
    std::vector<float> scores;
    std::vector<int64_t> docIds;

    uint32_t scoreResultIndex = 0;
    std::shared_ptr<GmBlock> resultChunk = nullptr;
};
}  // namespace NpuRetrieval
