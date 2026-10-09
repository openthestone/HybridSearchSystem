/*
 * report.h -- what fr_search prints and writes once the search is over: the per-query and
 * per-shard latency dumps, the [Result] summary line, and the CPU-recall check.
 *
 * Each of these is a no-op when the flag that asks for it is unset, so main() calls them
 * unconditionally.
 */
#pragma once

#include <cstdint>
#include <vector>

#include "../query/filter_setup.h"

namespace npur_harness {

// --latency_dump: 'idx ms topk' TSV, one line per query. Only meaningful at --batch_size 1.
void DumpQueryLatencies(const std::vector<double>& perQueryMs, const std::vector<uint32_t>& topks, size_t nq);

// --shard_latency_dump: per-query per-shard timing plus the host merge. Multi-card only.
void DumpShardLatencies(const std::vector<std::vector<double>>& shardMs, const std::vector<double>& mergeMs,
                        const std::vector<double>& perQueryMs, uint32_t nShards, size_t nq);

// The [Result] line. Sorts `latencies` in place.
void PrintLatencySummary(std::vector<double>& latencies, double wallMs, size_t bs);

// --recall_queries / --recall_ref_file: brute-force ground truth vs what the engine returned.
// Diagnostic only -- a failure here is reported, never fatal.
void CheckRecall(const std::vector<std::vector<float>>& queries, const FilterSet& filters, bool haveFilters,
                 const std::vector<uint32_t>& topks, const std::vector<std::vector<int64_t>>& resultDocIds,
                 uint64_t docNum, size_t nq);

}  // namespace npur_harness
