/*
 * Stub for NpuRetrieval `src/common/npuretrieval_constants.h`.
 *
 * Only two constants are referenced by `engine/` (in data_table_repository.cpp):
 *   - INDEX_SHARD_PREFIX_NUMS : zero-pad width of the per-shard sub-directory
 *                               name (multi-shard layout: <dataDir>/00000, ...).
 *   - NPU_MEMORY_WARNING_THRESHOLD : HBM usage ratio above which an alarm fires.
 *
 * NOTE: values below are reasonable defaults for bring-up. INDEX_SHARD_PREFIX_NUMS
 * must match the shard directory naming produced by the builder — verify against
 * the real constant before trusting multi-shard runs (single-shard path unaffected).
 */
#pragma once

#include <cstddef>

namespace NpuRetrieval {

constexpr int INDEX_SHARD_PREFIX_NUMS = 5;
constexpr double NPU_MEMORY_WARNING_THRESHOLD = 0.9;

}  // namespace NpuRetrieval
