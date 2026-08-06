#pragma once
#include <functional>
#include <string>
#include "src/full_recall/core/constant_definition.h"
#include <memory>
namespace NpuRetrieval {
// Entry point for building one index shard from either an on-disk data dir or,
// when --build_from_memory is set, a set of streaming IPC sockets.
bool Build(const std::string& schema, uint32_t shardId, const std::string& dataDir, const std::string& indexOutputDir);
}  // namespace NpuRetrieval
