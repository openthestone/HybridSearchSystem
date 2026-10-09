#pragma once
#include <string>
namespace NpuRetrieval {
/*
 * --build_from_memory: instead of reading an input dir, the builder is fed over a set of unix
 * domain sockets. `dataDir` is then a comma-separated list of socket paths, one receiver thread
 * each, draining into MemoryDataManager. A no-op when the flag is off, so Build() calls it either
 * way.
 *
 * The packet framing and the header's JSON keys are a wire contract with the sender; they live in
 * stream_input.cpp and must not change.
 */
bool LoadStreamingDataIfNeeded(const std::string& dataDir);
}  // namespace NpuRetrieval
