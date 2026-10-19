#pragma once

#include <string>
#include "murmurhash/MurmurHash3.h"

namespace NpuRetrieval {
// Computes a hash of a string |str|.
// WARNING: This hash function should not be used for any cryptographic purpose.
uint32_t Hash32(const char* data, uint32_t len);

inline uint32_t Hash32(const std::string& str) {
    return Hash32(str.data(), str.size());
}

uint64_t Hash64(const char* data, uint32_t len);

inline uint64_t Hash64(const std::string& str) {
    return Hash64(str.data(), str.size());
}
}  // namespace NpuRetrieval
