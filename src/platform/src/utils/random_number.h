/*
 * Stub for NpuRetrieval `src/utils/random_number.h`.
 *
 * data_table_repository.cpp includes this header but (in the extracted code)
 * does not call anything from it. We still provide a small thread-safe helper
 * in case other paths use it. Uniform integer in [min, max].
 */
#pragma once

#include <cstdint>
#include <random>

namespace NpuRetrieval {

inline uint64_t RandomNumber(uint64_t min, uint64_t max) {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    if (max <= min) {
        return min;
    }
    std::uniform_int_distribution<uint64_t> dist(min, max);
    return dist(rng);
}

}  // namespace NpuRetrieval
