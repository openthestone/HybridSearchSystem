/*
 * Stub for NpuRetrieval `src/utils/performance_recorder.h`.
 *
 * `engine/` code uses `RecordGuard guard{"stage name"};` as an RAII stage timer
 * scattered across the search path (22 sites). This stub records wall-clock
 * duration per stage tag; enable printing via env NPUR_PERF=1. Keeping it
 * lightweight avoids pulling in the real recorder infrastructure while still
 * being useful for the Phase 8 latency work.
 */
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace NpuRetrieval {

class RecordGuard {
   public:
    explicit RecordGuard(const std::string& tag) : m_tag(tag), m_start(std::chrono::steady_clock::now()) {}

    ~RecordGuard() {
        static const bool enabled = (std::getenv("NPUR_PERF") != nullptr);
        if (!enabled) {
            return;
        }
        auto us =
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - m_start).count();
        // Print straight to stderr (not the logger) so per-stage timings show
        // regardless of NPUR_LOG_LEVEL. Gate: NPUR_PERF set.
        std::fprintf(stderr, "[PERF] %-44s %8lld us\n", m_tag.c_str(), static_cast<long long>(us));
    }

    RecordGuard(const RecordGuard&) = delete;
    RecordGuard& operator=(const RecordGuard&) = delete;

   private:
    std::string m_tag;
    std::chrono::steady_clock::time_point m_start;
};

}  // namespace NpuRetrieval
