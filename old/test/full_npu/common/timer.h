// common/timer.h — latency measurement helpers.

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace full_npu {

using Clock = std::chrono::steady_clock;

struct StageLatency {
    double score_ms = 0;
    double filter_ms = 0;
    double agg_ms = 0;
    double topk_ms = 0;
    double total_ms = 0;
};

class Timer {
   public:
    void Start() {
        start_ = Clock::now();
    }
    double StopMs() {
        auto end = Clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

   private:
    Clock::time_point start_;
};

struct LatencyStats {
    double mean_ms = 0;
    double p50_ms = 0;
    double p99_ms = 0;
    double min_ms = 0;
    double max_ms = 0;
};

inline LatencyStats ComputeStats(std::vector<double>& samples) {
    LatencyStats s;
    if (samples.empty())
        return s;
    std::sort(samples.begin(), samples.end());
    size_t n = samples.size();
    double sum = 0;
    for (double v : samples)
        sum += v;
    s.mean_ms = sum / n;
    s.min_ms = samples.front();
    s.max_ms = samples.back();
    s.p50_ms = samples[(size_t)(n * 0.50)];
    s.p99_ms = samples[(size_t)(std::min<size_t>(n - 1, (size_t)(n * 0.99)))];
    return s;
}

inline void PrintStats(const char* name, const std::vector<double>& samples) {
    std::vector<double> s = samples;
    auto st = ComputeStats(s);
    std::printf("  %-12s mean=%7.3f ms  p50=%7.3f  p99=%7.3f  min=%7.3f  max=%7.3f  (n=%zu)\n", name, st.mean_ms,
                st.p50_ms, st.p99_ms, st.min_ms, st.max_ms, samples.size());
}

}  // namespace full_npu
