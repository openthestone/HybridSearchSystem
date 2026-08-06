// Latency timer: cold + warm, p50/p99 over N rounds.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace filter_cmp {

using Clock = std::chrono::steady_clock;
using us_double = double;

struct LatencyReport {
    double cold_us = 0.0;
    double warm_p50_us = 0.0;
    double warm_p99_us = 0.0;
    double mean_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
    uint32_t rounds = 0;
};

template <typename Fn>
LatencyReport MeasureLatency(uint32_t warmup, uint32_t rounds, Fn fn) {
    LatencyReport rep;
    rep.rounds = rounds;

    auto run_once = [&]() -> double {
        auto t0 = Clock::now();
        fn();
        auto t1 = Clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count();
    };

    // Cold run (first invocation)
    rep.cold_us = run_once();

    for (uint32_t i = 1; i < warmup; ++i)
        run_once();

    std::vector<double> samples;
    samples.reserve(rounds);
    for (uint32_t i = 0; i < rounds; ++i)
        samples.push_back(run_once());

    std::sort(samples.begin(), samples.end());
    auto pct = [&](double q) -> double {
        if (samples.empty())
            return 0.0;
        size_t idx = (size_t)((samples.size() - 1) * q);
        return samples[idx];
    };
    rep.warm_p50_us = pct(0.50);
    rep.warm_p99_us = pct(0.99);
    rep.min_us = samples.front();
    rep.max_us = samples.back();
    rep.mean_us = std::accumulate(samples.begin(), samples.end(), 0.0) / (double)samples.size();
    return rep;
}

}  // namespace filter_cmp
