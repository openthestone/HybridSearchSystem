/*
 * mem_report.h -- host RSS and per-device NPU HBM, sampled at three milestones (baseline, after
 * index load, after search) and printed as [MEM] lines under --mem_report.
 */
#pragma once

#include <cstdint>
#include <vector>

namespace npur_harness {

struct MemSnap {
    long rssKb = -1;
    std::vector<long> usedKb;
    std::vector<long> totalKb;
};

// One /proc/self/status key, e.g. "VmRSS:" or "VmHWM:". -1 when unavailable.
long ReadHostKb(const char* key);

// Binds each device in turn, so do not call it while another thread is driving them.
MemSnap SampleMem(const std::vector<int32_t>& devices);

void PrintMemReport(const std::vector<int32_t>& devices, const MemSnap& base, const MemSnap& load,
                    const MemSnap& search, long hostHwmKb);

}  // namespace npur_harness
