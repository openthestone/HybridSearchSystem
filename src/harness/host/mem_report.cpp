#include "mem_report.h"

#include <cstdio>
#include <cstring>

#include "acl/acl.h"

namespace npur_harness {

long ReadHostKb(const char* key) {
    long v = -1;
    if (std::FILE* f = std::fopen("/proc/self/status", "r")) {
        char line[256];
        const size_t klen = std::strlen(key);
        while (std::fgets(line, sizeof(line), f) != nullptr) {
            if (std::strncmp(line, key, klen) == 0) {
                std::sscanf(line + klen, "%ld", &v);
                break;
            }
        }
        std::fclose(f);
    }
    return v;
}

MemSnap SampleMem(const std::vector<int32_t>& devices) {
    MemSnap s;
    s.rssKb = ReadHostKb("VmRSS:");
    for (int32_t d : devices) {
        long usedKb = -1, totalKb = -1;
        if (aclrtSetDevice(d) == ACL_SUCCESS) {
            size_t freeB = 0, totalB = 0;
            if (aclrtGetMemInfo(ACL_HBM_MEM, &freeB, &totalB) == ACL_SUCCESS && totalB > 0) {
                usedKb = static_cast<long>((totalB - freeB) / 1024);
                totalKb = static_cast<long>(totalB / 1024);
            }
        }
        s.usedKb.push_back(usedKb);
        s.totalKb.push_back(totalKb);
    }
    return s;
}

void PrintMemReport(const std::vector<int32_t>& devices, const MemSnap& base, const MemSnap& load,
                    const MemSnap& search, long hostHwmKb) {
    auto mb = [](long kb) { return kb / 1024.0; };
    std::printf("[MEM] ==== memory report (delta vs previous milestone) ====\n");
    std::printf("[MEM] host RSS MB: baseline=%.1f  after_load=%.1f (%+.1f)  after_search=%.1f (%+.1f)  peak=%.1f\n",
                mb(base.rssKb), mb(load.rssKb), mb(load.rssKb - base.rssKb), mb(search.rssKb),
                mb(search.rssKb - load.rssKb), mb(hostHwmKb));
    for (size_t i = 0; i < devices.size(); ++i) {
        std::printf(
            "[MEM] dev%d HBM MB: baseline=%.1f  after_load=%.1f (%+.1f idx)  after_search=%.1f (%+.1f run)  "
            "total=%.1f\n",
            devices[i], mb(base.usedKb[i]), mb(load.usedKb[i]), mb(load.usedKb[i] - base.usedKb[i]),
            mb(search.usedKb[i]), mb(search.usedKb[i] - load.usedKb[i]), mb(search.totalKb[i]));
    }
}
}  // namespace npur_harness
