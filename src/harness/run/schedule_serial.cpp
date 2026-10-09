#include "schedule.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "../host/cpu_affinity.h"
#include "../query/filter_setup.h"
#include "acl/acl.h"
#include "flags.h"
#include "src/full_recall/retrieval/query/query_node_imp.h"
#include "src/full_recall/retrieval/searcher/full_recall_searcher.h"
#include "src/utils/logger.h"

namespace npur_harness {

bool RunSerial(RunContext& rc) {
    for (int w = 0; w < std::max(0, FLAGS_warmup) && rc.nq > 0; ++w)
        rc.RunBatch(0, std::min(rc.nq, rc.bs), /*record=*/false);

    rc.paceT0 = std::chrono::steady_clock::now();  // arrival clock origin: query 0 is offered now
    const auto serialT0 = rc.paceT0;
    for (size_t start = 0; start < rc.nq; start += rc.bs) {
        size_t end = std::min(rc.nq, start + rc.bs);
        auto arrival = std::chrono::steady_clock::now();
        if (rc.paced) {
            arrival = rc.arrivalOf(start);
            std::this_thread::sleep_until(arrival);  // no-op once we fall behind (saturated)
        }
        auto t0 = std::chrono::steady_clock::now();
        bool ok = rc.RunBatch(start, end, /*record=*/true);
        auto done = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(done - (rc.paced ? arrival : t0)).count();
        if (!ok) {
            LOG_ERROR("search failed at batch [" << start << "," << end << ")");
            return false;
        }
        for (size_t k = 0; k < (end - start); ++k) {
            rc.latencies_ms.push_back(ms);
            rc.perQueryMs[start + k] = ms;
        }
    }
    rc.wallMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - serialT0).count();
    return true;
}

}  // namespace npur_harness
