/*
 * Stub for NpuRetrieval `src/workflow/async/executor.h`.
 *
 * `engine/` uses this as a task-parallel primitive on the search path:
 *
 *     m_executor = CreateExecutor();
 *     auto ctx = m_executor->CreateExecuteContext(logContext);   // LogContext&
 *     ctx->AddTask([...]() -> ErrorCode::ResultType { ... });    // score
 *     ctx->AddTask([...]() -> ErrorCode::ResultType { ... });    // filter
 *     ctx->Wait([&](ErrorCode::ResultType r){ ... });            // join
 *
 * Concurrency is controlled by env NPUR_EXECUTOR_THREADS:
 *   - unset / <=1 (DEFAULT): tasks run SYNCHRONOUSLY in submission order. This
 *     is the known-good path (the searcher touches the NPU and is validated
 *     serial); keeps behaviour identical to bring-up.
 *   - N>1: up to N tasks run concurrently on a small worker pool. This unlocks
 *     the offline builder's per-segment parallelism (CPU-only, safe) and the
 *     searcher's score||filter overlap. Set it for `fr_builder` runs; leave it
 *     default (serial) for the searcher unless NPU concurrency is validated.
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "src/common/error_code.h"
#include "src/utils/env_switch.h"

namespace NpuRetrieval {

// Worker count from env NPUR_EXECUTOR_THREADS (default 1 = serial), read once.
inline int ExecutorThreads() {
    static int n = [] { return static_cast<int>(npur_env::PositiveOr("NPUR_EXECUTOR_THREADS", 1)); }();
    return n;
}

// Throttled progress printing for large task batches (e.g. the builder's per-
// segment tasks). Prints at most every ~2s to stderr. Enabled by default when a
// Wait() has many tasks; disable with NPUR_PROGRESS=0.
inline bool ProgressEnabled() {
    static bool on = [] { return npur_env::OnByDefaultNumeric("NPUR_PROGRESS"); }();
    return on;
}

// Opaque logging/trace context threaded through the real executor. Unused here.
class LogContext {
   public:
    LogContext() = default;
};

class ExecuteContext {
   public:
    using Task = std::function<ErrorCode::ResultType()>;

    void AddTask(Task task) {
        m_tasks.emplace_back(std::move(task));
    }

    // Runs every queued task and invokes `onResult` once per task with its
    // result (in submission order), mirroring the real Wait() callback fan-out.
    // Serial when NPUR_EXECUTOR_THREADS<=1, else up to N tasks concurrently.
    void Wait(const std::function<void(ErrorCode::ResultType)>& onResult) {
        const size_t total = m_tasks.size();
        const int threads = ExecutorThreads();
        const bool progress = ProgressEnabled() && total >= 8;

        // Throttled progress reporter (thread-safe, at most one line / ~2s).
        auto t0 = std::chrono::steady_clock::now();
        std::mutex printMtx;
        std::chrono::steady_clock::time_point lastPrint = t0;
        auto report = [&](size_t done) {
            if (!progress)
                return;
            std::lock_guard<std::mutex> lk(printMtx);
            auto now = std::chrono::steady_clock::now();
            // The final "done" line only when the run lasted long enough to be worth a
            // progress report; otherwise short, repeated executor runs (e.g. per-batch
            // aggregation, which finishes in ~0s) each spam one line and flood the output.
            const bool longRun = now - t0 >= std::chrono::seconds(2);
            const bool due = (done == total) ? longRun : (now - lastPrint >= std::chrono::seconds(2));
            if (due) {
                lastPrint = now;
                double sec = std::chrono::duration<double>(now - t0).count();
                std::fprintf(stderr, "[progress] %zu/%zu tasks done (%.1fs)\n", done, total, sec);
            }
        };

        if (threads <= 1 || total <= 1) {
            size_t done = 0;
            for (auto& task : m_tasks) {
                ErrorCode::ResultType ret = task ? task() : ErrorCode::ResultType::FAIL;
                if (onResult)
                    onResult(ret);
                report(++done);
            }
            m_tasks.clear();
            return;
        }
        // Bounded worker pool: each worker pulls the next task by atomic index.
        std::vector<ErrorCode::ResultType> results(total, ErrorCode::ResultType::FAIL);
        std::atomic<size_t> next{0};
        std::atomic<size_t> completed{0};
        auto worker = [&] {
            for (;;) {
                size_t i = next.fetch_add(1);
                if (i >= total)
                    break;
                results[i] = m_tasks[i] ? m_tasks[i]() : ErrorCode::ResultType::FAIL;
                report(completed.fetch_add(1) + 1);
            }
        };
        int nworkers = std::min<int>(threads, static_cast<int>(total));
        // Heartbeat: individual tasks can take many minutes (a 65536-doc segment
        // build), so completion-driven `report` can be silent for a long time.
        // Emit a liveness line every ~10s while a wave is in flight, coordinated
        // with `report` via lastPrint so we never double-print.
        std::atomic<bool> hbStop{false};
        std::thread heartbeat;
        if (progress) {
            heartbeat = std::thread([&] {
                while (!hbStop.load()) {
                    for (int s = 0; s < 100 && !hbStop.load(); ++s)
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    if (hbStop.load())
                        break;
                    std::lock_guard<std::mutex> lk(printMtx);
                    auto now = std::chrono::steady_clock::now();
                    if (now - lastPrint >= std::chrono::seconds(10)) {
                        lastPrint = now;
                        double sec = std::chrono::duration<double>(now - t0).count();
                        std::fprintf(stderr, "[progress] still building, %zu/%zu done, %d in flight (%.1fs)\n",
                                     completed.load(), total, nworkers, sec);
                    }
                }
            });
        }
        std::vector<std::thread> pool;
        pool.reserve(nworkers);
        for (int k = 0; k < nworkers; ++k)
            pool.emplace_back(worker);
        for (auto& th : pool)
            th.join();
        hbStop.store(true);
        if (heartbeat.joinable())
            heartbeat.join();
        if (onResult) {
            for (ErrorCode::ResultType r : results)
                onResult(r);
        }
        m_tasks.clear();
    }

   private:
    std::vector<Task> m_tasks;
};

class Executor {
   public:
    std::shared_ptr<ExecuteContext> CreateExecuteContext(LogContext& /*logContext*/) {
        return std::make_shared<ExecuteContext>();
    }
};

inline std::shared_ptr<Executor> CreateExecutor() {
    return std::make_shared<Executor>();
}

}  // namespace NpuRetrieval

// The real Executor is built on bthread (brpc); builder_impl.cpp calls
// bthread_setconcurrency() and relies on it being visible transitively through
// this header. With the synchronous stub there is no bthread runtime, so this
// is a no-op that reports success.
inline int bthread_setconcurrency(int /*num*/) {
    return 0;
}
