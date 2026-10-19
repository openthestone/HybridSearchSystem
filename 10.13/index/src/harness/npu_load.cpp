// fr_npuload: standalone NPU load generator for the contract's simulated load-fluctuation
// condition. Runs beside fr_search and genuinely occupies the listed cards, so the interference
// is real device contention rather than a host-side delay.
//
// Two components, both gated by --duty: `compute` runs the engine's MMad scorer kernel (Cube),
// `copy` runs device-to-device copies (HBM bandwidth). Neither should be run alone -- the
// scorer is HBM-bound, but a load that only moves bytes leaves the compute units visibly idle.
// --duty 1.0 models a permanently slower card, i.e. heterogeneity, not fluctuation.
//
//   fr_npuload --devices 0 --duty 0.5 --period_ms 1000

#include <acl/acl.h>
#include <gflags/gflags.h>

#include "ascend_device/aclrtlaunch_kernel_vector_mmad.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

DEFINE_string(devices, "0", "comma-separated NPU device ids to load (e.g. 0 or 0,3)");
DEFINE_double(duty, 0.5,
              "fraction of each period the cards are contended (0.5 = half the time). 1.0 makes "
              "the load constant, i.e. a permanently slower card rather than a fluctuating one.");
DEFINE_double(period_ms, 1000.0, "length of one contended/free cycle in ms");
DEFINE_double(seconds, 0.0, "stop after this many seconds (0 = run until signalled)");
DEFINE_double(report_s, 5.0, "print achieved load every this many seconds (0 = quiet)");

// ---- compute component (Cube, via the engine's MMad scorer kernel) ----------
DEFINE_int32(compute_workers, 1, "concurrent scorer-kernel streams per device (0 disables the compute load)");
DEFINE_int32(compute_docs, 1048576, "synthetic doc count per launch; must be a multiple of both b1N and --compute_zn");
DEFINE_int32(compute_dim, 64, "vector dimension: 32, 64 or 128 (picks b1N 2048/1024/512, as in the engine)");
DEFINE_int32(compute_zn, 16, "Zn split (the index's split_doc_num_zn); b1N must be divisible by it");
DEFINE_int32(compute_queries, 16, "synthetic query count per launch (padded to 16 like the engine)");
DEFINE_int32(compute_block_dim, 20, "cores per launch; 20 matches the engine's scorer block dim");

// ---- bandwidth component (HBM, via device-to-device copies) -----------------
DEFINE_int32(copy_workers, 1, "concurrent copy streams per device (0 disables the bandwidth load)");
DEFINE_int32(copy_buffer_mb, 256, "size of each copy buffer in MiB");
DEFINE_double(copy_gap_ms, 2.0,
              "idle gap after each copy round. This is the bandwidth throttle: 0 saturates HBM and "
              "starves the engine, while a small gap leaves it a usable share.");

namespace {

std::atomic<bool> g_stop{false};

void OnSignal(int) {
    g_stop.store(true, std::memory_order_relaxed);
}

std::vector<int32_t> ParseDevices(const std::string& csv) {
    std::vector<int32_t> out;
    size_t pos = 0;
    while (pos <= csv.size()) {
        const size_t comma = csv.find(',', pos);
        std::string tok = csv.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        const size_t b = tok.find_first_not_of(" \t");
        const size_t e = tok.find_last_not_of(" \t");
        if (b != std::string::npos)
            out.push_back(static_cast<int32_t>(std::strtol(tok.substr(b, e - b + 1).c_str(), nullptr, 10)));
        if (comma == std::string::npos)
            break;
        pos = comma + 1;
    }
    return out;
}

// Same table VectorScorerMmad::GetB1N uses -- sized so dimension*b1N*3*sizeof(fp16) fits L1.
uint16_t B1NFor(int dim) {
    switch (dim) {
        case 32:
            return 2048;
        case 64:
            return 1024;
        case 128:
            return 512;
        default:
            return 0;
    }
}

// True while this instant falls in the contended part of the current period.
bool InActiveWindow(std::chrono::steady_clock::time_point t0) {
    if (FLAGS_duty >= 1.0)
        return true;
    if (FLAGS_duty <= 0.0)
        return false;
    const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const double phase = elapsed - FLAGS_period_ms * std::floor(elapsed / FLAGS_period_ms);
    return phase < FLAGS_duty * FLAGS_period_ms;
}

void SleepThroughIdleWindow() {
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
}

void ComputeWorker(int32_t device, std::chrono::steady_clock::time_point t0, std::atomic<uint64_t>* launches) {
    if (aclrtSetDevice(device) != ACL_SUCCESS)
        return;
    const uint16_t b1N = B1NFor(FLAGS_compute_dim);
    const uint32_t zn = static_cast<uint32_t>(FLAGS_compute_zn);
    const uint32_t extendDocNum = static_cast<uint32_t>(FLAGS_compute_docs);
    const uint32_t extendQueryNum = static_cast<uint32_t>(FLAGS_compute_queries);
    const uint32_t dim = static_cast<uint32_t>(FLAGS_compute_dim);
    uint32_t blockDocNum = extendDocNum / static_cast<uint32_t>(FLAGS_compute_block_dim);
    blockDocNum = ((blockDocNum + b1N - 1) / b1N) * b1N;  // round up to a multiple of b1N, as the engine does
    const uint32_t blockDim = (extendDocNum + blockDocNum - 1) / blockDocNum;

    const size_t docBytes = static_cast<size_t>(extendDocNum) * dim * sizeof(uint16_t);
    const size_t queryBytes = static_cast<size_t>(extendQueryNum) * dim * sizeof(uint16_t);
    const size_t resultBytes = static_cast<size_t>(extendDocNum) * extendQueryNum * sizeof(float);

    void *doc = nullptr, *query = nullptr, *result = nullptr;
    aclrtStream stream = nullptr;
    const bool ok = aclrtMalloc(&doc, docBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
                    aclrtMalloc(&query, queryBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
                    aclrtMalloc(&result, resultBytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
                    aclrtCreateStream(&stream) == ACL_SUCCESS;
    if (!ok) {
        std::fprintf(stderr, "[NpuLoad] device %d: compute alloc failed (needs ~%.1f GB; lower --compute_docs)\n",
                     device, static_cast<double>(docBytes + resultBytes) / (1024.0 * 1024.0 * 1024.0));
    } else {
        aclrtMemset(doc, docBytes, 0, docBytes);
        aclrtMemset(query, queryBytes, 0, queryBytes);
        while (!g_stop.load(std::memory_order_relaxed)) {
            if (!InActiveWindow(t0)) {
                SleepThroughIdleWindow();
                continue;
            }
            // mOut == extendQueryNum: the disturbance wants the kernel's FULL cost.
            ACLRT_LAUNCH_KERNEL(kernel_vector_mmad)
            (blockDim, stream, reinterpret_cast<uint8_t*>(query), reinterpret_cast<uint8_t*>(doc),
             reinterpret_cast<uint8_t*>(result), static_cast<uint16_t>(extendQueryNum),
             static_cast<uint16_t>(extendQueryNum), extendDocNum, static_cast<uint16_t>(dim), b1N,
             static_cast<uint16_t>(zn), zn, blockDocNum);
            aclrtSynchronizeStream(stream);
            launches->fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (stream != nullptr)
        aclrtDestroyStream(stream);
    if (doc != nullptr)
        aclrtFree(doc);
    if (query != nullptr)
        aclrtFree(query);
    if (result != nullptr)
        aclrtFree(result);
    aclrtResetDevice(device);
}

void CopyWorker(int32_t device, std::chrono::steady_clock::time_point t0, std::atomic<uint64_t>* bytesMoved) {
    if (aclrtSetDevice(device) != ACL_SUCCESS)
        return;
    const size_t bytes = static_cast<size_t>(FLAGS_copy_buffer_mb) * 1024u * 1024u;
    void *src = nullptr, *dst = nullptr;
    aclrtStream stream = nullptr;
    const bool ok = aclrtMalloc(&src, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
                    aclrtMalloc(&dst, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS &&
                    aclrtCreateStream(&stream) == ACL_SUCCESS;
    if (!ok) {
        std::fprintf(stderr, "[NpuLoad] device %d: copy alloc failed (lower --copy_buffer_mb)\n", device);
    } else {
        while (!g_stop.load(std::memory_order_relaxed)) {
            if (!InActiveWindow(t0)) {
                SleepThroughIdleWindow();
                continue;
            }
            aclrtMemcpyAsync(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
            aclrtSynchronizeStream(stream);
            // A D2D copy both reads and writes HBM, so it costs twice the buffer in traffic.
            bytesMoved->fetch_add(static_cast<uint64_t>(bytes) * 2u, std::memory_order_relaxed);
            if (FLAGS_copy_gap_ms > 0.0)
                std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(FLAGS_copy_gap_ms));
        }
    }
    if (stream != nullptr)
        aclrtDestroyStream(stream);
    if (src != nullptr)
        aclrtFree(src);
    if (dst != nullptr)
        aclrtFree(dst);
    aclrtResetDevice(device);
}

}  // namespace

int main(int argc, char** argv) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    std::signal(SIGINT, OnSignal);
    std::signal(SIGTERM, OnSignal);

    const std::vector<int32_t> devices = ParseDevices(FLAGS_devices);
    if (devices.empty()) {
        std::fprintf(stderr, "[NpuLoad] --devices is empty\n");
        return 1;
    }
    if (FLAGS_compute_workers > 0) {
        // A launch with inconsistent tiling would read outside the synthetic buffers.
        const uint16_t b1N = B1NFor(FLAGS_compute_dim);
        if (b1N == 0) {
            std::fprintf(stderr, "[NpuLoad] --compute_dim must be 32, 64 or 128\n");
            return 1;
        }
        if (FLAGS_compute_zn <= 0 || (b1N % FLAGS_compute_zn) != 0) {
            std::fprintf(stderr, "[NpuLoad] b1N (%u) must be divisible by --compute_zn (%d)\n", b1N, FLAGS_compute_zn);
            return 1;
        }
        if (FLAGS_compute_docs <= 0 || (FLAGS_compute_docs % b1N) != 0 ||
            (FLAGS_compute_docs % FLAGS_compute_zn) != 0) {
            std::fprintf(stderr, "[NpuLoad] --compute_docs (%d) must be a multiple of b1N (%u) and --compute_zn (%d)\n",
                         FLAGS_compute_docs, b1N, FLAGS_compute_zn);
            return 1;
        }
        if (FLAGS_compute_queries <= 0 || (FLAGS_compute_queries % 16) != 0) {
            std::fprintf(stderr, "[NpuLoad] --compute_queries must be a positive multiple of 16\n");
            return 1;
        }
    }
    if (FLAGS_compute_workers <= 0 && FLAGS_copy_workers <= 0) {
        std::fprintf(stderr, "[NpuLoad] both load components are disabled; nothing to do\n");
        return 1;
    }
    if (aclInit(nullptr) != ACL_SUCCESS) {
        std::fprintf(stderr, "[NpuLoad] aclInit failed\n");
        return 1;
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<std::atomic<uint64_t>> launches(devices.size());
    std::vector<std::atomic<uint64_t>> moved(devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
        launches[i].store(0, std::memory_order_relaxed);
        moved[i].store(0, std::memory_order_relaxed);
    }

    std::vector<std::thread> workers;
    for (size_t i = 0; i < devices.size(); ++i) {
        for (int w = 0; w < FLAGS_compute_workers; ++w)
            workers.emplace_back(ComputeWorker, devices[i], t0, &launches[i]);
        for (int w = 0; w < FLAGS_copy_workers; ++w)
            workers.emplace_back(CopyWorker, devices[i], t0, &moved[i]);
        std::printf("[NpuLoad] device %d: %d compute worker(s) + %d copy worker(s), duty %.2f over %.0fms\n",
                    devices[i], FLAGS_compute_workers, FLAGS_copy_workers, FLAGS_duty, FLAGS_period_ms);
    }
    std::fflush(stdout);

    auto lastReport = std::chrono::steady_clock::now();
    std::vector<uint64_t> lastBytes(devices.size(), 0), lastLaunch(devices.size(), 0);
    while (!g_stop.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const auto now = std::chrono::steady_clock::now();
        if (FLAGS_seconds > 0.0 && std::chrono::duration<double>(now - t0).count() >= FLAGS_seconds)
            g_stop.store(true, std::memory_order_relaxed);
        if (FLAGS_report_s > 0.0 && std::chrono::duration<double>(now - lastReport).count() >= FLAGS_report_s) {
            const double dt = std::chrono::duration<double>(now - lastReport).count();
            for (size_t i = 0; i < devices.size(); ++i) {
                const uint64_t curB = moved[i].load(std::memory_order_relaxed);
                const uint64_t curL = launches[i].load(std::memory_order_relaxed);
                const double gbs = static_cast<double>(curB - lastBytes[i]) / dt / (1024.0 * 1024.0 * 1024.0);
                const double kps = static_cast<double>(curL - lastLaunch[i]) / dt;
                lastBytes[i] = curB;
                lastLaunch[i] = curL;
                std::printf("[NpuLoad] device %d: %.1f GB/s copy traffic + %.1f scorer kernels/s (duty-averaged)\n",
                            devices[i], gbs, kps);
            }
            std::fflush(stdout);
            lastReport = now;
        }
    }

    for (auto& t : workers)
        t.join();
    aclFinalize();
    std::printf("[NpuLoad] stopped\n");
    return 0;
}
