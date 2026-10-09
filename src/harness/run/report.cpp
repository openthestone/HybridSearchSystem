#include "report.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <unordered_set>

#include "../../io/dataset_hw.h"
#include "../recall/recall_ref.h"
#include "flags.h"
#include "src/utils/logger.h"

namespace npur_harness {

void DumpQueryLatencies(const std::vector<double>& perQueryMs, const std::vector<uint32_t>& topks, size_t nq) {
    if (FLAGS_latency_dump.empty())
        return;
    std::FILE* lf = std::fopen(FLAGS_latency_dump.c_str(), "w");
    if (lf == nullptr) {
        LOG_ERROR("cannot open latency_dump file: " << FLAGS_latency_dump);
    } else {
        std::fprintf(lf, "idx\tms\ttopk\n");
        for (size_t q = 0; q < nq; ++q)
            std::fprintf(lf, "%zu\t%.4f\t%d\n", q, perQueryMs[q], topks[q]);
        std::fclose(lf);
        LOG_INFO("wrote per-query latencies to " << FLAGS_latency_dump);
    }
}

void DumpShardLatencies(const std::vector<std::vector<double>>& shardMs, const std::vector<double>& mergeMs,
                        const std::vector<double>& perQueryMs, uint32_t nShards, size_t nq) {
    if (nShards <= 1 || FLAGS_shard_latency_dump.empty())
        return;
    std::FILE* sf = std::fopen(FLAGS_shard_latency_dump.c_str(), "w");
    if (sf == nullptr) {
        LOG_ERROR("cannot open shard_latency_dump file: " << FLAGS_shard_latency_dump);
    } else {
        std::fprintf(sf, "idx");
        for (uint32_t s = 0; s < nShards; ++s)
            std::fprintf(sf, "\tshard%u_ms", s);
        std::fprintf(sf, "\tmax_shard_ms\tmerge_ms\ttotal_ms\n");
        for (size_t q = 0; q < nq; ++q) {
            std::fprintf(sf, "%zu", q);
            double mx = 0.0;
            for (uint32_t s = 0; s < nShards; ++s) {
                std::fprintf(sf, "\t%.4f", shardMs[s][q]);
                if (shardMs[s][q] > mx)
                    mx = shardMs[s][q];
            }
            std::fprintf(sf, "\t%.4f\t%.4f\t%.4f\n", mx, mergeMs[q], perQueryMs[q]);
        }
        std::fclose(sf);
        LOG_INFO("wrote per-shard latencies to " << FLAGS_shard_latency_dump);
    }
}

void PrintLatencySummary(std::vector<double>& latencies_ms, double wallMs, size_t bs) {
    if (latencies_ms.empty())
        return;
    const double paceQps = FLAGS_target_qps;
    const bool paced = paceQps > 0.0;
    std::sort(latencies_ms.begin(), latencies_ms.end());  // latencies_ms holds each query's TRUE e2e
    auto pct = [&](double p) {
        size_t idx = std::min(latencies_ms.size() - 1, static_cast<size_t>(p * latencies_ms.size()));
        return latencies_ms[idx];
    };
    double sum = 0;
    for (double v : latencies_ms)
        sum += v;
    const double avg = sum / static_cast<double>(latencies_ms.size());
    const double qps = wallMs > 0 ? 1000.0 * static_cast<double>(latencies_ms.size()) / wallMs : 0.0;
    if (paced) {
        std::printf(
            "[Result] latency ms: avg=%.4f p50=%.4f p95=%.4f p99=%.4f qps=%.1f (per-query e2e incl. queueing, "
            "offered=%.1f qps, n=%zu, batch_size=%zu)\n",
            avg, pct(0.50), pct(0.95), pct(0.99), qps, paceQps, latencies_ms.size(), bs);
    } else {
        std::printf(
            "[Result] latency ms: avg=%.4f p50=%.4f p95=%.4f p99=%.4f qps=%.1f (per-query e2e, n=%zu, "
            "batch_size=%zu)\n",
            avg, pct(0.50), pct(0.95), pct(0.99), qps, latencies_ms.size(), bs);
    }
}

void CheckRecall(const std::vector<std::vector<float>>& queries, const FilterSet& filters, bool haveFilters,
                 const std::vector<uint32_t>& topks, const std::vector<std::vector<int64_t>>& resultDocIds,
                 uint64_t docNum, size_t nq) {
    if (FLAGS_dataset_hw.empty() || (FLAGS_recall_queries <= 0 && FLAGS_recall_ref_file.empty()))
        return;
    npur_port::HwDataset hw;
    if (!hw.Open(FLAGS_dataset_hw)) {
        LOG_ERROR("cannot open/mmap dataset_HW.bin: " << FLAGS_dataset_hw);
    } else if (hw.doc_num < FLAGS_cpu_doc_offset + docNum) {
        LOG_ERROR("dataset_HW.bin doc_num=" << hw.doc_num
                                            << " < cpu_doc_offset+docNum=" << (FLAGS_cpu_doc_offset + docNum));
    } else {
        const bool cached = !FLAGS_recall_ref_file.empty();
        const size_t rq = cached ? nq : std::min<size_t>(FLAGS_recall_queries, nq);
        RecallRefInputs refIn;
        refIn.datasetPath = FLAGS_dataset_hw;
        refIn.queryPath = FLAGS_query_file;
        refIn.filterPath = FLAGS_filter_file;
        refIn.topkPath = FLAGS_topk_file;
        refIn.docNum = docNum;
        refIn.nq = nq;
        refIn.docOffset = FLAGS_cpu_doc_offset;
        refIn.filterSeed = FLAGS_filter_seed;
        refIn.dim = hw.dim;
        refIn.topk = FLAGS_topk;
        refIn.numQueriesFlag = FLAGS_num_queries;
        refIn.fp32 = FLAGS_recall_fp32;
        const std::string fp = RecallFingerprint(refIn);
        std::vector<std::unordered_set<int64_t>> refs;
        const bool loaded = cached && LoadRecallRef(FLAGS_recall_ref_file, fp, nq, refs);
        if (loaded) {
            std::printf("[RecallRef] cache HIT: %s (%zu queries, %s ground truth)\n", FLAGS_recall_ref_file.c_str(),
                        static_cast<size_t>(nq), FLAGS_recall_fp32 ? "fp32" : "fp16");
            std::fflush(stdout);
        } else {
            unsigned nThreads = FLAGS_recall_ref_threads > 0 ? static_cast<unsigned>(FLAGS_recall_ref_threads)
                                                             : std::max(1u, std::thread::hardware_concurrency());
            std::printf(
                "[RecallRef] cache %s: computing %zu brute-force refs (%s ground truth, %u threads, "
                "one-time)...\n",
                cached ? "MISS/stale" : "disabled", static_cast<size_t>(rq), FLAGS_recall_fp32 ? "fp32" : "fp16",
                nThreads);
            std::fflush(stdout);
            const auto refT0 = std::chrono::steady_clock::now();
            refs.assign(nq, {});
            std::atomic<size_t> nextQ{0};
            auto worker = [&]() {
                for (size_t q = nextQ.fetch_add(1); q < rq; q = nextQ.fetch_add(1)) {
                    const FilterNode* ast = haveFilters ? filters.astFor(q) : nullptr;
                    refs[q] = CpuTopK(queries[q], ast, hw, docNum, static_cast<int>(topks[q]), FLAGS_cpu_doc_offset,
                                      FLAGS_recall_fp32);
                }
            };
            std::vector<std::thread> pool;
            pool.reserve(nThreads);
            for (unsigned t = 0; t < nThreads; ++t)
                pool.emplace_back(worker);
            for (std::thread& th : pool)
                th.join();
            const long long refMs =
                std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - refT0).count();
            if (cached) {
                if (WriteRecallRef(FLAGS_recall_ref_file, fp, refs)) {
                    std::printf("[RecallRef] built in %lld ms, cache written: %s\n", static_cast<long long>(refMs),
                                FLAGS_recall_ref_file.c_str());
                } else {
                    std::printf(
                        "[RecallRef] built in %lld ms, but WRITING THE CACHE FAILED: %s "
                        "(next run will recompute)\n",
                        static_cast<long long>(refMs), FLAGS_recall_ref_file.c_str());
                    LOG_ERROR("failed to write recall ref cache: " << FLAGS_recall_ref_file);
                }
            } else {
                std::printf("[RecallRef] built in %lld ms (no --recall_ref_file, not cached)\n",
                            static_cast<long long>(refMs));
            }
            std::fflush(stdout);
        }
        double recallSum = 0.0;
        size_t counted = 0;
        size_t mismatchLines = 0;
        double minRecall = 1.0;
        // NPUR_RECALL_LINES=N prints the first N mismatching queries.
        static const size_t kMaxMismatchLines = []() -> size_t {
            const char* v = std::getenv("NPUR_RECALL_LINES");
            return v != nullptr ? static_cast<size_t>(std::strtoul(v, nullptr, 10)) : 0;
        }();
        for (size_t q = 0; q < rq; ++q) {
            const std::unordered_set<int64_t>& ref = refs[q];
            if (ref.empty())
                continue;  // filter matched nothing; skip
            size_t hit = 0;
            for (int64_t id : resultDocIds[q])
                if (ref.count(id))
                    ++hit;
            double recall = static_cast<double>(hit) / static_cast<double>(ref.size());
            recallSum += recall;
            ++counted;
            if (recall < 1.0 || (!cached && q < 10)) {
                if (mismatchLines < kMaxMismatchLines)
                    std::printf("[Result] q%zu recall=%.4f (ref=%zu, engine=%zu)\n", q, recall, ref.size(),
                                resultDocIds[q].size());
                ++mismatchLines;
            }
            if (recall < minRecall)
                minRecall = recall;
        }
        if (mismatchLines > kMaxMismatchLines)
            std::printf("[Result] %zu quer%s below recall 1.0 (min %.4f)%s\n", mismatchLines,
                        mismatchLines == 1 ? "y" : "ies", minRecall,
                        kMaxMismatchLines == 0 ? "; NPUR_RECALL_LINES=20 to list them" : " (lines suppressed)");
        if (counted > 0)
            std::printf("[Result] CPU-recall over %zu queries: avg=%.2f%%\n", counted, (recallSum / counted) * 100.0);
        else
            std::printf("[Result] recall: no queries had non-empty reference results\n");
    }
    std::fflush(stdout);
}

}  // namespace npur_harness
