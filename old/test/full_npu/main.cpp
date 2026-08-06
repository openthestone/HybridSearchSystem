// main.cpp — full_npu Phase A.1 harness.
//
// Loads dataset_HW.bin (subset), runs N random queries through CPU brute force,
// reports per-stage latency stats. Phase A.2 will add NPU kernel paths that
// compare against this CPU reference.
//
// Build:
//   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
//   cmake --build build -j
//
// Run:
//   ./build/full_npu --dataset=/root/sks_hw/dataset_HW.bin --doc_subset=10000 \
//                    --leaves=100 --topk=100 --queries=20

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "common/cpu_ref.h"
#include "common/dataset_hw.h"
#include "common/expr.h"
#include "common/timer.h"

#ifdef FULL_NPU_ENABLE_NPU
#include "npu/aggregator_launcher.h"
#include "npu/filter_launcher.h"
#include "npu/inverted_index_loader.h"
#include "npu/fused_pipeline.h"
#include "npu/score_launcher.h"
#include "npu/session.h"
#include "npu/topk_launcher.h"
#endif

namespace {

struct Args {
    std::string dataset = "/root/sks_hw/dataset_HW.bin";
    uint64_t doc_subset = 10000;  // 0 = full corpus
    uint32_t leaves = 100;
    uint32_t topk = 100;
    uint32_t queries = 20;
    int threads = 16;  // LOCKED (avoid oversubscription)
    uint64_t data_seed = 0xC0FFEEULL;
    uint64_t expr_seed = 0xBEEFCAFEULL;
    bool print_matches = false;
    bool npu_score = false;       // Phase A.2: run NPU score + verify
    bool npu_filter = false;      // Phase A.2: run NPU filter + verify
    bool npu_invfilter = false;   // Phase A.3+: inverted-index filter
    bool npu_aggregator = false;  // Phase A.2: run NPU aggregator + verify
    bool npu_topk = false;        // Phase A.2: run NPU topk + verify
    bool npu_full = false;        // Phase A.3: end-to-end pipeline (chained launchers)
    bool npu_fused = false;       // Phase A.3+: device-fused pipeline
    bool npu_fused_inv = false;   // Phase A.3+: fused + inverted filter
    std::string inv_index = "/root/sks_hw/test/full_npu/dataset_HW_inv.bin";
    uint32_t npu_verify_count = 256;  // how many docs to verify vs CPU
    uint32_t npu_score_repeat = 5;    // sweep repeats for stable latency
    uint32_t npu_filter_repeat = 3;   // filter repeats
    uint32_t npu_agg_repeat = 5;      // aggregator repeats
    uint32_t npu_topk_repeat = 5;     // topk repeats
};

void PrintUsage(const char* prog) {
    std::fprintf(stderr,
                 "Usage: %s [opts]\n"
                 "  --dataset=PATH       default /root/sks_hw/dataset_HW.bin\n"
                 "  --doc_subset=N       default 10000 (0 = full 10M corpus)\n"
                 "  --leaves=N           default 100 (Boolean expr leaf count)\n"
                 "  --topk=N             default 100\n"
                 "  --queries=N          default 20 (random queries per config)\n"
                 "  --threads=N          default 16 (LOCKED — do not oversubscribe)\n"
                 "  --data_seed=N        default 0xC0FFEE\n"
                 "  --expr_seed=N        default 0xBEEFCAFE\n"
                 "  --print_matches      print per-query match counts\n"
                 "  --npu_score          Phase A.2: run NPU score kernel + verify vs CPU\n"
                 "  --npu_filter         Phase A.2: run NPU filter kernel + verify vs CPU\n"
                 "  --npu_invfilter      Phase A.3+: run inverted-index filter + verify\n"
                 "  --npu_aggregator     Phase A.2: run NPU aggregator + verify vs CPU compaction\n"
                 "  --npu_topk           Phase A.2: run NPU topk + verify vs CPU sort\n"
                 "  --npu_full           Phase A.3: end-to-end score+filter+aggregator+topk pipeline\n"
                 "  --npu_fused          Phase A.3+: device-side fused pipeline (skip inter-stage D2H/H2D)\n"
                 "  --npu_fused_inv      Phase A.3+: fused pipeline with inverted-index filter\n"
                 "  --inv_index=PATH     default /root/sks_hw/test/full_npu/dataset_HW_inv.bin\n"
                 "  --npu_verify_count=N default 256 (docs verified vs CPU dot product)\n"
                 "  --npu_score_repeat=N default 5 (kernel repeats for stable latency)\n"
                 "  --npu_filter_repeat=N default 3 (filter repeats)\n"
                 "  --npu_agg_repeat=N   default 5 (aggregator repeats)\n"
                 "  --npu_topk_repeat=N  default 5 (topk repeats)\n",
                 prog);
}

bool ParseU64(const char* s, uint64_t* out) {
    char* end = nullptr;
    unsigned long long v = std::strtoull(s, &end, 0);
    if (end == s || *end != '\0')
        return false;
    *out = (uint64_t)v;
    return true;
}
bool ParseU32(const char* s, uint32_t* out) {
    char* end = nullptr;
    unsigned long v = std::strtoul(s, &end, 0);
    if (end == s || *end != '\0')
        return false;
    *out = (uint32_t)v;
    return true;
}
bool ParseI(const char* s, int* out) {
    char* end = nullptr;
    long v = std::strtol(s, &end, 0);
    if (end == s || *end != '\0')
        return false;
    *out = (int)v;
    return true;
}

Args ParseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto eq = arg.find('=');
        std::string key = eq == std::string::npos ? arg : arg.substr(0, eq);
        std::string val = eq == std::string::npos ? "" : arg.substr(eq + 1);
        if (key == "--help" || key == "-h") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else if (key == "--dataset")
            a.dataset = val;
        else if (key == "--doc_subset")
            ParseU64(val.c_str(), &a.doc_subset);
        else if (key == "--leaves")
            ParseU32(val.c_str(), &a.leaves);
        else if (key == "--topk")
            ParseU32(val.c_str(), &a.topk);
        else if (key == "--queries")
            ParseU32(val.c_str(), &a.queries);
        else if (key == "--threads")
            ParseI(val.c_str(), &a.threads);
        else if (key == "--data_seed")
            ParseU64(val.c_str(), &a.data_seed);
        else if (key == "--expr_seed")
            ParseU64(val.c_str(), &a.expr_seed);
        else if (key == "--print_matches")
            a.print_matches = true;
        else if (key == "--npu_score")
            a.npu_score = true;
        else if (key == "--npu_filter")
            a.npu_filter = true;
        else if (key == "--npu_invfilter")
            a.npu_invfilter = true;
        else if (key == "--npu_fused_inv")
            a.npu_fused_inv = true;
        else if (key == "--inv_index")
            a.inv_index = val;
        else if (key == "--npu_aggregator")
            a.npu_aggregator = true;
        else if (key == "--npu_topk")
            a.npu_topk = true;
        else if (key == "--npu_full")
            a.npu_full = true;
        else if (key == "--npu_fused")
            a.npu_fused = true;
        else if (key == "--npu_verify_count")
            ParseU32(val.c_str(), &a.npu_verify_count);
        else if (key == "--npu_score_repeat")
            ParseU32(val.c_str(), &a.npu_score_repeat);
        else if (key == "--npu_filter_repeat")
            ParseU32(val.c_str(), &a.npu_filter_repeat);
        else if (key == "--npu_agg_repeat")
            ParseU32(val.c_str(), &a.npu_agg_repeat);
        else if (key == "--npu_topk_repeat")
            ParseU32(val.c_str(), &a.npu_topk_repeat);
        else {
            std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            PrintUsage(argv[0]);
            std::exit(2);
        }
    }
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    Args args = ParseArgs(argc, argv);

    if (args.threads > 16) {
        std::fprintf(stderr, "[warn] threads=%d exceeds 16 — oversubscription regression\n", args.threads);
    }

    full_npu::DatasetHW ds;
    if (!ds.Open(args.dataset)) {
        std::fprintf(stderr, "[fatal] failed to open dataset\n");
        return 1;
    }
    uint64_t doc_subset = ds.DocNum(args.doc_subset);
    std::printf("=== full_npu Phase A.1 (CPU baseline) ===\n");
    std::printf("dataset:     %s\n", args.dataset.c_str());
    std::printf("doc_num:     %llu (subset %llu)\n", (unsigned long long)ds.DocNum(0), (unsigned long long)doc_subset);
    std::printf("vector_dim:  %u\n", ds.VectorDim());
    std::printf("tag_num:     %u (stride %u word)\n", ds.TagNum(), ds.Stride());
    std::printf("leaves:      %u\n", args.leaves);
    std::printf("topk:        %u\n", args.topk);
    std::printf("queries:     %u\n", args.queries);
    std::printf("threads:     %d\n", args.threads);
    std::printf("\n");

    // Generate queries: random vector + random Boolean expr per query.
    std::mt19937_64 data_rng(args.data_seed);
    std::vector<std::vector<float>> query_vecs(args.queries);
    for (uint32_t q = 0; q < args.queries; ++q) {
        query_vecs[q].resize(ds.VectorDim());
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (uint32_t d = 0; d < ds.VectorDim(); ++d)
            query_vecs[q][d] = nd(data_rng);
    }
    std::vector<full_npu::BooleanExpr> exprs(args.queries);
    for (uint32_t q = 0; q < args.queries; ++q) {
        exprs[q] = full_npu::GenRandomExpr(ds.TagNum(), args.leaves, args.expr_seed + q);
    }

    // Warmup (1 query, not measured).
    {
        auto warm = full_npu::CpuBruteForce(ds, doc_subset, query_vecs[0].data(), exprs[0], args.topk, args.threads);
        std::printf("[warmup] match_count=%llu topk_size=%zu total=%.2f ms\n\n", (unsigned long long)warm.match_count,
                    warm.topk.size(), warm.timing.total_ms);
    }

    // Measured run.
    std::vector<double> total_ms;
    std::vector<double> filter_ms;
    std::vector<uint64_t> matches;
    total_ms.reserve(args.queries);
    filter_ms.reserve(args.queries);
    matches.reserve(args.queries);

    for (uint32_t q = 0; q < args.queries; ++q) {
        auto r = full_npu::CpuBruteForce(ds, doc_subset, query_vecs[q].data(), exprs[q], args.topk, args.threads);
        total_ms.push_back(r.timing.total_ms);
        filter_ms.push_back(r.timing.filter_ms);
        matches.push_back(r.match_count);
        if (args.print_matches) {
            std::printf("  q%u: matches=%llu total=%.2f ms\n", q, (unsigned long long)r.match_count, r.timing.total_ms);
        }
    }

    std::printf("=== Latency (per-query, ms) ===\n");
    full_npu::PrintStats("total", total_ms);
    full_npu::PrintStats("filter", filter_ms);

    // Match count stats.
    std::vector<uint64_t> m_sorted = matches;
    std::sort(m_sorted.begin(), m_sorted.end());
    double m_sum = 0;
    for (auto v : m_sorted)
        m_sum += v;
    std::printf("\n=== Match count ===\n");
    std::printf("  mean=%.1f  min=%llu  p50=%llu  p99=%llu  max=%llu\n", m_sum / m_sorted.size(),
                (unsigned long long)m_sorted.front(), (unsigned long long)m_sorted[m_sorted.size() / 2],
                (unsigned long long)m_sorted[(size_t)(m_sorted.size() * 0.99)], (unsigned long long)m_sorted.back());

    std::printf("\n=== done (Phase A.1 — CPU baseline only) ===\n");

#ifdef FULL_NPU_ENABLE_NPU
    if (args.npu_score) {
        std::printf("\n=== Phase A.2: NPU score kernel ===\n");
        std::printf("doc_subset:  %llu\n", (unsigned long long)doc_subset);
        std::printf("verify_count: %u (CPU dot product cross-check)\n", args.npu_verify_count);
        std::printf("repeat:      %u (averaged latency)\n", args.npu_score_repeat);

        try {
            full_npu::AclSession sess(0);
            full_npu::ScoreLauncher launcher(sess, ds.VectorDim());

            // Convert doc_subset docs to a flat FP32 buffer.
            // docs_f32[d * k + i] = ds.Vector(d)[i]
            const uint32_t k_dim = ds.VectorDim();
            std::vector<float> docs_f32((size_t)doc_subset * k_dim);
            for (uint64_t d = 0; d < doc_subset; ++d) {
                const float* v = ds.Vector(d);
                std::memcpy(&docs_f32[(size_t)d * k_dim], v, k_dim * sizeof(float));
            }

            // Use query 0.
            const float* q0 = query_vecs[0].data();

            // Run a few times for stable latency.
            std::vector<float> npu_scores(doc_subset, 0.0f);
            std::vector<double> npu_ms;
            for (uint32_t r = 0; r < args.npu_score_repeat; ++r) {
                double ms = launcher.Run(q0, docs_f32.data(), (uint32_t)doc_subset, &npu_scores);
                npu_ms.push_back(ms);
            }

            // CPU reference on first npu_verify_count docs.
            uint32_t nv = std::min<uint32_t>(args.npu_verify_count, (uint32_t)doc_subset);
            std::vector<float> cpu_scores(nv, 0.0f);
            full_npu::Timer t_cpu;
            t_cpu.Start();
#pragma omp parallel for num_threads(args.threads) schedule(static)
            for (uint32_t d = 0; d < nv; ++d) {
                const float* v = ds.Vector(d);
                float s = 0.0f;
                for (uint32_t i = 0; i < k_dim; ++i)
                    s += v[i] * q0[i];
                cpu_scores[d] = s;
            }
            double cpu_ms = t_cpu.StopMs();

            // Max abs diff + max rel diff.
            float max_abs = 0.0f, max_rel = 0.0f;
            uint32_t worst_d = 0;
            for (uint32_t d = 0; d < nv; ++d) {
                float diff = std::abs(npu_scores[d] - cpu_scores[d]);
                if (diff > max_abs) {
                    max_abs = diff;
                    worst_d = d;
                    float denom = std::max(std::abs(cpu_scores[d]), 1e-6f);
                    max_rel = diff / denom;
                }
            }

            std::printf("\n--- correctness (FP16 kernel vs FP32 CPU) ---\n");
            std::printf("verified docs: %u / %llu\n", nv, (unsigned long long)doc_subset);
            std::printf("max abs diff:  %.6f (doc %u, npu=%.4f cpu=%.4f)\n", max_abs, worst_d, npu_scores[worst_d],
                        cpu_scores[worst_d]);
            std::printf("max rel diff:  %.4f%%\n", max_rel * 100.0f);

            std::printf("\n--- latency (ms, H2D+kernel+D2H+sync) ---\n");
            full_npu::PrintStats("npu_score", npu_ms);
            std::printf("cpu_score (FP32 OMP):  %.3f ms  (verify subset, %u docs)\n", cpu_ms, nv);

            // Throughput ratio (rough): NPU processes all doc_subset, CPU only nv.
            double npu_mean = 0;
            for (auto v : npu_ms)
                npu_mean += v;
            npu_mean /= npu_ms.size();
            double cpu_proj = cpu_ms * ((double)doc_subset / nv);
            std::printf("cpu projected (full subset): %.3f ms\n", cpu_proj);
            std::printf("speedup: %.2fx\n", cpu_proj / npu_mean);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[npu_score] error: %s\n", e.what());
            return 3;
        }
    }

    if (args.npu_filter) {
        std::printf("\n=== Phase A.2: NPU filter kernel ===\n");
        std::printf("doc_subset:  %llu\n", (unsigned long long)doc_subset);
        std::printf("repeat:      %u\n", args.npu_filter_repeat);

        try {
            full_npu::AclSession sess(0);
            full_npu::FilterLauncher launcher(sess);

            // Use expr 0 against doc_subset docs.
            const auto& expr0 = exprs[0];
            std::printf("expr0: leaves=%u rpn_tokens=%zu\n", expr0.leaves, expr0.rpn.size());

            // Verify RPN encoding sanity.
            std::printf("rpn_u32 encoded: %zu tokens\n", launcher.RpnU32().size());

            // Run NPU filter a few times.
            std::vector<uint8_t> npu_result;
            std::vector<double> npu_ms;
            for (uint32_t r = 0; r < args.npu_filter_repeat; ++r) {
                double ms = launcher.Run(expr0, ds.Bitmap(0), ds.Stride(), (uint32_t)doc_subset, &npu_result);
                npu_ms.push_back(ms);
            }

            // CPU reference: EvalExpr on each doc.
            std::vector<uint8_t> cpu_result(doc_subset, 0);
            full_npu::Timer t_cpu;
            t_cpu.Start();
#pragma omp parallel for num_threads(args.threads) schedule(static)
            for (uint64_t d = 0; d < doc_subset; ++d) {
                cpu_result[d] = full_npu::EvalExpr(expr0, ds.Bitmap(d)) ? 1 : 0;
            }
            double cpu_ms = t_cpu.StopMs();

            // Compare.
            uint64_t mismatches = 0;
            uint64_t cpu_matches = 0, npu_matches = 0;
            for (uint64_t d = 0; d < doc_subset; ++d) {
                if (cpu_result[d] != npu_result[d])
                    ++mismatches;
                if (cpu_result[d])
                    ++cpu_matches;
                if (npu_result[d])
                    ++npu_matches;
            }
            double acc = 1.0 - (double)mismatches / (double)doc_subset;

            std::printf("\n--- correctness (NPU vs CPU EvalExpr) ---\n");
            std::printf("docs:           %llu\n", (unsigned long long)doc_subset);
            std::printf("mismatches:     %llu\n", (unsigned long long)mismatches);
            std::printf("accuracy:       %.6f\n", acc);
            std::printf("cpu matches:    %llu\n", (unsigned long long)cpu_matches);
            std::printf("npu matches:    %llu\n", (unsigned long long)npu_matches);

            std::printf("\n--- latency (ms, H2D bitmap+rpn + kernel + D2H) ---\n");
            full_npu::PrintStats("npu_filter", npu_ms);
            std::printf("cpu_filter (EvalExpr OMP): %.3f ms (%d threads)\n", cpu_ms, args.threads);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[npu_filter] error: %s\n", e.what());
            return 4;
        }
    }

    if (args.npu_invfilter) {
        std::printf("\n=== Phase A.3+: NPU inverted-index filter kernel ===\n");
        std::printf("doc_subset:  %llu\n", (unsigned long long)doc_subset);
        std::printf("inv_index:   %s\n", args.inv_index.c_str());
        std::printf("repeat:      %u\n", args.npu_filter_repeat);

        try {
            full_npu::InvertedIndex idx;
            if (!idx.Open(args.inv_index)) {
                std::fprintf(stderr, "[npu_invfilter] failed to load inv index\n");
                return 4;
            }
            if (idx.DocNum() < doc_subset) {
                std::fprintf(stderr, "[npu_invfilter] inv index doc_num=%llu < subset=%llu\n",
                             (unsigned long long)idx.DocNum(), (unsigned long long)doc_subset);
                return 4;
            }

            full_npu::AclSession sess(0);
            full_npu::InvertedFilterLauncher launcher(sess, idx);

            const auto& expr0 = exprs[0];
            std::printf("expr0: leaves=%u rpn_tokens=%zu\n", expr0.leaves, expr0.rpn.size());

            // Unique tag count.
            std::vector<uint32_t> uniq_tags;
            for (const auto& tok : expr0.rpn) {
                if (tok.kind == full_npu::TokenKind::TAG || tok.kind == full_npu::TokenKind::NOT_TAG) {
                    uniq_tags.push_back(tok.arg);
                }
            }
            std::sort(uniq_tags.begin(), uniq_tags.end());
            uniq_tags.erase(std::unique(uniq_tags.begin(), uniq_tags.end()), uniq_tags.end());
            std::printf("unique tags in expr: %zu\n", uniq_tags.size());
            size_t compact_bytes = uniq_tags.size() * idx.Segments() * sizeof(uint16_t);
            std::printf("compact postings: %zu bytes (%.1f KB) vs forward bitmap %llu bytes (%.1f MB)\n", compact_bytes,
                        compact_bytes / 1024.0, (unsigned long long)doc_subset * ds.Stride() * 8,
                        doc_subset * ds.Stride() * 8 / 1048576.0);

            std::vector<uint32_t> npu_result;
            std::vector<double> npu_ms;
            for (uint32_t r = 0; r < args.npu_filter_repeat; ++r) {
                double ms = launcher.Run(expr0, (uint32_t)doc_subset, &npu_result);
                npu_ms.push_back(ms);
            }

            // CPU reference.
            std::vector<uint8_t> cpu_result(doc_subset, 0);
            full_npu::Timer t_cpu;
            t_cpu.Start();
#pragma omp parallel for num_threads(args.threads) schedule(static)
            for (uint64_t d = 0; d < doc_subset; ++d) {
                cpu_result[d] = full_npu::EvalExpr(expr0, ds.Bitmap(d)) ? 1 : 0;
            }
            double cpu_ms = t_cpu.StopMs();

            uint64_t mismatches = 0;
            uint64_t cpu_matches = 0, npu_matches = 0;
            for (uint64_t d = 0; d < doc_subset; ++d) {
                uint32_t nv = npu_result[d] ? 1 : 0;
                uint8_t cv = cpu_result[d];
                if (cv != nv)
                    ++mismatches;
                if (cv)
                    ++cpu_matches;
                if (nv)
                    ++npu_matches;
            }
            double acc = 1.0 - (double)mismatches / (double)doc_subset;

            std::printf("\n--- correctness (NPU inverted vs CPU EvalExpr) ---\n");
            std::printf("docs:           %llu\n", (unsigned long long)doc_subset);
            std::printf("mismatches:     %llu\n", (unsigned long long)mismatches);
            std::printf("accuracy:       %.6f\n", acc);
            std::printf("cpu matches:    %llu\n", (unsigned long long)cpu_matches);
            std::printf("npu matches:    %llu\n", (unsigned long long)npu_matches);

            std::printf("\n--- latency (ms, H2D compact postings+rpn + kernel + D2H) ---\n");
            full_npu::PrintStats("npu_invfilter", npu_ms);
            std::printf("cpu_filter (EvalExpr OMP): %.3f ms (%d threads)\n", cpu_ms, args.threads);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[npu_invfilter] error: %s\n", e.what());
            return 4;
        }
    }

    if (args.npu_aggregator) {
        std::printf("\n=== Phase A.2: NPU aggregator kernel ===\n");
        std::printf("doc_subset:  %llu\n", (unsigned long long)doc_subset);
        std::printf("repeat:      %u\n", args.npu_agg_repeat);

        try {
            full_npu::AclSession sess(0);
            full_npu::AggregatorLauncher launcher(sess);

            const uint32_t k_dim = ds.VectorDim();
            const float* q0 = query_vecs[0].data();
            const auto& expr0 = exprs[0];

            // --- CPU reference: FP32 scores + EvalExpr filter + compact ---
            std::vector<float> cpu_scores(doc_subset, 0.0f);
            std::vector<uint8_t> cpu_filter(doc_subset, 0);
            full_npu::Timer t_cpu;
            t_cpu.Start();
#pragma omp parallel for num_threads(args.threads) schedule(static)
            for (uint64_t d = 0; d < doc_subset; ++d) {
                const float* v = ds.Vector(d);
                float s = 0.0f;
                for (uint32_t i = 0; i < k_dim; ++i)
                    s += v[i] * q0[i];
                cpu_scores[d] = s;
                cpu_filter[d] = full_npu::EvalExpr(expr0, ds.Bitmap(d)) ? 1 : 0;
            }
            double cpu_ms = t_cpu.StopMs();

            std::vector<uint32_t> cpu_ids;
            std::vector<float> cpu_sc;
            cpu_ids.reserve(doc_subset);
            cpu_sc.reserve(doc_subset);
            for (uint64_t d = 0; d < doc_subset; ++d) {
                if (cpu_filter[d]) {
                    cpu_ids.push_back((uint32_t)d);
                    cpu_sc.push_back(cpu_scores[d]);
                }
            }
            std::printf("cpu compacted: %zu matched / %llu total\n", cpu_ids.size(), (unsigned long long)doc_subset);

            // --- NPU aggregator inputs: scores (FP32) + filter (u32 0/1) ---
            std::vector<uint32_t> npu_filter_u32(doc_subset, 0);
            for (uint64_t d = 0; d < doc_subset; ++d) {
                npu_filter_u32[d] = cpu_filter[d] ? 1u : 0u;
            }

            // Reuse CPU scores directly (already FP32 on host).
            std::vector<uint32_t> npu_out_ids;
            std::vector<float> npu_out_scores;
            uint32_t npu_count = 0;
            std::vector<double> npu_ms;
            for (uint32_t r = 0; r < args.npu_agg_repeat; ++r) {
                double ms = launcher.Run(cpu_scores.data(), npu_filter_u32.data(), (uint32_t)doc_subset, &npu_out_ids,
                                         &npu_out_scores, &npu_count);
                npu_ms.push_back(ms);
            }

            // --- Correctness: count + id-set + scores ---
            std::printf("\n--- correctness (NPU vs CPU compaction) ---\n");
            std::printf("npu count:     %u\n", npu_count);
            std::printf("cpu count:     %zu\n", cpu_ids.size());
            bool count_ok = (npu_count == cpu_ids.size());
            std::printf("count match:   %s\n", count_ok ? "yes" : "NO");

            // Sort NPU output ids (write order is non-deterministic due to atomic).
            std::vector<uint32_t> npu_ids_sorted = npu_out_ids;
            std::sort(npu_ids_sorted.begin(), npu_ids_sorted.end());

            // Set equality.
            bool set_ok = (npu_ids_sorted.size() == cpu_ids.size());
            if (set_ok) {
                for (size_t i = 0; i < cpu_ids.size(); ++i) {
                    if (npu_ids_sorted[i] != cpu_ids[i]) {
                        set_ok = false;
                        break;
                    }
                }
            }
            std::printf("id set match:  %s\n", set_ok ? "yes" : "NO");

            // Score correctness: build map id → npu_score, compare vs cpu_scores.
            float max_abs = 0.0f;
            for (uint32_t i = 0; i < npu_count; ++i) {
                uint32_t id = npu_out_ids[i];
                float diff = std::abs(npu_out_scores[i] - cpu_scores[id]);
                if (diff > max_abs)
                    max_abs = diff;
            }
            std::printf("score max abs diff: %.6f (FP32 path, expect ~0)\n", max_abs);

            std::printf("\n--- latency (ms, H2D scores+filter + kernel + D2H count+ids+scores) ---\n");
            full_npu::PrintStats("npu_agg", npu_ms);
            std::printf("cpu_score+filter (OMP): %.3f ms (%d threads, FP32 + EvalExpr)\n", cpu_ms, args.threads);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[npu_aggregator] error: %s\n", e.what());
            return 5;
        }
    }

    if (args.npu_topk) {
        std::printf("\n=== Phase A.2: NPU topk kernel ===\n");
        std::printf("doc_subset:  %llu\n", (unsigned long long)doc_subset);
        std::printf("topk:        %u\n", args.topk);
        std::printf("repeat:      %u\n", args.npu_topk_repeat);

        try {
            full_npu::AclSession sess(0);
            full_npu::TopkLauncher launcher(sess);

            const uint32_t k_dim = ds.VectorDim();
            const float* q0 = query_vecs[0].data();
            const auto& expr0 = exprs[0];

            // CPU reference: score + filter + sort desc → top-K.
            struct ScoreId {
                float s;
                uint32_t id;
            };
            std::vector<ScoreId> cpu_si;
            cpu_si.reserve(doc_subset);
            full_npu::Timer t_cpu;
            t_cpu.Start();
#pragma omp parallel for num_threads(args.threads) schedule(static)
            for (uint64_t d = 0; d < doc_subset; ++d) {
                const float* v = ds.Vector(d);
                float s = 0.0f;
                for (uint32_t i = 0; i < k_dim; ++i)
                    s += v[i] * q0[i];
                bool m = full_npu::EvalExpr(expr0, ds.Bitmap(d));
                if (m) {
#pragma omp critical
                    cpu_si.push_back({s, (uint32_t)d});
                }
            }
            std::sort(cpu_si.begin(), cpu_si.end(), [](const ScoreId& a, const ScoreId& b) { return a.s > b.s; });
            double cpu_ms = t_cpu.StopMs();
            uint32_t cpu_k = std::min<uint32_t>(args.topk, (uint32_t)cpu_si.size());

            // Build NPU input arrays: compact scores + ids (matched docs only).
            std::vector<float> npu_in_scores(cpu_si.size());
            std::vector<uint32_t> npu_in_ids(cpu_si.size());
            for (size_t i = 0; i < cpu_si.size(); ++i) {
                npu_in_scores[i] = cpu_si[i].s;
                npu_in_ids[i] = cpu_si[i].id;
            }

            std::vector<float> npu_topk_scores;
            std::vector<uint32_t> npu_topk_ids;
            std::vector<double> npu_ms;
            for (uint32_t r = 0; r < args.npu_topk_repeat; ++r) {
                double ms = launcher.Run(npu_in_scores.data(), npu_in_ids.data(), (uint32_t)cpu_si.size(), args.topk,
                                         &npu_topk_scores, &npu_topk_ids);
                npu_ms.push_back(ms);
            }

            // Correctness: compare top-K.
            std::printf("\n--- correctness (NPU vs CPU sort, K=%u) ---\n", args.topk);
            std::printf("npu input N:   %zu matched\n", cpu_si.size());

            uint32_t match_count = 0;
            float max_score_diff = 0.0f;
            for (uint32_t i = 0; i < cpu_k; ++i) {
                float cs = cpu_si[i].s;
                uint32_t cid = cpu_si[i].id;
                float ns = (i < npu_topk_scores.size()) ? npu_topk_scores[i] : -1e30f;
                uint32_t nid = (i < npu_topk_ids.size()) ? npu_topk_ids[i] : 0xFFFFFFFFu;
                float diff = std::abs(cs - ns);
                if (diff > max_score_diff)
                    max_score_diff = diff;
                if (cid == nid && diff < 1e-3f)
                    ++match_count;
            }
            std::printf("topk position match: %u / %u\n", match_count, cpu_k);
            std::printf("max score diff:      %.6f\n", max_score_diff);

            // Top-K set equality (id-set, regardless of order within ties).
            std::vector<uint32_t> cpu_set, npu_set;
            for (uint32_t i = 0; i < cpu_k; ++i)
                cpu_set.push_back(cpu_si[i].id);
            for (uint32_t i = 0; i < args.topk && i < npu_topk_ids.size(); ++i) {
                npu_set.push_back(npu_topk_ids[i]);
            }
            std::sort(cpu_set.begin(), cpu_set.end());
            std::sort(npu_set.begin(), npu_set.end());
            bool set_eq = (cpu_set.size() == npu_set.size());
            if (set_eq) {
                for (size_t i = 0; i < cpu_set.size(); ++i) {
                    if (cpu_set[i] != npu_set[i]) {
                        set_eq = false;
                        break;
                    }
                }
            }
            std::printf("topk id set match:  %s\n", set_eq ? "yes" : "NO");

            // Top-3 sample.
            std::printf("\ntop-3 (cpu / npu):\n");
            for (uint32_t i = 0; i < 3 && i < cpu_k; ++i) {
                float ns = (i < npu_topk_scores.size()) ? npu_topk_scores[i] : 0.0f;
                uint32_t nid = (i < npu_topk_ids.size()) ? npu_topk_ids[i] : 0;
                std::printf("  [%u] cpu=(%.4f id=%u)  npu=(%.4f id=%u)\n", i, cpu_si[i].s, cpu_si[i].id, ns, nid);
            }

            std::printf("\n--- latency (ms, H2D scores+ids+count + kernel + D2H top-K) ---\n");
            full_npu::PrintStats("npu_topk", npu_ms);
            std::printf("cpu_score+filter+sort (OMP): %.3f ms (%d threads)\n", cpu_ms, args.threads);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[npu_topk] error: %s\n", e.what());
            return 6;
        }
    }

    if (args.npu_full) {
        std::printf("\n=== Phase A.3: end-to-end NPU pipeline ===\n");
        std::printf("doc_subset:  %llu\n", (unsigned long long)doc_subset);
        std::printf("topk:        %u\n", args.topk);
        std::printf("queries:     %u\n", args.queries);
        std::printf("path:        score(NPU) → filter(NPU) → aggregator(NPU) → topk(NPU)\n");

        try {
            full_npu::AclSession sess(0);
            full_npu::ScoreLauncher score_l(sess, ds.VectorDim());
            full_npu::FilterLauncher filter_l(sess);
            full_npu::AggregatorLauncher agg_l(sess);
            full_npu::TopkLauncher topk_l(sess);

            // Flatten docs to FP32 buffer once for all queries.
            const uint32_t k_dim = ds.VectorDim();
            std::vector<float> docs_f32((size_t)doc_subset * k_dim);
            for (uint64_t d = 0; d < doc_subset; ++d) {
                std::memcpy(&docs_f32[(size_t)d * k_dim], ds.Vector(d), k_dim * sizeof(float));
            }

            // CPU reference (query 0) for correctness check.
            auto cpu_ref =
                full_npu::CpuBruteForce(ds, doc_subset, query_vecs[0].data(), exprs[0], args.topk, args.threads);

            struct PipelineTimings {
                double score_ms, filter_ms, agg_ms, topk_ms, total_ms;
                uint32_t matched;
            };
            std::vector<PipelineTimings> timings;
            timings.reserve(args.queries);

            std::vector<float> npu_scores;
            std::vector<uint8_t> npu_filter_u8;
            std::vector<uint32_t> npu_filter_u32;
            std::vector<uint32_t> agg_ids;
            std::vector<float> agg_scores;
            std::vector<float> topk_scores_q0;
            std::vector<uint32_t> topk_ids_q0;
            uint32_t agg_count = 0;

            for (uint32_t q = 0; q < args.queries; ++q) {
                PipelineTimings tt;
                full_npu::Timer t_total;
                t_total.Start();

                // 1) Score: query · docs → scores (FP32 host out).
                tt.score_ms = score_l.Run(query_vecs[q].data(), docs_f32.data(), (uint32_t)doc_subset, &npu_scores);

                // 2) Filter: bitmap + expr → u8 result.
                tt.filter_ms = filter_l.Run(exprs[q], ds.Bitmap(0), ds.Stride(), (uint32_t)doc_subset, &npu_filter_u8);

                // Convert u8 → u32 for aggregator input.
                npu_filter_u32.assign(doc_subset, 0);
                for (uint64_t d = 0; d < doc_subset; ++d) {
                    npu_filter_u32[d] = npu_filter_u8[d] ? 1u : 0u;
                }

                // 3) Aggregator: compact (scores, filter) → (ids, scores, count).
                tt.agg_ms = agg_l.Run(npu_scores.data(), npu_filter_u32.data(), (uint32_t)doc_subset, &agg_ids,
                                      &agg_scores, &agg_count);
                tt.matched = agg_count;

                // 4) TopK: top-K by score.
                std::vector<float> q_topk_scores;
                std::vector<uint32_t> q_topk_ids;
                tt.topk_ms =
                    topk_l.Run(agg_scores.data(), agg_ids.data(), agg_count, args.topk, &q_topk_scores, &q_topk_ids);

                // Capture q0 result for correctness check.
                if (q == 0) {
                    topk_scores_q0 = std::move(q_topk_scores);
                    topk_ids_q0 = std::move(q_topk_ids);
                }

                tt.total_ms = t_total.StopMs();
                timings.push_back(tt);
            }

            // Latency breakdown.
            std::printf("\n--- per-stage latency (per query, ms) ---\n");
            auto stat = [&](const char* name, double PipelineTimings::* field) {
                std::vector<double> v;
                v.reserve(timings.size());
                for (auto& t : timings)
                    v.push_back(t.*field);
                full_npu::PrintStats(name, v);
            };
            stat("score", &PipelineTimings::score_ms);
            stat("filter", &PipelineTimings::filter_ms);
            stat("agg", &PipelineTimings::agg_ms);
            stat("topk", &PipelineTimings::topk_ms);
            stat("TOTAL", &PipelineTimings::total_ms);

            // Match count stats.
            double m_sum = 0;
            for (auto& t : timings)
                m_sum += t.matched;
            std::printf("\nmatched mean: %.1f (cpu q0 match: %llu)\n", m_sum / timings.size(),
                        (unsigned long long)cpu_ref.match_count);

            // Correctness vs CPU brute force on query 0.
            std::printf("\n--- correctness (query 0, vs CPU brute force top-%u) ---\n", args.topk);
            // CPU top-K set.
            std::vector<uint32_t> cpu_set;
            for (auto& r : cpu_ref.topk)
                cpu_set.push_back(r.doc_id);
            std::sort(cpu_set.begin(), cpu_set.end());

            std::vector<uint32_t> npu_set;
            for (uint32_t i = 0; i < args.topk && i < topk_ids_q0.size(); ++i) {
                npu_set.push_back(topk_ids_q0[i]);
            }
            std::sort(npu_set.begin(), npu_set.end());

            uint32_t set_intersect = 0;
            size_t i = 0, j = 0;
            while (i < cpu_set.size() && j < npu_set.size()) {
                if (cpu_set[i] == npu_set[j]) {
                    ++set_intersect;
                    ++i;
                    ++j;
                } else if (cpu_set[i] < npu_set[j])
                    ++i;
                else
                    ++j;
            }
            double recall = cpu_set.empty() ? 1.0 : (double)set_intersect / (double)cpu_set.size();
            std::printf("recall@%u: %.6f  (intersection %u / cpu %zu)\n", args.topk, recall, set_intersect,
                        cpu_set.size());

            // Top-3 sample (npu).
            std::printf("\ntop-3 NPU (query 0):\n");
            for (uint32_t k = 0; k < 3 && k < topk_scores_q0.size(); ++k) {
                std::printf("  [%u] score=%.4f id=%u\n", k, topk_scores_q0[k], topk_ids_q0[k]);
            }
            std::printf("top-3 CPU (query 0):\n");
            for (uint32_t k = 0; k < 3 && k < cpu_ref.topk.size(); ++k) {
                std::printf("  [%u] score=%.4f id=%u\n", k, cpu_ref.topk[k].score, cpu_ref.topk[k].doc_id);
            }

            // Compare against NPU score-only path FP16 noise expectation.
            // Note: scores use FP16 compute in NPU score kernel, so max diff
            // vs CPU FP32 dot product is bounded by FP16 quantization (~0.1%
            // typical, up to 12% on tiny-magnitude scores).
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[npu_full] error: %s\n", e.what());
            return 7;
        }
    }

    if (args.npu_fused || args.npu_fused_inv) {
        std::printf("\n=== Phase A.3+: device-side fused pipeline ===\n");
        std::printf("doc_subset:  %llu\n", (unsigned long long)doc_subset);
        std::printf("topk:        %u\n", args.topk);
        std::printf("queries:     %u\n", args.queries);
        std::printf("path:        score→filter→aggregator→topk (device-side chained)\n");

        try {
            full_npu::AclSession sess(0);
            full_npu::FusedPipeline fused(sess, ds.VectorDim());
            full_npu::InvertedIndex inv_idx;
            if (args.npu_fused_inv) {
                if (!inv_idx.Open(args.inv_index)) {
                    std::fprintf(stderr, "[npu_fused_inv] failed to load inv index: %s\n", args.inv_index.c_str());
                    return 8;
                }
                if (inv_idx.DocNum() < doc_subset) {
                    std::fprintf(stderr, "[npu_fused_inv] inv doc_num=%llu < subset=%llu\n",
                                 (unsigned long long)inv_idx.DocNum(), (unsigned long long)doc_subset);
                    return 8;
                }
                fused.SetInverted(&inv_idx);
                std::printf("filter:      INVERTED (tag-major postings)\n");
            }  // Flatten docs FP32 once.
            const uint32_t k_dim = ds.VectorDim();
            std::vector<float> docs_f32((size_t)doc_subset * k_dim);
            for (uint64_t d = 0; d < doc_subset; ++d) {
                std::memcpy(&docs_f32[(size_t)d * k_dim], ds.Vector(d), k_dim * sizeof(float));
            }

            auto cpu_ref =
                full_npu::CpuBruteForce(ds, doc_subset, query_vecs[0].data(), exprs[0], args.topk, args.threads);

            std::vector<float> topk_q0_scores;
            std::vector<uint32_t> topk_q0_ids;

            struct TT {
                double h2d, score, filter, agg, topk, d2h, total;
                uint32_t matched;
            };
            std::vector<TT> timings;
            timings.reserve(args.queries);

            std::vector<float> q_scores;
            std::vector<uint32_t> q_ids;
            for (uint32_t q = 0; q < args.queries; ++q) {
                uint32_t matched = 0;
                auto tt_raw = fused.Run(query_vecs[q].data(), docs_f32.data(), exprs[q], ds.Bitmap(0), ds.Stride(),
                                        (uint32_t)doc_subset, args.topk, &q_scores, &q_ids, &matched);
                TT tt{tt_raw.h2d_ms,  tt_raw.score_ms, tt_raw.filter_ms, tt_raw.agg_ms,
                      tt_raw.topk_ms, tt_raw.d2h_ms,   tt_raw.total_ms,  matched};
                timings.push_back(tt);
                if (q == 0) {
                    topk_q0_scores = q_scores;
                    topk_q0_ids = q_ids;
                }
            }

            std::printf("\n--- per-stage latency (per query, ms) ---\n");
            auto stat = [&](const char* name, double TT::* field) {
                std::vector<double> v;
                v.reserve(timings.size());
                for (auto& t : timings)
                    v.push_back(t.*field);
                full_npu::PrintStats(name, v);
            };
            stat("h2d", &TT::h2d);
            stat("score", &TT::score);
            stat("filter", &TT::filter);
            stat("agg", &TT::agg);
            stat("topk", &TT::topk);
            stat("d2h", &TT::d2h);
            stat("TOTAL", &TT::total);

            double m_sum = 0;
            for (auto& t : timings)
                m_sum += t.matched;
            std::printf("\nmatched mean: %.1f (cpu q0 match: %llu)\n", m_sum / timings.size(),
                        (unsigned long long)cpu_ref.match_count);

            // Correctness vs CPU brute force query 0.
            std::printf("\n--- correctness (query 0, vs CPU brute force top-%u) ---\n", args.topk);
            std::vector<uint32_t> cpu_set;
            for (auto& r : cpu_ref.topk)
                cpu_set.push_back(r.doc_id);
            std::sort(cpu_set.begin(), cpu_set.end());

            std::vector<uint32_t> npu_set;
            for (uint32_t i = 0; i < args.topk && i < topk_q0_ids.size(); ++i) {
                npu_set.push_back(topk_q0_ids[i]);
            }
            std::sort(npu_set.begin(), npu_set.end());

            uint32_t set_inter = 0;
            size_t i = 0, j = 0;
            while (i < cpu_set.size() && j < npu_set.size()) {
                if (cpu_set[i] == npu_set[j]) {
                    ++set_inter;
                    ++i;
                    ++j;
                } else if (cpu_set[i] < npu_set[j])
                    ++i;
                else
                    ++j;
            }
            double recall = cpu_set.empty() ? 1.0 : (double)set_inter / (double)cpu_set.size();
            std::printf("recall@%u: %.6f  (intersection %u / cpu %zu)\n", args.topk, recall, set_inter, cpu_set.size());

            std::printf("\ntop-3 NPU (query 0):\n");
            for (uint32_t k = 0; k < 3 && k < topk_q0_scores.size(); ++k) {
                std::printf("  [%u] score=%.4f id=%u\n", k, topk_q0_scores[k], topk_q0_ids[k]);
            }
            std::printf("top-3 CPU (query 0):\n");
            for (uint32_t k = 0; k < 3 && k < cpu_ref.topk.size(); ++k) {
                std::printf("  [%u] score=%.4f id=%u\n", k, cpu_ref.topk[k].score, cpu_ref.topk[k].doc_id);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[npu_fused] error: %s\n", e.what());
            return 8;
        }
    }
#endif

    return 0;
}
