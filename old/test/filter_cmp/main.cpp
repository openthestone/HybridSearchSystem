// main.cpp — CPU vs NPU attribute filter micro-benchmark harness.
//
// Three paths:
//   cpu     — forward bitmap, multi-thread CPU per-doc RPN eval (sks_hw style)
//   simple  — inverted BIT_SET, scalar AIV kernel (naive port)
//   tianji  — inverted BIT_SET, vectorized AIV kernel (Tianji port)
//
// Each path runs warmup+rounds iterations. Reports cold, warm p50/p99, mean.
// Verifies all NPU paths against CPU reference.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "common/cpu_filter.h"
#include "common/expr.h"
#include "common/raw_data.h"
#include "common/timer.h"
#include "npu_common/acl_guard.h"
#include "npu_common/npu_posting.h"
#include "npu_simple/simple_filter.h"
#include "npu_tianji/tianji_filter.h"

namespace {

struct Args {
    uint32_t doc_num = 100000;
    uint32_t tag_num = 1024;
    uint32_t segments_num = 10;
    double tag_density = 0.05;
    uint32_t expr_leaves = 6;
    uint32_t warmup = 30;
    uint32_t rounds = 300;
    uint32_t cpu_threads = std::thread::hardware_concurrency();
    uint32_t npu_block_dim = 8;
    uint64_t data_seed = 0xC0FFEEULL;
    uint64_t expr_seed = 0xBEEFCAFEULL;
    bool skip_npu = false;
    bool skip_simple = false;
    bool skip_tianji = false;
    bool verify = true;
    uint32_t verify_sweep = 0;  // if >0, run N random exprs through all paths
};

void PrintUsage(const char* prog) {
    std::fprintf(stderr,
                 "Usage: %s [opts]\n"
                 "  --doc_num=N          default 100000\n"
                 "  --tag_num=N          default 1024\n"
                 "  --segments_num=N     default 10\n"
                 "  --tag_density=F      default 0.05\n"
                 "  --expr_leaves=N      default 6\n"
                 "  --warmup=N           default 30\n"
                 "  --rounds=N           default 300\n"
                 "  --cpu_threads=N      default hardware_concurrency\n"
                 "  --npu_block_dim=N    default 8\n"
                 "  --data_seed=N        default 0xC0FFEE\n"
                 "  --expr_seed=N        default 0xBEEFCAFE\n"
                 "  --skip_npu           skip both NPU paths\n"
                 "  --skip_simple        skip npu_simple\n"
                 "  --skip_tianji        skip npu_tianji\n"
                 "  --no_verify          skip correctness check\n"
                 "  --verify_sweep=N     run N random exprs, verify each against CPU ref\n",
                 prog);
}

bool ParseUint(const char* s, uint32_t* out) {
    char* end = nullptr;
    unsigned long v = std::strtoul(s, &end, 0);
    if (end == s || *end != '\0')
        return false;
    *out = (uint32_t)v;
    return true;
}
bool ParseU64(const char* s, uint64_t* out) {
    char* end = nullptr;
    unsigned long long v = std::strtoull(s, &end, 0);
    if (end == s || *end != '\0')
        return false;
    *out = (uint64_t)v;
    return true;
}
bool ParseDouble(const char* s, double* out) {
    char* end = nullptr;
    double v = std::strtod(s, &end);
    if (end == s || *end != '\0')
        return false;
    *out = v;
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
        } else if (key == "--doc_num")
            ParseUint(val.c_str(), &a.doc_num);
        else if (key == "--tag_num")
            ParseUint(val.c_str(), &a.tag_num);
        else if (key == "--segments_num")
            ParseUint(val.c_str(), &a.segments_num);
        else if (key == "--tag_density")
            ParseDouble(val.c_str(), &a.tag_density);
        else if (key == "--expr_leaves")
            ParseUint(val.c_str(), &a.expr_leaves);
        else if (key == "--warmup")
            ParseUint(val.c_str(), &a.warmup);
        else if (key == "--rounds")
            ParseUint(val.c_str(), &a.rounds);
        else if (key == "--cpu_threads")
            ParseUint(val.c_str(), &a.cpu_threads);
        else if (key == "--npu_block_dim")
            ParseUint(val.c_str(), &a.npu_block_dim);
        else if (key == "--data_seed")
            ParseU64(val.c_str(), &a.data_seed);
        else if (key == "--expr_seed")
            ParseU64(val.c_str(), &a.expr_seed);
        else if (key == "--skip_npu")
            a.skip_npu = true;
        else if (key == "--skip_simple")
            a.skip_simple = true;
        else if (key == "--skip_tianji")
            a.skip_tianji = true;
        else if (key == "--no_verify")
            a.verify = false;
        else if (key == "--verify_sweep")
            ParseUint(val.c_str(), &a.verify_sweep);
        else {
            std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            PrintUsage(argv[0]);
            std::exit(2);
        }
    }
    return a;
}

void PrintReport(const char* name, const filter_cmp::LatencyReport& r) {
    std::printf(
        "  %-10s  cold=%9.1f us   warm p50=%9.1f us   p99=%9.1f us   "
        "mean=%9.1f   min=%9.1f   max=%9.1f   (rounds=%u)\n",
        name, r.cold_us, r.warm_p50_us, r.warm_p99_us, r.mean_us, r.min_us, r.max_us, r.rounds);
}

uint32_t MismatchCount(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    uint32_t diff = 0;
    if (a.size() != b.size())
        return (uint32_t)std::max(a.size(), b.size());
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i])
            ++diff;
    return diff;
}

uint32_t MatchCount(const std::vector<uint8_t>& r) {
    uint32_t c = 0;
    for (auto v : r)
        if (v)
            ++c;
    return c;
}

}  // namespace

int main(int argc, char** argv) {
    Args args = ParseArgs(argc, argv);

    // doc_num must divide segments_num for even segment split. Round down.
    uint32_t seg_docs = args.doc_num / args.segments_num;
    seg_docs = (seg_docs / 16u) * 16u;  // multiple of 16 for clean bitset
    if (seg_docs < 16u)
        seg_docs = 16u;
    uint32_t actual_doc_num = seg_docs * args.segments_num;
    if (actual_doc_num != args.doc_num) {
        std::fprintf(stderr, "[info] doc_num adjusted %u -> %u (seg_docs=%u, segs=%u)\n", args.doc_num, actual_doc_num,
                     seg_docs, args.segments_num);
    }

    std::printf("=== filter_cmp ===\n");
    std::printf("docs=%u tags=%u segs=%u seg_docs=%u density=%.4f expr_leaves=%u\n", actual_doc_num, args.tag_num,
                args.segments_num, seg_docs, args.tag_density, args.expr_leaves);
    std::printf("warmup=%u rounds=%u cpu_threads=%u npu_block_dim=%u\n", args.warmup, args.rounds, args.cpu_threads,
                args.npu_block_dim);

    // 1. Build raw forward bitmap
    auto t_build0 = filter_cmp::Clock::now();
    auto fwd = filter_cmp::BuildForwardBitmap(actual_doc_num, args.tag_num, args.tag_density, args.data_seed);
    auto t_build1 = filter_cmp::Clock::now();
    std::printf(
        "[setup] forward bitmap: %.1f ms (%zu bytes)\n",
        filter_cmp::us_double(std::chrono::duration_cast<std::chrono::microseconds>(t_build1 - t_build0).count()) /
            1000.0,
        fwd.bits.size() * sizeof(uint64_t));

    // 2. Build inverted postings
    auto t_inv0 = filter_cmp::Clock::now();
    auto posts = filter_cmp::BuildInvertedPostings(fwd, args.segments_num);
    auto t_inv1 = filter_cmp::Clock::now();
    std::printf(
        "[setup] inverted postings: %.1f ms (%zu bytes, seg_u16=%u padded=%u)\n",
        filter_cmp::us_double(std::chrono::duration_cast<std::chrono::microseconds>(t_inv1 - t_inv0).count()) / 1000.0,
        posts.ByteSize(), posts.seg_u16_count_real, posts.seg_u16_count);

    // 3. Build expr
    auto expr = filter_cmp::BuildRandomExpr(args.tag_num, args.expr_leaves, args.expr_seed);
    std::printf("[setup] expr: %s\n", expr.pretty.c_str());

    // 4. CPU reference + benchmark
    auto t_ref0 = filter_cmp::Clock::now();
    auto cpu_ref = filter_cmp::CpuFilterForward(fwd, expr, args.cpu_threads);
    auto t_ref1 = filter_cmp::Clock::now();
    std::printf(
        "[setup] CPU reference: %.1f ms (matches=%u/%u)\n",
        filter_cmp::us_double(std::chrono::duration_cast<std::chrono::microseconds>(t_ref1 - t_ref0).count()) / 1000.0,
        MatchCount(cpu_ref), actual_doc_num);

    std::printf("\n=== Latency (per-query, microseconds) ===\n");

    // CPU path
    auto cpu_report = filter_cmp::MeasureLatency(args.warmup, args.rounds, [&]() {
        auto r = filter_cmp::CpuFilterForward(fwd, expr, args.cpu_threads);
        if (r.empty())
            std::abort();
    });
    PrintReport("cpu", cpu_report);

    // NPU paths
    if (!args.skip_npu) {
        try {
            filter_cmp::AclStream acl(0);

            if (!args.skip_simple) {
                filter_cmp::SimpleFilter sf(acl, posts, expr, 64);
                if (args.verify) {
                    auto r = sf.Run(posts, expr);
                    uint32_t diff = MismatchCount(r, cpu_ref);
                    std::printf("[verify] simple: %u mismatches (matches=%u)\n", diff, MatchCount(r));
                }
                auto rep = filter_cmp::MeasureLatency(args.warmup, args.rounds, [&]() {
                    auto r = sf.Run(posts, expr);
                    if (r.empty())
                        std::abort();
                });
                PrintReport("npu_simple", rep);
            }

            if (!args.skip_tianji) {
                filter_cmp::TianjiFilter tf(acl, posts, expr, args.npu_block_dim);
                if (args.verify) {
                    auto r = tf.Run(posts, expr);
                    uint32_t diff = MismatchCount(r, cpu_ref);
                    std::printf("[verify] tianji: %u mismatches (matches=%u)\n", diff, MatchCount(r));
                }
                auto rep = filter_cmp::MeasureLatency(args.warmup, args.rounds, [&]() {
                    auto r = tf.Run(posts, expr);
                    if (r.empty())
                        std::abort();
                });
                PrintReport("npu_tianji", rep);
            }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] NPU path failed: %s\n", e.what());
            return 1;
        }
    }

    // Multi-expr correctness sweep
    if (args.verify_sweep > 0) {
        std::printf("\n=== Verify sweep: %u random exprs ===\n", args.verify_sweep);
        uint32_t simple_fails = 0, tianji_fails = 0;
        uint32_t simple_total_mismatch = 0, tianji_total_mismatch = 0;
        std::unique_ptr<filter_cmp::SimpleFilter> sf;
        std::unique_ptr<filter_cmp::TianjiFilter> tf;
        try {
            filter_cmp::AclStream acl(0);
            if (!args.skip_simple && !args.skip_npu) {
                sf = std::make_unique<filter_cmp::SimpleFilter>(acl, posts, expr, 64);
            }
            if (!args.skip_tianji && !args.skip_npu) {
                tf = std::make_unique<filter_cmp::TianjiFilter>(acl, posts, expr, args.npu_block_dim);
            }
            for (uint32_t k = 0; k < args.verify_sweep; ++k) {
                auto e = filter_cmp::BuildRandomExpr(args.tag_num,
                                                     1 + (k % 12),  // leaf count 1..12
                                                     args.expr_seed + k);
                auto ref = filter_cmp::CpuFilterForward(fwd, e, args.cpu_threads);
                uint32_t simple_diff = 0, tianji_diff = 0;
                if (sf) {
                    auto r = sf->Run(posts, e);
                    simple_diff = MismatchCount(r, ref);
                    if (simple_diff > 0) {
                        ++simple_fails;
                        simple_total_mismatch += simple_diff;
                    }
                }
                if (tf) {
                    auto r = tf->Run(posts, e);
                    tianji_diff = MismatchCount(r, ref);
                    if (tianji_diff > 0) {
                        ++tianji_fails;
                        tianji_total_mismatch += tianji_diff;
                    }
                }
                std::printf("  expr %u: leaves=%u rpn_len=%u ref_matches=%u", k, (uint32_t)e.tree.leaf_count,
                            (uint32_t)e.rpn.size(), MatchCount(ref));
                if (sf)
                    std::printf("  simple_diff=%u", simple_diff);
                if (tf)
                    std::printf("  tianji_diff=%u", tianji_diff);
                std::printf("\n");
            }
            std::printf("[sweep] simple: %u/%u exprs failed, %u total mismatched docs\n", simple_fails,
                        args.verify_sweep, simple_total_mismatch);
            std::printf("[sweep] tianji: %u/%u exprs failed, %u total mismatched docs\n", tianji_fails,
                        args.verify_sweep, tianji_total_mismatch);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "[error] sweep failed: %s\n", e.what());
            return 1;
        }
    }

    std::printf("\n=== done ===\n");
    return 0;
}
