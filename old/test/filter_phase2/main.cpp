// main.cpp — Phase 2 AICore forward-bitmap filter testbed skeleton.
//
// All paths (cpu_forward / aicore_forward / aicpu_mask) are STUBS.
// Filled during Phase 2 §5.2 implementation
// (see 纯NPU路径优化草案.md §5.2).
//
// Skeleton provides:
//   1. Argument parser matching real sks_hw data shape
//      (35672 tags, 1024 doc/bucket, ≤1000-tag expr, density < 0.05%)
//   2. Sweep loop over (bucket_count, leaves, density) matrix
//   3. Per-stage latency report scaffolding (H2D / kernel / D2H)
//
// Real bitmap gen, BucketPlan compiler, and kernel launchers are TODO.
// See README.md "Planned paths" and "Files".

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Args {
    // Real-shape defaults (see 真实数据测试结果-L2桶信息.md)
    uint32_t doc_per_bucket = 1024;  // sks_hw L2 bucket capacity (multiple of 16)
    uint32_t tag_num = 35672;        // production tag count, stride = 558 word
    double density = 0.0005;         // 0.05% — P99+ region (NOT 5%)
    uint32_t expr_leaves = 500;      // business ≤1000 tags
    uint32_t rpn_depth_ratio = 3;    // RPN ≤ 3× leaf count after NOT expansion
    uint32_t cpu_threads = 16;       // LOCKED — do NOT use hardware_concurrency
                                     // (filter_cmp oversubscription lesson, see README)
    // Sweep dimensions
    std::vector<uint32_t> bucket_counts = {1, 100, 800, 1500};  // P50 → P99+ query
    std::vector<uint32_t> leaf_counts = {10, 100, 500, 1000};
    std::vector<double> densities = {0.00005, 0.0005, 0.005, 0.05};

    bool skip_cpu = false;
    bool skip_aicore = false;
    bool skip_aicpu = false;
    bool verify = true;  // NPU paths must match CPU ref
};

void PrintUsage(const char* prog) {
    std::fprintf(stderr,
                 "Usage: %s [opts]\n"
                 "  --doc_per_bucket=N   default 1024 (L2 bucket capacity, multiple of 16)\n"
                 "  --tag_num=N          default 35672 (production tag count, stride 558)\n"
                 "  --density=F          default 0.0005 (0.05%% — P99+ region)\n"
                 "  --expr_leaves=N      default 500 (business ≤1000)\n"
                 "  --rpn_depth_ratio=N  default 3 (RPN ≤ 3× leaves after NOT expansion)\n"
                 "  --cpu_threads=N      default 16 (LOCKED — do not exceed)\n"
                 "  --buckets=N[,N,...]  default 1,100,800,1500 (P50→P99+ sweep)\n"
                 "  --leaves=N[,N,...]   default 10,100,500,1000\n"
                 "  --densities=F[,F,...] default 0.00005,0.0005,0.005,0.05\n"
                 "  --skip_cpu           skip cpu_forward\n"
                 "  --skip_aicore        skip aicore_forward\n"
                 "  --skip_aicpu         skip aicpu_mask\n"
                 "  --no_verify          skip correctness check vs CPU ref\n",
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
bool ParseDouble(const char* s, double* out) {
    char* end = nullptr;
    double v = std::strtod(s, &end);
    if (end == s || *end != '\0')
        return false;
    *out = v;
    return true;
}

std::vector<uint32_t> ParseUintList(const std::string& s) {
    std::vector<uint32_t> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos)
            j = s.size();
        uint32_t v;
        if (ParseUint(s.substr(i, j - i).c_str(), &v))
            out.push_back(v);
        i = j + 1;
    }
    return out;
}
std::vector<double> ParseDoubleList(const std::string& s) {
    std::vector<double> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos)
            j = s.size();
        double v;
        if (ParseDouble(s.substr(i, j - i).c_str(), &v))
            out.push_back(v);
        i = j + 1;
    }
    return out;
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
        } else if (key == "--doc_per_bucket")
            ParseUint(val.c_str(), &a.doc_per_bucket);
        else if (key == "--tag_num")
            ParseUint(val.c_str(), &a.tag_num);
        else if (key == "--density")
            ParseDouble(val.c_str(), &a.density);
        else if (key == "--expr_leaves")
            ParseUint(val.c_str(), &a.expr_leaves);
        else if (key == "--rpn_depth_ratio")
            ParseUint(val.c_str(), &a.rpn_depth_ratio);
        else if (key == "--cpu_threads")
            ParseUint(val.c_str(), &a.cpu_threads);
        else if (key == "--buckets")
            a.bucket_counts = ParseUintList(val);
        else if (key == "--leaves")
            a.leaf_counts = ParseUintList(val);
        else if (key == "--densities")
            a.densities = ParseDoubleList(val);
        else if (key == "--skip_cpu")
            a.skip_cpu = true;
        else if (key == "--skip_aicore")
            a.skip_aicore = true;
        else if (key == "--skip_aicpu")
            a.skip_aicpu = true;
        else if (key == "--no_verify")
            a.verify = false;
        else {
            std::fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            PrintUsage(argv[0]);
            std::exit(2);
        }
    }
    return a;
}

// ---- STUBS — implement during Phase 2 §5.2 ----

void BuildForwardBitmap(uint32_t doc_per_bucket, uint32_t bucket_count, uint32_t tag_num, double density) {
    // TODO: doc × uint64[stride] row-major, stride = ceil(tag_num/64)
    //   - match dataset_HW.bin Bitmaps section layout
    //   - support density sweep [0.00005, 0.05]
    (void)doc_per_bucket;
    (void)bucket_count;
    (void)tag_num;
    (void)density;
}

void BuildBucketPlan(uint32_t tag_num, uint32_t expr_leaves, uint32_t rpn_depth_ratio) {
    // TODO: generate Boolean expr ≤ expr_leaves tags
    //   - compile to BucketPlan (linear nodes/children/eval_order)
    //   - match FilterExpCompiler output shape (Query.h:963-1004)
    //   - selectivity-sort leaves by synthetic g_global_tag_freq
    (void)tag_num;
    (void)expr_leaves;
    (void)rpn_depth_ratio;
}

struct LatencyBreakdown {
    double h2d_bitmap_ms;
    double kernel_exec_ms;
    double d2h_ms;
    double total_ms;
    uint32_t matches;
};

LatencyBreakdown RunCpuForward() {
    // TODO: NEON vandq_u64 / vorrq_u64 8-word unroll (Query.h:936-956)
    //   - 16 threads (LOCKED)
    //   - matches sks_hw AND/OR short-circuit + sentinel folding
    return {0, 0, 0, 0, 0};
}
LatencyBreakdown RunAicoreForward() {
    // TODO: Phase 2 §5.2.2 original kernel
    //   - 32 doc/block
    //   - leaf tag word gather from sparse HBM bitmap
    //   - AND/OR short-circuit (Query.h:994-997 semantics)
    //   - GatherMask compact score+index output
    //   - read score from FP16 d_result_ws layout (TBD)
    return {0, 0, 0, 0, 0};
}
LatencyBreakdown RunAicpuMask() {
    // TODO: reuse aicpu_mask_filter_kernel.cpp as-is
    //   - bucket-segmented layout (npuAPI.cpp MFTask)
    //   - 3 D2H copies (current baseline)
    return {0, 0, 0, 0, 0};
}

void PrintReport(const char* name, const LatencyBreakdown& r) {
    std::printf(
        "  %-15s H2D=%6.3f ms  kernel=%6.3f ms  D2H=%6.3f ms  "
        "total=%6.3f ms  matches=%u\n",
        name, r.h2d_bitmap_ms, r.kernel_exec_ms, r.d2h_ms, r.total_ms, r.matches);
}

}  // namespace

int main(int argc, char** argv) {
    Args args = ParseArgs(argc, argv);

    if (args.doc_per_bucket % 16u != 0u) {
        std::fprintf(stderr, "[fatal] doc_per_bucket must be multiple of 16 (NPU alignment)\n");
        return 1;
    }
    if (args.cpu_threads > 16u) {
        std::fprintf(stderr,
                     "[warn] cpu_threads=%u exceeds 16 — oversubscription regression "
                     "(see filter_cmp README)\n",
                     args.cpu_threads);
    }

    std::printf("=== filter_phase2 (skeleton) ===\n");
    std::printf("doc_per_bucket=%u tag_num=%u (stride=%u) cpu_threads=%u\n", args.doc_per_bucket, args.tag_num,
                (args.tag_num + 63u) / 64u, args.cpu_threads);
    std::printf("sweep: buckets={");
    for (size_t i = 0; i < args.bucket_counts.size(); ++i)
        std::printf("%s%u", i ? "," : "", args.bucket_counts[i]);
    std::printf("} leaves={");
    for (size_t i = 0; i < args.leaf_counts.size(); ++i)
        std::printf("%s%u", i ? "," : "", args.leaf_counts[i]);
    std::printf("} densities={");
    for (size_t i = 0; i < args.densities.size(); ++i)
        std::printf("%s%.5f", i ? "," : "", args.densities[i]);
    std::printf("}\n\n");

    std::printf("[notice] all paths are STUBS — fill during Phase 2 §5.2\n\n");

    for (uint32_t bc : args.bucket_counts) {
        for (uint32_t leaves : args.leaf_counts) {
            for (double dens : args.densities) {
                std::printf("=== bucket_count=%u leaves=%u density=%.5f ===\n", bc, leaves, dens);
                BuildForwardBitmap(args.doc_per_bucket, bc, args.tag_num, dens);
                BuildBucketPlan(args.tag_num, leaves, args.rpn_depth_ratio);

                if (!args.skip_cpu)
                    PrintReport("cpu_forward", RunCpuForward());
                if (!args.skip_aicore)
                    PrintReport("aicore_forward", RunAicoreForward());
                if (!args.skip_aicpu)
                    PrintReport("aicpu_mask", RunAicpuMask());
                std::printf("\n");
            }
        }
    }

    std::printf("=== done (skeleton — no real measurements) ===\n");
    return 0;
}
