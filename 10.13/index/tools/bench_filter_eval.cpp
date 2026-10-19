#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "src/harness/query/filter_expr.h"
#include "src/harness/query/filter_flat.h"
#include "src/io/dataset_hw.h"

namespace {

using npur_harness::EvalFlatFilter;
using npur_harness::FilterNode;
using npur_harness::FilterOp;
using npur_harness::FlatFilter;
using npur_harness::FlattenFilter;
using Clock = std::chrono::steady_clock;

double MsSince(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct Shape {
    size_t nodes = 0, terms = 0, depth = 0;
};
void Measure(const FilterNode* n, size_t d, Shape& s) {
    if (n == nullptr)
        return;
    ++s.nodes;
    if (d > s.depth)
        s.depth = d;
    if (n->op == FilterOp::Term)
        ++s.terms;
    for (const auto& c : n->children)
        Measure(c.get(), d + 1, s);
}

uint64_t ArmAstBatch(const npur_port::HwDataset& hw, uint64_t docs, const std::vector<const FilterNode*>& fs) {
    uint64_t hits = 0;
    for (uint64_t d = 0; d < docs; ++d) {
        const uint64_t* bm = hw.bitmaps + d * hw.stride;
        for (const FilterNode* f : fs)
            if (npur_harness::EvalFilterBitmap(f, bm, hw.stride))
                ++hits;
    }
    return hits;
}
uint64_t ArmFlatBatch(const npur_port::HwDataset& hw, uint64_t docs, const std::vector<FlatFilter>& fs) {
    uint64_t hits = 0;
    for (uint64_t d = 0; d < docs; ++d) {
        const uint64_t* bm = hw.bitmaps + d * hw.stride;
        for (const FlatFilter& f : fs)
            if (EvalFlatFilter(f, bm))
                ++hits;
    }
    return hits;
}
uint64_t ArmAstSingle(const npur_port::HwDataset& hw, uint64_t docs, const std::vector<const FilterNode*>& fs) {
    uint64_t hits = 0;
    for (const FilterNode* f : fs)
        for (uint64_t d = 0; d < docs; ++d)
            if (npur_harness::EvalFilterBitmap(f, hw.bitmaps + d * hw.stride, hw.stride))
                ++hits;
    return hits;
}

int SelfTest() {
    std::printf("bench_filter_eval self-test\n");
    const uint32_t tagNum = 2048, words = (tagNum + 63u) / 64u;
    const uint64_t docs = 3000;
    std::vector<uint64_t> bitmaps(static_cast<size_t>(docs) * words, 0);
    std::mt19937_64 rng(20261012);
    for (uint64_t d = 0; d < docs; ++d)
        for (int t = 0; t < 120; ++t) {
            const uint32_t tag = static_cast<uint32_t>(rng() % tagNum);
            bitmaps[static_cast<size_t>(d) * words + (tag >> 6)] |= 1ULL << (tag & 63u);
        }
    const std::vector<std::string> exprs = {
        "7",
        "7 OR 9 OR 70 OR 71 OR 1000",
        "7 AND 9 AND 70",
        "(7 AND 9) OR (70 AND 71) OR 300",
        "NOT 7",
        "NOT (7 OR 9)",
        "7 AND NOT 9",
        "7 OR 5000",
        "7 AND 5000",
        "5000",
        "7 OR 7 OR 7",
        "6 OR 7",
        "(NOT 7) AND (9 OR 70) AND NOT (71 AND 72)",
        "((7)) OR ((9 AND 70))",
    };
    size_t mismatched = 0, matchedPairs = 0;
    for (const std::string& e : exprs) {
        bool ok = false;
        auto ast = npur_harness::ParseFilter(e, ok);
        if (!ok) {
            std::printf("  FAIL: cannot parse %s\n", e.c_str());
            ++mismatched;
            continue;
        }
        const FlatFilter flat = FlattenFilter(ast.get(), words);
        size_t differ = 0, hits = 0;
        for (uint64_t d = 0; d < docs; ++d) {
            const uint64_t* bm = bitmaps.data() + static_cast<size_t>(d) * words;
            const bool a = npur_harness::EvalFilterBitmap(ast.get(), bm, words);
            const bool b = EvalFlatFilter(flat, bm);
            if (a != b)
                ++differ;
            if (a)
                ++hits;
        }
        matchedPairs += hits;
        Shape s;
        Measure(ast.get(), 1, s);
        std::printf("  %-44s 命中 %4zu/%" PRIu64 "  不一致 %zu   节点 %zu -> 测试 %zu\n", e.c_str(), hits, docs, differ,
                    s.nodes, npur_harness::FlatFilterTests(flat));
        mismatched += differ;
    }
    {
        const FlatFilter flat = FlattenFilter(nullptr, words);
        size_t differ = 0;
        for (uint64_t d = 0; d < docs; ++d) {
            const uint64_t* bm = bitmaps.data() + static_cast<size_t>(d) * words;
            if (npur_harness::EvalFilterBitmap(nullptr, bm, words) != EvalFlatFilter(flat, bm))
                ++differ;
        }
        std::printf("  %-44s 命中 %4" PRIu64 "/%" PRIu64 "  不一致 %zu   节点 0 -> 测试 0\n", "(空表达式，全匹配)",
                    docs, docs, differ);
        mismatched += differ;
        matchedPairs += docs;
    }
    if (matchedPairs == 0) {
        std::printf("  FAIL: nothing matched, the comparison proved nothing\n");
        ++mismatched;
    }
    std::printf(mismatched == 0 ? "bench_filter_eval: OK\n" : "bench_filter_eval: %zu MISMATCH(ES)\n", mismatched);
    return mismatched == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3)
        return SelfTest();

    const std::string dsPath = argv[1], filterPath = argv[2];
    const uint64_t docs = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 100000;
    const size_t perThread = argc > 4 ? std::strtoul(argv[4], nullptr, 10) : 57;
    const unsigned threads = argc > 5 ? static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10))
                                      : std::max(1u, std::thread::hardware_concurrency());

    npur_port::HwDataset hw;
    std::string err;
    if (!hw.Open(dsPath, &err)) {
        std::fprintf(stderr, "[ERROR] %s: %s\n", dsPath.c_str(), err.c_str());
        return 1;
    }
    if (docs > hw.doc_num) {
        std::fprintf(stderr, "[ERROR] docs=%" PRIu64 " 超过数据集的 %" PRIu64 "\n", docs, hw.doc_num);
        return 1;
    }
    std::printf("数据集 %s: %" PRIu64 " 文档  dim=%u  标签 %u  每行位图 %u 字 = %.1fKB\n", dsPath.c_str(), hw.doc_num,
                hw.dim, hw.tag_num, hw.stride, hw.stride * 8.0 / 1024.0);

    std::vector<std::string> lines;
    {
        std::ifstream in(filterPath);
        if (!in) {
            std::fprintf(stderr, "[ERROR] 打不开过滤表达式文件：%s\n", filterPath.c_str());
            return 1;
        }
        std::string l;
        while (std::getline(in, l))
            if (!l.empty())
                lines.push_back(l);
    }
    if (lines.empty()) {
        std::fprintf(stderr, "[ERROR] %s 里没有表达式\n", filterPath.c_str());
        return 1;
    }

    const size_t total = static_cast<size_t>(threads) * perThread;
    std::vector<std::unique_ptr<FilterNode>> owned(total);
    std::vector<const FilterNode*> asts(total, nullptr);
    std::vector<FlatFilter> flats(total);
    Shape sum;
    size_t maxNodes = 0, flatTests = 0, parseFail = 0;
    for (size_t i = 0; i < total; ++i) {
        bool ok = false;
        owned[i] = npur_harness::ParseFilter(lines[i % lines.size()], ok);
        if (!ok) {
            owned[i].reset();
            ++parseFail;
        }
        asts[i] = owned[i].get();
        flats[i] = FlattenFilter(asts[i], hw.stride);
        Shape s;
        Measure(asts[i], 1, s);
        sum.nodes += s.nodes;
        sum.terms += s.terms;
        if (s.depth > sum.depth)
            sum.depth = s.depth;
        if (s.nodes > maxNodes)
            maxNodes = s.nodes;
        flatTests += npur_harness::FlatFilterTests(flats[i]);
    }
    std::printf(
        "表达式 %zu 条（文件里 %zu 条，解析失败 %zu）：平均 %.1f 节点 / %.1f 个 Term，最多 %zu 节点，"
        "最深 %zu 层\n",
        total, lines.size(), parseFail, static_cast<double>(sum.nodes) / total, static_cast<double>(sum.terms) / total,
        maxNodes, sum.depth);
    std::printf("压平后平均 %.1f 个 (字, 掩码) 测试：AST 每条 %.0fB 散落的节点 -> 连续 %.0fB\n",
                static_cast<double>(flatTests) / total, static_cast<double>(sum.nodes) / total * sizeof(FilterNode),
                static_cast<double>(flatTests) / total * 12.0);
    std::printf("每臂 %u 线程 x %zu 条表达式 x %" PRIu64 " 文档 = %.3f 亿对；整次构建 10800 x %" PRIu64
                " = %.0f 亿对，是这里的 %.0f 倍\n",
                threads, perThread, docs, static_cast<double>(threads) * perThread * docs / 1e8, hw.doc_num,
                10800.0 * static_cast<double>(hw.doc_num) / 1e8,
                10800.0 * static_cast<double>(hw.doc_num) / (static_cast<double>(threads) * perThread * docs));

    static const char* kArmName[3] = {"AST + 批量（压平之前）", "压平 + 批量（现在的代码）", "AST + 单条（批量之前）"};
    struct Sample {
        int arm;
        double ms;
        uint64_t hits;
    };
    std::vector<Sample> samples;

    {
        const auto t0 = Clock::now();
        uint64_t sink = 0;
        for (uint64_t d = 0; d < docs; ++d)
            sink += hw.bitmaps[d * hw.stride];
        std::printf("预热：把 %" PRIu64 " 行位图读进页缓存，%.1f ms（sink %" PRIu64 "）\n\n", docs, MsSince(t0),
                    sink & 1u);
        std::fflush(stdout);
    }
    auto runArm = [&](char arm) {
        std::vector<uint64_t> hits(threads, 0);
        const auto t0 = Clock::now();
        std::vector<std::thread> pool;
        pool.reserve(threads);
        for (unsigned t = 0; t < threads; ++t) {
            pool.emplace_back([&, t] {
                const size_t lo = static_cast<size_t>(t) * perThread;
                std::vector<const FilterNode*> myAst(asts.begin() + static_cast<long>(lo),
                                                     asts.begin() + static_cast<long>(lo + perThread));
                std::vector<FlatFilter> myFlat;
                if (arm == 'B') {
                    myFlat.reserve(perThread);
                    for (size_t i = 0; i < perThread; ++i)
                        myFlat.push_back(FlattenFilter(asts[lo + i], hw.stride));
                }
                switch (arm) {
                    case 'A':
                        hits[t] = ArmAstBatch(hw, docs, myAst);
                        break;
                    case 'B':
                        hits[t] = ArmFlatBatch(hw, docs, myFlat);
                        break;
                    default:
                        hits[t] = ArmAstSingle(hw, docs, myAst);
                        break;
                }
            });
        }
        for (std::thread& th : pool)
            th.join();
        uint64_t h = 0;
        for (uint64_t v : hits)
            h += v;
        const int k = arm == 'A' ? 0 : arm == 'B' ? 1 : 2;
        samples.push_back({k, MsSince(t0), h});
        const Sample& s = samples.back();
        const double pairs = static_cast<double>(threads) * perThread * docs;
        std::printf("  %9.1f ms  %6.1f ns/对  命中 %-10" PRIu64 " %s\n", s.ms, s.ms * 1e6 / pairs, s.hits, kArmName[k]);
        std::fflush(stdout);
    };

    std::printf("== 三臂，顺序 A B C C B A ==\n");
    for (char arm : {'A', 'B', 'C', 'C', 'B', 'A'})
        runArm(arm);

    std::printf("\n== 汇总 ==\n");
    double best[3] = {0, 0, 0}, worst[3] = {0, 0, 0};
    uint64_t hit[3] = {0, 0, 0};
    for (const Sample& smp : samples) {
        const int k = smp.arm;
        if (best[k] == 0 || smp.ms < best[k])
            best[k] = smp.ms;
        if (smp.ms > worst[k])
            worst[k] = smp.ms;
        hit[k] = smp.hits;
    }
    const double pairs = static_cast<double>(threads) * perThread * docs;
    const double full = 10800.0 * static_cast<double>(hw.doc_num);
    for (int k = 0; k < 3; ++k)
        std::printf("  两样本 %8.1f /%8.1f ms（臂内差 %4.1f%%）  折算整次构建 %6.0f s  %s\n", best[k], worst[k],
                    100.0 * (worst[k] - best[k]) / best[k], best[k] / 1000.0 * full / pairs, kArmName[k]);
    if (hit[0] != hit[1] || hit[0] != hit[2])
        std::printf("  [ERROR] 三臂命中数不同：%" PRIu64 " / %" PRIu64 " / %" PRIu64 "，压平的语义不等价\n", hit[0],
                    hit[1], hit[2]);
    else
        std::printf("  三臂命中数一致（%" PRIu64 "），只有布局不同\n", hit[0]);
    std::printf("  flat 相对 ast：%+.1f%%（取各自较快的样本）\n", 100.0 * (best[1] / best[0] - 1.0));
    std::printf(
        "\n折算的秒数只含过滤表达式判断。某一臂和当时真实构建耗时之差，就是判断之外的开销：\n"
        "点积、堆维护，以及所有线程一起等内存。\n");
    return 0;
}
