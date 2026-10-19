#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "src/harness/query/filter_expr.h"
#include "src/harness/recall/recall_ref.h"
#include "src/io/dataset_hw.h"

namespace {

int g_failures = 0;

void Check(bool ok, const std::string& what) {
    if (!ok) {
        std::printf("  FAIL: %s\n", what.c_str());
        ++g_failures;
    }
}

struct FakeCorpus {
    std::vector<float> vectors;
    std::vector<uint64_t> bitmaps;
    npur_port::HwDataset hw;

    FakeCorpus(uint64_t docNum, uint32_t dim, uint32_t tagNum, int tagsPerDoc, int valueLevels, uint64_t seed) {
        const uint32_t stride = (tagNum + 63u) / 64u;
        vectors.resize(static_cast<size_t>(docNum) * dim);
        bitmaps.assign(static_cast<size_t>(docNum) * stride, 0);
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<float> uf(-1.0f, 1.0f);
        std::uniform_int_distribution<uint32_t> ut(0, tagNum - 1);
        for (float& x : vectors)
            x = valueLevels > 0 ? static_cast<float>(rng() % static_cast<uint64_t>(valueLevels)) : uf(rng);
        for (uint64_t d = 0; d < docNum; ++d)
            for (int t = 0; t < tagsPerDoc; ++t) {
                const uint32_t tag = ut(rng);
                bitmaps[static_cast<size_t>(d) * stride + (tag >> 6)] |= 1ULL << (tag & 63u);
            }
        hw.vectors = vectors.data();
        hw.bitmaps = bitmaps.data();
        hw.dim = dim;
        hw.tag_num = tagNum;
        hw.stride = stride;
        hw.doc_num = docNum;
    }
};

std::vector<std::vector<float>> MakeQueries(size_t nq, uint32_t dim, int valueLevels, uint64_t seed) {
    std::vector<std::vector<float>> qs(nq, std::vector<float>(dim));
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> uf(-1.0f, 1.0f);
    for (auto& q : qs)
        for (float& x : q)
            x = valueLevels > 0 ? static_cast<float>(rng() % static_cast<uint64_t>(valueLevels)) : uf(rng);
    return qs;
}

std::vector<std::unique_ptr<npur_harness::FilterNode>> MakeFilters(size_t nq, uint32_t tagNum, uint64_t seed,
                                                                   std::vector<const npur_harness::FilterNode*>& out) {
    std::vector<std::unique_ptr<npur_harness::FilterNode>> owned;
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint32_t> ut(0, tagNum - 1);
    out.assign(nq, nullptr);
    for (size_t i = 0; i < nq; ++i) {
        if (i % 7 == 0)
            continue;
        auto n = std::make_unique<npur_harness::FilterNode>();
        if (i % 11 == 0) {
            n->op = npur_harness::FilterOp::And;
            for (int c = 0; c < 2; ++c) {
                auto t = std::make_unique<npur_harness::FilterNode>();
                t->op = npur_harness::FilterOp::Term;
                t->tag = ut(rng);
                n->children.push_back(std::move(t));
            }
        } else {
            n->op = npur_harness::FilterOp::Term;
            n->tag = ut(rng);
        }
        out[i] = n.get();
        owned.push_back(std::move(n));
    }
    return owned;
}

void RunCase(const char* name, uint64_t docNum, uint32_t dim, size_t nq, int topk, int valueLevels, bool fp32) {
    const uint32_t tagNum = 2048;
    const int tagsPerDoc = 104;
    FakeCorpus corpus(docNum, dim, tagNum, tagsPerDoc, valueLevels, 20261011);
    const auto queries = MakeQueries(nq, dim, valueLevels, 777);
    std::vector<const npur_harness::FilterNode*> filterPtrs;
    const auto ownedFilters = MakeFilters(nq, tagNum, 99, filterPtrs);

    std::vector<int> topks(nq);
    for (size_t i = 0; i < nq; ++i)
        topks[i] = (i % 3 == 0) ? topk : topk / 2 + 1;

    std::vector<std::unordered_set<int64_t>> single(nq);
    for (size_t i = 0; i < nq; ++i)
        single[i] = npur_harness::CpuTopK(queries[i], filterPtrs[i], corpus.hw, docNum, topks[i], 0, fp32);

    std::vector<npur_harness::RecallRefJob> jobs(nq);
    for (size_t i = 0; i < nq; ++i)
        jobs[i] = {&queries[i], filterPtrs[i], topks[i], i};
    const auto quantized = npur_harness::QuantizeCorpus(corpus.hw, docNum, 0, fp32, 4);
    std::vector<std::unordered_set<int64_t>> batched(nq);
    npur_harness::CpuTopKBatch(jobs, corpus.hw, quantized, docNum, 0, fp32, batched);

    size_t mismatched = 0, refTotal = 0, emptyRefs = 0;
    for (size_t i = 0; i < nq; ++i) {
        refTotal += single[i].size();
        if (single[i].empty())
            ++emptyRefs;
        if (single[i] != batched[i])
            ++mismatched;
    }
    Check(mismatched == 0, std::string(name) + ": " + std::to_string(mismatched) + "/" + std::to_string(nq) +
                               " queries differ between CpuTopK and CpuTopKBatch");
    Check(refTotal > 0 && emptyRefs < nq, std::string(name) + ": references are all empty, nothing was compared");
    std::printf("  %-34s nq=%zu refs=%zu 空=%zu 不一致=%zu\n", name, nq, refTotal, emptyRefs, mismatched);
}

}  // namespace

int main() {
    std::printf("test_recall_ref\n");

    RunCase("随机取值 fp16", 4000, 64, 48, 40, 0, false);
    RunCase("随机取值 fp32", 4000, 64, 48, 40, 0, true);
    RunCase("8 档取值（大量同分）", 4000, 64, 48, 40, 8, false);
    RunCase("3 档取值（更多同分）", 4000, 64, 48, 40, 3, false);
    RunCase("2 档取值（极端同分）", 4000, 64, 48, 40, 2, false);
    RunCase("K 大于命中数", 600, 64, 32, 5000, 0, false);
    RunCase("批内只有一条查询", 4000, 64, 1, 40, 3, false);
    RunCase("查询数多于文档数", 300, 64, 400, 20, 4, false);

    if (g_failures == 0)
        std::printf("test_recall_ref: OK\n");
    else
        std::printf("test_recall_ref: %d FAILURE(S)\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
