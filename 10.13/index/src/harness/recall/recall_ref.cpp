#include "recall_ref.h"

#include <sys/stat.h>

#include <algorithm>
#include <fstream>
#include <queue>
#include <thread>
#include <utility>

#include "../query/filter_flat.h"
#include "src/full_recall/core/number/trans_number.h"

namespace npur_harness {
namespace {
constexpr uint32_t kRecallRefMagic = 0x46455252u;  // "RREF"
constexpr uint32_t kRecallRefVersion = 2;

struct Better {
    bool operator()(const std::pair<float, int64_t>& a, const std::pair<float, int64_t>& b) const {
        if (a.first != b.first)
            return a.first > b.first;
        return a.second < b.second;
    }
};

inline float RoundTripF16(float v) {
    return NpuRetrieval::Float16ToFloat32(NpuRetrieval::Float32ToFloat16(v));
}
}  // namespace

// docOffset != 0: the engine returns absolute global ids, so the reference scans absolute rows.
std::unordered_set<int64_t> CpuTopK(const std::vector<float>& query, const FilterNode* filter,
                                    const npur_port::HwDataset& hw, uint64_t docNum, int topk, uint64_t docOffset,
                                    bool fp32) {
    std::vector<float> q16(hw.dim, 0.0f);
    for (uint32_t d = 0; d < hw.dim && d < query.size(); ++d)
        q16[d] = fp32 ? query[d] : RoundTripF16(query[d]);

    std::vector<std::pair<float, int64_t>> scored;
    scored.reserve(1024);
    for (uint64_t doc = docOffset; doc < docOffset + docNum; ++doc) {
        if (!npur_harness::EvalFilterBitmap(filter, hw.bitmaps + doc * hw.stride, hw.stride))
            continue;
        const float* v = hw.vectors + doc * hw.dim;
        float s = 0.0f;
        if (fp32) {
            for (uint32_t d = 0; d < hw.dim; ++d)
                s += q16[d] * v[d];
        } else {
            for (uint32_t d = 0; d < hw.dim; ++d)
                s += q16[d] * RoundTripF16(v[d]);
        }
        scored.emplace_back(s, static_cast<int64_t>(doc));
    }
    size_t k = std::min<size_t>(topk, scored.size());
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(), Better{});
    std::unordered_set<int64_t> ref;
    for (size_t i = 0; i < k; ++i)
        ref.insert(scored[i].second);
    return ref;
}

std::vector<float> QuantizeCorpus(const npur_port::HwDataset& hw, uint64_t docNum, uint64_t docOffset, bool fp32,
                                  unsigned threads) {
    if (fp32)
        return {};
    std::vector<float> out(static_cast<size_t>(docNum) * hw.dim);
    const unsigned nThreads = std::max(1u, threads);
    const uint64_t chunk = (docNum + nThreads - 1) / nThreads;
    std::vector<std::thread> pool;
    pool.reserve(nThreads);
    for (unsigned t = 0; t < nThreads; ++t) {
        pool.emplace_back([&, t] {
            const uint64_t lo = t * chunk;
            const uint64_t hi = std::min<uint64_t>(docNum, lo + chunk);
            for (uint64_t i = lo; i < hi; ++i) {
                const float* v = hw.vectors + (docOffset + i) * hw.dim;
                float* o = out.data() + static_cast<size_t>(i) * hw.dim;
                for (uint32_t d = 0; d < hw.dim; ++d)
                    o[d] = RoundTripF16(v[d]);
            }
        });
    }
    for (auto& th : pool)
        th.join();
    return out;
}

void CpuTopKBatch(const std::vector<RecallRefJob>& jobs, const npur_port::HwDataset& hw,
                  const std::vector<float>& quantized, uint64_t docNum, uint64_t docOffset, bool fp32,
                  std::vector<std::unordered_set<int64_t>>& out) {
    const size_t nJobs = jobs.size();
    if (nJobs == 0)
        return;
    std::vector<float> q16(nJobs * hw.dim, 0.0f);
    for (size_t j = 0; j < nJobs; ++j) {
        const std::vector<float>& q = *jobs[j].query;
        float* row = q16.data() + j * hw.dim;
        for (uint32_t d = 0; d < hw.dim && d < q.size(); ++d)
            row[d] = fp32 ? q[d] : RoundTripF16(q[d]);
    }
    using Heap = std::priority_queue<std::pair<float, int64_t>, std::vector<std::pair<float, int64_t>>, Better>;
    std::vector<Heap> heaps(nJobs);
    std::vector<FlatFilter> flat(nJobs);
    for (size_t j = 0; j < nJobs; ++j)
        flat[j] = FlattenFilter(jobs[j].filter, hw.stride);

    for (uint64_t i = 0; i < docNum; ++i) {
        const uint64_t doc = docOffset + i;
        const uint64_t* bm = hw.bitmaps + doc * hw.stride;
        const float* v = nullptr;
        for (size_t j = 0; j < nJobs; ++j) {
            if (!EvalFlatFilter(flat[j], bm))
                continue;
            if (v == nullptr)
                v = quantized.empty() ? hw.vectors + doc * hw.dim : quantized.data() + static_cast<size_t>(i) * hw.dim;
            const float* q = q16.data() + j * hw.dim;
            float s = 0.0f;
            for (uint32_t d = 0; d < hw.dim; ++d)
                s += q[d] * v[d];
            Heap& h = heaps[j];
            const std::pair<float, int64_t> cand(s, static_cast<int64_t>(doc));
            if (h.size() < static_cast<size_t>(jobs[j].topk))
                h.push(cand);
            else if (!h.empty() && Better{}(cand, h.top())) {
                h.pop();
                h.push(cand);
            }
        }
    }
    for (size_t j = 0; j < nJobs; ++j) {
        std::unordered_set<int64_t>& ref = out[jobs[j].outIndex];
        ref.clear();
        Heap& h = heaps[j];
        while (!h.empty()) {
            ref.insert(h.top().second);
            h.pop();
        }
    }
}

// The byte sequence below IS the cache key: changing it invalidates every cached reference.
std::string RecallFingerprint(const RecallRefInputs& in) {
    std::string b;
    auto put = [&](const void* p, size_t n) { b.append(static_cast<const char*>(p), n); };
    auto putFileSig = [&](const std::string& path) {
        uint64_t sz = 0;
        int64_t mt = 0;
        struct stat st{};
        if (!path.empty() && ::stat(path.c_str(), &st) == 0) {
            sz = static_cast<uint64_t>(st.st_size);
            mt = static_cast<int64_t>(st.st_mtime);
        }
        put(&sz, sizeof(sz));
        put(&mt, sizeof(mt));
    };
    uint32_t ver = kRecallRefVersion;
    put(&ver, sizeof(ver));
    const uint8_t fp32 = in.fp32 ? 1 : 0;  // the two modes' refs are different ground truths
    put(&fp32, sizeof(fp32));
    putFileSig(in.datasetPath);
    putFileSig(in.queryPath);
    putFileSig(in.filterPath);
    putFileSig(in.topkPath);
    put(&in.docNum, sizeof(in.docNum));
    put(&in.dim, sizeof(in.dim));
    put(&in.nq, sizeof(in.nq));
    put(&in.docOffset, sizeof(in.docOffset));
    put(&in.filterSeed, sizeof(in.filterSeed));
    put(&in.topk, sizeof(in.topk));
    put(&in.numQueriesFlag, sizeof(in.numQueriesFlag));
    return b;
}

bool LoadRecallRef(const std::string& path, const std::string& fp, uint64_t nq,
                   std::vector<std::unordered_set<int64_t>>& refs) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;
    uint32_t magic = 0, fpLen = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&fpLen), sizeof(fpLen));
    if (!in || magic != kRecallRefMagic || fpLen != fp.size())
        return false;
    std::string diskFp(fpLen, '\0');
    in.read(diskFp.data(), fpLen);
    uint64_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!in || diskFp != fp || count != nq)
        return false;  // stale or wrong shape
    refs.assign(static_cast<size_t>(nq), {});
    for (uint64_t q = 0; q < nq; ++q) {
        uint32_t n = 0;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (!in)
            return false;
        auto& r = refs[q];
        r.reserve(n);
        for (uint32_t i = 0; i < n; ++i) {
            int64_t id = 0;
            in.read(reinterpret_cast<char*>(&id), sizeof(id));
            if (!in)
                return false;
            r.insert(id);
        }
    }
    return true;
}

bool WriteRecallRef(const std::string& path, const std::string& fp,
                    const std::vector<std::unordered_set<int64_t>>& refs) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        return false;
    uint32_t magic = kRecallRefMagic, fpLen = static_cast<uint32_t>(fp.size());
    uint64_t count = refs.size();
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&fpLen), sizeof(fpLen));
    out.write(fp.data(), fp.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& r : refs) {
        uint32_t n = static_cast<uint32_t>(r.size());
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (int64_t id : r)
            out.write(reinterpret_cast<const char*>(&id), sizeof(id));
    }
    return static_cast<bool>(out);
}

}  // namespace npur_harness
