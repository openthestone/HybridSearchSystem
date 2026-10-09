#include "recall_ref.h"

#include <sys/stat.h>

#include <algorithm>
#include <fstream>
#include <utility>

#include "src/full_recall/core/number/trans_number.h"

namespace npur_harness {
namespace {
constexpr uint32_t kRecallRefMagic = 0x46455252u;  // "RREF"
constexpr uint32_t kRecallRefVersion = 1;          // bump if CpuTopK semantics change (e.g. the fp16 codec)
}  // namespace

// docOffset != 0: the engine returns absolute global ids, so the reference scans absolute rows.
std::unordered_set<int64_t> CpuTopK(const std::vector<float>& query, const FilterNode* filter,
                                    const npur_port::HwDataset& hw, uint64_t docNum, int topk, uint64_t docOffset,
                                    bool fp32) {
    std::vector<float> q16(hw.dim, 0.0f);
    for (uint32_t d = 0; d < hw.dim && d < query.size(); ++d)
        q16[d] = fp32 ? query[d] : NpuRetrieval::Float16ToFloat32(NpuRetrieval::Float32ToFloat16(query[d]));

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
            for (uint32_t d = 0; d < hw.dim; ++d) {
                float vd = NpuRetrieval::Float16ToFloat32(NpuRetrieval::Float32ToFloat16(v[d]));
                s += q16[d] * vd;
            }
        }
        scored.emplace_back(s, static_cast<int64_t>(doc));
    }
    size_t k = std::min<size_t>(topk, scored.size());
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    std::unordered_set<int64_t> ref;
    for (size_t i = 0; i < k; ++i)
        ref.insert(scored[i].second);
    return ref;
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
