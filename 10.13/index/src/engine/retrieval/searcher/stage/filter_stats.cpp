#include "filter_stats.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <iomanip>
#include <sstream>

#include "acl/acl.h"
#include "src/full_recall/core/log_definition.h"
#include "src/full_recall/retrieval/device/device_common_external.h"
#include "src/utils/env_switch.h"
#include "src/utils/logger.h"
#include "text_filter.h"

namespace NpuRetrieval {

uint32_t BitlistStatsLevel() {
    static const uint32_t level = npur_env::U32Capped("NPUR_BITLIST_STATS", 2, 2);
    return level;
}

namespace {
struct BitlistCallStats {
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> tableSum{0};    // (segments x postings) pairs the query referenced
    std::atomic<uint64_t> bitlistSum{0};  // of those, the ones that needed conversion
    std::atomic<uint64_t> bitlistMin{~0ULL};
    std::atomic<uint64_t> bitlistMax{0};
    std::atomic<uint64_t> bitsetBytesSum{0};  // bitlistNum * segmentByteSize, the dense output

    void Note(uint64_t table, uint64_t bitlist, uint64_t bytes) {
        calls.fetch_add(1, std::memory_order_relaxed);
        tableSum.fetch_add(table, std::memory_order_relaxed);
        bitlistSum.fetch_add(bitlist, std::memory_order_relaxed);
        bitsetBytesSum.fetch_add(bytes, std::memory_order_relaxed);
        uint64_t cur = bitlistMin.load(std::memory_order_relaxed);
        while (bitlist < cur && !bitlistMin.compare_exchange_weak(cur, bitlist, std::memory_order_relaxed)) {
        }
        cur = bitlistMax.load(std::memory_order_relaxed);
        while (bitlist > cur && !bitlistMax.compare_exchange_weak(cur, bitlist, std::memory_order_relaxed)) {
        }
    }
};
BitlistCallStats g_bitlistStats;
std::atomic<uint64_t> g_bitlistDeepCalls{0};

// NPUR_EXPR_STATS=1: shape of the post-order filter expression (operands per node, per operator).
struct ExprShapeStats {
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> nodes[4]{};     // indexed by FilterOpType order below
    std::atomic<uint64_t> postings[4]{};  // posting operands consumed by each op type
    std::atomic<uint64_t> stacks[4]{};    // stack operands consumed by each op type
    // operands per node, bucketed 1,2,3,4,5-8,9-16,17-32,>32
    std::atomic<uint64_t> andOrHist[8]{};
    std::atomic<uint64_t> conjPartHist[8]{};
    std::atomic<uint64_t> maxOperands{0};
    std::atomic<uint64_t> maxConjParts{0};

    static uint32_t Bucket(uint64_t n) {
        if (n == 0) {
            return 0;
        }
        if (n <= 4) {
            return static_cast<uint32_t>(n - 1);
        }
        if (n <= 8) {
            return 4;
        }
        if (n <= 16) {
            return 5;
        }
        return n <= 32 ? 6 : 7;
    }
    void NoteMax(std::atomic<uint64_t>& slot, uint64_t v) {
        uint64_t cur = slot.load(std::memory_order_relaxed);
        while (v > cur && !slot.compare_exchange_weak(cur, v, std::memory_order_relaxed)) {
        }
    }
};
ExprShapeStats g_exprStats;

struct EmptyOperandStats {
    std::atomic<uint64_t> calls{0};
    std::atomic<uint64_t> pairs{0};         // every (posting, segment) pair the filter table covers
    std::atomic<uint64_t> orPairs{0};       // ... of those, the ones an OR node consumes
    std::atomic<uint64_t> orEmpty{0};       // ... of those, the ones whose posting is all-zero
    std::atomic<uint64_t> orEmptyDense{0};  // ... of those, the ones stored as a full 16KB bitmap
};
EmptyOperandStats g_emptyStats;

static uint32_t ExprOpSlot(uint32_t opType) {
    switch (opType) {
        case FilterOpType::AND:
            return 0;
        case FilterOpType::OR:
            return 1;
        case FilterOpType::NOT:
            return 2;
        case FilterOpType::CONJUNCTION:
            return 3;
        default:
            return 3;
    }
}
}  // namespace

// NPUR_EMPTY_STATS=1: walks every pair a second time, so leave it off for timing runs.
bool EmptyStatsEnabled() {
    static const bool on = npur_env::On("NPUR_EMPTY_STATS");
    return on;
}

static uint64_t EmptyStatsEvery() {
    static const uint64_t every = npur_env::PositiveOr("NPUR_EMPTY_STATS_EVERY", 5000);
    return every;
}

void NoteEmptyOperands(const std::vector<std::vector<uint8_t>*>& postingTypes,
                       const std::vector<std::vector<uint16_t>*>& postingWeights,
                       const std::vector<uint8_t>& sparseDirect, uint32_t postingsNum, uint32_t segmentsNum) {
    uint64_t pairs = 0;
    uint64_t orPairs = 0;
    uint64_t orEmpty = 0;
    uint64_t orEmptyDense = 0;
    for (size_t postingIdx = 0; postingIdx < postingsNum; postingIdx++) {
        if (postingIdx >= postingWeights.size() || postingWeights[postingIdx] == nullptr ||
            postingTypes[postingIdx] == nullptr) {
            continue;
        }
        const std::vector<uint16_t>& w = *(postingWeights[postingIdx]);
        const std::vector<uint8_t>& types = *(postingTypes[postingIdx]);
        const bool orOperand = postingIdx < sparseDirect.size() && sparseDirect[postingIdx] != 0;
        for (size_t segIdx = 0; segIdx < segmentsNum && segIdx < w.size() && segIdx < types.size(); segIdx++) {
            pairs += 1;
            if (!orOperand) {
                continue;
            }
            orPairs += 1;
            if (w[segIdx] == 0) {
                orEmpty += 1;
                if (types[segIdx] == 0) {
                    orEmptyDense += 1;
                }
            }
        }
    }
    g_emptyStats.pairs.fetch_add(pairs, std::memory_order_relaxed);
    g_emptyStats.orPairs.fetch_add(orPairs, std::memory_order_relaxed);
    g_emptyStats.orEmpty.fetch_add(orEmpty, std::memory_order_relaxed);
    g_emptyStats.orEmptyDense.fetch_add(orEmptyDense, std::memory_order_relaxed);
    const uint64_t seen = g_emptyStats.calls.fetch_add(1, std::memory_order_relaxed) + 1;
    if (seen % EmptyStatsEvery() != 0) {
        return;
    }
    const uint64_t allPairs = g_emptyStats.pairs.load(std::memory_order_relaxed);
    const uint64_t allOr = g_emptyStats.orPairs.load(std::memory_order_relaxed);
    const uint64_t allEmpty = g_emptyStats.orEmpty.load(std::memory_order_relaxed);
    const uint64_t allEmptyDense = g_emptyStats.orEmptyDense.load(std::memory_order_relaxed);
    std::ostringstream oss;
    oss << "[EMPTYSTAT] calls=" << seen << " pairs=" << allPairs << " orPairs=" << allOr << " orEmpty=" << allEmpty
        << " orEmptyDense=" << allEmptyDense;
    if (allOr > 0) {
        oss << std::fixed << std::setprecision(2) << " emptyOfOr=" << (100.0 * allEmpty / allOr) << "%"
            << " denseEmptyOfOr=" << (100.0 * allEmptyDense / allOr) << "%";
    }
    if (allPairs > 0) {
        oss << std::fixed << std::setprecision(2) << " emptyOfAll=" << (100.0 * allEmpty / allPairs) << "%";
    }
    fprintf(stdout, "%s\n", oss.str().c_str());
    fflush(stdout);
}

bool ExprStatsEnabled() {
    static const bool on = npur_env::On("NPUR_EXPR_STATS");
    return on;
}

// Node encodings, same as CheckPostExpr validates: AND, OR and NOT are [op, postingNum, stackNum];
// CONJUNCTION is [op, partNum, n0..n_{partNum-1}]. Called only after CheckPostExpr returned true.
void NoteExprShape(const std::vector<uint32_t>& postExpr) {
    for (uint32_t idx = 0; idx < postExpr.size(); idx++) {
        const uint32_t opType = postExpr[idx];
        const uint32_t slot = ExprOpSlot(opType);
        g_exprStats.nodes[slot].fetch_add(1, std::memory_order_relaxed);
        if (opType == FilterOpType::CONJUNCTION) {
            const uint32_t partNum = postExpr[idx + 1];
            g_exprStats.conjPartHist[ExprShapeStats::Bucket(partNum)].fetch_add(1, std::memory_order_relaxed);
            g_exprStats.NoteMax(g_exprStats.maxConjParts, partNum);
            uint64_t total = 0;
            for (uint32_t i = 1; i <= partNum; i++) {
                total += postExpr[idx + 1 + i];
            }
            g_exprStats.postings[slot].fetch_add(total, std::memory_order_relaxed);
            g_exprStats.NoteMax(g_exprStats.maxOperands, total);
            idx = idx + partNum + 1;
            continue;
        }
        const uint32_t postingNum = postExpr[idx + 1];
        const uint32_t stackNum = postExpr[idx + 2];
        g_exprStats.postings[slot].fetch_add(postingNum, std::memory_order_relaxed);
        g_exprStats.stacks[slot].fetch_add(stackNum, std::memory_order_relaxed);
        if (opType != FilterOpType::NOT) {
            g_exprStats.andOrHist[ExprShapeStats::Bucket(postingNum + stackNum)].fetch_add(1,
                                                                                           std::memory_order_relaxed);
        }
        g_exprStats.NoteMax(g_exprStats.maxOperands, postingNum + stackNum);
        idx += 2;
    }
    const uint64_t seen = g_exprStats.calls.fetch_add(1, std::memory_order_relaxed) + 1;
    const uint64_t every = npur_env::PositiveOr("NPUR_EXPR_STATS_EVERY", 5000);
    if (seen != 1 && seen % every != 0) {
        return;
    }
    std::ostringstream oss;
    static const char* kNames[4] = {"and", "or", "not", "conj"};
    oss << "[EXPRSTAT] exprs=" << seen;
    for (uint32_t i = 0; i < 4; i++) {
        oss << " " << kNames[i] << "={nodes=" << g_exprStats.nodes[i].load(std::memory_order_relaxed)
            << ",postings=" << g_exprStats.postings[i].load(std::memory_order_relaxed)
            << ",stacks=" << g_exprStats.stacks[i].load(std::memory_order_relaxed) << "}";
    }
    oss << " maxOperands=" << g_exprStats.maxOperands.load(std::memory_order_relaxed)
        << " maxConjParts=" << g_exprStats.maxConjParts.load(std::memory_order_relaxed);
    oss << " andOrOperandHist(1,2,3,4,5-8,9-16,17-32,32+)=[";
    for (uint32_t b = 0; b < 8; b++) {
        oss << g_exprStats.andOrHist[b].load(std::memory_order_relaxed) << (b == 7 ? "" : ",");
    }
    oss << "] conjPartHist=[";
    for (uint32_t b = 0; b < 8; b++) {
        oss << g_exprStats.conjPartHist[b].load(std::memory_order_relaxed) << (b == 7 ? "" : ",");
    }
    oss << "]";
    fprintf(stdout, "%s\n", oss.str().c_str());
    fflush(stdout);
}

void NoteBitlistCall(const BitlistCallShape& s) {
    g_bitlistStats.Note(static_cast<uint64_t>(s.tableCount), static_cast<uint64_t>(s.bitlistNum),
                        static_cast<uint64_t>(s.bitlistNum) * s.segmentByteSize);
    const uint64_t every = npur_env::PositiveOr("NPUR_BITLIST_STATS_EVERY", 10000);
    const uint64_t seen = g_bitlistStats.calls.load(std::memory_order_relaxed);
    if (seen == 1 || seen % every == 0) {
        const uint64_t tableSum = g_bitlistStats.tableSum.load(std::memory_order_relaxed);
        const uint64_t bitlistSum = g_bitlistStats.bitlistSum.load(std::memory_order_relaxed);
        std::ostringstream oss;
        oss << "[BITSTAT-QUERY] dev=" << s.deviceId << " calls=" << seen << " segments=" << s.segmentsNum
            << " postingsPerQuery=" << s.postingsNum
            << " pairsPerCall=" << static_cast<double>(tableSum) / static_cast<double>(seen)
            << " convertedPerCall=" << static_cast<double>(bitlistSum) / static_cast<double>(seen) << " convertedRatio="
            << (tableSum == 0 ? 0.0 : static_cast<double>(bitlistSum) / static_cast<double>(tableSum))
            << " convertedMin=" << g_bitlistStats.bitlistMin.load(std::memory_order_relaxed)
            << " convertedMax=" << g_bitlistStats.bitlistMax.load(std::memory_order_relaxed) << " denseOutBytesPerCall="
            << static_cast<double>(g_bitlistStats.bitsetBytesSum.load(std::memory_order_relaxed)) /
                   static_cast<double>(seen)
            << " segmentByteSize=" << s.segmentByteSize << " blockDim=" << s.blockDim
            << " maxPostingLengthByte=" << s.maxPostingLengthByte << " ubUnitCapacity=" << s.maxPostingLengthByte / 6;
        fprintf(stdout, "%s\n", oss.str().c_str());
        fflush(stdout);
    }
}

void NoteBitlistDeep(const BitlistCallShape& s) {
    if (g_bitlistDeepCalls.fetch_add(1, std::memory_order_relaxed) >=
        npur_env::PositiveOr("NPUR_BITLIST_STATS_CALLS", 2)) {
        return;
    }
    std::vector<uint32_t> kPerPosting(s.bitlistNum, 0);
    bool readOk = true;
    for (size_t i = 0; i < s.bitlistNum && readOk; i++) {
        uint8_t hdr[8] = {0};
        if (aclrtMemcpy(hdr, sizeof(hdr), s.srcList[i], sizeof(hdr), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
            LOG_ERROR("[BITSTAT-DEEP] header read failed at " << i);
            readOk = false;
            break;
        }
        kPerPosting[i] = *reinterpret_cast<uint32_t*>(hdr + 4) / 6;
    }
    if (readOk) {
        uint64_t kSum = 0;
        uint32_t kMax = 0;
        uint64_t kHist[16] = {0};
        for (size_t i = 0; i < s.bitlistNum; i++) {
            const uint32_t k = kPerPosting[i];
            kSum += k;
            kMax = std::max(kMax, k);
            uint32_t bucket = 0;
            while (bucket < 15 && (k >> bucket) != 0) {
                bucket += 1;
            }
            kHist[bucket] += 1;
        }
        uint64_t coreKMax = 0;
        uint64_t coreKMin = ~0ULL;
        uint32_t corePostingsMax = 0;
        for (uint32_t b = 0; b < s.blockDim; b++) {
            uint32_t begin = 0;
            uint32_t end = 0;
            if (b < s.formerNum) {
                begin = b * s.formerLength;
                end = begin + s.formerLength;
            } else {
                begin = s.formerNum * s.formerLength + s.tailLength * (b - s.formerNum);
                end = begin + s.tailLength;
            }
            end = std::min<uint32_t>(end, static_cast<uint32_t>(s.bitlistNum));
            uint64_t coreK = 0;
            for (uint32_t i = begin; i < end; i++) {
                coreK += kPerPosting[i];
            }
            coreKMax = std::max(coreKMax, coreK);
            coreKMin = std::min(coreKMin, coreK);
            corePostingsMax = std::max<uint32_t>(corePostingsMax, end > begin ? end - begin : 0);
        }
        const double coreKMean = static_cast<double>(kSum) / static_cast<double>(s.blockDim);
        std::ostringstream oss;
        oss << "[BITSTAT-DEEP] dev=" << s.deviceId << " converted=" << s.bitlistNum << " kSum=" << kSum
            << " kMean=" << static_cast<double>(kSum) / static_cast<double>(s.bitlistNum) << " kMax=" << kMax
            << " ubUnitCapacity=" << s.maxPostingLengthByte / 6 << " ubFillMean="
            << (s.maxPostingLengthByte < 6 ? 0.0
                                           : static_cast<double>(kSum) / static_cast<double>(s.bitlistNum) /
                                                 static_cast<double>(s.maxPostingLengthByte / 6))
            << " bytesInPerPosting=" << static_cast<double>(kSum) * 6.0 / static_cast<double>(s.bitlistNum)
            << " bytesOutPerPosting=" << s.segmentByteSize << " coreKMax=" << coreKMax << " coreKMin=" << coreKMin
            << " coreKMean=" << coreKMean
            << " coreImbalance=" << (coreKMean == 0.0 ? 0.0 : static_cast<double>(coreKMax) / coreKMean)
            << " corePostingsMax=" << corePostingsMax << " kHist=[";
        for (uint32_t b = 0; b < 16; b++) {
            oss << kHist[b] << (b == 15 ? "" : ",");
        }
        oss << "]";
        fprintf(stdout, "%s\n", oss.str().c_str());
        fflush(stdout);
    }
}

// ---- the two debug dumps, both reached only through PRINT_RES_ENABLE -------
namespace {
const uint32_t PRINT_TARGET_SEGMENT_INDEX = 0;  // print info for this target segment
}  // namespace

void TextFilter::PrintDevicePostings(uint8_t*& devicePostings, size_t postingsByteSize, uint32_t postingsNum,
                                     uint32_t segmentsNum, std::vector<uint8_t*> postings) {
    uint64_t* devicePostingsHost;
    uint16_t* segHost;
    auto ret = aclrtMallocHost((void**)&devicePostingsHost, postingsByteSize);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("PrintDevicePostings aclrtMallocHost fail, error code is:" << ret);
        return;
    }
    ret = aclrtMallocHost((void**)&segHost, m_segmentByteSize);
    if (ret != ACL_SUCCESS) {
        CHECK_ACL_ONLY_LOG(aclrtFreeHost(devicePostingsHost));
        LOG_ERROR("PrintDevicePostings aclrtMallocHost fail, error code is:" << ret);
        return;
    }
    ret =
        aclrtMemcpy(devicePostingsHost, postingsByteSize, devicePostings, postingsByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        CHECK_ACL_ONLY_LOG(aclrtFreeHost(devicePostingsHost));
        CHECK_ACL_ONLY_LOG(aclrtFreeHost(segHost));
        LOG_ERROR("PrintDevicePostings aclrtMemcpy fail, error code is:" << ret);
        return;
    }

    for (size_t postingsNumIdx = 0; postingsNumIdx < postingsNum; postingsNumIdx++) {
        for (size_t segmentsNumIdx = 0; segmentsNumIdx < segmentsNum; segmentsNumIdx++) {
            if (segmentsNumIdx != PRINT_TARGET_SEGMENT_INDEX || postingsNumIdx != 0) {
                continue;
            }
            aclrtMemcpy(segHost, m_segmentByteSize,
                        (uint8_t*)(*(devicePostingsHost + segmentsNumIdx * postingsNum + postingsNumIdx)),
                        m_segmentByteSize, ACL_MEMCPY_DEVICE_TO_HOST);

            std::string postingsRawStr;
            for (size_t i = 0; i < m_segmentByteSize / sizeof(uint16_t); i++) {
                std::ostringstream oss;
                uint64_t value = *(segHost + i);
                oss << std::hex << std::uppercase << (uint64_t)value;
                postingsRawStr += oss.str() + ",";
            }
            LOG_INFO(" --- after PostingBitListToSet: posting["
                     << postingsNumIdx << "], seg[" << segmentsNumIdx
                     << "], device_addr:" << *(devicePostingsHost + segmentsNumIdx * postingsNum + postingsNumIdx)
                     << " = [" << postingsRawStr << "]");
        }
    }
    for (size_t segmentsNumIdx = 0; segmentsNumIdx < segmentsNum; segmentsNumIdx++) {
        if (segmentsNumIdx != PRINT_TARGET_SEGMENT_INDEX) {
            continue;
        }
        std::string addrStr;
        for (size_t postingsNumIdx = 0; postingsNumIdx < postingsNum; postingsNumIdx++) {
            std::ostringstream oss;
            uint64_t value = (uint64_t)(postings[segmentsNumIdx * postingsNum + postingsNumIdx]);
            oss << std::hex << std::uppercase << (uint64_t)value;
            addrStr += oss.str() + ",";
        }
        LOG_INFO("before PostingBitListToSet, devicePostings seg[" << segmentsNumIdx << "], addrs:[" << addrStr << "]");
    }
    for (size_t segmentsNumIdx = 0; segmentsNumIdx < segmentsNum; segmentsNumIdx++) {
        if (segmentsNumIdx != PRINT_TARGET_SEGMENT_INDEX) {
            continue;
        }
        std::string addrStr;
        for (size_t postingsNumIdx = 0; postingsNumIdx < postingsNum; postingsNumIdx++) {
            std::ostringstream oss;
            uint64_t value = *(devicePostingsHost + segmentsNumIdx * postingsNum + postingsNumIdx);
            oss << std::hex << std::uppercase << (uint64_t)value;
            addrStr += oss.str() + ",";
        }
        LOG_INFO("after PostingBitListToSet, devicePostings: seg[" << segmentsNumIdx << "], addrs:[" << addrStr << "]");
    }
    CHECK_ACL_ONLY_LOG(aclrtFreeHost(devicePostingsHost));
    CHECK_ACL_ONLY_LOG(aclrtFreeHost(segHost));
}

void TextFilter::PrintFilterRes(const uint8_t* resultInDevice) {
    const uint32_t segmentsNum = m_dataTable->GetSegmentNum();
    uint16_t* resultHost;
    uint32_t resultHostByteSize = segmentsNum * m_segmentByteSize;
    auto ret = aclrtMallocHost((void**)&resultHost, resultHostByteSize);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("PrintFilterRes aclrtMallocHost fail, error code is:" << ret);
        return;
    }
    ret = aclrtMemcpy(resultHost, resultHostByteSize, resultInDevice, resultHostByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        CHECK_ACL_ONLY_LOG(aclrtFreeHost(resultHost));
        LOG_ERROR("PrintFilterRes aclrtMemcpy fail, error code is:" << ret);
        return;
    }
    for (uint32_t segIdx = 0; segIdx < segmentsNum; segIdx++) {
        if (segIdx != PRINT_TARGET_SEGMENT_INDEX) {
            continue;
        }
        std::string postingsRawStr;
        for (uint32_t i = 0; i < m_segmentLength; i++) {
            std::ostringstream oss;
            uint64_t value = *(resultHost + segIdx * m_segmentLength + i);
            oss << std::hex << std::uppercase << (uint64_t)value;
            postingsRawStr += oss.str() + ",";
        }
        LOG_INFO("after BitmapTextFilter:  seg[" << segIdx << "] = [" << postingsRawStr << "]");
    }
    CHECK_ACL_ONLY_LOG(aclrtFreeHost(resultHost));
}

}  // namespace NpuRetrieval
