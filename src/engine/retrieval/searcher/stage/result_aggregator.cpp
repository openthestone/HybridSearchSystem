#include "result_aggregator.h"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include "acl/acl.h"
#include "ascend_device/aclrtlaunch_Aggregator.h"
#include "ascend_device/aclrtlaunch_TOPK.h"
#include "configuration/develop_configuration.h"
#include "src/utils/logger.h"
#include "src/full_recall/core/constant_definition.h"
#include "src/full_recall/core/log_definition.h"
#include "src/utils/performance_recorder.h"
#include "src/full_recall/retrieval/searcher/runtime/gm_memory_manager.h"
#include "src/full_recall/retrieval/searcher/runtime/stream_manager.h"
#include "src/utils/env_switch.h"

namespace NpuRetrieval {
const uint32_t SIZEOF_FLOAT = sizeof(float);
const uint32_t TOPK_MIN_LIMIT = 1024;
const uint32_t RESULT_MESSAGE_SIZE_IN_DEVICE = 16;
const uint32_t RESULT_MESSAGE_BYTE_SIZE_IN_DEVICE = RESULT_MESSAGE_SIZE_IN_DEVICE * SIZEOF_FLOAT * 48;
const uint32_t MAX_BLOCK_SIZE = 5888;  // 256*23; must be a multiple of 256 and keep the aggregator kernel within UB
// A core count that does not divide blockNumber makes the tail core the straggler; at 5M
// docs/shard blockNumber is 850 = 2*5^2*17.
const uint32_t BLOCK_DIM_MAX = 40;
// Where TOPK writes its own counts under NPUR_FUSE_AGG_TOPK. The block is 16*48 = 768 uint32
// and the Aggregator uses core i's slot at i*16, i < 40, so 640..703 is free for kernel_topk.h.
const uint32_t kTopkCountElemOffset = BLOCK_DIM_MAX * 16;

bool compareDesc(const ScoreWithIndex& a, const ScoreWithIndex& b) {
    return a.score > b.score;
}

// IEEE-754 monotonic transform, so (score, index) packs into a uint64 that sorts as a plain integer.
static inline uint32_t ScoreToSortable(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return bits ^ ((0u - (bits >> 31)) | 0x80000000u);
}

// NPUR_SHARD_TOPK_RATIO in (0,1], default 1: each shard aggregates only its top ceil(topK*ratio).
static double ShardTopKRatio() {
    static const double ratio = npur_env::DoubleInRange("NPUR_SHARD_TOPK_RATIO", 1.0, 0.0, 1.0);
    return ratio;
}

// NPUR_PACKED_SORT=1: sort 8-byte packed (score,index) keys instead of 16-byte ScoreWithIndex.
static bool PackedSortMode() {
    static const bool enabled = npur_env::On("NPUR_PACKED_SORT");
    return enabled;
}

// NPUR_RADIX_SORT=1: LSD radix sort of the packed keys instead of std::sort. Implies the packed path.
static bool RadixSortMode() {
    static const bool enabled = npur_env::On("NPUR_RADIX_SORT");
    return enabled;
}

// Keys are unique, so this is byte-identical to std::sort with greater<>. firstShift 32 sorts
// the high 4 bytes only and leaves ties in input order -- every pass is a stable counting sort.
static void RadixSortDescU64(std::vector<uint64_t>& keys, int firstShift = 0) {
    const size_t n = keys.size();
    if (n < 2) {
        return;
    }
    std::vector<uint64_t> tmp(n);
    uint64_t* src = keys.data();
    uint64_t* dst = tmp.data();
    for (int shift = firstShift; shift < 64; shift += 8) {
        size_t cnt[256] = {0};
        for (size_t i = 0; i < n; i++) {
            cnt[(src[i] >> shift) & 0xFFu]++;
        }
        size_t pos = 0;
        size_t start[256];
        for (int b = 255; b >= 0; b--) {
            start[b] = pos;
            pos += cnt[b];
        }
        for (size_t i = 0; i < n; i++) {
            dst[start[(src[i] >> shift) & 0xFFu]++] = src[i];
        }
        std::swap(src, dst);
    }
    // 8 (even) passes -> the sorted data ends up back in keys.data(); no final copy.
}

// Process-global fence: two shards' synchronous D2H copies collide on shared ACL-runtime state.
static std::mutex g_d2hCopyMutex;

void ResultAggregator::SetTopK(uint32_t topK) {
    const double ratio = ShardTopKRatio();
    m_topK = (ratio < 1.0) ? std::max<uint32_t>(1, static_cast<uint32_t>(std::ceil(topK * ratio))) : topK;
}

bool ResultAggregator::AggrAndTopKDevice(uint8_t* filterResultInDevice, uint8_t* docScoreInDevice,
                                         size_t docAllByteSize, uint8_t*& docLocationInDeviceConst,
                                         AggrDeviceResult& out) {
    out = AggrDeviceResult{};  // stays empty unless we stage host buffers below
    uint32_t effectiveCount = 0;
    uint32_t effectiveCountSmaller = 0;
    GmBlock docLocationInDeviceResultChunk{};
    {
        RecordGuard guardAlloc{"AggrAndTopK_allocBlocks"};
        GmMemoryManager::GetByDeviceId(m_deviceId)
            ->AllocateBlock(GmPoolName::AGGREGATOR_POOL, docAllByteSize, docLocationInDeviceResultChunk);
    }
    if (docLocationInDeviceResultChunk.data == nullptr) {
        LOG_ERROR("docLocationInDeviceResultChunk is nullptr. AggrAndTopK fail" << ",m_deviceId=" << m_deviceId);
        return false;
    }
    uint8_t* docLocationInDeviceResult = reinterpret_cast<uint8_t*>(docLocationInDeviceResultChunk.data);

    GmBlock resultNextInDeviceChunk{};
    {
        RecordGuard guardAlloc{"AggrAndTopK_allocBlocks"};
        GmMemoryManager::GetByDeviceId(m_deviceId)
            ->AllocateBlock(GmPoolName::AGGREGATOR_POOL, docAllByteSize, resultNextInDeviceChunk);
    }
    if (resultNextInDeviceChunk.data == nullptr) {
        LOG_ERROR("resultNextInDeviceChunk is nullptr. AggrAndTopK fail" << ",m_deviceId=" << m_deviceId);
        GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(docLocationInDeviceResultChunk);
        return false;
    }
    uint8_t* resultNextInDevice = reinterpret_cast<uint8_t*>(resultNextInDeviceChunk.data);

    GmBlock resultLocationNextInDeviceChunk{};
    {
        RecordGuard guardAlloc{"AggrAndTopK_allocBlocks"};
        GmMemoryManager::GetByDeviceId(m_deviceId)
            ->AllocateBlock(GmPoolName::AGGREGATOR_POOL, docAllByteSize, resultLocationNextInDeviceChunk);
    }
    if (resultLocationNextInDeviceChunk.data == nullptr) {
        LOG_ERROR("resultLocationNextInDeviceChunk is nullptr. AggrAndTopK fail" << ",m_deviceId=" << m_deviceId);
        FreeAggregator(nullptr, nullptr, docLocationInDeviceResultChunk, resultNextInDeviceChunk,
                       resultLocationNextInDeviceChunk);
        return false;
    }
    uint8_t* resultLocationNextInDevice = reinterpret_cast<uint8_t*>(resultLocationNextInDeviceChunk.data);

    if (!Aggregator((uint8_t*)filterResultInDevice, docScoreInDevice, docAllByteSize, docLocationInDeviceConst,
                    docLocationInDeviceResult, resultNextInDevice, resultLocationNextInDevice, m_topK, effectiveCount,
                    effectiveCountSmaller)) {
        LOG_ERROR("Aggregator fail" << ",m_deviceId=" << m_deviceId);
        FreeAggregator(nullptr, nullptr, docLocationInDeviceResultChunk, resultNextInDeviceChunk,
                       resultLocationNextInDeviceChunk);
        return false;
    }

    uint32_t effectiveCountSum = effectiveCount + effectiveCountSmaller;
    LOG_DEBUG("effectiveCount =" << effectiveCount << ",effectiveCountSmaller =" << effectiveCountSmaller
                                 << ",effectiveCountSum =" << effectiveCountSum);
    if (effectiveCountSum == 0) {
        FreeAggregator(nullptr, nullptr, docLocationInDeviceResultChunk, resultNextInDeviceChunk,
                       resultLocationNextInDeviceChunk);
        return true;
    }
    if (effectiveCountSum > docAllByteSize / SIZEOF_FLOAT) {
        LOG_ERROR("effectiveCountSum is spill over , effectiveCountSum="
                  << effectiveCountSum << ", docAllByteSize=" << docAllByteSize << ",m_deviceId=" << m_deviceId);
        FreeAggregator(nullptr, nullptr, docLocationInDeviceResultChunk, resultNextInDeviceChunk,
                       resultLocationNextInDeviceChunk);
        return false;
    }
    uint8_t* hostTOPKResult = nullptr;
    uint8_t* hostDocLocation = nullptr;
    uint32_t effectiveSumByteSize = effectiveCountSum * SIZEOF_FLOAT;
    aclError ret = ACL_SUCCESS;
    {
        RecordGuard guardMallocHost{"AggrAndTopK_mallocHost"};
        ret = aclrtMallocHost((void**)(&hostTOPKResult), effectiveSumByteSize);
        if (ret == ACL_SUCCESS)
            ret = aclrtMallocHost((void**)(&hostDocLocation), effectiveSumByteSize);
    }
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtMallocHost fail, error code is:" << ret << ",m_deviceId=" << m_deviceId);
        FreeAggregator(hostTOPKResult, hostDocLocation, docLocationInDeviceResultChunk, resultNextInDeviceChunk,
                       resultLocationNextInDeviceChunk);
        return false;
    }
    bool d2hOk = AggregatorResultFillD2H(docScoreInDevice, docLocationInDeviceResult, resultNextInDevice,
                                         resultLocationNextInDevice, hostTOPKResult, hostDocLocation, effectiveCount,
                                         effectiveCountSmaller);
    {
        RecordGuard guardFree{"AggrAndTopK_FreeAggregator"};
        FreeAggregator(d2hOk ? nullptr : hostTOPKResult, d2hOk ? nullptr : hostDocLocation,
                       docLocationInDeviceResultChunk, resultNextInDeviceChunk, resultLocationNextInDeviceChunk);
    }
    if (!d2hOk)
        return false;
    out.hostTOPKResult = hostTOPKResult;
    out.hostDocLocation = hostDocLocation;
    out.effectiveCount = effectiveCount;
    out.effectiveCountSmaller = effectiveCountSmaller;
    out.empty = false;
    return true;
}

bool ResultAggregator::AggrAndTopKExtract(AggrDeviceResult& dr, std::shared_ptr<FullRecallResult>& fullRecallResult) {
    if (dr.empty || dr.hostTOPKResult == nullptr)
        return true;  // effectiveCountSum==0: empty result, nothing was staged
    bool state = AggregatorResultFillExtract(dr.hostTOPKResult, dr.hostDocLocation, dr.effectiveCount,
                                             dr.effectiveCountSmaller, fullRecallResult);
    {
        RecordGuard guardFree{"AggrAndTopK_FreeExtract"};
        if (dr.hostTOPKResult != nullptr)
            CHECK_ACL_ONLY_LOG(aclrtFreeHost(dr.hostTOPKResult));
        if (dr.hostDocLocation != nullptr)
            CHECK_ACL_ONLY_LOG(aclrtFreeHost(dr.hostDocLocation));
    }
    dr = AggrDeviceResult{};
    return state;
}

bool ResultAggregator::BatchAggregateMode() {
    static const bool on = npur_env::On("NPUR_BATCH_AGGREGATE");
    return on;
}

// NPUR_TOPK_CONCURRENT=1: streams of their own for the second and later TOPK kernels. TOPK is
// single-core by construction (blockDim 1, and kernel_topk.h never reads GetBlockIdx).
// NPUR_EXTRACT_FAST=1: sort keys built in DESCENDING index order so a stable 4-pass radix
// reproduces the 8-pass order, and the result loop split into one pass per memory.
static bool ExtractFastMode() {
    static const bool on = npur_env::On("NPUR_EXTRACT_FAST");
    return on;
}

static bool TopkConcurrentMode() {
    static const bool on = npur_env::On("NPUR_TOPK_CONCURRENT");
    return on;
}

// NPUR_FUSE_AGG_TOPK=1: TOPK queues behind its Aggregator on the same stream, writing its counts
// at kTopkCountElemOffset so one D2H brings back both. A query that matched nothing then reaches
// the kernel -- that is what kernel_topk.h's m_realDocNum guard is for.
static bool FuseAggTopkMode() {
    static const bool on = npur_env::On("NPUR_FUSE_AGG_TOPK");
    return on;
}

// NPUR_TOPK_COUNTS_IN_PLACE=1: TOPK reads the counts at the stride the Aggregator wrote them,
// not from a host-packed upload.
static bool TopkCountsInPlaceMode() {
    static const bool on = npur_env::On("NPUR_TOPK_COUNTS_IN_PLACE");
    return on;
}

// NPUR_AGG_CONCURRENT=<N> (default 1 = off): Aggregator kernels over N streams. Never on a
// shared stream -- it would no longer be ordered behind the filter kernels it reads.
static uint32_t AggConcurrentStreams() {
    static const uint32_t v = npur_env::U32Clamped("NPUR_AGG_CONCURRENT", 1, 1, 8);
    return v;
}

// NPUR_TOPK_LOOP_STATS=1: aggregate the kernel's counts[2] iteration count, print [Topk] at exit.
namespace {
struct TopkLoopStats {
    std::atomic<bool> enabled{false};
    std::atomic<uint64_t> queries{0};
    std::atomic<uint64_t> iterSum{0};
    std::atomic<uint64_t> iterMax{0};
    std::atomic<uint64_t> survivorSum{0};
    std::atomic<uint64_t> scanSum{0};     // survivors scanned, summed over iterations (approx traffic)
    std::atomic<uint64_t> iterHist[9]{};  // 1..8, then 9+ in [8]
    // Buckets of 100K, last one 700K+.
    std::atomic<uint64_t> survivorMin{~0ull};
    std::atomic<uint64_t> survivorMax{0};
    std::atomic<uint64_t> survivorHist[8]{};
    ~TopkLoopStats() {
        const uint64_t nq = queries.load();
        if (!enabled.load() || nq == 0)
            return;
        std::printf(
            "[Topk] loop stats over %llu topk-queries: iters avg=%.2f max=%llu; "
            "survivors avg=%.0f; scanned avg=%.0f (%.2fx survivors)\n",
            static_cast<unsigned long long>(nq), static_cast<double>(iterSum.load()) / nq,
            static_cast<unsigned long long>(iterMax.load()), static_cast<double>(survivorSum.load()) / nq,
            static_cast<double>(scanSum.load()) / nq,
            survivorSum.load() ? static_cast<double>(scanSum.load()) / survivorSum.load() : 0.0);
        std::printf("[Topk] survivors: min=%llu max=%llu; histogram (100K buckets):",
                    static_cast<unsigned long long>(survivorMin.load()),
                    static_cast<unsigned long long>(survivorMax.load()));
        for (int b = 0; b < 7; ++b)
            std::printf(" %dK:%llu", b * 100, static_cast<unsigned long long>(survivorHist[b].load()));
        std::printf(" 700K+:%llu\n", static_cast<unsigned long long>(survivorHist[7].load()));
        std::printf("[Topk] iter histogram:");
        for (int b = 0; b < 8; ++b)
            std::printf(" %d:%llu", b + 1, static_cast<unsigned long long>(iterHist[b].load()));
        std::printf(" 9+:%llu\n", static_cast<unsigned long long>(iterHist[8].load()));
        std::fflush(stdout);
    }
};
TopkLoopStats g_topkLoopStats;
bool TopkLoopStatsEnabled() {
    static const bool on = [] {
        const bool v = npur_env::On("NPUR_TOPK_LOOP_STATS");
        g_topkLoopStats.enabled.store(v);
        return v;
    }();
    return on;
}
inline void TopkLoopStatsAdd(uint32_t iters, uint32_t survivors) {
    g_topkLoopStats.queries.fetch_add(1, std::memory_order_relaxed);
    g_topkLoopStats.iterSum.fetch_add(iters, std::memory_order_relaxed);
    uint64_t prev = g_topkLoopStats.iterMax.load(std::memory_order_relaxed);
    while (iters > prev && !g_topkLoopStats.iterMax.compare_exchange_weak(prev, iters)) {
    }
    g_topkLoopStats.survivorSum.fetch_add(survivors, std::memory_order_relaxed);
    uint64_t lo = g_topkLoopStats.survivorMin.load(std::memory_order_relaxed);
    while (survivors < lo && !g_topkLoopStats.survivorMin.compare_exchange_weak(lo, survivors)) {
    }
    uint64_t hi = g_topkLoopStats.survivorMax.load(std::memory_order_relaxed);
    while (survivors > hi && !g_topkLoopStats.survivorMax.compare_exchange_weak(hi, survivors)) {
    }
    const uint32_t sBucket = survivors / 100000 >= 7 ? 7 : survivors / 100000;
    g_topkLoopStats.survivorHist[sBucket].fetch_add(1, std::memory_order_relaxed);
    // A MODEL, not a measurement: sum survivors / 2^i, capped at the iteration count actually run.
    uint64_t scanned = 0;
    uint64_t sz = survivors;
    for (uint32_t i = 0; i < iters && sz > 0; ++i) {
        scanned += sz;
        sz /= 2;
    }
    g_topkLoopStats.scanSum.fetch_add(scanned, std::memory_order_relaxed);
    const int bucket = iters >= 9 ? 8 : static_cast<int>(iters) - 1;
    if (bucket >= 0)
        g_topkLoopStats.iterHist[bucket].fetch_add(1, std::memory_order_relaxed);
}
}  // namespace

static uint32_t AggTopKOf(uint32_t topK) {
    const double ratio = ShardTopKRatio();
    return (ratio < 1.0) ? std::max<uint32_t>(1, static_cast<uint32_t>(std::ceil(topK * ratio))) : topK;
}

namespace {
struct BatchSlot {
    uint32_t aggTopK{0};
    uint8_t* filterResult{nullptr};
    uint8_t* docScore{nullptr};
    GmBlock docLocationResultChunk{};
    GmBlock resultNextChunk{};
    GmBlock resultLocationNextChunk{};
    uint32_t effectiveCount{0};
    uint32_t effectiveCountSmaller{0};
    uint32_t docNumberShards[BLOCK_DIM_MAX]{};
    bool held{false};    // device blocks allocated and still to be freed
    bool active{false};  // participates in the remaining phases
    bool topk{false};    // goes through the TOPK kernel
    bool merge{false};   // goes through the D2D Merge instead
};
}  // namespace

bool ResultAggregator::AggrAndTopKDeviceBatch(const std::vector<uint8_t*>& filterResults,
                                              const std::vector<uint8_t*>& docScores, size_t docAllByteSize,
                                              uint8_t*& docLocationInDeviceConst, const std::vector<uint32_t>& topKs,
                                              std::vector<AggrDeviceResult>& out, aclrtStream sharedStream,
                                              bool* sharedStreamSynced) {
    const size_t n = topKs.size();
    out.assign(n, AggrDeviceResult{});
    if (n == 0)
        return true;
    if (filterResults.size() != n || docScores.size() != n) {
        LOG_ERROR("AggrAndTopKDeviceBatch size mismatch: " << filterResults.size() << "/" << docScores.size() << " vs "
                                                           << n << ",m_deviceId=" << m_deviceId);
        return false;
    }

    uint32_t blockDimAggregator = FLAGS_full_recall_aggregator_block_dim;
    if (blockDimAggregator > BLOCK_DIM_MAX)
        blockDimAggregator = BLOCK_DIM_MAX;
    if (blockDimAggregator < 1)
        blockDimAggregator = 1;
    const uint32_t docNumber = static_cast<uint32_t>(docAllByteSize / sizeof(float));
    const uint32_t blockLength = MAX_BLOCK_SIZE;
    const uint32_t blockNumber = (docNumber + blockLength - 1) / blockLength;
    const uint32_t lastBlockLength = docNumber - blockLength * (blockNumber - 1);
    if (blockNumber < 20)  // small data: no point spreading it over cores
        blockDimAggregator = 1;

    auto mm = GmMemoryManager::GetByDeviceId(m_deviceId);
    std::vector<BatchSlot> slot(n);
    auto releaseBlocks = [&]() {
        for (size_t q = 0; q < n; ++q) {
            if (!slot[q].held)
                continue;
            mm->FreeBlock(slot[q].docLocationResultChunk);
            mm->FreeBlock(slot[q].resultNextChunk);
            mm->FreeBlock(slot[q].resultLocationNextChunk);
            slot[q].held = false;
        }
    };

    {
        RecordGuard guardAlloc{"AggrAndTopK_allocBlocks"};
        for (size_t q = 0; q < n; ++q) {
            BatchSlot& s = slot[q];
            s.aggTopK = AggTopKOf(topKs[q]);
            s.filterResult = filterResults[q];
            s.docScore = docScores[q];
            if (s.filterResult == nullptr || s.docScore == nullptr)
                continue;  // no query at this slot; out[q] stays empty
            mm->AllocateBlock(GmPoolName::AGGREGATOR_POOL, docAllByteSize, s.docLocationResultChunk);
            mm->AllocateBlock(GmPoolName::AGGREGATOR_POOL, docAllByteSize, s.resultNextChunk);
            mm->AllocateBlock(GmPoolName::AGGREGATOR_POOL, docAllByteSize, s.resultLocationNextChunk);
            if (s.docLocationResultChunk.data == nullptr || s.resultNextChunk.data == nullptr ||
                s.resultLocationNextChunk.data == nullptr) {
                LOG_ERROR("AggrAndTopKDeviceBatch AllocateBlock fail, q=" << q << ",m_deviceId=" << m_deviceId);
                mm->FreeBlock(s.docLocationResultChunk);
                mm->FreeBlock(s.resultNextChunk);
                mm->FreeBlock(s.resultLocationNextChunk);
                releaseBlocks();
                return false;
            }
            s.held = true;
            s.active = true;
        }
    }
    const uint32_t countStride = RESULT_MESSAGE_BYTE_SIZE_IN_DEVICE;
    GmBlock countChunk{};
    mm->AllocateBlock(GmPoolName::AGGREGATOR_SCRATCH_POOL, static_cast<uint32_t>(n) * countStride, countChunk);
    if (countChunk.data == nullptr) {
        LOG_ERROR("AggrAndTopKDeviceBatch count scratch AllocateBlock fail,m_deviceId=" << m_deviceId);
        releaseBlocks();
        return false;
    }
    uint8_t* countDevice = reinterpret_cast<uint8_t*>(countChunk.data);
    const bool countsInPlace = TopkCountsInPlaceMode();
    const bool fuse = FuseAggTopkMode() && countsInPlace;
    bool fused = false;  // set once the launches actually went out that way
    auto releaseCount = [&]() {
        if (countChunk.data != nullptr) {
            mm->FreeBlock(countChunk);
            countChunk.data = nullptr;
        }
    };

    // Under NPUR_SHARE_FILTER_STREAM the caller owns this stream, so it is never pooled here.
    aclrtStream stream = sharedStream != nullptr ? sharedStream : StreamManager::GetInstance()->GetStream(m_deviceId);
    auto releaseStream = [&]() {
        if (sharedStream == nullptr) {
            StreamManager::GetInstance()->FreeStream(m_deviceId, stream);
        }
    };
    {
        RecordGuard guard{"AggrAndTopK_Aggregator_NPU"};
        std::vector<aclrtStream> lanes{stream};
        const uint32_t wantLanes = sharedStream == nullptr ? AggConcurrentStreams() : 1u;
        for (uint32_t i = 1; i < wantLanes; ++i) {
            aclrtStream extra = StreamManager::GetInstance()->GetStream(m_deviceId);
            if (extra == nullptr)
                break;
            lanes.push_back(extra);
        }
        size_t lane = 0;
        for (size_t q = 0; q < n; ++q) {
            if (!slot[q].active)
                continue;
            aclrtStream launchStream = lanes[lane % lanes.size()];
            lane++;
            ACLRT_LAUNCH_KERNEL(Aggregator)
            (blockDimAggregator, launchStream, slot[q].filterResult, slot[q].docScore, docLocationInDeviceConst,
             reinterpret_cast<uint8_t*>(slot[q].docLocationResultChunk.data), countDevice + q * countStride, docNumber,
             blockNumber, blockLength, lastBlockLength);
        }
        // Only with a single lane: on another stream TOPK is not ordered behind its input.
        if (fuse && lanes.size() == 1) {
            const uint32_t blockNumberShardAgg = blockNumber / blockDimAggregator;
            for (size_t q = 0; q < n; ++q) {
                BatchSlot& s = slot[q];
                if (!s.active)
                    continue;
                ACLRT_LAUNCH_KERNEL(TOPK)
                (1, stream, s.docScore, reinterpret_cast<uint8_t*>(s.docLocationResultChunk.data),
                 reinterpret_cast<uint8_t*>(s.resultNextChunk.data),
                 reinterpret_cast<uint8_t*>(s.resultLocationNextChunk.data),
                 countDevice + q * countStride + kTopkCountElemOffset * sizeof(uint32_t), countDevice + q * countStride,
                 FLAGS_full_recall_npu_topk_loop_count, blockDimAggregator, blockNumberShardAgg, s.aggTopK,
                 static_cast<uint32_t>(s.aggTopK * FLAGS_full_recall_npu_topk_finish_buffer_ratio),
                 RESULT_MESSAGE_SIZE_IN_DEVICE);
            }
            fused = true;
        }
        aclError ret = aclrtSynchronizeStream(stream);
        for (size_t i = 1; i < lanes.size(); ++i) {
            const aclError laneRet = aclrtSynchronizeStream(lanes[i]);
            if (laneRet != ACL_SUCCESS) {
                LOG_ERROR("aclrtSynchronizeStream (Aggregator lane) fail, error code is:" << laneRet << ",m_deviceId="
                                                                                          << m_deviceId);
                if (ret == ACL_SUCCESS)
                    ret = laneRet;
            }
            StreamManager::GetInstance()->FreeStream(m_deviceId, lanes[i]);
        }
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("aclrtSynchronizeStream fail, error code is:" << ret << ",m_deviceId=" << m_deviceId);
            releaseStream();
            releaseCount();
            releaseBlocks();
            return false;
        }
        if (sharedStreamSynced != nullptr) {
            *sharedStreamSynced = true;
        }
    }

    uint8_t* hostCounts = nullptr;
    {
        RecordGuard guardMallocHost{"AggrAndTopK_mallocHost"};
        if (aclrtMallocHost((void**)(&hostCounts), static_cast<size_t>(n) * countStride) != ACL_SUCCESS) {
            LOG_ERROR("hostCounts aclrtMallocHost fail,m_deviceId=" << m_deviceId);
            releaseStream();
            releaseCount();
            releaseBlocks();
            return false;
        }
    }
    aclError ret;
    {
        RecordGuard guardCounts{"AggrAndTopK_counts_d2h"};
        ret = aclrtMemcpy(hostCounts, static_cast<size_t>(n) * countStride, countDevice,
                          static_cast<size_t>(n) * countStride, ACL_MEMCPY_DEVICE_TO_HOST);
    }
    if (!countsInPlace) {
        releaseCount();
    }
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("hostCounts aclrtMemcpy fail, error code is:" << ret << ",m_deviceId=" << m_deviceId);
        releaseStream();
        releaseCount();
        CHECK_ACL_ONLY_LOG(aclrtFreeHost(hostCounts));
        releaseBlocks();
        return false;
    }
    for (size_t q = 0; q < n; ++q) {
        BatchSlot& s = slot[q];
        if (!s.active)
            continue;
        const uint32_t* counts = reinterpret_cast<const uint32_t*>(hostCounts + q * countStride);
        for (uint32_t i = 0; i < blockDimAggregator; i++) {
            const uint32_t effectiveCountBlock = counts[i * RESULT_MESSAGE_SIZE_IN_DEVICE];
            s.effectiveCount += effectiveCountBlock;
            s.docNumberShards[i] = effectiveCountBlock;
        }
        if (s.effectiveCount == 0) {
            s.active = false;  // nothing matched: out[q] stays empty, blocks freed at the end
            continue;
        }
        if (fused) {
            const uint32_t* c = counts + kTopkCountElemOffset;
            if (TopkLoopStatsEnabled())
                TopkLoopStatsAdd(c[2], s.effectiveCount);
            s.effectiveCount = c[0];
            s.effectiveCountSmaller = c[1];
            continue;
        }
        uint32_t topKJudge = s.aggTopK * FLAGS_full_recall_npu_topk_enters_threshold_ratio;
        if (topKJudge < TOPK_MIN_LIMIT)
            topKJudge = TOPK_MIN_LIMIT;
        if (s.effectiveCount <= topKJudge)
            s.merge = (blockDimAggregator > 1);  // single core: already contiguous, nothing to do
        else
            s.topk = true;
    }
    CHECK_ACL_ONLY_LOG(aclrtFreeHost(hostCounts));

    // A synchronous D2H costs 18-20us whatever it carries: calls are the cost here, not bytes.
    size_t topkCount = 0;
    for (size_t q = 0; q < n; ++q)
        if (slot[q].topk)
            topkCount++;
    if (topkCount > 0) {
        RecordGuard guard{"AggrAndTopK_TopK_NPU"};
        const uint32_t shardsStride = BLOCK_DIM_MAX * sizeof(uint32_t);
        GmBlock docNumberChunk{}, topkCountChunk{};
        if (!countsInPlace) {
            mm->AllocateBlock(GmPoolName::AGGREGATOR_SCRATCH_POOL, static_cast<uint32_t>(n) * shardsStride,
                              docNumberChunk);
        }
        mm->AllocateBlock(GmPoolName::AGGREGATOR_SCRATCH_POOL, static_cast<uint32_t>(n) * countStride, topkCountChunk);
        if ((!countsInPlace && docNumberChunk.data == nullptr) || topkCountChunk.data == nullptr) {
            LOG_ERROR("AggrAndTopKDeviceBatch topk scratch AllocateBlock fail,m_deviceId=" << m_deviceId);
            mm->FreeBlock(docNumberChunk);
            mm->FreeBlock(topkCountChunk);
            releaseCount();
            releaseStream();
            releaseBlocks();
            return false;
        }
        uint8_t* docNumberDevice = reinterpret_cast<uint8_t*>(docNumberChunk.data);
        uint8_t* topkCountDevice = reinterpret_cast<uint8_t*>(topkCountChunk.data);
        if (!countsInPlace) {
            std::vector<uint32_t> shardsHost(n * BLOCK_DIM_MAX, 0);
            for (size_t q = 0; q < n; ++q)
                if (slot[q].topk)
                    std::memcpy(&shardsHost[q * BLOCK_DIM_MAX], slot[q].docNumberShards, shardsStride);
            ret = aclrtMemcpy(docNumberDevice, static_cast<size_t>(n) * shardsStride, shardsHost.data(),
                              static_cast<size_t>(n) * shardsStride, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("docNumberDevice aclrtMemcpy fail, error code is:" << ret << ",m_deviceId=" << m_deviceId);
                mm->FreeBlock(docNumberChunk);
                mm->FreeBlock(topkCountChunk);
                releaseStream();
                releaseBlocks();
                return false;
            }
        }
        const uint32_t blockNumberShardAggregator = blockNumber / blockDimAggregator;
        // Falls back to the shared stream, and once the pool is empty it is not asked again.
        std::vector<aclrtStream> sideStreams;
        bool firstLaunch = true;
        bool poolDry = false;
        for (size_t q = 0; q < n; ++q) {
            BatchSlot& s = slot[q];
            if (!s.topk)
                continue;
            aclrtStream launchStream = stream;
            if (TopkConcurrentMode() && !firstLaunch && !poolDry) {
                aclrtStream extra = StreamManager::GetInstance()->GetStream(m_deviceId);
                if (extra != nullptr) {
                    sideStreams.push_back(extra);
                    launchStream = extra;
                } else {
                    poolDry = true;
                }
            }
            firstLaunch = false;
            uint8_t* docLocationResult = reinterpret_cast<uint8_t*>(s.docLocationResultChunk.data);
            uint8_t* resultNext = reinterpret_cast<uint8_t*>(s.resultNextChunk.data);
            uint8_t* resultLocationNext = reinterpret_cast<uint8_t*>(s.resultLocationNextChunk.data);
            const uint32_t topKEarlyQuit = s.aggTopK * FLAGS_full_recall_npu_topk_finish_buffer_ratio;
            uint8_t* blockCounts = countsInPlace ? countDevice + q * countStride : docNumberDevice + q * shardsStride;
            uint32_t blockCountsStride = countsInPlace ? RESULT_MESSAGE_SIZE_IN_DEVICE : 1u;
            ACLRT_LAUNCH_KERNEL(TOPK)
            (1, launchStream, s.docScore, docLocationResult, resultNext, resultLocationNext,
             topkCountDevice + q * countStride, blockCounts, FLAGS_full_recall_npu_topk_loop_count, blockDimAggregator,
             blockNumberShardAggregator, s.aggTopK, topKEarlyQuit, blockCountsStride);
        }
        std::unique_ptr<RecordGuard> guardTopkSync(new RecordGuard("AggrAndTopK_TopK_sync"));
        ret = aclrtSynchronizeStream(stream);
        // Every side stream is waited on and handed back: a kernel on one still reads and writes
        // blocks released below.
        for (aclrtStream side : sideStreams) {
            const aclError sideRet = aclrtSynchronizeStream(side);
            if (sideRet != ACL_SUCCESS) {
                LOG_ERROR("aclrtSynchronizeStream (TopK side stream) fail, error code is:" << sideRet << ",m_deviceId="
                                                                                           << m_deviceId);
                if (ret == ACL_SUCCESS)
                    ret = sideRet;
            }
            StreamManager::GetInstance()->FreeStream(m_deviceId, side);
        }
        guardTopkSync.reset();
        mm->FreeBlock(docNumberChunk);
        releaseCount();
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("aclrtSynchronizeStream fail, error code is:" << ret << ",m_deviceId=" << m_deviceId);
            mm->FreeBlock(topkCountChunk);
            releaseStream();
            releaseBlocks();
            return false;
        }
        uint8_t* hostTopkCounts = nullptr;
        {
            RecordGuard guardMallocHost{"AggrAndTopK_mallocHost"};
            if (aclrtMallocHost((void**)(&hostTopkCounts), static_cast<size_t>(n) * countStride) != ACL_SUCCESS) {
                LOG_ERROR("hostTopkCounts aclrtMallocHost fail,m_deviceId=" << m_deviceId);
                mm->FreeBlock(topkCountChunk);
                releaseStream();
                releaseBlocks();
                return false;
            }
        }
        {
            RecordGuard guardTopkCounts{"AggrAndTopK_topk_counts_d2h"};
            ret = aclrtMemcpy(hostTopkCounts, static_cast<size_t>(n) * countStride, topkCountDevice,
                              static_cast<size_t>(n) * countStride, ACL_MEMCPY_DEVICE_TO_HOST);
        }
        mm->FreeBlock(topkCountChunk);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("hostTopkCounts aclrtMemcpy fail, error code is:" << ret << ",m_deviceId=" << m_deviceId);
            CHECK_ACL_ONLY_LOG(aclrtFreeHost(hostTopkCounts));
            releaseStream();
            releaseBlocks();
            return false;
        }
        for (size_t q = 0; q < n; ++q) {
            if (!slot[q].topk)
                continue;
            const uint32_t* c = reinterpret_cast<const uint32_t*>(hostTopkCounts + q * countStride);
            if (TopkLoopStatsEnabled()) {
                // c[2] is the kernel's iteration count; effectiveCount still holds the TopK input size.
                TopkLoopStatsAdd(c[2], slot[q].effectiveCount);
            }
            slot[q].effectiveCount = c[0];
            slot[q].effectiveCountSmaller = c[1];
        }
        CHECK_ACL_ONLY_LOG(aclrtFreeHost(hostTopkCounts));
    }
    releaseCount();  // no TOPK query in the batch: nothing read it past phase 2
    releaseStream();

    for (size_t q = 0; q < n; ++q) {
        BatchSlot& s = slot[q];
        if (!s.merge)
            continue;
        RecordGuard guard{"AggrAndTopK_Aggregator_NPU_Memcpy"};
        const uint32_t shardAllByte = blockNumber / blockDimAggregator * blockLength * SIZEOF_FLOAT;
        uint8_t* docLocationResult = reinterpret_cast<uint8_t*>(s.docLocationResultChunk.data);
        if (!Merge(s.docScore, docLocationResult, blockDimAggregator, s.docNumberShards, shardAllByte)) {
            LOG_ERROR("Merge fail in batch, q=" << q << ",m_deviceId=" << m_deviceId);
            releaseBlocks();
            return false;
        }
    }

    for (size_t q = 0; q < n; ++q) {
        BatchSlot& s = slot[q];
        if (!s.active)
            continue;
        const uint32_t effectiveCountSum = s.effectiveCount + s.effectiveCountSmaller;
        if (effectiveCountSum == 0)
            continue;
        if (effectiveCountSum > docAllByteSize / SIZEOF_FLOAT) {
            LOG_ERROR("effectiveCountSum is spill over , effectiveCountSum="
                      << effectiveCountSum << ", docAllByteSize=" << docAllByteSize << ",m_deviceId=" << m_deviceId);
            releaseBlocks();
            return false;
        }
        uint8_t* hostTOPKResult = nullptr;
        uint8_t* hostDocLocation = nullptr;
        const uint32_t effectiveSumByteSize = effectiveCountSum * SIZEOF_FLOAT;
        {
            RecordGuard guardMallocHost{"AggrAndTopK_mallocHost"};
            aclError r = aclrtMallocHost((void**)(&hostTOPKResult), effectiveSumByteSize);
            if (r == ACL_SUCCESS)
                r = aclrtMallocHost((void**)(&hostDocLocation), effectiveSumByteSize);
            if (r != ACL_SUCCESS) {
                LOG_ERROR("aclrtMallocHost fail, error code is:" << r << ",m_deviceId=" << m_deviceId);
                if (hostTOPKResult != nullptr)
                    CHECK_ACL_ONLY_LOG(aclrtFreeHost(hostTOPKResult));
                releaseBlocks();
                return false;
            }
        }
        const bool d2hOk =
            AggregatorResultFillD2H(s.docScore, reinterpret_cast<uint8_t*>(s.docLocationResultChunk.data),
                                    reinterpret_cast<uint8_t*>(s.resultNextChunk.data),
                                    reinterpret_cast<uint8_t*>(s.resultLocationNextChunk.data), hostTOPKResult,
                                    hostDocLocation, s.effectiveCount, s.effectiveCountSmaller);
        if (!d2hOk) {
            CHECK_ACL_ONLY_LOG(aclrtFreeHost(hostTOPKResult));
            CHECK_ACL_ONLY_LOG(aclrtFreeHost(hostDocLocation));
            releaseBlocks();
            return false;
        }
        out[q].hostTOPKResult = hostTOPKResult;
        out[q].hostDocLocation = hostDocLocation;
        out[q].effectiveCount = s.effectiveCount;
        out[q].effectiveCountSmaller = s.effectiveCountSmaller;
        out[q].empty = false;
    }
    {
        RecordGuard guardFree{"AggrAndTopK_FreeAggregator"};
        releaseBlocks();
    }
    return true;
}

bool ResultAggregator::AggrAndTopK(uint8_t* filterResultInDevice, uint8_t* docScoreInDevice, size_t docAllByteSize,
                                   uint8_t*& docLocationInDeviceConst,
                                   std::shared_ptr<FullRecallResult>& fullRecallResult) {
    AggrDeviceResult dr;
    if (!AggrAndTopKDevice(filterResultInDevice, docScoreInDevice, docAllByteSize, docLocationInDeviceConst, dr))
        return false;
    return AggrAndTopKExtract(dr, fullRecallResult);
}

bool ResultAggregator::AggregatorResultFill(uint8_t* docScoreInDevice, uint8_t* docLocationInDeviceResult,
                                            uint8_t* resultNextInDevice, uint8_t* resultLocationNextInDevice,
                                            uint8_t* hostTOPKResult, uint8_t* hostDocLocation, uint32_t effectiveCount,
                                            uint32_t effectiveCountSmaller,
                                            std::shared_ptr<FullRecallResult>& fullRecallResult) {
    if (!AggregatorResultFillD2H(docScoreInDevice, docLocationInDeviceResult, resultNextInDevice,
                                 resultLocationNextInDevice, hostTOPKResult, hostDocLocation, effectiveCount,
                                 effectiveCountSmaller))
        return false;
    return AggregatorResultFillExtract(hostTOPKResult, hostDocLocation, effectiveCount, effectiveCountSmaller,
                                       fullRecallResult);
}

bool ResultAggregator::AggregatorResultFillD2H(uint8_t* docScoreInDevice, uint8_t* docLocationInDeviceResult,
                                               uint8_t* resultNextInDevice, uint8_t* resultLocationNextInDevice,
                                               uint8_t* hostTOPKResult, uint8_t* hostDocLocation,
                                               uint32_t effectiveCount, uint32_t effectiveCountSmaller) {
    uint32_t effectiveByteSize = effectiveCount * SIZEOF_FLOAT;
    {
        RecordGuard guardCopy{"AggrAndTopK_fill_d2h_copy"};
        std::unique_lock<std::mutex> d2hLock(g_d2hCopyMutex, std::defer_lock);
        {
            RecordGuard guardLock{"AggrAndTopK_fill_d2h_lock"};
            d2hLock.lock();
        }
        if (effectiveCount > 0) {
            auto ret = aclrtMemcpy(hostTOPKResult, effectiveByteSize, docScoreInDevice, effectiveByteSize,
                                   ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("hostTOPKResult aclrtMemcpy fail, error code is:" << ret << ",m_deviceId=" << m_deviceId);
                return false;
            }
            ret = aclrtMemcpy(hostDocLocation, effectiveByteSize, docLocationInDeviceResult, effectiveByteSize,
                              ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("hostDocLocation aclrtMemcpy fail, error code is:" << ret << ",m_deviceId=" << m_deviceId);
                return false;
            }
        }
        if (effectiveCountSmaller > 0) {
            uint32_t effectiveCountSmallerByteSize = effectiveCountSmaller * SIZEOF_FLOAT;
            auto ret = aclrtMemcpy((float*)hostTOPKResult + effectiveCount, effectiveCountSmallerByteSize,
                                   resultNextInDevice, effectiveCountSmallerByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("hostTOPKResult aclrtMemcpy fail, error code is:" << ret << ",m_deviceId=" << m_deviceId);
                return false;
            }
            ret = aclrtMemcpy((uint32_t*)hostDocLocation + effectiveCount, effectiveCountSmallerByteSize,
                              resultLocationNextInDevice, effectiveCountSmallerByteSize, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("hostDocLocation aclrtMemcpy fail, error code is:" << ret << ",m_deviceId=" << m_deviceId);
                return false;
            }
        }
    }
    return true;
}

bool ResultAggregator::AggregatorResultFillExtract(uint8_t* hostTOPKResult, uint8_t* hostDocLocation,
                                                   uint32_t effectiveCount, uint32_t effectiveCountSmaller,
                                                   std::shared_ptr<FullRecallResult>& fullRecallResult) {
    uint32_t effectiveCountSum = effectiveCount + effectiveCountSmaller;
    std::vector<ScoreWithIndex>& results = fullRecallResult->scoreWithIndexes;
    const float* scorePtr = reinterpret_cast<const float*>(hostTOPKResult);
    const uint32_t* locationPtr = reinterpret_cast<const uint32_t*>(hostDocLocation);
    uint32_t count = std::min(effectiveCountSum, m_topK);
    // Diagnostic for the invalid-ldocid corruption: a tail index means the D2H copied more than
    // the kernel wrote, an interior one means the kernel paired a score with a garbage location.
    auto reportBadLoc = [&](uint32_t idx, uint32_t loc) {
        float locF;
        std::memcpy(&locF, &loc, sizeof(locF));
        std::printf(
            "[Ldocid] idx=%u eff=%u smaller=%u sum=%u region=%s tailDist=%ld loc=%u locAsFloat=%.4f "
            "score=%.4f device=%d\n",
            idx, effectiveCount, effectiveCountSmaller, effectiveCountSum, idx < effectiveCount ? "topk" : "smaller",
            idx < effectiveCount ? static_cast<long>(effectiveCount) - 1 - idx
                                 : static_cast<long>(effectiveCountSum) - 1 - idx,
            loc, locF, scorePtr[idx], m_deviceId);
        std::fflush(stdout);
    };
    if (PackedSortMode() || RadixSortMode()) {
        const bool fast = ExtractFastMode();
        const bool halfRadix = fast && RadixSortMode();
        std::vector<uint64_t> keys;
        {
            RecordGuard guardBuild{"AggrAndTopK_fill_build"};
            keys.reserve(effectiveCountSum);
            if (halfRadix) {
                // Descending index order, so the 4-pass sort breaks score ties index-descending.
                for (uint32_t i = effectiveCountSum; i-- > 0;) {
                    keys.push_back((static_cast<uint64_t>(ScoreToSortable(scorePtr[i])) << 32) | i);
                }
            } else {
                for (uint32_t i = 0; i < effectiveCountSum; i++) {
                    keys.push_back((static_cast<uint64_t>(ScoreToSortable(scorePtr[i])) << 32) | i);
                }
            }
        }
        {
            // Ties break by candidate index: recall-neutral, since the merge is set-based.
            RecordGuard guard2{"AggrAndTopK_std_sort_sort"};
            if (RadixSortMode()) {
                RadixSortDescU64(keys, halfRadix ? 32 : 0);
            } else {
                auto keyGreater = [](uint64_t a, uint64_t b) { return a > b; };
                if (effectiveCountSum > count) {
                    std::nth_element(keys.begin(), keys.begin() + count, keys.end(), keyGreater);
                }
                std::sort(keys.begin(), keys.begin() + count, keyGreater);
            }
        }
        {
            RecordGuard guard{"AggrAndTopK_create_result"};
            if (fast) {
                constexpr uint32_t kAhead = 32;
                const uint64_t* gIds = m_docIdMapping->GDocIdData();
                const size_t gIdCount = m_docIdMapping->GDocIdCount();
                std::vector<uint32_t> locs(count);
                {
                    RecordGuard guardLocs{"AggrAndTopK_create_result_locs"};
                    for (uint32_t i = 0; i < count; i++) {
                        locs[i] = locationPtr[static_cast<uint32_t>(keys[i] & 0xFFFFFFFFu)];
                    }
                }
                std::vector<uint64_t> gids(count, 0);
                {
                    RecordGuard guardGather{"AggrAndTopK_create_result_gather"};
                    for (uint32_t i = 0; i < count; i++) {
                        if (i + kAhead < count && locs[i + kAhead] < gIdCount) {
                            __builtin_prefetch(gIds + locs[i + kAhead]);
                        }
                        if (locs[i] < gIdCount) {  // GetGDocId's own bound
                            gids[i] = gIds[locs[i]];
                        }
                    }
                }
                RecordGuard guardBuild{"AggrAndTopK_create_result_build"};
                auto bad = [&](uint32_t i) {
                    uint64_t unused = 0;
                    (void)m_docIdMapping->GetGDocId(locs[i], unused);  // its "invalid ldocid" error, as before
                    reportBadLoc(static_cast<uint32_t>(keys[i] & 0xFFFFFFFFu), locs[i]);
                    LOG_WARN("skip out-of-range result index:" << locs[i]);
                };
                if (fullRecallResult->isMultiShard) {
                    results.reserve(count);
                    for (uint32_t i = 0; i < count; i++) {
                        if (locs[i] >= gIdCount) {
                            bad(i);
                            continue;
                        }
                        ScoreWithIndex entry{scorePtr[static_cast<uint32_t>(keys[i] & 0xFFFFFFFFu)], locs[i]};
                        entry.id = gids[i];
                        results.push_back(entry);
                    }
                } else {
                    fullRecallResult->docIds.reserve(count);
                    fullRecallResult->scores.reserve(count);
                    for (uint32_t i = 0; i < count; i++) {
                        if (locs[i] >= gIdCount) {
                            bad(i);
                            continue;
                        }
                        fullRecallResult->docIds.emplace_back(gids[i]);
                        fullRecallResult->scores.emplace_back(scorePtr[static_cast<uint32_t>(keys[i] & 0xFFFFFFFFu)]);
                    }
                }
            } else if (fullRecallResult->isMultiShard) {
                results.reserve(count);
                for (uint32_t i = 0; i < count; i++) {
                    uint32_t idx = static_cast<uint32_t>(keys[i] & 0xFFFFFFFFu);
                    uint32_t location = locationPtr[idx];
                    uint64_t gDocid;
                    if (!m_docIdMapping->GetGDocId(location, gDocid)) {
                        reportBadLoc(idx, location);
                        LOG_WARN("skip out-of-range result index:" << location);
                        continue;
                    }
                    ScoreWithIndex entry{scorePtr[idx], location};
                    entry.id = gDocid;
                    results.push_back(entry);
                }
            } else {
                fullRecallResult->docIds.reserve(count);
                fullRecallResult->scores.reserve(count);
                for (uint32_t i = 0; i < count; i++) {
                    uint32_t idx = static_cast<uint32_t>(keys[i] & 0xFFFFFFFFu);
                    uint32_t location = locationPtr[idx];
                    uint64_t gDocid;
                    if (!m_docIdMapping->GetGDocId(location, gDocid)) {
                        reportBadLoc(idx, location);
                        LOG_WARN("skip out-of-range result index:" << location);
                        continue;
                    }
                    fullRecallResult->docIds.emplace_back(gDocid);
                    fullRecallResult->scores.emplace_back(scorePtr[idx]);
                }
            }
        }
    } else {
        {
            RecordGuard guardBuild{"AggrAndTopK_fill_build"};
            results.reserve(effectiveCountSum);
            for (uint32_t i = 0; i < effectiveCountSum; i++) {
                results.emplace_back(scorePtr[i], locationPtr[i]);
            }
        }
        {
            RecordGuard guard2{"AggrAndTopK_std_sort_sort"};
            if (effectiveCountSum > count) {
                std::nth_element(results.begin(), results.begin() + count, results.end(), compareDesc);
            }
            std::sort(results.begin(), results.begin() + count, compareDesc);
        }
        {
            RecordGuard guard{"AggrAndTopK_create_result"};
            if (fullRecallResult->isMultiShard) {
                for (uint32_t i = 0; i < count; i++) {
                    uint64_t gDocid;
                    if (!m_docIdMapping->GetGDocId(results[i].index, gDocid)) {
                        // Safety net: a stray out-of-range index must not abort the whole query.
                        LOG_WARN("skip out-of-range result index:" << results[i].index);
                        continue;
                    }
                    results[i].id = gDocid;
                }
                // erase, not resize: ScoreWithIndex has no default constructor
                results.erase(results.begin() + count, results.end());
            } else {
                fullRecallResult->docIds.reserve(count);
                fullRecallResult->scores.reserve(count);
                for (uint32_t i = 0; i < count; i++) {
                    uint64_t gDocid;
                    if (!m_docIdMapping->GetGDocId(results[i].index, gDocid)) {
                        LOG_WARN("skip out-of-range result index:" << results[i].index);
                        continue;
                    }
                    fullRecallResult->docIds.emplace_back(gDocid);
                    fullRecallResult->scores.emplace_back(results[i].score);
                }
                results.clear();
            }
        }
    }
    return true;
}

void ResultAggregator::FreeAggregator(uint8_t* hostTOPKResult, uint8_t* hostDocLocation,
                                      GmBlock& docLocationInDeviceResultChunk, GmBlock& resultNextInDeviceChunk,
                                      GmBlock& resultLocationNextInDeviceChunk) {
    if (hostTOPKResult != nullptr) {
        CHECK_ACL_ONLY_LOG(aclrtFreeHost(hostTOPKResult));
    }
    if (hostDocLocation != nullptr) {
        CHECK_ACL_ONLY_LOG(aclrtFreeHost(hostDocLocation));
    }
    if (docLocationInDeviceResultChunk.data != nullptr) {
        GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(docLocationInDeviceResultChunk);
    }
    if (resultNextInDeviceChunk.data != nullptr) {
        GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(resultNextInDeviceChunk);
    }
    if (resultLocationNextInDeviceChunk.data != nullptr) {
        GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(resultLocationNextInDeviceChunk);
    }
}

bool ResultAggregator::Aggregator(uint8_t* filterResultInDevice, uint8_t*& docScoreInDevice, size_t docAllByteSize,
                                  uint8_t*& docLocationInDeviceConst, uint8_t*& docLocationInDeviceResult,
                                  uint8_t*& resultNextInDevice, uint8_t*& resultLocationNextInDevice, uint32_t topK,
                                  uint32_t& effectiveCount, uint32_t& topKEffectiveCountSmaller) {
    uint32_t blockDimAggregator = FLAGS_full_recall_aggregator_block_dim;
    if (blockDimAggregator > BLOCK_DIM_MAX) {
        blockDimAggregator = BLOCK_DIM_MAX;
        LOG_WARN("blockDim set to BLOCK_DIM_MAX. config value is " << FLAGS_full_recall_aggregator_block_dim);
    }
    if (blockDimAggregator < 1) {
        blockDimAggregator = 1;
        LOG_WARN("blockDim set to 1. config value is " << FLAGS_full_recall_aggregator_block_dim);
    }
    uint32_t docNumber = docAllByteSize / sizeof(float);
    uint32_t blockLength = MAX_BLOCK_SIZE;
    uint32_t blockNumber = (docNumber + blockLength - 1) / blockLength;
    uint32_t lastBlockLength = docNumber - blockLength * (blockNumber - 1);
    LOG_DEBUG("docNumber:" << docNumber << " blockNumber:" << blockNumber << " blockLength:" << blockLength
                           << " lastBlockLength:" << lastBlockLength << " topK:" << topK);
    if (blockNumber < 20) {  // fewer than 20 blocks (small data): no need to parallelize
        LOG_INFO("blockNumber:" << blockNumber << " is small, blockDimAggregator reset 1");
        blockDimAggregator = 1;
    }
    // Pooled scratch: freeing a HUGE_FIRST device block syncs the stream and occasionally stalls ms.
    aclError ret = ACL_SUCCESS;
    GmBlock effectiveCountChunk{};
    GmMemoryManager::GetByDeviceId(m_deviceId)
        ->AllocateBlock(GmPoolName::AGGREGATOR_SCRATCH_POOL, RESULT_MESSAGE_BYTE_SIZE_IN_DEVICE, effectiveCountChunk);
    if (effectiveCountChunk.data == nullptr) {
        LOG_ERROR("effectiveCountDevice AllocateBlock fail" << ",m_deviceId=" << m_deviceId);
        return false;
    }
    uint8_t* effectiveCountDevice = reinterpret_cast<uint8_t*>(effectiveCountChunk.data);
    {
        aclrtStream stream = StreamManager::GetInstance()->GetStream(m_deviceId);
        RecordGuard guard{"AggrAndTopK_Aggregator_NPU"};
        ACLRT_LAUNCH_KERNEL(Aggregator)
        (blockDimAggregator, stream, filterResultInDevice, docScoreInDevice, docLocationInDeviceConst,
         docLocationInDeviceResult, effectiveCountDevice, docNumber, blockNumber, blockLength, lastBlockLength);
        ret = aclrtSynchronizeStream(stream);
        StreamManager::GetInstance()->FreeStream(m_deviceId, stream);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("aclrtSynchronizeStream fail, error code is:" << ret << ",m_deviceId=" << m_deviceId);
            GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(effectiveCountChunk);
            return false;
        }
    }

    uint8_t* hostEffectiveCount = nullptr;
    ret = aclrtMallocHost((void**)(&hostEffectiveCount), RESULT_MESSAGE_BYTE_SIZE_IN_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("hostEffectiveCount aclrtMalloc fail, error code is:" << ret << ",m_deviceId=" << m_deviceId);
        GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(effectiveCountChunk);
        return false;
    }
    ret = aclrtMemcpy(hostEffectiveCount, RESULT_MESSAGE_BYTE_SIZE_IN_DEVICE, effectiveCountDevice,
                      RESULT_MESSAGE_BYTE_SIZE_IN_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("hostEffectiveCount aclrtMemcpy fail, error code is:" << ret << ",m_deviceId=" << m_deviceId);
        GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(effectiveCountChunk);
        CHECK_ACL_ONLY_LOG(aclrtFreeHost(hostEffectiveCount));
        return false;
    }
    uint32_t docNumberShards[BLOCK_DIM_MAX] = {0};
    for (uint32_t i = 0; i < blockDimAggregator; i++) {
        uint32_t effectiveCountBlock = *((uint32_t*)hostEffectiveCount + i * RESULT_MESSAGE_SIZE_IN_DEVICE);
        effectiveCount += effectiveCountBlock;
        docNumberShards[i] = effectiveCountBlock;
        LOG_DEBUG("deviceId:" << m_deviceId << ",i:" << i << ",effectiveCount:" << effectiveCount
                              << ";effectiveCountBlock:" << effectiveCountBlock << ",m_deviceId=" << m_deviceId);
    }
    GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(effectiveCountChunk);
    CHECK_ACL_ONLY_LOG(aclrtFreeHost(hostEffectiveCount));
    if (effectiveCount == 0) {
        return true;
    }
    uint32_t topKJudge = topK * FLAGS_full_recall_npu_topk_enters_threshold_ratio;
    if (topKJudge < TOPK_MIN_LIMIT) {
        topKJudge = TOPK_MIN_LIMIT;
        LOG_DEBUG("topKJudge:" << topKJudge << " is small.");
    }
    if (effectiveCount <= topKJudge) {
        if (blockDimAggregator > 1) {
            RecordGuard guard{"AggrAndTopK_Aggregator_NPU_Memcpy"};
            uint32_t shardAllByte = blockNumber / blockDimAggregator * blockLength * SIZEOF_FLOAT;
            return Merge(docScoreInDevice, docLocationInDeviceResult, blockDimAggregator, docNumberShards,
                         shardAllByte);
        }
    } else {
        return TopK(docScoreInDevice, docLocationInDeviceResult, resultNextInDevice, resultLocationNextInDevice,
                    blockNumber, blockDimAggregator, docNumberShards, topK, effectiveCount, topKEffectiveCountSmaller);
    }
    return true;
}

bool ResultAggregator::Merge(uint8_t*& docScoreInDevice, uint8_t*& docLocationInDeviceResult,
                             uint32_t blockDimAggregator, uint32_t* docNumberShards, uint32_t shardAllByte) {
    uint32_t effectiveCountCopy = 0;
    for (uint32_t i = 0; i < blockDimAggregator; i++) {
        uint32_t effectiveCountBlock = docNumberShards[i];
        uint32_t effectiveCountBlockCopy = effectiveCountBlock * SIZEOF_FLOAT;
        if (i > 0 && effectiveCountBlock > 0) {
            LOG_DEBUG("effectiveCountCopy:" << effectiveCountCopy << ", i:" << i << ", shardAllByte:" << shardAllByte
                                            << ", effectiveCountBlockCopy:" << effectiveCountBlockCopy);
            auto ret =
                aclrtMemcpy(docScoreInDevice + effectiveCountCopy * SIZEOF_FLOAT, effectiveCountBlockCopy,
                            docScoreInDevice + i * shardAllByte, effectiveCountBlockCopy, ACL_MEMCPY_DEVICE_TO_DEVICE);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("docScoreInDevice aclrtMemcpy fail, error code is:" << ret << ", i:" << i
                                                                              << ",m_deviceId=" << m_deviceId);
                return false;
            }
            ret = aclrtMemcpy(docLocationInDeviceResult + effectiveCountCopy * SIZEOF_FLOAT, effectiveCountBlockCopy,
                              docLocationInDeviceResult + i * shardAllByte, effectiveCountBlockCopy,
                              ACL_MEMCPY_DEVICE_TO_DEVICE);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("docLocationInDeviceResult aclrtMemcpy fail, error code is:" << ret << ", i:" << i
                                                                                       << ",m_deviceId=" << m_deviceId);
                return false;
            }
        }
        effectiveCountCopy += effectiveCountBlock;
    }
    return true;
}

bool ResultAggregator::TopK(uint8_t*& docScoreInDevice, uint8_t*& docLocationInDeviceResult,
                            uint8_t*& resultNextInDevice, uint8_t*& resultLocationNextInDevice,
                            uint32_t blockNumberAggregator, uint32_t blockDimAggregator, uint32_t* docNumberShards,
                            uint32_t topK, uint32_t& effectiveCount, uint32_t& topKEffectiveCountSmaller) {
    RecordGuard guard{"AggrAndTopK_TopK_NPU"};
    if (blockDimAggregator == 0) {
        LOG_ERROR("get blockDimAggregator value is zero" << ",m_deviceId=" << m_deviceId);
        return false;
    }
    aclError ret = ACL_SUCCESS;
    GmBlock docNumberChunk{};
    GmMemoryManager::GetByDeviceId(m_deviceId)
        ->AllocateBlock(GmPoolName::AGGREGATOR_SCRATCH_POOL, BLOCK_DIM_MAX * sizeof(uint32_t), docNumberChunk);
    if (docNumberChunk.data == nullptr) {
        LOG_ERROR("docNumberInDevice AllocateBlock fail" << ",m_deviceId=" << m_deviceId);
        return false;
    }
    uint8_t* docNumberInDevice = reinterpret_cast<uint8_t*>(docNumberChunk.data);
    ret = aclrtMemcpy(docNumberInDevice, BLOCK_DIM_MAX * sizeof(uint32_t), docNumberShards,
                      BLOCK_DIM_MAX * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("docNumberInDevice aclrtMemcpy fail, error code is:" << ret << ",m_deviceId=" << m_deviceId);
        GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(docNumberChunk);
        return false;
    }
    GmBlock topkCountChunk{};
    uint32_t blockNumberShardAggregator = blockNumberAggregator / blockDimAggregator;
    GmMemoryManager::GetByDeviceId(m_deviceId)
        ->AllocateBlock(GmPoolName::AGGREGATOR_SCRATCH_POOL, RESULT_MESSAGE_BYTE_SIZE_IN_DEVICE, topkCountChunk);
    if (topkCountChunk.data == nullptr) {
        LOG_ERROR("TOPKResultCountInDevice AllocateBlock fail" << ",m_deviceId=" << m_deviceId);
        GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(docNumberChunk);
        return false;
    }
    uint8_t* TOPKResultCountInDevice = reinterpret_cast<uint8_t*>(topkCountChunk.data);
    aclrtStream streamTopK = StreamManager::GetInstance()->GetStream(m_deviceId);
    ACLRT_LAUNCH_KERNEL(TOPK)(1, streamTopK, docScoreInDevice, docLocationInDeviceResult, resultNextInDevice,
                              resultLocationNextInDevice, TOPKResultCountInDevice, docNumberInDevice,
                              FLAGS_full_recall_npu_topk_loop_count, blockDimAggregator, blockNumberShardAggregator,
                              topK, topK * FLAGS_full_recall_npu_topk_finish_buffer_ratio,
                              /*docNumberStride=*/1u);
    ret = aclrtSynchronizeStream(streamTopK);
    StreamManager::GetInstance()->FreeStream(m_deviceId, streamTopK);
    GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(docNumberChunk);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtSynchronizeStream fail, error code is:" << ret << ",m_deviceId=" << m_deviceId);
        GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(topkCountChunk);
        return false;
    }
    uint8_t* TOPKEffectiveCountHost = nullptr;
    ret = aclrtMallocHost((void**)(&TOPKEffectiveCountHost), RESULT_MESSAGE_BYTE_SIZE_IN_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("TOPKEffectiveCountHost fail, error code is:" << ret << ",m_deviceId=" << m_deviceId);
        GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(topkCountChunk);
        return false;
    }
    ret = aclrtMemcpy(TOPKEffectiveCountHost, RESULT_MESSAGE_BYTE_SIZE_IN_DEVICE, TOPKResultCountInDevice,
                      RESULT_MESSAGE_BYTE_SIZE_IN_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("TOPKEffectiveCountHost fail, error code is:" << ret << ",m_deviceId=" << m_deviceId);
        GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(topkCountChunk);
        CHECK_ACL_ONLY_LOG(aclrtFreeHost(TOPKEffectiveCountHost));
        return false;
    }
    effectiveCount = *((uint32_t*)TOPKEffectiveCountHost);
    topKEffectiveCountSmaller = *((uint32_t*)TOPKEffectiveCountHost + 1);
    uint32_t lookCountReal = *((uint32_t*)TOPKEffectiveCountHost + 2);  // value at index 2
    LOG_DEBUG("NPU TOPK effectiveCount=" << effectiveCount << ",effectiveCountSmaller=" << topKEffectiveCountSmaller
                                         << ",lookCountReal=" << lookCountReal << ",m_deviceId=" << m_deviceId);
    GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(topkCountChunk);
    CHECK_ACL_ONLY_LOG(aclrtFreeHost(TOPKEffectiveCountHost));
    return true;
}
}  // namespace NpuRetrieval
