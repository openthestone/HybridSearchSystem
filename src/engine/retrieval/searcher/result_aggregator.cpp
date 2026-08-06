#include "result_aggregator.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
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

namespace NpuRetrieval {
const uint32_t SIZEOF_FLOAT = sizeof(float);
const uint32_t TOPK_MIN_LIMIT = 1024;
const uint32_t RESULT_MESSAGE_SIZE_IN_DEVICE = 16;
const uint32_t RESULT_MESSAGE_BYTE_SIZE_IN_DEVICE = RESULT_MESSAGE_SIZE_IN_DEVICE * SIZEOF_FLOAT * 20;
const uint32_t MAX_BLOCK_SIZE = 5888;  // 256*23; must be a multiple of 256 and keep the aggregator kernel within UB
const uint32_t BLOCK_DIM_MAX = 16;

// comparator for descending score order (default struct-sort path)
bool compareDesc(const ScoreWithIndex& a, const ScoreWithIndex& b) {
    return a.score > b.score;
}

// Map a float to a uint32 whose unsigned order matches the float's numeric order (higher
// float -> higher uint32), so (score, index) can be packed into one uint64 and sorted as a
// plain integer. Standard IEEE-754 monotonic transform; correct for negative scores too.
static inline uint32_t ScoreToSortable(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    return bits ^ ((0u - (bits >> 31)) | 0x80000000u);
}

// Read an on/off toggle env var once.
static bool EnvOn(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] == '1';
}

// NPUR_SHARD_TOPK_RATIO in (0,1], default 1: each shard aggregates only its top
// ceil(topK*ratio) instead of the full topK. With N shards the global top-K splits
// ~evenly across shards (~topK/N each), so a shard rarely needs its full topK; the host
// merge still assembles the full top-K from the union. Trades a little recall for a
// smaller per-shard NPU TopK + host sort. Read once; out-of-range values fall back to 1.
static double ShardTopKRatio() {
    static const double ratio = [] {
        const char* v = std::getenv("NPUR_SHARD_TOPK_RATIO");
        double r = (v != nullptr) ? std::atof(v) : 1.0;
        return (r > 0.0 && r <= 1.0) ? r : 1.0;
    }();
    return ratio;
}

// Opt-in (NPUR_PACKED_SORT=1): in the per-shard host top-K sort, sort 8-byte packed
// (score,index) keys instead of the 16-byte ScoreWithIndex. Default off keeps the struct
// sort, so main's default behavior is unchanged. Read once.
static bool PackedSortMode() {
    static const bool enabled = EnvOn("NPUR_PACKED_SORT");
    return enabled;
}

// Opt-in (NPUR_RADIX_SORT=1): sort the packed keys with an LSD radix sort instead of
// std::sort. Implies the packed path.
static bool RadixSortMode() {
    static const bool enabled = EnvOn("NPUR_RADIX_SORT");
    return enabled;
}

// LSD radix sort of the packed keys, descending by the full 64-bit value (8 passes of
// 8-bit digits). Keys are unique (low 32 bits = distinct candidate index), so the order is
// fully determined and byte-identical to std::sort with a greater<> comparator.
static void RadixSortDescU64(std::vector<uint64_t>& keys) {
    const size_t n = keys.size();
    if (n < 2) {
        return;
    }
    std::vector<uint64_t> tmp(n);
    uint64_t* src = keys.data();
    uint64_t* dst = tmp.data();
    for (int shift = 0; shift < 64; shift += 8) {
        size_t cnt[256] = {0};
        for (size_t i = 0; i < n; i++) {
            cnt[(src[i] >> shift) & 0xFFu]++;
        }
        // Descending: place the highest byte value at the front.
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

// Serialize the AggregatorResultFill D2H copies across shard threads. The copies
// are the multi-card p99 tail: two shards run in lockstep, so their synchronous
// aclrtMemcpy D2H calls collide and one occasionally stalls for ms on shared
// ACL-runtime state (single-card, no concurrency, shows no such stall). The
// expensive compute kernels do NOT contend (their tails are flat), so we keep
// those concurrent and only fence this ~33us copy region: at most one shard copies
// at a time, which trades a tiny wait for the other shard against eliminating the
// pathological collision. Process-global (both shards share this one process).
static std::mutex g_d2hCopyMutex;

void ResultAggregator::SetTopK(uint32_t topK) {
    // Per-shard aggregation top-K = ceil(topK * ratio) (ratio<1 only). The final merge
    // still assembles the full top-K from all shards, so the reduced per-shard set only
    // needs to cover this shard's slice of the global top-K.
    const double ratio = ShardTopKRatio();
    m_topK = (ratio < 1.0) ? std::max<uint32_t>(1, static_cast<uint32_t>(std::ceil(topK * ratio))) : topK;
}

bool ResultAggregator::AggrAndTopK(uint8_t* filterResultInDevice, uint8_t* docScoreInDevice, size_t docAllByteSize,
                                   uint8_t*& docLocationInDeviceConst,
                                   std::shared_ptr<FullRecallResult>& fullRecallResult) {
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
    bool state = AggregatorResultFill(docScoreInDevice, docLocationInDeviceResult, resultNextInDevice,
                                      resultLocationNextInDevice, hostTOPKResult, hostDocLocation, effectiveCount,
                                      effectiveCountSmaller, fullRecallResult);
    {
        RecordGuard guardFree{"AggrAndTopK_FreeAggregator"};
        FreeAggregator(hostTOPKResult, hostDocLocation, docLocationInDeviceResultChunk, resultNextInDeviceChunk,
                       resultLocationNextInDeviceChunk);
    }
    return state;
}

bool ResultAggregator::AggregatorResultFill(uint8_t* docScoreInDevice, uint8_t* docLocationInDeviceResult,
                                            uint8_t* resultNextInDevice, uint8_t* resultLocationNextInDevice,
                                            uint8_t* hostTOPKResult, uint8_t* hostDocLocation, uint32_t effectiveCount,
                                            uint32_t effectiveCountSmaller,
                                            std::shared_ptr<FullRecallResult>& fullRecallResult) {
    uint32_t effectiveCountSum = effectiveCount + effectiveCountSmaller;
    uint32_t effectiveByteSize = effectiveCount * SIZEOF_FLOAT;
    std::vector<ScoreWithIndex>& results = fullRecallResult->scoreWithIndexes;
    const float* scorePtr = reinterpret_cast<const float*>(hostTOPKResult);
    const uint32_t* locationPtr = reinterpret_cast<const uint32_t*>(hostDocLocation);
    {
        // The 4 synchronous D2H copies, fenced across shard threads (see g_d2hCopyMutex).
        RecordGuard guardCopy{"AggrAndTopK_fill_d2h_copy"};
        std::lock_guard<std::mutex> d2hLock(g_d2hCopyMutex);
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
    uint32_t count = std::min(effectiveCountSum, m_topK);
    if (PackedSortMode() || RadixSortMode()) {
        // Tier 1: sort 8-byte packed keys ([sortable(score):32 | candidateIdx:32]) instead of
        // the 16-byte ScoreWithIndex, halving what the sort moves. The score/location payload is
        // re-read bit-exact from the host buffers afterward; the packed high bits only order it.
        std::vector<uint64_t> keys;
        {
            RecordGuard guardBuild{"AggrAndTopK_fill_build"};
            keys.reserve(effectiveCountSum);
            for (uint32_t i = 0; i < effectiveCountSum; i++) {
                keys.push_back((static_cast<uint64_t>(ScoreToSortable(scorePtr[i])) << 32) | i);
            }
        }
        {
            // Descending: highest scores first; ties break by candidate index (deterministic,
            // recall-neutral since the cross-shard merge is set-based).
            RecordGuard guard2{"AggrAndTopK_std_sort_sort"};
            if (RadixSortMode()) {
                // Radix sorts all n; the top `count` are then keys[0..count).
                RadixSortDescU64(keys);
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
            // Low 32 bits index back into the bit-exact host buffers; resolve the global id.
            if (fullRecallResult->isMultiShard) {
                results.reserve(count);
                for (uint32_t i = 0; i < count; i++) {
                    uint32_t idx = static_cast<uint32_t>(keys[i] & 0xFFFFFFFFu);
                    uint32_t location = locationPtr[idx];
                    uint64_t gDocid;
                    if (!m_docIdMapping->GetGDocId(location, gDocid)) {
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
                        LOG_WARN("skip out-of-range result index:" << location);
                        continue;
                    }
                    fullRecallResult->docIds.emplace_back(gDocid);
                    fullRecallResult->scores.emplace_back(scorePtr[idx]);
                }
            }
        }
    } else {
        // Default path: build and sort the 16-byte ScoreWithIndex in place.
        {
            RecordGuard guardBuild{"AggrAndTopK_fill_build"};
            results.reserve(effectiveCountSum);
            for (uint32_t i = 0; i < effectiveCountSum; i++) {
                results.emplace_back(scorePtr[i], locationPtr[i]);
            }
        }
        {
            // Only the top-`count` are ever returned, so select them (linear nth_element)
            // and sort just those in place, instead of full-sorting every candidate.
            RecordGuard guard2{"AggrAndTopK_std_sort_sort"};
            if (effectiveCountSum > count) {
                std::nth_element(results.begin(), results.begin() + count, results.end(), compareDesc);
            }
            std::sort(results.begin(), results.begin() + count, compareDesc);
        }
        {
            RecordGuard guard{"AggrAndTopK_create_result"};
            // convert indices to global doc ids for the kept top-K only
            if (fullRecallResult->isMultiShard) {
                for (uint32_t i = 0; i < count; i++) {
                    uint64_t gDocid;
                    if (!m_docIdMapping->GetGDocId(results[i].index, gDocid)) {
                        // Safety net: keep a stray out-of-range index from aborting the whole
                        // query. The device fix (kernel_topk.h) removes the root cause.
                        LOG_WARN("skip out-of-range result index:" << results[i].index);
                        continue;
                    }
                    results[i].id = gDocid;
                }
                // drop the unsorted tail beyond the top-K (erase, not resize:
                // ScoreWithIndex has no default constructor)
                results.erase(results.begin() + count, results.end());
            } else {
                fullRecallResult->docIds.reserve(count);
                fullRecallResult->scores.reserve(count);
                for (uint32_t i = 0; i < count; i++) {
                    uint64_t gDocid;
                    if (!m_docIdMapping->GetGDocId(results[i].index, gDocid)) {
                        // Safety net (see above): skip a stray index rather than abort.
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
    // Pooled scratch instead of a per-query raw aclrtMalloc/aclrtFree: freeing a HUGE_FIRST
    // device block syncs the stream and occasionally stalls ms (a p99 tail spike). The pool
    // reuses a pre-allocated fixed block; FreeBlock returns it to the free list (no driver free).
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
    // gather the per-shard counts when running multi-core
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
            // merge the per-shard results into a contiguous region when running multi-core
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
                              topK, topK * FLAGS_full_recall_npu_topk_finish_buffer_ratio);
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
