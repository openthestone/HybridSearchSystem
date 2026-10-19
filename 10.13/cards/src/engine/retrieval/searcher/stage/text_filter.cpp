#include "text_filter.h"
#include "filter_stats.h"
#include <cstring>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include "acl/acl.h"
#include "src/utils/env_switch.h"
#include "ascend_device/aclrtlaunch_BitmapTextFilter.h"
#include "ascend_device/aclrtlaunch_PostingBitListToSet.h"
#include "src/utils/logger.h"
#include "src/utils/performance_recorder.h"
#include "src/full_recall/core/constant_definition.h"
#include "src/full_recall/core/log_definition.h"
#include "src/full_recall/retrieval/searcher/runtime/stream_manager.h"

namespace NpuRetrieval {
static const bool PRINT_RES_ENABLE = false;

// NPUR_POOL_POSTINGS=0: raw per-query aclrtMalloc/aclrtFree instead of the pool.
static bool PoolPostingsEnabled() {
    static const bool enabled = npur_env::OnByDefaultExact("NPUR_POOL_POSTINGS");
    return enabled;
}

// NPUR_OVERLAP_POSTING_BITLIST=1: run the bitlist->bitset conversion inside the scorer's window.
static bool BitlistOverlapEnabled() {
    static const bool on = npur_env::On("NPUR_OVERLAP_POSTING_BITLIST");
    return on;
}

// NPUR_SPARSE_DIRECT=1: hand an OR node its sparse postings in their encoded form instead of
// converting them to 16KB bitsets. Filter switches: see FilterFlag in device_common_external.h.
static uint32_t FilterFlags() {
    static const uint32_t flags = []() -> uint32_t {
        uint32_t f = 0;
        if (npur_env::On("NPUR_OR_NO_TILE_CHECK")) {
            f |= FILTER_FLAG_OR_NO_TILE_CHECK;
        }
        if (npur_env::On("NPUR_OR_SKIP_EMPTY")) {
            f |= FILTER_FLAG_OR_SKIP_EMPTY;
        }
        if (npur_env::On("NPUR_OR_ALIGNED_COPY")) {
            f |= FILTER_FLAG_OR_ALIGNED_COPY;
        }
        // Not a bit: 1..3 pick which step of ApplySparse to leave out, in bits 8..10.
        const char* ablate = std::getenv("NPUR_OR_ABLATE");
        if (ablate != nullptr) {
            const long level = strtol(ablate, nullptr, 10);
            if (level > 0 && level <= OR_ABLATE_FLOOR) {
                f |= (static_cast<uint32_t>(level) << FILTER_ABLATE_SHIFT) & FILTER_ABLATE_MASK;
            }
        }
        return f;
    }();
    return flags;
}

static uint32_t EffectiveFilterFlags() {
    uint32_t f = FilterFlags();
    if (SparsePostingsArePacked()) {
        f |= FILTER_FLAG_SPARSE_PACKED;
    }
    return f;
}

// NPUR_ONE_COUNT_PASS=1: answer both counts in one walk over the posting types instead of two.
static bool OneCountPassEnabled() {
    static const bool on = npur_env::On("NPUR_ONE_COUNT_PASS");
    return on;
}

// NPUR_CLASSIFY_FAST=1: an OR operand's classify loop writes the table directly, bypassing emit.
static bool ClassifyFastEnabled() {
    static const bool on = npur_env::On("NPUR_CLASSIFY_FAST");
    return on;
}

static bool SparseDirectEnabled() {
    static const bool on = npur_env::On("NPUR_SPARSE_DIRECT");
    return on;
}

static std::vector<uint8_t> BuildSparseDirectMask(const std::vector<uint32_t>& postExpr, size_t postingNum) {
    const std::vector<uint8_t> none(postingNum, 0);
    std::vector<uint8_t> mask(postingNum, 0);
    size_t next = 0;
    for (size_t idx = 0; idx < postExpr.size(); idx++) {
        const uint32_t opType = postExpr[idx];
        if (opType == FilterOpType::CONJUNCTION) {
            if (idx + 1 >= postExpr.size()) {
                LOG_ERROR("BuildSparseDirectMask: conj node truncated at " << idx);
                return none;
            }
            const uint32_t partNum = postExpr[idx + 1];
            if (partNum > postExpr.size() - idx - 2) {
                LOG_ERROR("BuildSparseDirectMask: conj partNum " << partNum << " truncated at " << idx);
                return none;
            }
            for (uint32_t i = 1; i <= partNum; i++) {
                next += postExpr[idx + 1 + i];
            }
            // Keeps next <= postingNum, so `postingNum - next` below cannot wrap as a size_t.
            if (next > postingNum) {
                LOG_ERROR("BuildSparseDirectMask: conj node at " << idx << " overruns " << postingNum << " operands");
                return none;
            }
            idx = idx + partNum + 1;
            continue;
        }
        if (opType != FilterOpType::AND && opType != FilterOpType::OR && opType != FilterOpType::NOT) {
            LOG_ERROR("BuildSparseDirectMask: invalid opType " << opType << " at " << idx);
            return none;
        }
        if (idx + 2 >= postExpr.size()) {
            LOG_ERROR("BuildSparseDirectMask: and/or/not node truncated at " << idx);
            return none;
        }
        const uint32_t nodePostings = postExpr[idx + 1];
        if (nodePostings > postingNum - next) {  // next <= postingNum holds by induction
            LOG_ERROR("BuildSparseDirectMask: node at " << idx << " wants " << nodePostings << " of "
                                                        << (postingNum - next) << " operands left");
            return none;
        }
        if (opType == FilterOpType::OR) {
            for (uint32_t i = 0; i < nodePostings; i++) {
                mask[next + i] = 1;
            }
        }
        next += nodePostings;
        idx += 2;
    }
    if (next != postingNum) {
        LOG_ERROR("BuildSparseDirectMask: consumed " << next << " operands, expected " << postingNum);
        return none;
    }
    return mask;
}

// NPUR_BATCH_FILTER=1: run the chunk's filter as phases across the batch instead of query by query.
static bool BatchFilterEnabled() {
    static const bool on = npur_env::On("NPUR_BATCH_FILTER");
    return on;
}

namespace {
// Slice bases are rounded up so no kernel ever sees a misaligned pointer.
constexpr uint32_t SLICE_ALIGN = 512;
inline uint32_t AlignUpSlice(uint32_t v) {
    return (v + SLICE_ALIGN - 1) / SLICE_ALIGN * SLICE_ALIGN;
}
}  // namespace

bool TextFilter::FusedPathApplies(const std::vector<FilterExpr>* prepared) {
    return BatchFilterEnabled() && prepared != nullptr;
}

bool TextFilter::BatchComputeFused(const std::vector<QueryNode*>& queryTrees, uint8_t* resultInDevice,
                                   const std::vector<FilterExpr>& prepared) {
    FusedFilterBatch batch;
    if (!FusedPrepare(queryTrees, resultInDevice, prepared, batch)) {
        return false;
    }
    FusedLaunch(batch, /*timeKernel=*/true);
    return FusedFinish(batch, /*streamSynced=*/false);
}

bool TextFilter::FusedPrepare(const std::vector<QueryNode*>& queryTrees, uint8_t* resultInDevice,
                              const std::vector<FilterExpr>& prepared, FusedFilterBatch& batch) {
    const size_t n = queryTrees.size();
    const uint32_t resultByteSize = m_dataTable->GetDocNumPerSegment() / 8 * m_dataTable->GetSegmentNum();
    auto mm = GmMemoryManager::GetByDeviceId(m_deviceId);

    auto ret = aclrtSetDevice(m_deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtSetDevice failed, m_deviceId:" << m_deviceId << ", error code is:" << ret);
        return false;
    }

    batch = FusedFilterBatch{};
    batch.slot.assign(n, FilterSlot{});
    std::vector<FilterSlot>& slot = batch.slot;
    uint32_t exprTotal = 0;
    uint32_t stackTotal = 0;
    size_t activeCount = 0;
    auto releasePostings = [&]() {
        for (size_t q = 0; q < n; ++q) {
            if (!slot[q].ownPostings)
                continue;
            FreeDevicePostings(slot[q].devicePostingsChunk);
            mm->FreeBlock(slot[q].bitsetPostingsChunk);
            slot[q].ownPostings = false;
        }
    };
    auto fail = [&]() {
        releasePostings();
        CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
        batch = FusedFilterBatch{};
        return false;
    };

    for (size_t q = 0; q < n; ++q) {
        FilterSlot& s = slot[q];
        s.result = resultInDevice + static_cast<size_t>(resultByteSize) * q;
        const FilterExpr& e = prepared[q];
        if (queryTrees[q] == nullptr || !e.valid) {
            LOG_ERROR("invalid query tree or prepared expression at q=" << q);
            return fail();
        }
        if (e.emptyTree) {
            ret = aclrtMemset(s.result, resultByteSize, 0xff, resultByteSize);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("aclrtMemset fail, error code is:" << ret);
                return fail();
            }
            continue;  // stays inactive
        }
        if (!CheckPostExpr(e.postExpr, e.postingDeviceAddrs.size())) {
            LOG_ERROR("CheckPostExpr failed at q=" << q);
            return fail();
        }
        CreateContextData(e.postExpr.size(), e.opNum, e.postingDeviceAddrs.size(), s.context);
        if (e.postingsPrepared) {
            s.devicePostings = reinterpret_cast<uint8_t*>(e.devicePostingsChunk.data);
        } else {
            const std::vector<uint8_t> mask = SparseDirectEnabled()
                                                  ? BuildSparseDirectMask(e.postExpr, e.postingDeviceAddrs.size())
                                                  : std::vector<uint8_t>{};
            if (!PostingBitListToSet(e.postingTypes, e.postingDeviceAddrs,
                                     static_cast<uint32_t>(e.postingDeviceAddrs.size()), s.bitsetPostingsChunk,
                                     s.devicePostingsChunk, mask.empty() ? nullptr : &mask, &e.postingWeights)) {
                LOG_ERROR("posting bitList to set failed at q=" << q);
                FreeDevicePostings(s.devicePostingsChunk);
                return fail();
            }
            s.devicePostings = reinterpret_cast<uint8_t*>(s.devicePostingsChunk.data);
            s.ownPostings = true;
        }
        s.postExpr = &e.postExpr;
        s.exprOffset = exprTotal;
        exprTotal += AlignUpSlice(static_cast<uint32_t>(e.postExpr.size() * sizeof(uint32_t)));
        s.stackOffset = stackTotal;
        stackTotal += AlignUpSlice(s.context.segmentNum * s.context.segmentLength *
                                   static_cast<uint32_t>(sizeof(uint16_t)) * s.context.opNum);
        s.active = true;
        activeCount++;
    }
    batch.deviceSet = true;
    batch.activeCount = activeCount;
    if (activeCount == 0) {  // every query was an empty tree; the memsets already did the work
        return true;         // FusedLaunch has nothing to queue, FusedFinish only drops the device
    }

    {
        RecordGuard guard{"TextFilter_BitmapTextFilter_postExpr_malloc_h2d"};
        if (PoolSmallH2dEnabled()) {
            mm->AllocateBlock(GmPoolName::TEXT_FILTER_EXPR_POOL, exprTotal, batch.exprChunk);
            if (batch.exprChunk.data == nullptr) {
                LOG_ERROR("batch devicePostExpr pool allocate fail");
                return fail();
            }
            batch.deviceExprBase = reinterpret_cast<uint32_t*>(batch.exprChunk.data);
        } else {
            ret = aclrtMalloc((void**)&batch.deviceExprBase, exprTotal, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("batch devicePostExpr aclrtMalloc fail, error code is:" << ret);
                batch.deviceExprBase = nullptr;
                return fail();
            }
        }
        std::vector<uint8_t> staging(exprTotal, 0);
        for (size_t q = 0; q < n; ++q) {
            if (!slot[q].active)
                continue;
            std::memcpy(staging.data() + slot[q].exprOffset, slot[q].postExpr->data(),
                        slot[q].postExpr->size() * sizeof(uint32_t));
        }
        ret = aclrtMemcpy(batch.deviceExprBase, exprTotal, staging.data(), exprTotal, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("batch devicePostExpr aclrtMemcpy fail, error code is:" << ret);
            ReleaseFusedExpr(batch);
            return fail();
        }
    }
    {
        RecordGuard guard{"TextFilter_BitmapTextFilter_resultStack_allocate"};
        mm->AllocateBlock(GmPoolName::TEXT_FILTER_STACK_POOL, stackTotal, batch.stackChunk);
    }
    if (batch.stackChunk.data == nullptr || batch.stackChunk.size < stackTotal) {
        LOG_ERROR("GmMemoryManager allocate failed for the batch result stack.");
        mm->FreeBlock(batch.stackChunk);
        ReleaseFusedExpr(batch);
        return fail();
    }
    for (size_t q = 0; q < n; ++q) {
        if (!slot[q].active)
            continue;
        slot[q].devicePostExpr =
            reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(batch.deviceExprBase) + slot[q].exprOffset);
        slot[q].resultStack =
            reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(batch.stackChunk.data) + slot[q].stackOffset);
    }
    return true;
}

void TextFilter::ReleaseFusedExpr(FusedFilterBatch& batch) {
    if (batch.exprChunk.data != nullptr) {
        GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(batch.exprChunk);
        batch.exprChunk.data = nullptr;  // FreeBlock does not clear it; a second call must be a no-op
    } else if (batch.deviceExprBase != nullptr) {
        CHECK_ACL_ONLY_LOG(aclrtFree(batch.deviceExprBase));
    }
    batch.deviceExprBase = nullptr;
}

void TextFilter::FusedLaunch(FusedFilterBatch& batch, bool timeKernel) {
    if (!batch.deviceSet || batch.activeCount == 0) {
        return;
    }
    // timeKernel false is NPUR_SHARE_FILTER_STREAM: the wait happens in the aggregator's sync.
    batch.stream = StreamManager::GetInstance()->GetStream(m_deviceId);
    std::unique_ptr<RecordGuard> launchOnly;
    if (timeKernel) {
        batch.allGuard.reset(new RecordGuard("TextFilter_BitmapTextFilter_all"));
        batch.kernelGuard.reset(new RecordGuard("TextFilter_BitmapTextFilter_Kernel"));
    } else {
        launchOnly.reset(new RecordGuard("TextFilter_BitmapTextFilter_launch_only"));
    }
    for (size_t q = 0; q < batch.slot.size(); ++q) {
        FilterSlot& s = batch.slot[q];
        if (!s.active)
            continue;
        ACLRT_LAUNCH_KERNEL(BitmapTextFilter)
        (FLAGS_full_recall_text_filter_block_dim, batch.stream, s.devicePostings, s.devicePostExpr, s.resultStack,
         &s.context, s.result);
    }
    batch.launched = true;
}

bool TextFilter::FusedFinish(FusedFilterBatch& batch, bool streamSynced) {
    if (!batch.deviceSet) {
        return true;  // FusedPrepare failed and already cleaned up, or was never run
    }
    bool ok = true;
    if (batch.launched) {
        if (!streamSynced) {
            aclError ret = aclrtSynchronizeStream(batch.stream);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("aclrtSynchronizeStream fail, error code is:" << ret);
                ok = false;
            }
        }
        batch.kernelGuard.reset();  // same order the two scoped guards used to close in
        batch.allGuard.reset();
        StreamManager::GetInstance()->FreeStream(m_deviceId, batch.stream);
        batch.stream = nullptr;
        batch.launched = false;
    }

    auto mm = GmMemoryManager::GetByDeviceId(m_deviceId);
    if (batch.activeCount > 0) {
        RecordGuard guard{"TextFilter_BitmapTextFilter_teardown_free"};
        ReleaseFusedExpr(batch);
        mm->FreeBlock(batch.stackChunk);
        batch.stackChunk.data = nullptr;
    }
    for (FilterSlot& s : batch.slot) {
        if (!s.ownPostings)
            continue;
        FreeDevicePostings(s.devicePostingsChunk);
        mm->FreeBlock(s.bitsetPostingsChunk);
        s.ownPostings = false;
    }
    CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
    batch.deviceSet = false;
    return ok;
}

bool TextFilter::BatchCompute(const std::vector<QueryNode*> queryTrees, uint8_t* resultInDevice,
                              const std::vector<FilterExpr>* prepared) {
    RecordGuard guard{"TextFilter BatchCompute"};
    LOG_DEBUG("TextFilter::BatchCompute- queryTrees.size():" << queryTrees.size());
    if (prepared != nullptr && prepared->size() != queryTrees.size()) {
        LOG_ERROR("prepared expression count " << prepared->size() << " != queryTrees count " << queryTrees.size());
        return false;
    }
    if (BatchFilterEnabled() && prepared != nullptr) {
        return BatchComputeFused(queryTrees, resultInDevice, *prepared);
    }
    uint32_t segmentsNum = m_dataTable->GetSegmentNum();
    uint32_t docNumPerSegment = m_dataTable->GetDocNumPerSegment();
    LogContext logContext;
    auto asyncContext = m_executor->CreateExecuteContext(logContext);
    uint32_t resultByteSize = docNumPerSegment / 8 * segmentsNum;
    for (size_t i = 0; i < queryTrees.size(); i++) {
        uint8_t* resultInDeviceNew = (uint8_t*)resultInDevice + resultByteSize * i;
        QueryNode* queryTree = queryTrees[i];
        const FilterExpr* expr = prepared == nullptr ? nullptr : &(*prepared)[i];
        auto computeFunc = [this, queryTree, resultInDeviceNew, resultByteSize, expr]() -> ErrorCode::ResultType {
            if (queryTree == nullptr) {
                LOG_ERROR("one queryTree is nullptr.");
                return ErrorCode::ResultType::FAIL;
            }
            auto ret = aclrtSetDevice(m_deviceId);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("aclrtSetDevice failed, m_deviceId:" << m_deviceId << ", error code is:" << ret);
                return ErrorCode::ResultType::FAIL;
            }
            if (!Compute(queryTree, resultInDeviceNew, resultByteSize, expr)) {
                LOG_ERROR("one text compute failed");
                CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
                return ErrorCode::ResultType::FAIL;
            }
            CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
            return ErrorCode::ResultType::SUCCESS;
        };
        asyncContext->AddTask(computeFunc);
    }
    bool isSuccess = true;
    asyncContext->Wait([&isSuccess](ErrorCode::ResultType ret) {
        if (ret != ErrorCode::ResultType::SUCCESS) {
            isSuccess = false;
        }
    });
    return isSuccess;
}

bool TextFilter::PrepareExpr(QueryNode* queryTree, FilterExpr& out) {
    RecordGuard guard{"TextFilter_Compute_GetPostOrderExpression"};
    {
        std::unique_lock<std::shared_mutex> lock(queryTree->m_adjustNodeMutex);
        if (!queryTree->m_isAdjustNodeDone) {
            queryTree->AdjustNode(m_dataTable->GetPostingFields());
            queryTree->m_isAdjustNodeDone = true;
        }
    }
    {
        std::shared_lock<std::shared_mutex> lock(queryTree->m_adjustNodeMutex);
        if (queryTree->GetChildrenNum() == 0) {  // empty query tree
            LOG_WARN("empty postExpr, text filter skiped.");
            out.emptyTree = true;
            out.valid = true;
            return true;
        }

        if (!queryTree->GetPostOrderExpression(*m_dataTable, out.postExpr, out.postingTypes, out.postingWeights,
                                               out.postingDeviceAddrs, out.opNum)) {
            LOG_ERROR("get post order expression failed");
            return false;
        }
    }
    // types, weights and addrs run in lockstep: emit indexes all three by the same postingIdx.
    if (out.postingWeights.size() != out.postingTypes.size()) {
        LOG_ERROR("posting weights size " << out.postingWeights.size() << " != types size " << out.postingTypes.size());
        return false;
    }
    if (out.postingTypes.size() != out.postingDeviceAddrs.size()) {
        LOG_ERROR("the size of postingTypes is not equal to postingDeviceAddrs");
        return false;
    }
    out.valid = true;
    return true;
}

bool TextFilter::BatchPrepareExpr(const std::vector<QueryNode*> queryTrees, std::vector<FilterExpr>& out) {
    RecordGuard guard{"TextFilter BatchPrepareExpr"};
    out.clear();
    out.resize(queryTrees.size());
    for (size_t i = 0; i < queryTrees.size(); i++) {
        if (queryTrees[i] == nullptr) {
            LOG_ERROR("one queryTree is nullptr.");
            return false;
        }
        if (!PrepareExpr(queryTrees[i], out[i])) {
            return false;
        }
    }
    return true;
}

bool TextFilter::BatchPreparePostings(std::vector<FilterExpr>& exprs, bool deferConvSync) {
    RecordGuard guard{"TextFilter BatchPreparePostings"};
    for (auto& e : exprs) {
        if (!e.valid || e.emptyTree || e.postingsPrepared) {
            continue;
        }
        if (!BitlistOverlapEnabled() && GetBitlistTypeNum(e.postingTypes) > 0) {
            continue;
        }
        e.devicePostingsChunk = GmBlock{GmPoolName::TEXT_FILTER_POSTINGS_POOL, nullptr, 0};
        uint32_t postingsNum = e.postingDeviceAddrs.size();
        const std::vector<uint8_t> mask = SparseDirectEnabled()
                                              ? BuildSparseDirectMask(e.postExpr, e.postingDeviceAddrs.size())
                                              : std::vector<uint8_t>{};
        if (!PostingBitListToSet(e.postingTypes, e.postingDeviceAddrs, postingsNum, e.bitsetPostingsChunk,
                                 e.devicePostingsChunk, mask.empty() ? nullptr : &mask, &e.postingWeights,
                                 deferConvSync ? &e.pendingConvStream : nullptr)) {
            LOG_ERROR("BatchPreparePostings: PostingBitListToSet failed");
            FreeDevicePostings(e.devicePostingsChunk);
            return false;
        }
        e.postingsPrepared = true;
    }
    return true;
}

bool TextFilter::DrainPreparedConversions(std::vector<FilterExpr>& exprs) {
    bool ok = true;
    for (auto& e : exprs) {
        if (e.pendingConvStream == nullptr) {
            continue;
        }
        RecordGuard guard{"TextFilter_PostingBitListToSet_deferred_sync"};
        if (aclrtSynchronizeStream(e.pendingConvStream) != ACL_SUCCESS) {
            LOG_ERROR("aclrtSynchronizeStream (deferred bitlist conversion) fail, m_deviceId=" << m_deviceId);
            ok = false;
        }
        StreamManager::GetInstance()->FreeStream(m_deviceId, e.pendingConvStream);
        e.pendingConvStream = nullptr;
    }
    return ok;
}

void TextFilter::FreePreparedPostings(std::vector<FilterExpr>& exprs) {
    DrainPreparedConversions(exprs);
    for (auto& e : exprs) {
        if (!e.postingsPrepared) {
            continue;
        }
        FreeDevicePostings(e.devicePostingsChunk);
        GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(e.bitsetPostingsChunk);
        e.postingsPrepared = false;
    }
}

bool TextFilter::Compute(QueryNode* queryTree, uint8_t* resultInDevice, uint32_t resultByteSize,
                         const FilterExpr* prepared) {
    FilterExpr built;
    if (prepared == nullptr) {
        if (!PrepareExpr(queryTree, built)) {
            return false;
        }
        prepared = &built;
    }
    if (!prepared->valid) {
        LOG_ERROR("prepared filter expression is invalid");
        return false;
    }
    if (prepared->emptyTree) {
        auto ret = aclrtMemset(resultInDevice, resultByteSize, 0xff, resultByteSize);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("aclrtMemset fail, error code is:" << ret);
            return false;
        }
        return true;
    }
    const std::vector<uint32_t>& postExpr = prepared->postExpr;
    const std::vector<std::vector<uint8_t>*>& postingTypes = prepared->postingTypes;
    const std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs = prepared->postingDeviceAddrs;
    const std::vector<std::vector<uint16_t>*>& postingWeights = prepared->postingWeights;
    const uint32_t opNum = prepared->opNum;

    if (NpuRetrieval::Logger::GetLogLevel() <= log4cplus::DEBUG_LOG_LEVEL) {
        std::string postExprStr;
        for (const uint32_t& i : postExpr) {
            postExprStr += std::to_string(i) + ',';
        }
        LOG_DEBUG("postExpr:" << postExprStr << " opNum:" << opNum);
    }

    if (!CheckPostExpr(postExpr, postingDeviceAddrs.size())) {
        LOG_ERROR("CheckPostExpr failed");
        return false;
    }

    TextFilterContextData contextData{};
    CreateContextData(postExpr.size(), opNum, postingDeviceAddrs.size(), contextData);

    GmBlock bitsetPostingsChunk{};
    GmBlock devicePostingsChunk{GmPoolName::TEXT_FILTER_POSTINGS_POOL, nullptr, 0};
    uint8_t* devicePostings = nullptr;
    bool ownPostings = false;
    if (prepared != nullptr && prepared->postingsPrepared) {
        devicePostings = reinterpret_cast<uint8_t*>(prepared->devicePostingsChunk.data);
    } else {
        uint32_t postingsNum = postingDeviceAddrs.size();
        const std::vector<uint8_t> mask =
            SparseDirectEnabled() ? BuildSparseDirectMask(postExpr, postingDeviceAddrs.size()) : std::vector<uint8_t>{};
        if (!PostingBitListToSet(postingTypes, postingDeviceAddrs, postingsNum, bitsetPostingsChunk,
                                 devicePostingsChunk, mask.empty() ? nullptr : &mask, &postingWeights)) {
            LOG_ERROR("posting bitList to set failed");
            FreeDevicePostings(devicePostingsChunk);
            return false;
        }
        devicePostings = reinterpret_cast<uint8_t*>(devicePostingsChunk.data);
        ownPostings = true;
    }
    {
        RecordGuard guard{"TextFilter_BitmapTextFilter_all"};
        if (!BitmapTextFilter(postExpr, contextData, devicePostings, resultInDevice)) {
            LOG_ERROR("text filter failed");
            if (ownPostings) {
                FreeDevicePostings(devicePostingsChunk);
                GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(bitsetPostingsChunk);
            }
            return false;
        }
    }
    if (ownPostings) {
        {
            RecordGuard guard{"TextFilter_Compute_teardown_aclrtFree_devicePostings"};
            FreeDevicePostings(devicePostingsChunk);
        }
        {
            RecordGuard guard{"TextFilter_Compute_teardown_FreeBlock_bitset"};
            GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(bitsetPostingsChunk);
        }
    }
    return true;
}

// NPUR_BITLIST_ONE_H2D=1: the table and the kernel's two address lists share one allocation and
// one copy. NPUR_BITLIST_POSTING_MAJOR=1 walks the classifying pass posting-major; needs ONE_H2D.
static bool BitlistPostingMajorEnabled() {
    static const bool on = npur_env::On("NPUR_BITLIST_POSTING_MAJOR");
    return on;
}

static bool BitlistOneH2dEnabled() {
    static const bool on = npur_env::On("NPUR_BITLIST_ONE_H2D");
    return on;
}

bool TextFilter::PostingBitListToSet(const std::vector<std::vector<uint8_t>*>& postingTypes,
                                     const std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs,
                                     uint32_t postingsNum, GmBlock& bitsetPostingsChunk, GmBlock& devicePostingsChunk,
                                     const std::vector<uint8_t>* sparseDirect,
                                     const std::vector<std::vector<uint16_t>*>* postingWeights,
                                     aclrtStream* deferSync) {
    const bool skipEmpty = (EffectiveFilterFlags() & FILTER_FLAG_OR_SKIP_EMPTY) != 0;
    const std::vector<std::vector<uint16_t>*>* weights = skipEmpty ? postingWeights : nullptr;
    if (weights != nullptr && weights->size() != postingsNum) {
        LOG_ERROR("posting weights size " << weights->size() << " != postingsNum " << postingsNum);
        weights = nullptr;
    }
    RecordGuard guard{"TextFilter_PostingBitListToSet"};
    uint32_t docNumPerSegment = m_dataTable->GetDocNumPerSegment();
    // GetMaxPostingLength can exceed FilterOrOp's 16KB tile buffer past den~0.03, so refuse to tag.
    if (sparseDirect != nullptr &&
        GetMaxPostingLength(docNumPerSegment, FLAGS_full_recall_bitlist_denseness_threshold) > m_segmentByteSize) {
        sparseDirect = nullptr;
    }
    const uint32_t segmentsNum = m_dataTable->GetSegmentNum();
    uint32_t blockNum = 0;
    uint32_t anyBitlist = 0;
    {
        RecordGuard guardCount{"TextFilter_PostingBitListToSet_count"};
        // The transpose path below offsets every posting by 8, which is wrong for a tagged one.
        if (OneCountPassEnabled() && sparseDirect != nullptr) {
            CountBitlistTypes(postingTypes, sparseDirect, blockNum, anyBitlist);
        } else {
            blockNum = GetBitlistTypeNum(postingTypes, sparseDirect);  // number of blocks to allocate
            anyBitlist = sparseDirect == nullptr ? blockNum : GetBitlistTypeNum(postingTypes, nullptr);
        }
    }
    LOG_DEBUG("deviceId:" << m_deviceId << " postingsNum:" << postingsNum << " docNumPerSegment:" << docNumPerSegment
                          << " segmentsNum:" << segmentsNum << " segmentByteSize:" << m_segmentByteSize
                          << " blockNum:" << blockNum);
    if (anyBitlist == 0) {  // no list->set needed; just transpose the pointer array and offset by 8 bytes
        return PostingBitListTransposeAndOffset(postingDeviceAddrs, segmentsNum, postingsNum, devicePostingsChunk);
    }

    {
        RecordGuard guardCheck{"TextFilter_PostingBitListToSet_check"};
        if (!CheckPostingTypes(postingTypes, segmentsNum, postingsNum)) {
            LOG_ERROR("invalid posting types");
            return false;
        }
    }

    uint8_t* bitsetPostings = nullptr;
    {
        RecordGuard guard{"TextFilter_PostingBitListToSet_GmMemoryPool_Allocate"};
        // Never zero: blockNum can be 0 with the classify pass still to run, and a zero-byte
        // block would come back null.
        uint32_t bitsetPostingsByteSize = (blockNum == 0 ? 1 : blockNum) * m_segmentByteSize;
        GmMemoryManager::GetByDeviceId(m_deviceId)
            ->AllocateBlock(GmPoolName::TEXT_FILTER_BITLIST2SET_POOL, bitsetPostingsByteSize, bitsetPostingsChunk);
        bitsetPostings = reinterpret_cast<uint8_t*>(bitsetPostingsChunk.data);
    }
    // Every failure from here on has to hand the pool block back; the caller frees only the table.
    auto failAfterAlloc = [&]() {
        GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(bitsetPostingsChunk);
        bitsetPostingsChunk.data = nullptr;
        bitsetPostingsChunk.size = 0;
        return false;
    };
    if (bitsetPostings == nullptr || bitsetPostingsChunk.size < (blockNum == 0 ? 1 : blockNum) * m_segmentByteSize) {
        LOG_ERROR("GmMemoryManager allocate failed.");
        return failAfterAlloc();
    }

    // One buffer, three regions: the table, then blockNum source addresses, then blockNum
    // destination addresses. `postings` aliases the first.
    const size_t tableCount = static_cast<size_t>(segmentsNum) * postingsNum;
    const bool oneH2d = BitlistOneH2dEnabled();
    std::vector<uint8_t*> staging;
    std::vector<uint8_t*> postings;               // pointers to the converted bitsets, used by the later filtering
    std::vector<uint8_t*> bitlistPostings;        // addresses of the bitlists that need conversion
    std::vector<uint8_t*> bitlistResultPostings;  // addresses of the converted bitlist results
    size_t bitlistNum = 0;
    uint8_t** tableOut = nullptr;
    uint8_t** srcOut = nullptr;
    uint8_t** dstOut = nullptr;
    if (oneH2d) {
        staging.resize(tableCount + 2 * static_cast<size_t>(blockNum));
        tableOut = staging.data();
        srcOut = staging.data() + tableCount;
        dstOut = srcOut + blockNum;
    } else {
        postings.reserve(tableCount);
        bitlistPostings.reserve(blockNum);
        bitlistResultPostings.reserve(blockNum);
    }
    auto setTable = [&](size_t idx, uint8_t* v) {
        if (oneH2d) {
            tableOut[idx] = v;
        } else {
            postings.emplace_back(v);  // legacy path only ever visits idx in ascending order
            (void)idx;
        }
    };
    auto pushBitlist = [&](uint8_t* src, uint8_t* dst) {
        if (oneH2d) {
            srcOut[bitlistNum] = src;
            dstOut[bitlistNum] = dst;
        } else {
            bitlistPostings.emplace_back(src);
            bitlistResultPostings.emplace_back(dst);
        }
    };
    bool classifyOk = true;
    auto emit = [&](size_t tableIdx, size_t segIdx, bool orOperand, const uint16_t* w,
                    const std::vector<uint8_t>& types, const std::vector<uint8_t*>& addrs) {
        if (orOperand && w != nullptr) {
            if (skipEmpty && w[segIdx] == 0) {
                setTable(tableIdx, reinterpret_cast<uint8_t*>(EMPTY_OPERAND_TAG));
                return;
            }
        }
        if (types[segIdx] == 0) {
            setTable(tableIdx, addrs[segIdx] + 8);  // +8 bytes: the bitset address
            return;
        }
        if (orOperand) {
            setTable(tableIdx,
                     reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(addrs[segIdx]) | SPARSE_OPERAND_TAG));
            return;
        }
        if (bitlistNum >= blockNum) {
            LOG_ERROR("bitlistNum out of range");
            classifyOk = false;
            return;
        }
        uint8_t* dst = bitsetPostings + bitlistNum * m_segmentByteSize;
        setTable(tableIdx, dst);
        pushBitlist(addrs[segIdx], dst);
        bitlistNum += 1;
    };
    const bool postingMajor = BitlistPostingMajorEnabled() && oneH2d;
    {
        RecordGuard guardClassify{"TextFilter_PostingBitListToSet_classify"};
        auto orOperandAt = [&](size_t postingIdx) {
            return sparseDirect != nullptr && (*sparseDirect)[postingIdx] != 0;
        };
        auto weightsAt = [&](size_t postingIdx) -> const uint16_t* {
            if (weights == nullptr || (*weights)[postingIdx] == nullptr) {
                return nullptr;
            }
            return (*weights)[postingIdx]->data();
        };
        const bool fast = ClassifyFastEnabled() && oneH2d;
        if (postingMajor) {
            for (size_t postingIdx = 0; postingIdx < postingsNum && classifyOk; postingIdx++) {
                const std::vector<uint8_t>& types = *(postingTypes[postingIdx]);
                const std::vector<uint8_t*>& addrs = *(postingDeviceAddrs[postingIdx]);
                const bool orOperand = orOperandAt(postingIdx);
                const uint16_t* w = weightsAt(postingIdx);
                if (fast && orOperand && !(skipEmpty && w != nullptr)) {
                    for (size_t segIdx = 0; segIdx < segmentsNum; segIdx++) {
                        uint8_t* a = addrs[segIdx];
                        tableOut[segIdx * postingsNum + postingIdx] =
                            types[segIdx] == 0
                                ? a + 8
                                : reinterpret_cast<uint8_t*>(reinterpret_cast<uintptr_t>(a) | SPARSE_OPERAND_TAG);
                    }
                    continue;
                }
                for (size_t segIdx = 0; segIdx < segmentsNum && classifyOk; segIdx++) {
                    emit(segIdx * postingsNum + postingIdx, segIdx, orOperand, w, types, addrs);
                }
            }
        } else {
            for (size_t segIdx = 0; segIdx < segmentsNum && classifyOk; segIdx++) {
                for (size_t postingIdx = 0; postingIdx < postingsNum && classifyOk; postingIdx++) {
                    emit(segIdx * postingsNum + postingIdx, segIdx, orOperandAt(postingIdx), weightsAt(postingIdx),
                         *(postingTypes[postingIdx]), *(postingDeviceAddrs[postingIdx]));
                }
            }
        }
    }
    if (EmptyStatsEnabled() && postingWeights != nullptr && sparseDirect != nullptr) {
        NoteEmptyOperands(postingTypes, *postingWeights, *sparseDirect, postingsNum, segmentsNum);
    }
    if (!classifyOk) {
        return failAfterAlloc();
    }
    const size_t postingsByteSize = tableCount * sizeof(uint8_t*);
    uint8_t* deviceBitlistPostings = nullptr;
    uint8_t* deviceBitlistResultPostings = nullptr;
    const size_t bitlistPostingsByteSize = bitlistNum * sizeof(uint8_t*);
    aclError ret = ACL_SUCCESS;
    uint8_t* devicePostings = nullptr;  // used by the debug print at the end of the function
    if (oneH2d) {
        const size_t totalByteSize = staging.size() * sizeof(uint8_t*);
        AllocateDevicePostings(totalByteSize, devicePostingsChunk);
        uint8_t* base = reinterpret_cast<uint8_t*>(devicePostingsChunk.data);
        devicePostings = base;
        if (base == nullptr || devicePostingsChunk.size < totalByteSize) {
            LOG_ERROR("devicePostings pool allocate failed");
            return failAfterAlloc();
        }
        {
            RecordGuard guardTable{"TextFilter_PostingBitListToSet_table_h2d"};
            ret = aclrtMemcpy(base, totalByteSize, staging.data(), totalByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        }
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("bitlist staging aclrtMemcpy fail,  error code is:" << ret);
            return failAfterAlloc();
        }
        deviceBitlistPostings = base + postingsByteSize;
        deviceBitlistResultPostings = deviceBitlistPostings + bitlistPostingsByteSize;
    } else {
        AllocateDevicePostings(postingsByteSize, devicePostingsChunk);
        devicePostings = reinterpret_cast<uint8_t*>(devicePostingsChunk.data);
        if (devicePostings == nullptr || devicePostingsChunk.size < postingsByteSize) {
            LOG_ERROR("devicePostings pool allocate failed");
            return failAfterAlloc();
        }
        {
            RecordGuard guardTable{"TextFilter_PostingBitListToSet_table_h2d"};
            ret = aclrtMemcpy(devicePostings, postingsByteSize, postings.data(), postingsByteSize,
                              ACL_MEMCPY_HOST_TO_DEVICE);
        }
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("devicePostings aclrtMemcpy fail,  error code is:" << ret);
            return failAfterAlloc();
        }
    }
    // Exiting here avoids aclrtMalloc(0) on the non-ONE_H2D path below, which fails the query.
    if (bitlistNum == 0) {
        return true;
    }

    if (!oneH2d) {
        RecordGuard guardAddrs{"TextFilter_PostingBitListToSet_addrs_h2d"};
        ret = aclrtMalloc((void**)&deviceBitlistPostings, bitlistPostingsByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret == ACL_SUCCESS) {
            ret = aclrtMemcpy(deviceBitlistPostings, bitlistPostingsByteSize, bitlistPostings.data(),
                              bitlistPostingsByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        }
        if (ret == ACL_SUCCESS) {
            ret = aclrtMalloc((void**)&deviceBitlistResultPostings, bitlistPostingsByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        if (ret == ACL_SUCCESS) {
            ret = aclrtMemcpy(deviceBitlistResultPostings, bitlistPostingsByteSize, bitlistResultPostings.data(),
                              bitlistPostingsByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
        }
    }
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("bitlist address arrays malloc/memcpy fail, error code is:" << ret);
        if (deviceBitlistPostings != nullptr)
            CHECK_ACL_ONLY_LOG(aclrtFree(deviceBitlistPostings));
        if (deviceBitlistResultPostings != nullptr)
            CHECK_ACL_ONLY_LOG(aclrtFree(deviceBitlistResultPostings));
        return failAfterAlloc();
    }
    auto releaseAddrs = [&]() {
        if (oneH2d)
            return;  // slices of devicePostingsChunk; the caller frees it
        CHECK_ACL_ONLY_LOG(aclrtFree(deviceBitlistPostings));
        CHECK_ACL_ONLY_LOG(aclrtFree(deviceBitlistResultPostings));
    };
    uint32_t blockDim = FLAGS_full_recall_text_filter_block_dim;
    uint32_t alignNum = blockDim;
    uint32_t partIdxAligned = ((bitlistNum + alignNum - 1) / alignNum) * alignNum;
    uint32_t formerNum = bitlistNum % blockDim;
    uint32_t tailNum = blockDim - formerNum;
    uint32_t tailLength = bitlistNum / blockDim;
    uint32_t formerLength = partIdxAligned / blockDim;
    LOG_INFO(" bitlistNum:" << bitlistNum << " formerNum:" << formerNum << " formerLength:" << formerLength
                            << " tailNum:" << tailNum << " tailLength:" << tailLength);
    uint32_t maxPostingLengthByte =
        GetMaxPostingLength(docNumPerSegment, FLAGS_full_recall_bitlist_denseness_threshold);
    const uint32_t statsLevel = BitlistStatsLevel();
    if (statsLevel > 0) {
        BitlistCallShape shape;
        shape.deviceId = m_deviceId;
        shape.segmentsNum = segmentsNum;
        shape.postingsNum = postingsNum;
        shape.tableCount = tableCount;
        shape.bitlistNum = bitlistNum;
        shape.segmentByteSize = m_segmentByteSize;
        shape.blockDim = blockDim;
        shape.maxPostingLengthByte = maxPostingLengthByte;
        shape.formerNum = formerNum;
        shape.formerLength = formerLength;
        shape.tailLength = tailLength;
        shape.srcList = oneH2d ? srcOut : bitlistPostings.data();
        NoteBitlistCall(shape);
        if (statsLevel >= 2 && bitlistNum > 0) {
            NoteBitlistDeep(shape);  // one D2H per posting; the call budget is counted in there
        }
    }
    bool state = true;
    aclrtStream stream = StreamManager::GetInstance()->GetStream(m_deviceId);
    // NPUR_DEFER_CONV_SYNC: the caller waits on the stream later, before any filter kernel reads it.
    const bool defer = deferSync != nullptr && oneH2d && stream != nullptr && !npur_env::Present("NPUR_DEBUG_BITLIST");
    {
        RecordGuard guard{defer ? "TextFilter_PostingBitListToSet_launch_only"
                                : "TextFilter_PostingBitListToSet_Kernel"};
        // NPUR_BITLIST_CLEAR_SPARSE=1: clear only the units written, not the full 16KB.
        static const uint32_t clearSparse = npur_env::On("NPUR_BITLIST_CLEAR_SPARSE") ? 1u : 0u;
        // NPUR_BITLIST_ABLATE=1|2|3|4: skip one step. WRONG RESULTS BY DESIGN.
        static const uint32_t ablate = npur_env::U32Capped("NPUR_BITLIST_ABLATE", 4, 0);
        // NPUR_BITLIST_SCATTER=1: AscendC::Scatter, a silent no-op on dav_c220. Recall is the test.
        // 1 = byte offsets as stored, 2 = halved to indices
        static const uint32_t useScatter = npur_env::U32Capped("NPUR_BITLIST_SCATTER", 2, 0);
        ACLRT_LAUNCH_KERNEL(PostingBitListToSet)
        (blockDim, stream, deviceBitlistPostings, m_segmentByteSize, maxPostingLengthByte, deviceBitlistResultPostings,
         formerNum, formerLength, tailNum, tailLength, clearSparse, ablate, useScatter);
        if (!defer) {
            ret = aclrtSynchronizeStream(stream);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("aclrtSynchronizeStream fail,  error code is:" << ret);
                state = false;
            }
        }
    }
    if (defer) {
        *deferSync = stream;  // the caller syncs it and hands it back (DrainPreparedConversions)
    } else {
        StreamManager::GetInstance()->FreeStream(m_deviceId, stream);
    }
    // NPUR_DEBUG_BITLIST=1: verify the device conversion against a host recompute.
    if (npur_env::Present("NPUR_DEBUG_BITLIST") && state && !bitlistPostings.empty()) {
        uint32_t nshow = std::min<uint32_t>(6, static_cast<uint32_t>(bitlistPostings.size()));
        for (uint32_t j = 0; j < nshow; j++) {
            uint8_t hdr[8] = {0};
            CHECK_ACL_ONLY_LOG(aclrtMemcpy(hdr, 8, bitlistPostings[j], 8, ACL_MEMCPY_DEVICE_TO_HOST));
            uint32_t plen = *reinterpret_cast<uint32_t*>(hdr + 4);
            std::vector<uint8_t> in(8 + plen);
            CHECK_ACL_ONLY_LOG(
                aclrtMemcpy(in.data(), in.size(), bitlistPostings[j], in.size(), ACL_MEMCPY_DEVICE_TO_HOST));
            std::vector<uint8_t> outDev(m_segmentByteSize, 0);
            CHECK_ACL_ONLY_LOG(aclrtMemcpy(outDev.data(), m_segmentByteSize, bitlistResultPostings[j],
                                           m_segmentByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
            uint32_t K = (plen / 3) / static_cast<uint32_t>(sizeof(uint16_t));
            std::vector<uint8_t> outExp(m_segmentByteSize, 0);
            const uint32_t* offs = reinterpret_cast<const uint32_t*>(in.data() + 8);
            const uint16_t* data = reinterpret_cast<const uint16_t*>(in.data() + 8 + K * sizeof(uint32_t));
            bool inRange = true;
            for (uint32_t k = 0; k < K; k++) {
                uint32_t byteOff = offs[k];
                if (byteOff + sizeof(uint16_t) > m_segmentByteSize) {
                    inRange = false;
                    continue;
                }
                *reinterpret_cast<uint16_t*>(outExp.data() + byteOff) = data[k];
            }
            int diff = 0;
            for (uint32_t b = 0; b < m_segmentByteSize; b++)
                diff += (outDev[b] != outExp[b]);
            std::fprintf(stderr, "[BITLIST] j=%u plen=%u K=%u inRange=%d diffBytes=%d\n", j, plen, K, (int)inRange,
                         diff);
            if (diff) {
                std::string od, oe, off;
                char t[16];
                for (uint32_t b = 0; b < m_segmentByteSize; b++) {
                    std::snprintf(t, sizeof(t), "%02x", outDev[b]);
                    od += t;
                    std::snprintf(t, sizeof(t), "%02x", outExp[b]);
                    oe += t;
                }
                for (uint32_t k = 0; k < K && k < 24; k++) {
                    std::snprintf(t, sizeof(t), "%u,", offs[k]);
                    off += t;
                }
                std::fprintf(stderr, "[BITLIST]   offsets(byte)=%s\n[BITLIST]   dev=%s\n[BITLIST]   exp=%s\n",
                             off.c_str(), od.c_str(), oe.c_str());
            }
        }
    }
    releaseAddrs();
    if (PRINT_RES_ENABLE && NpuRetrieval::Logger::GetLogLevel() <= log4cplus::DEBUG_LOG_LEVEL) {
        PrintDevicePostings(devicePostings, postingsByteSize, postingsNum, segmentsNum, postings);
    }
    return state;
}

bool TextFilter::PostingBitListTransposeAndOffset(const std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs,
                                                  uint32_t segmentsNum, uint32_t postingsNum,
                                                  GmBlock& devicePostingsChunk) {
    std::vector<uint8_t*> postings;  // pointers to the converted bitsets, used by the later filtering
    postings.reserve(segmentsNum * postingsNum);
    for (size_t segIdx = 0; segIdx < segmentsNum; segIdx++) {
        for (size_t postingIdx = 0; postingIdx < postingsNum; postingIdx++) {
            postings.emplace_back((*(postingDeviceAddrs[postingIdx]))[segIdx] + 8);  // +8 bytes: the bitset address
        }
    }
    size_t postingsByteSize = segmentsNum * postingsNum * sizeof(uint8_t*);
    AllocateDevicePostings(postingsByteSize, devicePostingsChunk);
    uint8_t* devicePostings = reinterpret_cast<uint8_t*>(devicePostingsChunk.data);
    if (devicePostings == nullptr || devicePostingsChunk.size < postingsByteSize) {
        LOG_ERROR("devicePostings pool allocate failed");
        return false;
    }
    auto ret =
        aclrtMemcpy(devicePostings, postingsByteSize, postings.data(), postingsByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("devicePostings aclrtMemcpy fail, error code is:" << ret);
        return false;
    }
    return true;
}

void TextFilter::AllocateDevicePostings(size_t postingsByteSize, GmBlock& devicePostingsChunk) {
    if (PoolPostingsEnabled()) {
        GmMemoryManager::GetByDeviceId(m_deviceId)
            ->AllocateBlock(GmPoolName::TEXT_FILTER_POSTINGS_POOL, postingsByteSize, devicePostingsChunk);
        return;
    }
    uint8_t* raw = nullptr;
    auto ret = aclrtMalloc((void**)&raw, postingsByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("devicePostings aclrtMalloc fail, error code is:" << ret);
        devicePostingsChunk.data = nullptr;
        devicePostingsChunk.size = 0;
        return;
    }
    devicePostingsChunk.data = reinterpret_cast<char*>(raw);
    devicePostingsChunk.size = static_cast<uint32_t>(postingsByteSize);
}

void TextFilter::FreeDevicePostings(GmBlock& devicePostingsChunk) {
    if (PoolPostingsEnabled()) {
        GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(devicePostingsChunk);
        return;
    }
    if (devicePostingsChunk.data != nullptr) {
        CHECK_ACL_ONLY_LOG(aclrtFree(devicePostingsChunk.data));
        devicePostingsChunk.data = nullptr;
        devicePostingsChunk.size = 0;
    }
}

bool TextFilter::BitmapTextFilter(const std::vector<uint32_t>& postExpr, TextFilterContextData& context,
                                  uint8_t*& devicePostings, uint8_t* resultInDevice) {
    uint32_t postExprByteSize = postExpr.size() * sizeof(uint32_t);
    uint32_t* devicePostExpr;
    uint16_t* resultStack;
    GmBlock resultStackChunk{GmPoolName::TEXT_FILTER_STACK_POOL, nullptr, 0};
    {
        size_t resultStackByteSize = context.segmentNum * context.segmentLength * sizeof(uint16_t) * context.opNum;
        {
            RecordGuard guard{"TextFilter_BitmapTextFilter_postExpr_malloc_h2d"};
            auto ret = aclrtMalloc((void**)&devicePostExpr, postExprByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("devicePostExpr aclrtMalloc fail, error code is:" << ret);
                return false;
            }
            ret = aclrtMemcpy(devicePostExpr, postExprByteSize, postExpr.data(), postExprByteSize,
                              ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("devicePostExpr aclrtMemcpy fail,  error code is:" << ret);
                CHECK_ACL_ONLY_LOG(aclrtFree(devicePostExpr));
                return false;
            }
        }
        {
            RecordGuard guard{"TextFilter_BitmapTextFilter_resultStack_allocate"};
            GmMemoryManager::GetByDeviceId(m_deviceId)
                ->AllocateBlock(GmPoolName::TEXT_FILTER_STACK_POOL, resultStackByteSize, resultStackChunk);
        }
        resultStack = reinterpret_cast<uint16_t*>(resultStackChunk.data);
        if (resultStack == nullptr || resultStackChunk.size < resultStackByteSize) {
            LOG_ERROR("GmMemoryManager allocate failed.");
            GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(resultStackChunk);
            CHECK_ACL_ONLY_LOG(aclrtFree(devicePostExpr));
            return false;
        }
    }

    aclrtStream stream = nullptr;
    stream = StreamManager::GetInstance()->GetStream(m_deviceId);
    {
        RecordGuard guard{"TextFilter_BitmapTextFilter_Kernel"};
        ACLRT_LAUNCH_KERNEL(BitmapTextFilter)
        (FLAGS_full_recall_text_filter_block_dim, stream, devicePostings, devicePostExpr, resultStack, &context,
         resultInDevice);
        CHECK_ACL_ONLY_LOG(aclrtSynchronizeStream(stream));
    }
    StreamManager::GetInstance()->FreeStream(m_deviceId, stream);

    if (PRINT_RES_ENABLE && NpuRetrieval::Logger::GetLogLevel() <= log4cplus::INFO_LOG_LEVEL) {
        PrintFilterRes(resultInDevice);
    }

    {
        RecordGuard guard{"TextFilter_BitmapTextFilter_teardown_free"};
        CHECK_ACL_ONLY_LOG(aclrtFree(devicePostExpr));
        GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(resultStackChunk);
    }
    return true;
}

uint32_t TextFilter::GetMaxPostingLength(const uint32_t docNumPerSegment, const double denseness) {
    // Bitlist encoding: a u32 offset plus a u16 mask per unit (6 bytes); SPARSE_PACKED folds
    // them into one u32, so 4.
    const uint32_t bytesPerUnit = SparsePostingsArePacked() ? 4 : 6;
    uint32_t maxPostingLength = std::min(static_cast<uint32_t>(docNumPerSegment * denseness * bytesPerUnit),
                                         docNumPerSegment / 8 * (bytesPerUnit / 2));
    LOG_DEBUG("bitlist, denseness:" << denseness << " maxPostingLength:" << maxPostingLength);
    return maxPostingLength;
}

void TextFilter::CountBitlistTypes(const std::vector<std::vector<uint8_t>*>& postingTypes,
                                   const std::vector<uint8_t>* sparseDirect, uint32_t& withMask, uint32_t& all) {
    withMask = 0;
    all = 0;
    for (size_t postingIdx = 0; postingIdx < postingTypes.size(); postingIdx++) {
        const auto& typesPtr = postingTypes[postingIdx];
        if (typesPtr == nullptr) {
            LOG_ERROR("typesPtr is null");
            continue;
        }
        const bool masked =
            sparseDirect != nullptr && postingIdx < sparseDirect->size() && (*sparseDirect)[postingIdx] != 0;
        uint32_t n = 0;
        for (const uint8_t& type : *typesPtr) {
            if (type != 0) {
                n += 1;
            }
        }
        all += n;
        if (!masked) {
            withMask += n;
        }
    }
}

uint32_t TextFilter::GetBitlistTypeNum(const std::vector<std::vector<uint8_t>*>& postingTypes,
                                       const std::vector<uint8_t>* sparseDirect) {
    uint32_t blockNum = 0;
    for (size_t postingIdx = 0; postingIdx < postingTypes.size(); postingIdx++) {
        const auto& typesPtr = postingTypes[postingIdx];
        if (typesPtr == nullptr) {
            LOG_ERROR("typesPtr is null");
            continue;
        }
        if (sparseDirect != nullptr && postingIdx < sparseDirect->size() && (*sparseDirect)[postingIdx] != 0) {
            continue;
        }
        for (const uint8_t& type : *typesPtr) {
            if (type != 0) {
                blockNum += 1;
            }
        }
    }
    return blockNum;
}

void TextFilter::CreateContextData(const uint32_t exprLen, const uint32_t opNum, const int32_t postingNum,
                                   TextFilterContextData& context) {
    context.exprLen = exprLen;
    context.opNum = opNum;
    context.postingNum = postingNum;

    context.segmentLength = m_segmentLength;
    context.segmentNum = m_dataTable->GetSegmentNum();
    // segmentByteSize/sizeof(uint16_t)/tileNum/BUFFER_NUM must be a multiple of 16, so every
    // compute/copy meets the 32-byte requirement; segmentByteSize is already 32-aligned.
    context.tileNum = 1;
    context.flags = EffectiveFilterFlags();
}

bool TextFilter::CheckPostingTypes(const std::vector<std::vector<uint8_t>*>& postingTypes, uint32_t segmentsNum,
                                   uint32_t postingsNum) {
    if (postingTypes.size() != postingsNum) {
        LOG_ERROR("invalid posting types, not equal to postingsNum");
        return false;
    }
    for (const auto& postingTypeTypePtr : postingTypes) {
        if (postingTypeTypePtr == nullptr) {
            LOG_ERROR("invalid posting types, has nullptr element");
            return false;
        }
        if (postingTypeTypePtr->size() != segmentsNum) {
            LOG_ERROR("invalid posting types, not equal to segmentsNum");
            return false;
        }
    }
    return true;
}

bool TextFilter::CheckPostExprNotNode(const std::vector<uint32_t>& postExpr, uint32_t& idx, uint32_t& postingRemained,
                                      uint32_t& numInStack) {
    static const uint32_t postingNumOffset = 1;
    static const uint32_t stackNumOffset = 2;
    if (postExpr.size() <= (idx + stackNumOffset)) {
        LOG_ERROR("invalid not node expr, postExpr invalid");
        return false;
    }
    uint32_t postingNum = postExpr[idx + postingNumOffset];
    uint32_t stackNum = postExpr[idx + stackNumOffset];
    if (postingNum + stackNum != 1) {
        LOG_ERROR("invalid not node expr, postingNum:0, stackNum:0");
        return false;
    }
    if (postingRemained < postingNum) {
        LOG_ERROR("invalid not node expr, postingNum is not enough");
        return false;
    }
    if (numInStack < stackNum) {
        LOG_ERROR("invalid not node expr, stackNum is not enough");
        return false;
    }
    numInStack -= stackNum;
    postingRemained -= postingNum;
    numInStack += 1;
    idx += 2;  // 2 params
    return true;
}

bool TextFilter::CheckPostExprAndOrNode(const std::vector<uint32_t>& postExpr, uint32_t& idx, uint32_t& postingRemained,
                                        uint32_t& numInStack) {
    static const uint32_t postingNumOffset = 1;
    static const uint32_t stackNumOffset = 2;
    if (postExpr.size() <= (idx + stackNumOffset)) {
        LOG_ERROR("invalid and/or node expr, postExpr invalid");
        return false;
    }
    uint32_t postingNum = postExpr[idx + postingNumOffset];
    uint32_t stackNum = postExpr[idx + stackNumOffset];
    if (postingNum + stackNum == 0) {
        LOG_ERROR("invalid and/or node expr, postingNum:0, stackNum:0");
        return false;
    }
    if (postingRemained < postingNum) {
        LOG_ERROR("invalid and/or node expr, postingNum is not enough");
        return false;
    }
    if (numInStack < stackNum) {
        LOG_ERROR("invalid and/or node expr, stackNum is not enough");
        return false;
    }
    numInStack -= stackNum;
    postingRemained -= postingNum;
    numInStack += 1;
    idx += 2;  // 2 params
    return true;
}

bool TextFilter::CheckPostExprConjNode(const std::vector<uint32_t>& postExpr, uint32_t& idx, uint32_t& postingRemained,
                                       uint32_t& numInStack) {
    size_t postExprLength = postExpr.size();
    if (postExprLength <= (idx + 1)) {
        LOG_ERROR("invalid conj node expr, postExpr invalid");
        return false;
    }
    uint32_t postingNums = 0;
    uint32_t partNum = postExpr[idx + 1];
    if (partNum == 0) {
        LOG_ERROR("invalid conj node expr, partNum is 0");
        return false;
    }
    if (postExprLength <= (idx + 1 + partNum)) {
        LOG_ERROR("invalid conj node expr, postExpr invalid");
        return false;
    }
    for (size_t i = 2; i < partNum + 2; ++i) {  // postingNum entries start 2 positions after opType
        const uint32_t& postingNum = postExpr[idx + i];
        if (postingNum == 0) {
            LOG_ERROR("invalid conj node expr, postingNum is 0");
            return false;
        }
        postingNums += postExpr[idx + i];
    }

    if (postingRemained < postingNums) {
        LOG_ERROR("invalid conj node expr, postingNum is not enough");
        return false;
    }
    postingRemained -= postingNums;
    numInStack += 1;
    idx = idx + partNum + 1;
    return true;
}

bool TextFilter::CheckPostExpr(const std::vector<uint32_t>& postExpr, const uint32_t postingNums) {
    uint32_t numInStack = 0;
    uint32_t postingRemained = postingNums;
    for (uint32_t exprIdx = 0; exprIdx < postExpr.size(); exprIdx++) {
        uint32_t opType = postExpr[exprIdx];
        bool res = true;
        switch (opType) {
            case FilterOpType::CONJUNCTION: {
                res = CheckPostExprConjNode(postExpr, exprIdx, postingRemained, numInStack);
                break;
            }
            case FilterOpType::AND:
            case FilterOpType::OR: {
                res = CheckPostExprAndOrNode(postExpr, exprIdx, postingRemained, numInStack);
                break;
            }
            case FilterOpType::NOT: {
                res = CheckPostExprNotNode(postExpr, exprIdx, postingRemained, numInStack);
                break;
            }
            default: {
                LOG_ERROR("invalid opType:" << opType);
                return false;
            }
        }
        if (!res) {
            LOG_ERROR("invalid postExpr");
            return false;
        }
    }
    if (postingRemained != 0 || numInStack != 1) {
        LOG_ERROR("invalid postExpr, postingRemained:" << postingRemained << " numInStack:" << numInStack);
        return false;
    }
    if (ExprStatsEnabled()) {
        NoteExprShape(postExpr);
    }
    return true;
}

}  // namespace NpuRetrieval
