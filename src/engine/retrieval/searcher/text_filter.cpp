#include "text_filter.h"
#include <sstream>
#include <iomanip>
#include <chrono>
#include "acl/acl.h"
#include "ascend_device/aclrtlaunch_BitmapTextFilter.h"
#include "ascend_device/aclrtlaunch_PostingBitListToSet.h"
#include "src/utils/logger.h"
#include "src/utils/performance_recorder.h"
#include "src/full_recall/core/constant_definition.h"
#include "src/full_recall/core/log_definition.h"
#include "src/full_recall/retrieval/searcher/runtime/stream_manager.h"

namespace NpuRetrieval {
static const uint32_t PRINT_TARGET_SEGMENT_INDEX = 0;  // print info for this target segment
static const bool PRINT_RES_ENABLE = false;

// A/B switch for the devicePostings pointer-table allocation. Default: pooled
// (TEXT_FILTER_POSTINGS_POOL). NPUR_POOL_POSTINGS=0 restores the pre-pooling
// baseline (raw per-query aclrtMalloc/aclrtFree) so both paths can be profiled
// from the same binary. Read once; constant for the process lifetime, so an
// Allocate and its matching Free always agree on which path to take.
static bool PoolPostingsEnabled() {
    static const bool enabled = []() {
        const char* v = std::getenv("NPUR_POOL_POSTINGS");
        return v == nullptr || std::string(v) != "0";
    }();
    return enabled;
}

bool TextFilter::BatchCompute(const std::vector<QueryNode*> queryTrees, uint8_t* resultInDevice,
                              const std::vector<FilterExpr>* prepared) {
    RecordGuard guard{"TextFilter BatchCompute"};
    LOG_DEBUG("TextFilter::BatchCompute- queryTrees.size():" << queryTrees.size());
    if (prepared != nullptr && prepared->size() != queryTrees.size()) {
        LOG_ERROR("prepared expression count " << prepared->size() << " != queryTrees count " << queryTrees.size());
        return false;
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
        // Adjust queryTree to drop invalid nodes. GetPostingFields is identical
        // across shards, so we can lock here to avoid concurrency issues; otherwise
        // we would have to copy the queryTree.
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

        if (!queryTree->GetPostOrderExpression(*m_dataTable, out.postExpr, out.postingTypes, out.postingDeviceAddrs,
                                               out.opNum)) {
            LOG_ERROR("get post order expression failed");
            return false;
        }
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

bool TextFilter::BatchPreparePostings(std::vector<FilterExpr>& exprs) {
    RecordGuard guard{"TextFilter BatchPreparePostings"};
    for (auto& e : exprs) {
        // emptyTree filters via an all-ones memset in Compute -- no postings to build.
        if (!e.valid || e.emptyTree || e.postingsPrepared) {
            continue;
        }
        // Pre-seed the pool name so a failure still FreeBlocks into the right pool.
        e.devicePostingsChunk = GmBlock{GmPoolName::TEXT_FILTER_POSTINGS_POOL, nullptr, 0};
        uint32_t postingsNum = e.postingDeviceAddrs.size();
        if (!PostingBitListToSet(e.postingTypes, e.postingDeviceAddrs, postingsNum, e.bitsetPostingsChunk,
                                 e.devicePostingsChunk)) {
            LOG_ERROR("BatchPreparePostings: PostingBitListToSet failed");
            FreeDevicePostings(e.devicePostingsChunk);
            return false;
        }
        e.postingsPrepared = true;
    }
    return true;
}

void TextFilter::FreePreparedPostings(std::vector<FilterExpr>& exprs) {
    for (auto& e : exprs) {
        if (!e.postingsPrepared) {
            continue;
        }
        FreeDevicePostings(e.devicePostingsChunk);
        // bitsetPostingsChunk is null on the all-bitset transpose path; FreeBlock handles that.
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
    const uint32_t opNum = prepared->opNum;

    // debug logging
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

    // ============ PostingBitListToSet ============
    // Use the pointer table prepared during the scorer window (overlap path) if present;
    // else build it here. `ownPostings` = this Compute allocated the chunks and must free
    // them (in the overlap path the caller owns them via FreePreparedPostings).
    GmBlock bitsetPostingsChunk{};
    GmBlock devicePostingsChunk{GmPoolName::TEXT_FILTER_POSTINGS_POOL, nullptr, 0};
    uint8_t* devicePostings = nullptr;
    bool ownPostings = false;
    if (prepared != nullptr && prepared->postingsPrepared) {
        devicePostings = reinterpret_cast<uint8_t*>(prepared->devicePostingsChunk.data);
    } else {
        uint32_t postingsNum = postingDeviceAddrs.size();
        if (!PostingBitListToSet(postingTypes, postingDeviceAddrs, postingsNum, bitsetPostingsChunk,
                                 devicePostingsChunk)) {
            LOG_ERROR("posting bitList to set failed");
            FreeDevicePostings(devicePostingsChunk);
            return false;
        }
        devicePostings = reinterpret_cast<uint8_t*>(devicePostingsChunk.data);
        ownPostings = true;
    }
    // ============BitmapTextFilter============
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

bool TextFilter::PostingBitListToSet(const std::vector<std::vector<uint8_t>*>& postingTypes,
                                     const std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs,
                                     uint32_t postingsNum, GmBlock& bitsetPostingsChunk, GmBlock& devicePostingsChunk) {
    RecordGuard guard{"TextFilter_PostingBitListToSet"};
    uint32_t docNumPerSegment = m_dataTable->GetDocNumPerSegment();
    const uint32_t segmentsNum = m_dataTable->GetSegmentNum();
    const uint32_t blockNum = GetBitlistTypeNum(postingTypes);  // number of memory blocks to allocate
    LOG_DEBUG("deviceId:" << m_deviceId << " postingsNum:" << postingsNum << " docNumPerSegment:" << docNumPerSegment
                          << " segmentsNum:" << segmentsNum << " segmentByteSize:" << m_segmentByteSize
                          << " blockNum:" << blockNum);
    if (blockNum == 0) {  // no list->set needed; just transpose the pointer array and offset by 8 bytes
        return PostingBitListTransposeAndOffset(postingDeviceAddrs, segmentsNum, postingsNum, devicePostingsChunk);
    }

    if (!CheckPostingTypes(postingTypes, segmentsNum, postingsNum)) {
        LOG_ERROR("invalid posting types");
        return false;
    }

    // memory used for the list->set conversion
    uint8_t* bitsetPostings = nullptr;
    {
        RecordGuard guard{"TextFilter_PostingBitListToSet_GmMemoryPool_Allocate"};
        uint32_t bitsetPostingsByteSize = blockNum * m_segmentByteSize;
        GmMemoryManager::GetByDeviceId(m_deviceId)
            ->AllocateBlock(GmPoolName::TEXT_FILTER_BITLIST2SET_POOL, bitsetPostingsByteSize, bitsetPostingsChunk);
        bitsetPostings = reinterpret_cast<uint8_t*>(bitsetPostingsChunk.data);
        if (bitsetPostings == nullptr || bitsetPostingsChunk.size < bitsetPostingsByteSize) {
            LOG_ERROR("GmMemoryManager allocate failed.");
            return false;
        }
    }

    std::vector<uint8_t*> postings;               // pointers to the converted bitsets, used by the later filtering
    std::vector<uint8_t*> bitlistPostings;        // addresses of the bitlists that need conversion
    std::vector<uint8_t*> bitlistResultPostings;  // addresses of the converted bitlist results
    size_t bitlistNum = 0;
    postings.reserve(segmentsNum * postingsNum);
    bitlistPostings.reserve(blockNum);
    bitlistResultPostings.reserve(blockNum);
    for (size_t segIdx = 0; segIdx < segmentsNum; segIdx++) {
        for (size_t postingIdx = 0; postingIdx < postingsNum; postingIdx++) {
            auto& types = *(postingTypes[postingIdx]);
            if (types[segIdx] == 0) {
                postings.emplace_back((*(postingDeviceAddrs[postingIdx]))[segIdx] + 8);  // +8 bytes: the bitset address
                continue;
            }
            if (bitlistNum >= blockNum) {
                LOG_ERROR("bitlistNum out of range");
                GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(bitsetPostingsChunk);
                bitsetPostingsChunk.data = nullptr;
                bitsetPostingsChunk.size = 0;
                return false;
            }
            postings.emplace_back(bitsetPostings + bitlistNum * m_segmentByteSize);
            bitlistPostings.emplace_back((*(postingDeviceAddrs[postingIdx]))[segIdx]);
            bitlistResultPostings.emplace_back(bitsetPostings + bitlistNum * m_segmentByteSize);
            bitlistNum += 1;
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
        LOG_ERROR("devicePostings aclrtMemcpy fail,  error code is:" << ret);
        return false;
    }
    // prepare the bitlist->bitset kernel inputs
    uint8_t* deviceBitlistPostings = nullptr;
    uint8_t* deviceBitlistResultPostings = nullptr;
    size_t bitlistPostingsByteSize = bitlistPostings.size() * sizeof(uint8_t*);
    ret = aclrtMalloc((void**)&deviceBitlistPostings, bitlistPostingsByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("deviceBitlistPostings aclrtMalloc fail, error code is:" << ret);
        return false;
    }
    ret = aclrtMemcpy(deviceBitlistPostings, bitlistPostingsByteSize, bitlistPostings.data(), bitlistPostingsByteSize,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("deviceBitlistPostings aclrtMemcpy fail,  error code is:" << ret);
        CHECK_ACL_ONLY_LOG(aclrtFree(deviceBitlistPostings));
        return false;
    }
    ret = aclrtMalloc((void**)&deviceBitlistResultPostings, bitlistPostingsByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("deviceBitlistResultPostings aclrtMalloc fail, error code is:" << ret);
        return false;
    }
    ret = aclrtMemcpy(deviceBitlistResultPostings, bitlistPostingsByteSize, bitlistResultPostings.data(),
                      bitlistPostingsByteSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("deviceBitlistResultPostings aclrtMemcpy fail,  error code is:" << ret);
        CHECK_ACL_ONLY_LOG(aclrtFree(deviceBitlistPostings));
        CHECK_ACL_ONLY_LOG(aclrtFree(deviceBitlistResultPostings));
        return false;
    }
    // tiling across cores
    uint32_t blockDim = FLAGS_full_recall_text_filter_block_dim;
    uint32_t alignNum = blockDim;
    uint32_t partIdxAligned = ((bitlistNum + alignNum - 1) / alignNum) * alignNum;
    uint32_t formerNum = bitlistNum % blockDim;
    uint32_t tailNum = blockDim - formerNum;
    uint32_t tailLength = bitlistNum / blockDim;
    uint32_t formerLength = partIdxAligned / blockDim;
    LOG_INFO(" bitlistNum:" << bitlistNum << " formerNum:" << formerNum << " formerLength:" << formerLength
                            << " tailNum:" << tailNum << " tailLength:" << tailLength);
    // bitlist_denseness_threshold could later be read from the index
    uint32_t maxPostingLengthByte =
        GetMaxPostingLength(docNumPerSegment, FLAGS_full_recall_bitlist_denseness_threshold);
    bool state = true;
    aclrtStream stream = StreamManager::GetInstance()->GetStream(m_deviceId);
    {
        RecordGuard guard{"TextFilter_PostingBitListToSet_Kernel"};
        ACLRT_LAUNCH_KERNEL(PostingBitListToSet)
        (blockDim, stream, deviceBitlistPostings, m_segmentByteSize, maxPostingLengthByte, deviceBitlistResultPostings,
         formerNum, formerLength, tailNum, tailLength);
        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("aclrtSynchronizeStream fail,  error code is:" << ret);
            state = false;
        }
    }
    StreamManager::GetInstance()->FreeStream(m_deviceId, stream);
    // [DEBUG] env-gated: verify the device bitlist->bitset conversion against a
    // host recompute (proven-correct algorithm). Set NPUR_DEBUG_BITLIST=1.
    if (std::getenv("NPUR_DEBUG_BITLIST") != nullptr && state && !bitlistPostings.empty()) {
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
    CHECK_ACL_ONLY_LOG(aclrtFree(deviceBitlistPostings));
    CHECK_ACL_ONLY_LOG(aclrtFree(deviceBitlistResultPostings));
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
    // Baseline (NPUR_POOL_POSTINGS=0): raw per-query device malloc, matched by the
    // aclrtFree in FreeDevicePostings. On failure leave the chunk empty so the
    // caller's null/size check rejects it.
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
    // Baseline: raw free. No-op when the chunk was never allocated (early error
    // paths leave data == nullptr), matching the pool FreeBlock no-op.
    if (devicePostingsChunk.data != nullptr) {
        CHECK_ACL_ONLY_LOG(aclrtFree(devicePostingsChunk.data));
        devicePostingsChunk.data = nullptr;
        devicePostingsChunk.size = 0;
    }
}

bool TextFilter::BitmapTextFilter(const std::vector<uint32_t>& postExpr, TextFilterContextData& context,
                                  uint8_t*& devicePostings, uint8_t* resultInDevice) {
    // copy postExpr
    uint32_t postExprByteSize = postExpr.size() * sizeof(uint32_t);
    uint32_t* devicePostExpr;
    // prepare resultStack
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
    // Posting denseness = (hit docs for a token) / (total docs). Below the threshold we use
    // bitlist, otherwise bitset. For the bitlist type the posting encoding splits into a
    // posting-offset section and a posting-data section:
    //   posting data:   bitset form, uint16 unit, theoretical max = docNumPerSegment*denseness*sizeof(u16)
    //   posting offset: byte offsets of each data entry, uint32 unit, max = docNumPerSegment*denseness*sizeof(u32)
    uint32_t maxPostingLength =
        std::min(static_cast<uint32_t>(docNumPerSegment * denseness * 6), docNumPerSegment / 8 * 3);
    LOG_DEBUG("bitlist, denseness:" << denseness << " maxPostingLength:" << maxPostingLength);
    return maxPostingLength;
}

uint32_t TextFilter::GetBitlistTypeNum(const std::vector<std::vector<uint8_t>*>& postingTypes) {
    uint32_t blockNum = 0;
    for (const auto& typesPtr : postingTypes) {
        if (typesPtr == nullptr) {
            LOG_ERROR("typesPtr is null");
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
    // tile info: depends on segmentByteSize. segmentByteSize/sizeof(uint16_t)/tileNum/BUFFER_NUM
    // must be a multiple of 16 so each compute/copy meets the 32-byte-multiple requirement;
    // segmentByteSize is already a multiple of 32.
    context.tileNum = 1;
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
    return true;
}

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
