#include "full_recall_searcher.h"
#include <cstdlib>
#include <string>
#include <thread>
#include "acl/acl.h"
#include "src/utils/env_switch.h"
#include "src/utils/logger.h"
#include "src/full_recall/core/constant_definition.h"
#include "src/full_recall/core/log_definition.h"
#include "full_recall/core/number/trans_number.h"
#include "configuration/develop_configuration.h"
#include "src/utils/performance_recorder.h"
#include "full_recall/retrieval/searcher/runtime/gm_memory_manager.h"

namespace NpuRetrieval {

// NPUR_PARALLEL_SCORE_FILTER=1: scorer (Cube) and text filter (Vector) on two host threads, so
// they co-reside. Off by default: p50 2.66 -> 2.40ms but p99 3.82 -> 6.83ms.
static bool ParallelScoreFilterEnabled() {
    static const bool enabled = npur_env::OnExact("NPUR_PARALLEL_SCORE_FILTER");
    return enabled;
}

// NPUR_OVERLAP_POSTEXPR (default 1): hide the filter's postfix-expression build inside the scorer
// kernel's window. Ignored when NPUR_PARALLEL_SCORE_FILTER=1.
//   1  the build runs inside the scorer kernel's execution window.
//   2  same calls in the same order, Sync right after the launch (control arm).
//   0  pre-overlap baseline: both tasks through the executor.
static int OverlapPostExprMode() {
    static const int mode = []() {
        const char* v = std::getenv("NPUR_OVERLAP_POSTEXPR");
        if (v == nullptr) {
            return 1;
        }
        const std::string s(v);
        if (s == "0") {
            return 0;
        }
        if (s == "2") {
            return 2;
        }
        return 1;
    }();
    return mode;
}

// NPUR_OVERLAP_POSTING=1: also build the posting pointer table inside the scorer's window.
static bool OverlapPostingMode() {
    static const bool enabled = npur_env::On("NPUR_OVERLAP_POSTING");
    return enabled;
}

// NPUR_EARLY_FILTER_PREP=1: also run TextFilter::FusedPrepare inside the scorer's window. Needs
// NPUR_BATCH_FILTER, NPUR_OVERLAP_POSTING and ReadyForEarlyPrep.
static bool EarlyFilterPrepMode() {
    static const bool enabled = npur_env::On("NPUR_EARLY_FILTER_PREP");
    return enabled;
}

// NPUR_SHARE_FILTER_STREAM=1 (needs NPUR_BATCH_FILTER and NPUR_BATCH_AGGREGATE): hand the filter
// kernels' stream to the aggregator, whose first sync waits for both. Their time is then
// reported inside AggrAndTopK_Aggregator_NPU.
static bool ShareFilterStreamMode() {
    static const bool enabled = npur_env::On("NPUR_SHARE_FILTER_STREAM");
    return enabled;
}

// NPUR_DEFER_CONV_SYNC=1 (needs NPUR_OVERLAP_POSTING and NPUR_BITLIST_ONE_H2D): a conversion
// launched in the scorer window is waited on after the scorer sync, before any filter kernel.
static bool DeferConvSyncMode() {
    static const bool enabled = npur_env::On("NPUR_DEFER_CONV_SYNC");
    return enabled;
}

// NPUR_OVERLAP_FILTER_KERNEL=1: also LAUNCH the filter kernels inside the scorer window. Needs
// NPUR_STREAM_POOL for a second stream, implies the early prepare, and drains any pending
// conversion first since the filter reads what it writes.
static bool OverlapFilterKernelMode() {
    static const bool enabled = npur_env::On("NPUR_OVERLAP_FILTER_KERNEL");
    return enabled;
}

// FusedPrepare stays cheap only when every query already has a posting table and none is empty.
static bool ReadyForEarlyPrep(const std::vector<FilterExpr>& exprs) {
    for (const FilterExpr& e : exprs) {
        if (!e.valid || e.emptyTree || !e.postingsPrepared) {
            return false;
        }
    }
    return true;
}

void FullRecallSearcher::AddQuery(QueryNode* queryTrees, const std::vector<float>& queryVectors, uint32_t topK) {
    m_queryVectors.reserve(m_queryVectors.size() + queryVectors.size());
    float vectorSum = 0.0f;
    for (float data : queryVectors) {
        m_queryVectors.emplace_back(Float32ToFloat16(data));
        vectorSum += data;
        LOG_DEBUG("m_queryVectors data = " << data);
    }
    LOG_DEBUG("query sum=" << vectorSum);
    m_topKs.emplace_back(topK);
    m_queries.emplace_back(queryTrees);
    m_dimension = queryVectors.size();
}

bool FullRecallSearcher::BatchSearch(std::vector<std::shared_ptr<FullRecallResult>>& fullRecallResults,
                                     const std::string& vectorFieldName, bool isMultiShard) {
    RecordGuard guard{"BatchSearch"};
    LOG_DEBUG("FullRecallSearcher BatchSearch start");
    if (m_dataTable == nullptr) {
        LOG_ERROR("m_dataTable is nullptr, vectorFieldName" << vectorFieldName);
        return false;
    }
    uint32_t docNumPerSegment = m_dataTable->GetDocNumPerSegment();
    uint32_t segmentsNum = m_dataTable->GetSegmentNum();

    GmBlock resultChunk{};        // holds the scoring / aggregation result
    GmBlock filterResultChunk{};  // holds the filter (targeting) result
    auto ret = aclrtSetDevice(m_deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtSetDevice fail, deviceId is:" << m_deviceId << " error code is:" << ret);
        return false;
    }
    // The offline bitset unit is uint16, so this is always divisible by 8 (checked at load).
    uint32_t allResultsByteSize =
        segmentsNum * docNumPerSegment / 8 * m_queries.size();  // all filter results for the batch
    GmMemoryManager::GetByDeviceId(m_deviceId)
        ->AllocateBlock(GmPoolName::TEXT_FILTER_RESULT_POOL, allResultsByteSize, filterResultChunk);
    uint8_t* filterResultInDevice = reinterpret_cast<uint8_t*>(filterResultChunk.data);
    if (filterResultInDevice == nullptr) {
        LOG_ERROR("GmMemoryManager allocate failed.");
        CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
        return false;
    }

    std::shared_ptr<LogContext> logContext = std::make_shared<LogContext>();
    auto asyncContext = m_executor->CreateExecuteContext(*logContext);

    auto scoreTask = [this, &resultChunk, &vectorFieldName]() -> ErrorCode::ResultType {
        auto ret = aclrtSetDevice(m_deviceId);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("aclrtSetDevice fail, deviceId is:" << m_deviceId << " error code is:" << ret);
            return ErrorCode::ResultType::FAIL;
        }
        if (!m_vectorScorerMmad.BatchCompute(m_queryVectors, m_dimension, vectorFieldName, resultChunk)) {
            LOG_ERROR("m_vectorScorerMmad.BatchCompute failed.");
            CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
            return ErrorCode::ResultType::FAIL;
        }
        CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
        return ErrorCode::ResultType::SUCCESS;
    };

    auto filterTask = [this, &filterResultInDevice]() -> ErrorCode::ResultType {
        auto ret = aclrtSetDevice(m_deviceId);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("aclrtSetDevice fail, deviceId is:" << m_deviceId << " error code is:" << ret);
            return ErrorCode::ResultType::FAIL;
        }
        if (!m_textFilter.BatchCompute(m_queries, filterResultInDevice)) {
            LOG_ERROR("textFilter.BatchCompute failed.");
            CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
            return ErrorCode::ResultType::FAIL;
        }
        CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
        return ErrorCode::ResultType::SUCCESS;
    };

    bool isSuccess = true;
    if (ParallelScoreFilterEnabled()) {
        ErrorCode::ResultType scoreRet = ErrorCode::ResultType::FAIL;
        std::thread scoreThread([&scoreRet, &scoreTask]() { scoreRet = scoreTask(); });
        ErrorCode::ResultType filterRet = filterTask();
        scoreThread.join();
        if (scoreRet != ErrorCode::ResultType::SUCCESS || filterRet != ErrorCode::ResultType::SUCCESS) {
            isSuccess = false;
        }
    } else if (OverlapPostExprMode() != 0) {
        const bool overlap = OverlapPostExprMode() == 1;
        const bool overlapPosting = overlap && OverlapPostingMode();
        ScorerLaunch launch;
        std::vector<FilterExpr> exprs;
        isSuccess =
            m_vectorScorerMmad.BatchComputeLaunch(m_queryVectors, m_dimension, vectorFieldName, resultChunk, launch);
        if (!isSuccess) {
            LOG_ERROR("m_vectorScorerMmad.BatchComputeLaunch failed.");
        } else if (overlap && !m_textFilter.BatchPrepareExpr(m_queries, exprs)) {
            LOG_ERROR("m_textFilter.BatchPrepareExpr failed.");
            isSuccess = false;
        } else if (overlapPosting && !m_textFilter.BatchPreparePostings(exprs)) {
            LOG_ERROR("m_textFilter.BatchPreparePostings failed.");
            isSuccess = false;
        }
        // Sync even when the build failed: skipping it would free resultChunk out from under a
        // kernel still writing it.
        if (!m_vectorScorerMmad.BatchComputeSync(launch)) {
            LOG_ERROR("m_vectorScorerMmad.BatchComputeSync failed.");
            isSuccess = false;
        }
        if (isSuccess && !overlap && !m_textFilter.BatchPrepareExpr(m_queries, exprs)) {
            LOG_ERROR("m_textFilter.BatchPrepareExpr failed.");
            isSuccess = false;
        }
        if (isSuccess && !m_textFilter.BatchCompute(m_queries, filterResultInDevice, &exprs)) {
            LOG_ERROR("textFilter.BatchCompute failed.");
            isSuccess = false;
        }
        if (overlapPosting) {
            m_textFilter.FreePreparedPostings(exprs);
        }
    } else {
        asyncContext->AddTask(scoreTask);
        asyncContext->AddTask(filterTask);
        asyncContext->Wait([&isSuccess](ErrorCode::ResultType ret) {
            if (ret != ErrorCode::ResultType::SUCCESS) {
                isSuccess = false;
            }
        });
    }
    if (isSuccess) {
        uint8_t* resultInDevice = reinterpret_cast<uint8_t*>(resultChunk.data);  // scoring / aggregation result
        isSuccess = RunAggregation(filterResultInDevice, resultInDevice, fullRecallResults, isMultiShard);
    } else {
        LOG_ERROR("score and filter fail, please check");
    }
    GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(resultChunk);
    GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(filterResultChunk);
    CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
    return isSuccess;
}

bool FullRecallSearcher::OneFilter(GmBlock& filterResultChunk) {
    RecordGuard guard{"FullRecallSearcher OneFilter"};
    if (m_dataTable == nullptr) {
        LOG_ERROR("m_dataTable is nullptr, deviceId is:" << m_deviceId);
        return false;
    }
    if (m_queries.size() != 1) {
        LOG_ERROR("m_queries size is not 1, deviceId is:" << m_deviceId);
        return false;
    }
    auto ret = aclrtSetDevice(m_deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtSetDevice fail, deviceId is:" << m_deviceId << " error code is:" << ret);
        return false;
    }
    uint32_t docNumPerSegment = m_dataTable->GetDocNumPerSegment();
    uint32_t segmentsNum = m_dataTable->GetSegmentNum();
    // The offline bitset unit is uint16, so this is always divisible by 8 (checked at load).
    uint32_t resultsByteSize = segmentsNum * docNumPerSegment / 8;  // 8 bits per byte
    GmMemoryManager::GetByDeviceId(m_deviceId)
        ->AllocateBlock(GmPoolName::TEXT_FILTER_RESULT_POOL, resultsByteSize, filterResultChunk);
    uint8_t* filterResultInDevice = reinterpret_cast<uint8_t*>(filterResultChunk.data);
    if (filterResultInDevice == nullptr) {
        LOG_ERROR("GmMemoryManager allocate failed.");
        CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
        return false;
    }
    if (!m_textFilter.Compute(m_queries[0], filterResultInDevice, resultsByteSize)) {
        LOG_ERROR("textFilter.BatchCompute failed.");
        GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(filterResultChunk);
        CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
        return false;
    }
    CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
    return true;
}

bool FullRecallSearcher::BatchScore(const std::string& vectorFieldName, GmBlock& resultChunk) {
    RecordGuard guard{"FullRecallSearcher BatchScore"};
    if (m_dataTable == nullptr) {
        LOG_ERROR("m_dataTable is nullptr, vectorFieldName" << vectorFieldName);
        return false;
    }
    auto ret = aclrtSetDevice(m_deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtSetDevice fail, deviceId is:" << m_deviceId << " error code is:" << ret);
        return false;
    }
    if (!m_vectorScorerMmad.BatchCompute(m_queryVectors, m_dimension, vectorFieldName, resultChunk)) {
        LOG_ERROR("m_vectorScorerMmad.BatchCompute failed. m_deviceId=" << m_deviceId);
        GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(resultChunk);
        CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
        return false;
    }
    CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
    return true;
}

bool FullRecallSearcher::OneAggregator(GmBlock& resultChunk, GmBlock& filterResultChunk, uint32_t resultInDeviceOffset,
                                       bool isMultiShard,
                                       std::vector<std::shared_ptr<FullRecallResult>>& fullRecallResults) {
    RecordGuard guard{"FullRecallSearcher OneAggregator"};
    if (m_dataTable == nullptr) {
        LOG_ERROR("m_dataTable is nullptr, deviceId is:" << m_deviceId);
        return false;
    }
    auto ret = aclrtSetDevice(m_deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtSetDevice fail, deviceId is:" << m_deviceId << " error code is:" << ret);
        return false;
    }
    uint8_t* filterResultInDevice = reinterpret_cast<uint8_t*>(filterResultChunk.data);
    uint8_t* resultInDevice = reinterpret_cast<uint8_t*>(resultChunk.data);  // scoring / aggregation result

    uint32_t docNum = m_dataTable->GetDocNum();
    uint32_t docAllByteSize = docNum * sizeof(float);
    uint32_t vectorResultOffsetOne = m_dataTable->GetScoreExtendDocNum() * sizeof(float);
    uint8_t* resultInDeviceNew = resultInDevice + resultInDeviceOffset * vectorResultOffsetOne;

    std::shared_ptr<FullRecallResult> result = std::make_shared<FullRecallResult>();
    result->isMultiShard = isMultiShard;
    ResultAggregator resultAggregator{};
    resultAggregator.SetDocIdMapping(m_docIdMapping);
    resultAggregator.SetTopK(m_topKs[0]);  // reaching here guarantees the array is non-empty
    resultAggregator.SetDeviceId(m_deviceId);
    bool isSuccess = resultAggregator.AggrAndTopK(filterResultInDevice, resultInDeviceNew, docAllByteSize,
                                                  m_docLocationInDevice, result);
    if (isSuccess) {
        fullRecallResults.emplace_back(result);
    }
    if (resultChunk.readCount == nullptr) {
        LOG_ERROR("resultChunk.readCount is nullptr");
    } else {
        if (resultChunk.readCount->fetch_sub(1) == 1) {  // fetch_sub returns the previous value
            GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(resultChunk);
        }
    }

    GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(filterResultChunk);
    CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
    return true;
}

bool FullRecallSearcher::RunAggregation(uint8_t* filterResultInDevice, uint8_t* resultInDevice,
                                        std::vector<std::shared_ptr<FullRecallResult>>& fullRecallResults,
                                        bool isMultiShard) {
    RecordGuard guard{"Aggregator all"};
    uint32_t docNum = m_dataTable->GetDocNum();
    uint32_t segmentsNum = m_dataTable->GetSegmentNum();
    uint32_t vectorResultOffsetOne = m_dataTable->GetScoreExtendDocNum() * sizeof(float);
    uint32_t docAllByteSize = docNum * sizeof(float);
    uint32_t resultInDeviceOffset = 0;
    uint32_t filterResultInDeviceByteOffset = 0;
    uint32_t filterResultInDeviceByteSize = segmentsNum * (m_dataTable->GetSegmentByteSize());
    std::shared_ptr<LogContext> logContext = std::make_shared<LogContext>();
    auto asyncAggregator = m_executor->CreateExecuteContext(*logContext);
    fullRecallResults.resize(m_topKs.size());
    uint32_t index = 0;
    for (uint32_t topK : m_topKs) {
        if (m_queries[index] == nullptr) {
            LOG_ERROR("m_queries[index] is nullptr.");
            std::vector<uint64_t> resultId;
            std::vector<float> resultScore;
            fullRecallResults[index] = std::make_shared<FullRecallResult>();
            index++;
            continue;
        }
        uint8_t* filterResultInDeviceNew = filterResultInDevice + filterResultInDeviceByteOffset;
        uint8_t* resultInDeviceNew = resultInDevice + resultInDeviceOffset;
        auto aggTask = [this, topK, filterResultInDeviceNew, resultInDeviceNew, docAllByteSize, &fullRecallResults,
                        index, isMultiShard]() -> ErrorCode::ResultType {
            auto ret = aclrtSetDevice(m_deviceId);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("aclrtSetDevice fail, deviceId is:" << m_deviceId << " error code is:" << ret);
                return ErrorCode::ResultType::FAIL;
            }
            std::shared_ptr<FullRecallResult> result = std::make_shared<FullRecallResult>();
            result->isMultiShard = isMultiShard;
            ResultAggregator resultAggregator{};
            resultAggregator.SetDocIdMapping(m_docIdMapping);
            resultAggregator.SetTopK(topK);
            resultAggregator.SetDeviceId(m_deviceId);
            if (!resultAggregator.AggrAndTopK(filterResultInDeviceNew, resultInDeviceNew, docAllByteSize,
                                              m_docLocationInDevice, result)) {
                LOG_ERROR("AggrAndTopK failed.");
                CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
                return ErrorCode::ResultType::FAIL;
            }
            fullRecallResults[index] = result;
            CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
            return ErrorCode::ResultType::SUCCESS;
        };
        asyncAggregator->AddTask(aggTask);
        resultInDeviceOffset += vectorResultOffsetOne;
        filterResultInDeviceByteOffset += filterResultInDeviceByteSize;
        LOG_DEBUG("index:" << index << " resultInDeviceOffset=" << resultInDeviceOffset
                           << " filterResultInDeviceByteOffset=" << filterResultInDeviceByteOffset);
        index++;
    }
    bool isSuccessAggregator = true;
    asyncAggregator->Wait([&isSuccessAggregator](ErrorCode::ResultType ret) {
        if (ret != ErrorCode::ResultType::SUCCESS) {
            isSuccessAggregator = false;
        }
    });
    if (isSuccessAggregator) {
        LOG_DEBUG("Aggregator and topK success.");
    } else {
        LOG_ERROR("Aggregator and topK fail, please check");
    }
    return isSuccessAggregator;
}

bool FullRecallSearcher::RunAggregationDevice(uint8_t* filterResultInDevice, uint8_t* resultInDevice,
                                              std::vector<AggrDeviceResult>& deviceResults, aclrtStream sharedStream,
                                              bool* sharedStreamSynced) {
    RecordGuard guard{"Aggregator all"};
    uint32_t docNum = m_dataTable->GetDocNum();
    uint32_t segmentsNum = m_dataTable->GetSegmentNum();
    uint32_t vectorResultOffsetOne = m_dataTable->GetScoreExtendDocNum() * sizeof(float);
    uint32_t docAllByteSize = docNum * sizeof(float);
    uint32_t resultInDeviceOffset = 0;
    uint32_t filterResultInDeviceByteOffset = 0;
    uint32_t filterResultInDeviceByteSize = segmentsNum * (m_dataTable->GetSegmentByteSize());
    // NPUR_BATCH_AGGREGATE: one phased call for the whole batch.
    if (ResultAggregator::BatchAggregateMode()) {
        std::vector<uint8_t*> filterResults(m_topKs.size(), nullptr);
        std::vector<uint8_t*> docScores(m_topKs.size(), nullptr);
        for (size_t q = 0; q < m_topKs.size(); ++q) {
            if (m_queries[q] == nullptr)
                continue;  // offsets still advance only for real queries, as in the loop below
            filterResults[q] = filterResultInDevice + filterResultInDeviceByteOffset;
            docScores[q] = resultInDevice + resultInDeviceOffset;
            resultInDeviceOffset += vectorResultOffsetOne;
            filterResultInDeviceByteOffset += filterResultInDeviceByteSize;
        }
        ResultAggregator batchAggregator{};
        batchAggregator.SetDocIdMapping(m_docIdMapping);
        batchAggregator.SetDeviceId(m_deviceId);
        return batchAggregator.AggrAndTopKDeviceBatch(filterResults, docScores, docAllByteSize, m_docLocationInDevice,
                                                      m_topKs, deviceResults, sharedStream, sharedStreamSynced);
    }
    // The per-query tasks take streams of their own, so drain the shared one rather than race it.
    if (sharedStream != nullptr) {
        if (aclrtSynchronizeStream(sharedStream) != ACL_SUCCESS) {
            LOG_ERROR("aclrtSynchronizeStream (shared filter stream) fail, m_deviceId=" << m_deviceId);
            return false;
        }
        if (sharedStreamSynced != nullptr) {
            *sharedStreamSynced = true;
        }
    }
    std::shared_ptr<LogContext> logContext = std::make_shared<LogContext>();
    auto asyncAggregator = m_executor->CreateExecuteContext(*logContext);
    deviceResults.assign(m_topKs.size(), AggrDeviceResult{});
    uint32_t index = 0;
    for (uint32_t topK : m_topKs) {
        if (m_queries[index] == nullptr) {
            index++;
            continue;  // deviceResults[index] stays empty; offsets advance only for real queries
        }
        uint8_t* filterResultInDeviceNew = filterResultInDevice + filterResultInDeviceByteOffset;
        uint8_t* resultInDeviceNew = resultInDevice + resultInDeviceOffset;
        auto devTask = [this, topK, filterResultInDeviceNew, resultInDeviceNew, docAllByteSize, &deviceResults,
                        index]() -> ErrorCode::ResultType {
            auto ret = aclrtSetDevice(m_deviceId);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("aclrtSetDevice fail, deviceId is:" << m_deviceId << " error code is:" << ret);
                return ErrorCode::ResultType::FAIL;
            }
            ResultAggregator resultAggregator{};
            resultAggregator.SetDocIdMapping(m_docIdMapping);
            resultAggregator.SetTopK(topK);
            resultAggregator.SetDeviceId(m_deviceId);
            if (!resultAggregator.AggrAndTopKDevice(filterResultInDeviceNew, resultInDeviceNew, docAllByteSize,
                                                    m_docLocationInDevice, deviceResults[index])) {
                LOG_ERROR("AggrAndTopKDevice failed.");
                CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
                return ErrorCode::ResultType::FAIL;
            }
            CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
            return ErrorCode::ResultType::SUCCESS;
        };
        asyncAggregator->AddTask(devTask);
        resultInDeviceOffset += vectorResultOffsetOne;
        filterResultInDeviceByteOffset += filterResultInDeviceByteSize;
        index++;
    }
    bool ok = true;
    asyncAggregator->Wait([&ok](ErrorCode::ResultType ret) {
        if (ret != ErrorCode::ResultType::SUCCESS)
            ok = false;
    });
    return ok;
}

bool FullRecallSearcher::RunAggregationExtract(std::vector<AggrDeviceResult>& deviceResults,
                                               std::vector<std::shared_ptr<FullRecallResult>>& fullRecallResults,
                                               bool isMultiShard) {
    RecordGuard guard{"Aggregator extract"};
    std::shared_ptr<LogContext> logContext = std::make_shared<LogContext>();
    auto asyncExtract = m_executor->CreateExecuteContext(*logContext);
    fullRecallResults.assign(m_topKs.size(), nullptr);
    uint32_t index = 0;
    for (uint32_t topK : m_topKs) {
        auto result = std::make_shared<FullRecallResult>();
        result->isMultiShard = isMultiShard;
        fullRecallResults[index] = result;
        if (m_queries[index] == nullptr || index >= deviceResults.size() || deviceResults[index].empty) {
            index++;
            continue;  // empty result (no candidates / no query), nothing staged to extract
        }
        // Capture by reference: AggrAndTopKExtract takes a non-const reference, and a by-value
        // capture is const inside a non-mutable lambda.
        auto exTask = [this, topK, index, &deviceResults, &fullRecallResults]() -> ErrorCode::ResultType {
            auto ret = aclrtSetDevice(m_deviceId);  // aclrtFreeHost inside Extract needs a device context
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("aclrtSetDevice fail, deviceId is:" << m_deviceId << " error code is:" << ret);
                return ErrorCode::ResultType::FAIL;
            }
            ResultAggregator resultAggregator{};
            resultAggregator.SetDocIdMapping(m_docIdMapping);
            resultAggregator.SetTopK(topK);
            resultAggregator.SetDeviceId(m_deviceId);
            bool ok = resultAggregator.AggrAndTopKExtract(deviceResults[index], fullRecallResults[index]);
            CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
            return ok ? ErrorCode::ResultType::SUCCESS : ErrorCode::ResultType::FAIL;
        };
        asyncExtract->AddTask(exTask);
        index++;
    }
    bool ok = true;
    asyncExtract->Wait([&ok](ErrorCode::ResultType ret) {
        if (ret != ErrorCode::ResultType::SUCCESS)
            ok = false;
    });
    return ok;
}

bool FullRecallSearcher::BatchSearchDevice(const std::string& vectorFieldName, bool isMultiShard,
                                           std::vector<AggrDeviceResult>& deviceResults) {
    RecordGuard guard{"BatchSearchDevice"};
    (void)isMultiShard;  // isMultiShard is applied in Extract (result flag); Device stages raw top-K
    if (m_dataTable == nullptr) {
        LOG_ERROR("m_dataTable is nullptr, vectorFieldName" << vectorFieldName);
        return false;
    }
    uint32_t docNumPerSegment = m_dataTable->GetDocNumPerSegment();
    uint32_t segmentsNum = m_dataTable->GetSegmentNum();
    GmBlock resultChunk{};
    GmBlock filterResultChunk{};
    auto ret = aclrtSetDevice(m_deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtSetDevice fail, deviceId is:" << m_deviceId << " error code is:" << ret);
        return false;
    }
    uint32_t allResultsByteSize = segmentsNum * docNumPerSegment / 8 * m_queries.size();
    GmMemoryManager::GetByDeviceId(m_deviceId)
        ->AllocateBlock(GmPoolName::TEXT_FILTER_RESULT_POOL, allResultsByteSize, filterResultChunk);
    uint8_t* filterResultInDevice = reinterpret_cast<uint8_t*>(filterResultChunk.data);
    if (filterResultInDevice == nullptr) {
        LOG_ERROR("GmMemoryManager allocate failed.");
        CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
        return false;
    }
    // ParallelScoreFilterEnabled() is deliberately NOT replicated here: it spawns a thread per batch.
    bool ok = true;
    // Under NPUR_SHARE_FILTER_STREAM the filter batch outlives this branch, so its state and the
    // posting tables its kernels read have to as well.
    std::vector<FilterExpr> exprs;
    FusedFilterBatch fused;
    bool filterPending = false;  // filter kernels queued on fused.stream; FusedFinish owed after the aggregator
    bool overlapPosting = false;
    if (OverlapPostExprMode() != 0) {
        const bool overlap = OverlapPostExprMode() == 1;
        overlapPosting = overlap && OverlapPostingMode();
        ScorerLaunch launch;
        bool fusedPrepared = false;  // NPUR_EARLY_FILTER_PREP ran FusedPrepare inside the window
        ok = m_vectorScorerMmad.BatchComputeLaunch(m_queryVectors, m_dimension, vectorFieldName, resultChunk, launch);
        if (!ok) {
            LOG_ERROR("m_vectorScorerMmad.BatchComputeLaunch failed. m_deviceId=" << m_deviceId);
        } else if (overlap && !m_textFilter.BatchPrepareExpr(m_queries, exprs)) {
            LOG_ERROR("m_textFilter.BatchPrepareExpr failed.");
            ok = false;
        } else if (overlapPosting && !m_textFilter.BatchPreparePostings(exprs, DeferConvSyncMode())) {
            LOG_ERROR("m_textFilter.BatchPreparePostings failed.");
            ok = false;
        } else if (overlapPosting && (EarlyFilterPrepMode() || OverlapFilterKernelMode()) &&
                   TextFilter::FusedPathApplies(&exprs) && ReadyForEarlyPrep(exprs)) {
            if (!m_textFilter.FusedPrepare(m_queries, filterResultInDevice, exprs, fused)) {
                LOG_ERROR("m_textFilter.FusedPrepare (in the scorer window) failed.");
                ok = false;
            } else {
                fusedPrepared = true;
                if (OverlapFilterKernelMode()) {
                    if (!m_textFilter.DrainPreparedConversions(exprs)) {
                        LOG_ERROR("m_textFilter.DrainPreparedConversions (before the early filter) failed.");
                        ok = false;
                    } else {
                        m_textFilter.FusedLaunch(fused, /*timeKernel=*/false);
                    }
                }
            }
        }
        // Sync even when the build failed: skipping it would free resultChunk out from under a
        // kernel still writing it.
        if (!m_vectorScorerMmad.BatchComputeSync(launch)) {
            LOG_ERROR("m_vectorScorerMmad.BatchComputeSync failed.");
            ok = false;
        }
        // Pending conversions must finish before any filter kernel reads their bitsets.
        if (!m_textFilter.DrainPreparedConversions(exprs)) {
            LOG_ERROR("m_textFilter.DrainPreparedConversions failed.");
            ok = false;
        }
        if (ok && !overlap && !m_textFilter.BatchPrepareExpr(m_queries, exprs)) {
            LOG_ERROR("m_textFilter.BatchPrepareExpr failed.");
            ok = false;
        }
        const bool share =
            ShareFilterStreamMode() && ResultAggregator::BatchAggregateMode() && TextFilter::FusedPathApplies(&exprs);
        if (ok && (fusedPrepared || share)) {
            if (!fusedPrepared && !m_textFilter.FusedPrepare(m_queries, filterResultInDevice, exprs, fused)) {
                LOG_ERROR("m_textFilter.FusedPrepare failed.");
                ok = false;
            } else if (share) {
                if (!fused.launched) {
                    m_textFilter.FusedLaunch(fused, /*timeKernel=*/false);
                }
                filterPending = true;
            } else {
                RecordGuard guard{"TextFilter BatchCompute"};  // what is left of it after the window
                if (!fused.launched) {
                    m_textFilter.FusedLaunch(fused, /*timeKernel=*/true);
                }
                if (!m_textFilter.FusedFinish(fused, /*streamSynced=*/false)) {
                    LOG_ERROR("textFilter.FusedFinish failed.");
                    ok = false;
                }
            }
        } else if (ok && !m_textFilter.BatchCompute(m_queries, filterResultInDevice, &exprs)) {
            LOG_ERROR("textFilter.BatchCompute failed.");
            ok = false;
        }
        // An unfinished in-window FusedPrepare still holds its blocks and the device reference.
        if (fusedPrepared && !filterPending) {
            m_textFilter.FusedFinish(fused, /*streamSynced=*/false);
        }
        // Not while the filter kernels that read them may still be running.
        if (overlapPosting && !filterPending) {
            m_textFilter.FreePreparedPostings(exprs);
        }
    } else {
        ok = m_vectorScorerMmad.BatchCompute(m_queryVectors, m_dimension, vectorFieldName, resultChunk);
        if (!ok)
            LOG_ERROR("m_vectorScorerMmad.BatchCompute failed. m_deviceId=" << m_deviceId);
        if (ok && !m_textFilter.BatchCompute(m_queries, filterResultInDevice)) {
            LOG_ERROR("textFilter.BatchCompute failed.");
            ok = false;
        }
    }
    bool aggSynced = false;
    if (ok) {
        uint8_t* resultInDevice = reinterpret_cast<uint8_t*>(resultChunk.data);
        ok = RunAggregationDevice(filterResultInDevice, resultInDevice, deviceResults,
                                  filterPending ? fused.stream : nullptr, &aggSynced);
    }
    if (filterPending) {
        // Either the aggregator's first sync waited for the filter kernels or FusedFinish does.
        if (!m_textFilter.FusedFinish(fused, aggSynced)) {
            LOG_ERROR("textFilter.FusedFinish (shared stream) failed.");
            ok = false;
        }
        if (overlapPosting) {
            m_textFilter.FreePreparedPostings(exprs);
        }
    }
    GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(resultChunk);
    GmMemoryManager::GetByDeviceId(m_deviceId)->FreeBlock(filterResultChunk);
    CHECK_ACL_ONLY_LOG(aclrtResetDevice(m_deviceId));
    return ok;
}

bool FullRecallSearcher::BatchSearchExtract(std::vector<AggrDeviceResult>& deviceResults,
                                            std::vector<std::shared_ptr<FullRecallResult>>& fullRecallResults,
                                            bool isMultiShard) {
    RecordGuard guard{"BatchSearchExtract"};
    return RunAggregationExtract(deviceResults, fullRecallResults, isMultiShard);
}
}  // namespace NpuRetrieval
