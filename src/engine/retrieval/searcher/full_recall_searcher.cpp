#include "full_recall_searcher.h"
#include <cstdlib>
#include <string>
#include <thread>
#include "acl/acl.h"
#include "src/utils/logger.h"
#include "src/full_recall/core/constant_definition.h"
#include "src/full_recall/core/log_definition.h"
#include "full_recall/core/number/trans_number.h"
#include "configuration/develop_configuration.h"
#include "src/utils/performance_recorder.h"
#include "full_recall/retrieval/searcher/runtime/gm_memory_manager.h"

namespace NpuRetrieval {

// A/B switch for overlapping the scorer and text filter. Default OFF: they run
// serially via the single-threaded executor. NPUR_PARALLEL_SCORE_FILTER=1 runs
// them on two host threads -> two distinct StreamManager streams, so the MMad
// scorer (Cube) and the bitmap filter (Vector) co-reside on the NPU. Read once;
// constant for the process.
//
// The overlap is real -- the scorer kernel costs the same (~818us) whether serial
// or parallel, because Cube and Vector are separate units, so it completes for
// free inside the filter's window. But it is off by default because it trades the
// tail for the median. Same-window measurement on 910B3 (2-shard, scorer 12 /
// filter 40) vs serial:
//
//     avg  2.704 -> 2.650ms   p50  2.663 -> 2.404ms   p90 3.025 -> 2.777ms
//     p95  3.223 -> 3.336ms   p99  3.817 -> 6.832ms   max 26 -> 41ms
//
// The p99 cost is not noise: all three parallel runs were >= 6.2ms, all three
// serial runs <= 4.85ms. The extra thread inflates the known cross-shard ACL D2H
// contention (fill_d2h_copy p99 70/71/69us serial vs 85/114/6534us parallel).
// Turn it on only if median latency matters more than the tail.
static bool ParallelScoreFilterEnabled() {
    static const bool enabled = []() {
        const char* v = std::getenv("NPUR_PARALLEL_SCORE_FILTER");
        return v != nullptr && std::string(v) == "1";
    }();
    return enabled;
}

// A/B switch for hiding the text filter's host-side postfix-expression build inside
// the scorer kernel's execution window. Default OFF. Read once; constant for the
// process. Ignored when NPUR_PARALLEL_SCORE_FILTER=1, which already overlaps far more.
//
// Serially the device goes idle mid-query for no reason: the scorer's
// aclrtSynchronizeStream returns before the filter task even starts, and the filter
// then spends TextFilter_Compute_GetPostOrderExpression walking the query tree on the
// host with nothing running on the NPU. This launches the scorer, builds the
// expression while its kernel runs, and only then waits.
//
// Why this is not the parallel path that failed:
//   - No second thread, so none of the spawn jitter that --shard_worker_pool exists
//     to avoid.
//   - The build issues zero ACL calls (query_node_imp.cpp includes only its own
//     header and the logger), so it cannot reproduce the ACL contention that made
//     NPUR_PARALLEL_SCORE_FILTER's D2H go 39.2 -> 71.3us.
//   - The scorer is synced before the filter touches the device, so the two kernels
//     still never co-reside and cannot fight over HBM bandwidth (filter 233 -> 719us).
//
// Modes (default 1):
//   1  the postfix build runs inside the scorer kernel's execution window.
//   2  same calls in the same order, but the Sync sits right after the launch so
//      nothing overlaps. The control arm that isolates the overlap.
//   0  the pre-overlap baseline: both tasks through the executor, each re-entering
//      aclrtSetDevice/aclrtResetDevice.
//
// 2-shard, 3 interleaved rounds, medians, recall 100% on all nine runs:
//
//                    avg      p50      p90      p99
//     0 baseline   2.5765   2.5489   2.8243   3.2674
//     2 control    2.5560   2.5362   2.8270   3.2171
//     1 overlap    2.4497   2.4327   2.7175   2.9864
//
// The control arm is what makes this readable. 2-vs-0 is worth -20us avg (ranges
// overlap -- not a result) and -50us p99: dropping the executor round trip and those
// redundant per-task Set/Reset buys almost nothing. The whole win is 1-vs-2, whose
// p99 ranges do not overlap.
//
// And it lands where the mechanism says it should. The build itself costs 101.1us avg
// / 251us p99 per shard, and both shards hide their own concurrently, so those are the
// ceilings; the overlap measures -106.3us avg (105% of ceiling) and -230.7us p99
// (92%). p99 gains more than avg precisely because the build's own p99 is 2.5x its
// avg -- more to hide on the slow queries. Two independent points predicted by one
// stage's distribution is what makes this causal rather than a coincidence.
//
// Safe where NPUR_PARALLEL_SCORE_FILTER was not: no second thread (no spawn jitter),
// the build issues no ACL calls at all (query_node_imp.cpp includes only its own
// header and the logger, so no repeat of D2H 39.2 -> 71.3us), and the scorer is synced
// before the filter touches the device, so the kernels never co-reside and cannot
// fight over HBM (filter 233 -> 719us).
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

// NPUR_OVERLAP_POSTING=1 (default off): also build the filter's posting pointer table
// (PostingBitListToSet) inside the scorer's execution window, not after the sync. Only
// applies on the overlap path (OverlapPostExprMode()==1). Nearly pure host + a small H2D,
// so it does not fight the scorer for HBM the way the filter kernel would.
static bool OverlapPostingMode() {
    static const bool enabled = [] {
        const char* v = std::getenv("NPUR_OVERLAP_POSTING");
        return v != nullptr && v[0] == '1';
    }();
    return enabled;
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
    // The query and doc-side embeddings correspond. Model/doc dimension alignment is
    // validated by the model-index consistency management, and only requests from the
    // same model are accumulated together.
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
    // In the offline data the bitset uses uint16 as its unit, so it is always divisible by 8; validated at load time.
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

    // Wait for scoring and filtering to finish. They write disjoint device buffers
    // (resultChunk vs filterResultInDevice), so they are independent. The outer
    // aclrtSetDevice above holds a device refcount for the whole BatchSearch, so the
    // per-task Set/ResetDevice in the parallel threads only bump the count and never
    // tear the context down.
    bool isSuccess = true;
    if (ParallelScoreFilterEnabled()) {
        // Overlap on two host threads -> two distinct StreamManager streams. The two
        // kernels never contend for compute (Cube vs Vector are separate units); what
        // they share is HBM bandwidth, and the scorer -- which is bandwidth-bound --
        // wins it, so the filter absorbs the whole slowdown (kernel 233 -> 719us) and
        // becomes the long pole. Filter runs inline on this thread; only the scorer
        // gets a spawned thread.
        ErrorCode::ResultType scoreRet = ErrorCode::ResultType::FAIL;
        std::thread scoreThread([&scoreRet, &scoreTask]() { scoreRet = scoreTask(); });
        ErrorCode::ResultType filterRet = filterTask();
        scoreThread.join();
        if (scoreRet != ErrorCode::ResultType::SUCCESS || filterRet != ErrorCode::ResultType::SUCCESS) {
            isSuccess = false;
        }
    } else if (OverlapPostExprMode() != 0) {
        // All on this thread; BatchSearch's outer aclrtSetDevice already covers it, so no
        // Set/Reset here. Mode 1 and mode 2 run the identical calls in the identical
        // order -- the only difference is whether the postfix build lands before or after
        // the Sync, which is exactly the thing being priced.
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
            // Still done inside the scorer window: it's host bookkeeping + a small H2D.
            LOG_ERROR("m_textFilter.BatchPreparePostings failed.");
            isSuccess = false;
        }
        // Sync even when the build failed: the launched kernel still owns the stream and
        // the query buffer, and skipping this would leak them and free resultChunk out
        // from under a kernel still writing to it.
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
        // Free the pre-built posting tables (if any) -- BatchCompute is done consuming them.
        // Runs on success and failure alike; FreePreparedPostings skips un-prepared exprs.
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
        // wait for aggregation to finish
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
    // In the offline data the bitset uses uint16 as its unit, so it is always divisible by 8; validated at load time.
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
}  // namespace NpuRetrieval
