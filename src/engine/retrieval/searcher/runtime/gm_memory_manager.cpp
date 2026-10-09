#include "gm_memory_manager.h"
#include "utils/logger.h"
#include "acl/acl.h"
#include "configuration/develop_configuration.h"
#include "src/full_recall/core/constant_definition.h"

namespace NpuRetrieval {

GmMemoryManager::GmMemoryManager() {
    m_memoryPools.resize(static_cast<size_t>(GmPoolName::VALID_POOL_NUM), nullptr);

    m_memoryPools[static_cast<size_t>(GmPoolName::VECTOR_SCORE_RESULT_POOL)] = std::make_shared<GmMemoryPool>(
        GmPoolName::VECTOR_SCORE_RESULT_POOL, FLAGS_full_recall_batch_search_thread_num, 0);
    m_memoryPools[static_cast<size_t>(GmPoolName::TEXT_FILTER_RESULT_POOL)] = std::make_shared<GmMemoryPool>(
        GmPoolName::TEXT_FILTER_RESULT_POOL,
        FLAGS_full_recall_batch_search_thread_num * FLAGS_full_recall_batch_accumulation_max_size, 0);
    m_memoryPools[static_cast<size_t>(GmPoolName::TEXT_FILTER_STACK_POOL)] = std::make_shared<GmMemoryPool>(
        GmPoolName::TEXT_FILTER_STACK_POOL,
        FLAGS_full_recall_batch_search_thread_num * FLAGS_full_recall_batch_accumulation_max_size, 0);
    // Hundreds of MB per query; a raw aclrtFree of one measured ~36ms. blockSize 0 sizes it lazily
    // on the first query, and the pool auto-grows when a block is too small.
    m_memoryPools[static_cast<size_t>(GmPoolName::TEXT_FILTER_BITLIST2SET_POOL)] = std::make_shared<GmMemoryPool>(
        GmPoolName::TEXT_FILTER_BITLIST2SET_POOL, FLAGS_full_recall_batch_search_thread_num, 0);
    // 3 blocks per query, and NPUR_BATCH_AGGREGATE holds every query's 3 at once, so the cap has
    // to scale with the batch: everything past it is a raw aclrtFree of a ~20MB block per chunk
    // (513.8 qps against 1416.0 unbatched at BATCH_SIZE=16). The 4th block per query is reserve.
    m_memoryPools[static_cast<size_t>(GmPoolName::AGGREGATOR_POOL)] = std::make_shared<GmMemoryPool>(
        GmPoolName::AGGREGATOR_POOL,
        FLAGS_full_recall_batch_search_thread_num * 4 * FLAGS_full_recall_batch_accumulation_max_size, 0);
    // Small fixed scratch for the aggregation stage. A raw device free syncs the stream and
    // occasionally stalls for ms -- the p99 tail. blockSize 1280B =
    // RESULT_MESSAGE_BYTE_SIZE_IN_DEVICE covers the largest of the three and avoids a regrow spike.
    m_memoryPools[static_cast<size_t>(GmPoolName::AGGREGATOR_SCRATCH_POOL)] = std::make_shared<GmMemoryPool>(
        GmPoolName::AGGREGATOR_SCRATCH_POOL, FLAGS_full_recall_batch_search_thread_num * 4, 1280);
    // Small host->device pointer table rebuilt per query in TextFilter::Compute. blockSize 0 sizes
    // lazily and the pool grows to the largest table seen (postingsNum varies per filter).
    // NPUR_OVERLAP_POSTING holds one table per query in the batch, not one per in-flight query.
    m_memoryPools[static_cast<size_t>(GmPoolName::TEXT_FILTER_POSTINGS_POOL)] = std::make_shared<GmMemoryPool>(
        GmPoolName::TEXT_FILTER_POSTINGS_POOL,
        FLAGS_full_recall_batch_search_thread_num * FLAGS_full_recall_batch_accumulation_max_size, 0);
    // The scorer's padded query matrix and the filter's concatenated postfix expressions, one of
    // each per batch in flight. Only used under NPUR_POOL_SMALL_H2D=1.
    m_memoryPools[static_cast<size_t>(GmPoolName::SCORER_QUERY_POOL)] =
        std::make_shared<GmMemoryPool>(GmPoolName::SCORER_QUERY_POOL, FLAGS_full_recall_batch_search_thread_num, 0);
    m_memoryPools[static_cast<size_t>(GmPoolName::TEXT_FILTER_EXPR_POOL)] =
        std::make_shared<GmMemoryPool>(GmPoolName::TEXT_FILTER_EXPR_POOL, FLAGS_full_recall_batch_search_thread_num, 0);
}

const std::shared_ptr<GmMemoryPool> GmMemoryManager::GetMemoryPool(const GmPoolName poolName) {
    const size_t idx = static_cast<size_t>(poolName);
    if (idx >= m_memoryPools.size()) {
        LOG_ERROR("invalid poolName:" << idx);
        return nullptr;
    }
    return m_memoryPools[idx];
}

void GmMemoryManager::AllocateBlock(const GmPoolName poolName, uint32_t size, GmBlock& chunk) {
    auto poolPtr = GetMemoryPool(poolName);
    if (poolPtr == nullptr) {
        LOG_ERROR("Get Memory Pool failed, it is null");
        return;
    }
    chunk = poolPtr->Allocate(size);
}

void GmMemoryManager::FreeBlock(GmBlock& chunk) {
    auto poolPtr = GetMemoryPool(chunk.poolName);
    if (poolPtr == nullptr) {
        LOG_ERROR("Get Memory Pool failed, it is null");
        return;
    }
    poolPtr->Deallocate(chunk);
}

}  // namespace NpuRetrieval
