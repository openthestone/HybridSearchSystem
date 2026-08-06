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
    // The bitlist2set bitset buffer is hundreds of MB per query; without pooling each
    // query would aclrtMalloc/aclrtFree a huge HBM block (measured free ~36ms, 61% of
    // query latency). Enable reuse with a non-zero blockNum; leave blockSize 0 so the
    // first query sizes it lazily, and the pool auto-grows and refills when a block is
    // too small (see GmMemoryPool::Allocate/Deallocate).
    m_memoryPools[static_cast<size_t>(GmPoolName::TEXT_FILTER_BITLIST2SET_POOL)] = std::make_shared<GmMemoryPool>(
        GmPoolName::TEXT_FILTER_BITLIST2SET_POOL, FLAGS_full_recall_batch_search_thread_num, 0);
    m_memoryPools[static_cast<size_t>(GmPoolName::AGGREGATOR_POOL)] =
        std::make_shared<GmMemoryPool>(GmPoolName::AGGREGATOR_POOL, FLAGS_full_recall_batch_search_thread_num * 4,
                                       0);  // 4x the scoring pool; 3x should suffice, keep one extra in reserve.
    // Small fixed scratch buffers for the aggregation stage (effectiveCountDevice /
    // docNumberInDevice / TOPKResultCountInDevice). These were raw aclrtMalloc/aclrtFree
    // per query -- a device free syncs the stream and occasionally stalls for ms = the
    // p99 tail. Use a dedicated small pool: non-zero blockSize preallocates (1280B =
    // RESULT_MESSAGE_BYTE_SIZE_IN_DEVICE, covers the largest of the three and avoids a
    // lazy-regrow spike), blockNum=thread_num*4 (peak concurrency is 2 blocks per query,
    // ample margin).
    m_memoryPools[static_cast<size_t>(GmPoolName::AGGREGATOR_SCRATCH_POOL)] = std::make_shared<GmMemoryPool>(
        GmPoolName::AGGREGATOR_SCRATCH_POOL, FLAGS_full_recall_batch_search_thread_num * 4, 1280);
    // Small host->device pointer table (segmentsNum * postingsNum * 8B, a few KB) rebuilt
    // for every query in TextFilter::Compute. It used to be a raw aclrtMalloc/aclrtFree pair
    // per query; the device free syncs the stream and occasionally stalls for ms -- the
    // PostingBitListToSet p99 tail. Pool it: blockSize 0 sizes lazily on the first query and
    // the pool auto-grows to the largest table seen (postingsNum varies per filter), blockNum
    // = thread_num since at most one table is live per in-flight query.
    m_memoryPools[static_cast<size_t>(GmPoolName::TEXT_FILTER_POSTINGS_POOL)] = std::make_shared<GmMemoryPool>(
        GmPoolName::TEXT_FILTER_POSTINGS_POOL, FLAGS_full_recall_batch_search_thread_num, 0);
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
