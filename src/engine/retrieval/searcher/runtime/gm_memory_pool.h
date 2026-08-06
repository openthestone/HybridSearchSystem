#pragma once

#include <vector>
#include "acl/acl.h"
#include "configuration/develop_configuration.h"
#include "utils/logger.h"
#include "src/full_recall/core/constant_definition.h"
#include "src/full_recall/core/log_definition.h"
#include "src/full_recall/core/full_recall_result.h"
#include <atomic>
namespace NpuRetrieval {

class GmMemoryPool {
   public:
    GmMemoryPool(GmPoolName name, uint32_t blockNum, uint32_t blockSize)
        : m_name(name), m_defaultBlockNum(blockNum), m_defaultBlockSize(blockSize) {
        if (m_defaultBlockSize == 0) {
            return;
        }
        // preallocate the blocks
        for (uint32_t i = 0; i < m_defaultBlockNum; ++i) {
            GmBlock block{m_name, nullptr, blockSize};
            auto ret = aclrtMalloc((void**)&block.data, m_defaultBlockSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("aclrtMalloc default block fail, error code is:" << ret);
                continue;
            }
            m_freeList.emplace_back(block);
        }
    }

    ~GmMemoryPool() {
        while (!m_freeList.empty()) {
            auto block = m_freeList.back();
            CHECK_ACL_ONLY_LOG(aclrtFree(block.data));
            m_freeList.pop_back();
        }
    }

    GmBlock Allocate(const uint32_t size) {
        GmBlock block{m_name, nullptr, 0};
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_freeList.empty()) {
                LOG_INFO("pool:" << (int)m_name << " block num is not enough, try to expand");
                block.size = size;
                auto ret = aclrtMalloc((void**)&block.data, size, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    LOG_ERROR("Allocate fail, aclrtMalloc fail, error code is:" << ret);
                    block.data = nullptr;
                    return block;
                }
                m_count.fetch_add(1, std::memory_order_relaxed);
                LOG_INFO("pool:" << (int)m_name << " count:" << m_count.load(std::memory_order_relaxed));
                return block;
            }
            block = m_freeList.back();
            m_freeList.pop_back();
        }
        if (block.size < size) {
            LOG_INFO("pool:" << (int)m_name << " block size not enough, current:" << block.size << " applied:" << size);
            CHECK_ACL_ONLY_LOG(aclrtFree(block.data));
            block.data = nullptr;
            block.size = size;
            auto ret = aclrtMalloc((void**)&block.data, size, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("Allocate fail, aclrtMalloc fail, error code is:" << ret);
                block.data = nullptr;
                return block;
            }
            return block;
        }
        m_count.fetch_add(1, std::memory_order_relaxed);
        LOG_DEBUG("pool:" << (int)m_name << " count:" << m_count.load(std::memory_order_relaxed));
        return block;
    }

    void Deallocate(const NpuRetrieval::GmBlock& block) {
        if (block.data == nullptr) {
            return;
        }
        m_count.fetch_sub(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_freeList.size() < m_defaultBlockNum) {
                m_freeList.emplace_back(block);
            } else {
                CHECK_ACL_ONLY_LOG(aclrtFree(block.data));
            }
        }
    }

   private:
    GmPoolName m_name;
    uint32_t m_defaultBlockNum;                     // number of preallocated blocks
    uint32_t m_defaultBlockSize;                    // size of each block in bytes
    std::vector<NpuRetrieval::GmBlock> m_freeList;  // list of free blocks
    std::mutex m_mutex;                             // guards m_freeList
    std::atomic<uint32_t> __attribute__((aligned(64))) m_count = 0;
};
}  // namespace NpuRetrieval
