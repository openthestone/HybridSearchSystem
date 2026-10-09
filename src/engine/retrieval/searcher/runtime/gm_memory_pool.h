#pragma once

#include <cstdlib>
#include <cstdio>
#include <cstdint>
#include <vector>
#include "acl/acl.h"
#include "src/utils/env_switch.h"
#include "configuration/develop_configuration.h"
#include "utils/logger.h"
#include "src/full_recall/core/constant_definition.h"
#include "src/full_recall/core/log_definition.h"
#include "src/full_recall/core/full_recall_result.h"
#include <atomic>
namespace NpuRetrieval {

// NPUR_POOL_HIGH_WATER (default ON; 0 restores the old behaviour): grow a block to the largest
// size this pool has ever been asked for. Otherwise a query needing a big block can pull a small
// one and trigger a free+malloc of the OLD block -- hundreds of MB for the bitlist2set pool. The
// ceiling is unchanged; this only reaches it sooner. m_maxSize is atomic because the resize
// branch runs outside m_mutex, and a lost update just costs one more resize later.
static inline bool PoolHighWaterEnabled() {
    static const bool on = npur_env::OnByDefault("NPUR_POOL_HIGH_WATER");
    return on;
}

static inline bool PoolStatsEnabled() {
    static const bool on = npur_env::OnUnlessZero("NPUR_POOL_STATS");
    return on;
}

// NPUR_POOL_SMALL_H2D=1 (default off): take the scorer's query matrix and the filter's postfix
// expressions from their pools instead of a raw aclrtMalloc/aclrtFree every batch. Neither is
// released before the stream that reads it has been synced.
static inline bool PoolSmallH2dEnabled() {
    static const bool on = npur_env::On("NPUR_POOL_SMALL_H2D");
    return on;
}

class GmMemoryPool {
   public:
    GmMemoryPool(GmPoolName name, uint32_t blockNum, uint32_t blockSize)
        : m_name(name), m_defaultBlockNum(blockNum), m_defaultBlockSize(blockSize) {
        if (m_defaultBlockSize == 0) {
            return;
        }
        for (uint32_t i = 0; i < m_defaultBlockNum; ++i) {
            GmBlock block{m_name, nullptr, blockSize};
            auto ret = aclrtMalloc((void**)&block.data, m_defaultBlockSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("aclrtMalloc default block fail, error code is:" << ret);
                continue;
            }
            NoteAlloc(m_defaultBlockSize);
            m_freeList.emplace_back(block);
        }
    }

    ~GmMemoryPool() {
        if (PoolStatsEnabled()) {
            Report("final");
        }
        while (!m_freeList.empty()) {
            auto block = m_freeList.back();
            NoteFree(block.size);
            CHECK_ACL_ONLY_LOG(aclrtFree(block.data));
            m_freeList.pop_back();
        }
    }

    GmBlock Allocate(const uint32_t size) {
        GmBlock block{m_name, nullptr, 0};
        uint32_t want = size;
        if (PoolHighWaterEnabled()) {
            uint32_t seen = m_maxSize.load(std::memory_order_relaxed);
            while (size > seen && !m_maxSize.compare_exchange_weak(seen, size, std::memory_order_relaxed)) {
            }
            want = m_maxSize.load(std::memory_order_relaxed);
        }
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (m_freeList.empty()) {
                LOG_INFO("pool:" << (int)m_name << " block num is not enough, try to expand");
                block.size = want;
                auto ret = aclrtMalloc((void**)&block.data, want, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS) {
                    LOG_ERROR("Allocate fail, aclrtMalloc fail, error code is:" << ret);
                    block.data = nullptr;
                    return block;
                }
                NoteAlloc(want);
                m_count.fetch_add(1, std::memory_order_relaxed);
                LOG_INFO("pool:" << (int)m_name << " count:" << m_count.load(std::memory_order_relaxed));
                return block;
            }
            block = m_freeList.back();
            m_freeList.pop_back();
        }
        if (block.size < size) {
            LOG_INFO("pool:" << (int)m_name << " block size not enough, current:" << block.size << " applied:" << size);
            const uint32_t oldSize = block.size;
            CHECK_ACL_ONLY_LOG(aclrtFree(block.data));
            NoteFree(oldSize);
            block.data = nullptr;
            block.size = want;
            auto ret = aclrtMalloc((void**)&block.data, want, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("Allocate fail, aclrtMalloc fail, error code is:" << ret);
                block.data = nullptr;
                return block;
            }
            NoteAlloc(want);
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
                NoteFree(block.size);
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
    std::atomic<uint32_t> m_maxSize{0};             // largest size ever requested; see PoolHighWaterEnabled
    std::atomic<uint32_t> __attribute__((aligned(64))) m_count = 0;
    // NPUR_POOL_STATS: bytes held (free list + handed out), high-water mark, last peak reported.
    std::atomic<uint64_t> m_liveBytes{0};
    std::atomic<uint64_t> m_peakBytes{0};
    std::atomic<uint64_t> m_reportedPeak{0};
    std::atomic<uint64_t> m_mallocCount{0};

    void NoteAlloc(uint64_t bytes) {
        const uint64_t live = m_liveBytes.fetch_add(bytes, std::memory_order_relaxed) + bytes;
        m_mallocCount.fetch_add(1, std::memory_order_relaxed);
        uint64_t peak = m_peakBytes.load(std::memory_order_relaxed);
        while (live > peak && !m_peakBytes.compare_exchange_weak(peak, live, std::memory_order_relaxed)) {
        }
        if (!PoolStatsEnabled()) {
            return;
        }
        const uint64_t reported = m_reportedPeak.load(std::memory_order_relaxed);
        const uint64_t current = m_peakBytes.load(std::memory_order_relaxed);
        if (reported != 0 && current < reported + reported / 20) {
            return;
        }
        m_reportedPeak.store(current, std::memory_order_relaxed);
        Report("peak");
    }

    void NoteFree(uint64_t bytes) {
        m_liveBytes.fetch_sub(bytes, std::memory_order_relaxed);
    }

    void Report(const char* why) {
        // No device id: with two cards in one process each pool id appears twice, so sum them.
        fprintf(stdout, "[POOLSTAT] pool=%d %s liveMB=%.1f peakMB=%.1f blockNum=%u mallocs=%llu\n",
                static_cast<int>(m_name), why,
                static_cast<double>(m_liveBytes.load(std::memory_order_relaxed)) / 1048576.0,
                static_cast<double>(m_peakBytes.load(std::memory_order_relaxed)) / 1048576.0, m_defaultBlockNum,
                static_cast<unsigned long long>(m_mallocCount.load(std::memory_order_relaxed)));
        fflush(stdout);
    }
};
}  // namespace NpuRetrieval
