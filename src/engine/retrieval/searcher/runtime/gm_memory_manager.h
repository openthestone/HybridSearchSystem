#pragma once

#include <memory>
#include <vector>
#include "gm_memory_pool.h"
#include "utils/safe_unordered_map.h"

namespace NpuRetrieval {

// Manages the per-device set of memory pools.
class GmMemoryManager {
   public:
    GmMemoryManager();
    ~GmMemoryManager() = default;
    static GmMemoryManager* GetByDeviceId(int32_t deviceId) {
        static SafeUnorderedMap<int32_t, std::shared_ptr<GmMemoryManager>> m_deviceMemoryManager;
        std::shared_ptr<GmMemoryManager> instance = nullptr;
        m_deviceMemoryManager.Find(deviceId, instance);
        if (instance == nullptr) {
            std::shared_ptr<GmMemoryManager> instance = std::make_shared<GmMemoryManager>();
            m_deviceMemoryManager.Insert(deviceId, instance);
            return instance.get();
        } else {
            return instance.get();
        }
    }

    // allocate a block from the named pool
    void AllocateBlock(const GmPoolName poolName, uint32_t size, GmBlock& chunk);
    // return a block to its pool
    void FreeBlock(GmBlock& chunk);

   private:
    const std::shared_ptr<GmMemoryPool> GetMemoryPool(const GmPoolName poolName);
    std::vector<std::shared_ptr<GmMemoryPool>> m_memoryPools;
};

}  // namespace NpuRetrieval
