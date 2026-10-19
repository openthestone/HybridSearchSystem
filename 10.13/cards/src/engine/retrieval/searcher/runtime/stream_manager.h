#pragma once

#include <memory>
#include <vector>
#include <stack>
#include <mutex>
#include <unordered_map>
#include "common/uncopyable.h"
#include "acl/acl.h"

namespace NpuRetrieval {

class StreamManager {
   public:
    static StreamManager* GetInstance() {
        static StreamManager instance;
        return &instance;
    }
    bool Init(const int32_t deviceId, uint32_t size);
    aclrtStream GetStream(const int32_t deviceId);
    void FreeStream(const int32_t deviceId, aclrtStream stream);
    NPURETRIEVAL_DECLARE_UNCOPYABLE(StreamManager);

   private:
    StreamManager() = default;
    ~StreamManager() = default;

   private:
    std::mutex m_mutex;
    std::unordered_map<int32_t, std::stack<aclrtStream>> m_streams;
};

}  // namespace NpuRetrieval
