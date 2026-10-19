#include "stream_manager.h"
#include "utils/logger.h"
#include "acl/acl.h"
#include "configuration/develop_configuration.h"
#include "src/full_recall/core/constant_definition.h"
#include "src/full_recall/core/log_definition.h"

namespace NpuRetrieval {
namespace {
constexpr int kMaxStreams = 1024;  // per-device stream cap
}

bool StreamManager::Init(const int32_t deviceId, uint32_t size) {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_streams.find(deviceId) != m_streams.end()) {
        LOG_WARN("stream of device:" << deviceId << " has been inited, not need to init it again.");
        return true;
    }
    // No aclrtResetDevice needed here: this process's device resources are released
    // automatically on process exit.
    auto ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtSetDevice fail, deviceId is:" << deviceId << " error code is:" << ret);
        return false;
    }
    std::stack<aclrtStream> streamStack;
    uint32_t created = 0;
    for (int i = 0; i < kMaxStreams && created < size; i++) {
        aclrtStream stream = nullptr;
        ret = aclrtCreateStream(&stream);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("aclrtCreateStream fail, deviceId is:" << deviceId << " error code is:" << ret << ",i:" << i);
            CHECK_ACL_ONLY_LOG(aclrtDestroyStream(stream));
            continue;
        }
        streamStack.push(stream);
        created++;
    }
    if (created < size) {
        LOG_WARN("stream of device:" << deviceId << " not enough, created:" << created << " requested:" << size);
        return false;
    }
    m_streams[deviceId] = std::move(streamStack);
    LOG_INFO("stream init success.");
    return true;
}

aclrtStream StreamManager::GetStream(const int32_t deviceId) {
    aclrtStream stream = nullptr;
    size_t remaining = 0;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto iter = m_streams.find(deviceId);
        if (iter == m_streams.end()) {
            return nullptr;
        }
        std::stack<aclrtStream>& streamStack = iter->second;
        if (!streamStack.empty()) {
            stream = streamStack.top();
            streamStack.pop();
            remaining = streamStack.size();
        } else {
            LOG_WARN("stream is not enough:");
        }
    }
    LOG_DEBUG("stream stack remain size:" << remaining << " stream:" << (uint64_t)stream);
    return stream;
}

void StreamManager::FreeStream(const int32_t deviceId, aclrtStream stream) {
    if (stream == nullptr) {
        LOG_DEBUG("stream is NULL");  // benign: nothing to return to the pool
        return;
    }
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto iter = m_streams.find(deviceId);
        if (iter == m_streams.end()) {
            LOG_WARN("invalid device id:" << deviceId);
            return;
        }
        iter->second.push(stream);
    }
    return;
}

}  // namespace NpuRetrieval
