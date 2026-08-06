// ACL RAII: device set + stream create/destroy.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "acl/acl.h"

namespace filter_cmp {

class AclStream {
   public:
    AclStream(int32_t device_id) : device_id_(device_id) {
        auto ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
            throw std::runtime_error("aclInit failed: " + std::to_string(ret));
        }
        ret = aclrtSetDevice(device_id_);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtSetDevice failed: " + std::to_string(ret));
        }
        ret = aclrtCreateStream(&stream_);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtCreateStream failed: " + std::to_string(ret));
        }
    }
    ~AclStream() {
        if (stream_) {
            aclrtSynchronizeStream(stream_);
            aclrtDestroyStream(stream_);
        }
        aclrtResetDevice(device_id_);
        // Do not aclFinalize: leave for the process teardown (other streams may exist).
    }
    aclrtStream Get() const {
        return stream_;
    }
    int32_t DeviceId() const {
        return device_id_;
    }

   private:
    int32_t device_id_;
    aclrtStream stream_ = nullptr;
};

template <typename T>
T* DevAlloc(size_t count) {
    T* p = nullptr;
    auto ret = aclrtMalloc((void**)&p, count * sizeof(T), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        throw std::runtime_error("aclrtMalloc failed: " + std::to_string(ret));
    }
    return p;
}

inline void DevFree(void* p) {
    if (p)
        aclrtFree(p);
}

inline void H2D(void* dst, const void* src, size_t bytes, aclrtStream stream) {
    auto ret = aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS)
        throw std::runtime_error("aclrtMemcpy H2D failed: " + std::to_string(ret));
    (void)stream;
}

inline void D2H(void* dst, const void* src, size_t bytes, aclrtStream stream) {
    auto ret = aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS)
        throw std::runtime_error("aclrtMemcpy D2H failed: " + std::to_string(ret));
    (void)stream;
}

}  // namespace filter_cmp
