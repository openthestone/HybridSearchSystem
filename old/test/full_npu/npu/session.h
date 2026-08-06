// npu/session.h — minimal ACL runtime RAII wrapper.
//
// aclInit on construct, teardown on destruct. One stream per session.
// Phase A.2: only device 0. Multi-device comes in Phase A.3.

#pragma once

#include <acl/acl.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace full_npu {

class AclSession {
   public:
    AclSession(int32_t device_id = 0) : device_id_(device_id) {
        aclError ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_INITIALIZE) {
            throw std::runtime_error("aclInit failed: " + std::to_string(ret));
        }
        owned_init_ = (ret == ACL_SUCCESS);

        ret = aclrtSetDevice(device_id_);
        if (ret != ACL_SUCCESS) {
            if (owned_init_)
                aclFinalize();
            throw std::runtime_error("aclrtSetDevice failed: " + std::to_string(ret));
        }

        ret = aclrtCreateStream(&stream_);
        if (ret != ACL_SUCCESS) {
            aclrtResetDevice(device_id_);
            if (owned_init_)
                aclFinalize();
            throw std::runtime_error("aclrtCreateStream failed: " + std::to_string(ret));
        }
    }

    ~AclSession() {
        if (stream_) {
            aclrtSynchronizeStream(stream_);
            aclrtDestroyStream(stream_);
        }
        aclrtResetDevice(device_id_);
        if (owned_init_)
            aclFinalize();
    }

    aclrtStream Stream() const {
        return stream_;
    }
    int32_t DeviceId() const {
        return device_id_;
    }

    // Synchronize the stream.
    void Sync() {
        aclError ret = aclrtSynchronizeStream(stream_);
        if (ret != ACL_SUCCESS) {
            std::fprintf(stderr, "[acl] Sync failed: %d\n", (int)ret);
        }
    }

    AclSession(const AclSession&) = delete;
    AclSession& operator=(const AclSession&) = delete;

   private:
    int32_t device_id_ = 0;
    aclrtStream stream_ = nullptr;
    bool owned_init_ = false;
};

}  // namespace full_npu
