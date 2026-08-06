// npu/topk_launcher.h — host launcher for kernel_topk.
//
// Pipeline:
//   1. H2D compacted scores + ids + count (from aggregator output)
//   2. Launch kernel_topk (single block, serial partial sort)
//   3. Sync + D2H top-K scores + ids
//
// Host reverses min-heap output to descending order.

#pragma once

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "aclrtlaunch_kernel_topk.h"
#include "session.h"
#include "timer.h"

namespace full_npu {

class TopkLauncher {
   public:
    TopkLauncher(AclSession& sess) : sess_(sess) {}

    ~TopkLauncher() {
        if (d_scores_)
            aclrtFree(d_scores_);
        if (d_ids_)
            aclrtFree(d_ids_);
        if (d_count_)
            aclrtFree(d_count_);
        if (d_out_sc_)
            aclrtFree(d_out_sc_);
        if (d_out_ids_)
            aclrtFree(d_out_ids_);
    }

    // Run top-K over compacted (scores, ids) of size `count`.
    // Fills out_topk_scores / out_topk_ids with K entries (descending).
    double Run(const float* host_scores, const uint32_t* host_ids, uint32_t count, uint32_t k,
               std::vector<float>* out_topk_scores, std::vector<uint32_t>* out_topk_ids) {
        if (k > 128)
            throw std::runtime_error("topk k must be ≤ 128");

        out_topk_scores->assign(k, -1e30f);
        out_topk_ids->assign(k, 0xFFFFFFFFu);

        size_t scores_bytes = (size_t)count * sizeof(float);
        size_t ids_bytes = (size_t)count * sizeof(uint32_t);
        size_t out_sc_bytes = (size_t)k * sizeof(float);
        size_t out_id_bytes = (size_t)k * sizeof(uint32_t);

        // Lazy alloc.
        if (!d_scores_ || scores_bytes > d_scores_cap_) {
            if (d_scores_)
                aclrtFree(d_scores_);
            d_scores_cap_ = scores_bytes;
            aclrtMalloc((void**)&d_scores_, d_scores_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        if (!d_ids_ || ids_bytes > d_ids_cap_) {
            if (d_ids_)
                aclrtFree(d_ids_);
            d_ids_cap_ = ids_bytes;
            aclrtMalloc((void**)&d_ids_, d_ids_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        if (!d_count_) {
            aclrtMalloc((void**)&d_count_, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
        }
        if (!d_out_sc_ || out_sc_bytes > d_out_sc_cap_) {
            if (d_out_sc_)
                aclrtFree(d_out_sc_);
            d_out_sc_cap_ = out_sc_bytes;
            aclrtMalloc((void**)&d_out_sc_, d_out_sc_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        if (!d_out_ids_ || out_id_bytes > d_out_ids_cap_) {
            if (d_out_ids_)
                aclrtFree(d_out_ids_);
            d_out_ids_cap_ = out_id_bytes;
            aclrtMalloc((void**)&d_out_ids_, d_out_ids_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        Timer t;
        t.Start();

        // Handle empty input: write sentinel output directly, skip kernel.
        if (count == 0) {
            for (uint32_t i = 0; i < k; ++i) {
                (*out_topk_scores)[i] = -1e30f;
                (*out_topk_ids)[i] = 0xFFFFFFFFu;
            }
            return t.StopMs();
        }

        aclError ret = aclrtMemcpy(d_scores_, scores_bytes, host_scores, scores_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS)
            throw std::runtime_error("topk H2D scores: " + std::to_string(ret));
        ret = aclrtMemcpy(d_ids_, ids_bytes, host_ids, ids_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS)
            throw std::runtime_error("topk H2D ids: " + std::to_string(ret));
        ret = aclrtMemcpy(d_count_, sizeof(uint32_t), &count, sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS)
            throw std::runtime_error("topk H2D count: " + std::to_string(ret));

        // Single-block launch.
        ACLRT_LAUNCH_KERNEL(kernel_topk)(1, sess_.Stream(), d_scores_, d_ids_, d_count_, d_out_sc_, d_out_ids_, k);
        sess_.Sync();

        // D2H output.
        ret = aclrtMemcpy(out_topk_scores->data(), out_sc_bytes, d_out_sc_, out_sc_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS)
            throw std::runtime_error("topk D2H scores: " + std::to_string(ret));
        ret = aclrtMemcpy(out_topk_ids->data(), out_id_bytes, d_out_ids_, out_id_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS)
            throw std::runtime_error("topk D2H ids: " + std::to_string(ret));

        // Kernel already produces descending order via in-kernel heap-sort.
        return t.StopMs();
    }

   private:
    AclSession& sess_;

    void* d_scores_ = nullptr;
    size_t d_scores_cap_ = 0;
    void* d_ids_ = nullptr;
    size_t d_ids_cap_ = 0;
    void* d_count_ = nullptr;
    void* d_out_sc_ = nullptr;
    size_t d_out_sc_cap_ = 0;
    void* d_out_ids_ = nullptr;
    size_t d_out_ids_cap_ = 0;
};

}  // namespace full_npu
