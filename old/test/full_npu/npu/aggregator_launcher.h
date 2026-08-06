// npu/aggregator_launcher.h — host launcher for kernel_aggregator.
//
// Pipeline:
//   1. H2D scores (FP32, from CPU score path for now)
//   2. H2D filter mask (u32 0/1)
//   3. Zero per-block counts array (total_blocks u32s)
//   4. Launch kernel_aggregator in chunks of 32 blocks (910B3 AIV cliff)
//   5. Sync + D2H counts, reduce on host → compact out_doc_ids/out_scores
//
// Per-block layout (no atomics): block b writes compacted entries to
// outDocIds[b*16 .. b*16+n), outScores[b*16 .. b*16+n), count[b]=n.

#pragma once

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "aclrtlaunch_kernel_aggregator.h"
#include "session.h"
#include "timer.h"

namespace full_npu {

class AggregatorLauncher {
   public:
    AggregatorLauncher(AclSession& sess) : sess_(sess) {}

    ~AggregatorLauncher() {
        if (d_scores_)
            aclrtFree(d_scores_);
        if (d_filter_)
            aclrtFree(d_filter_);
        if (d_out_ids_)
            aclrtFree(d_out_ids_);
        if (d_out_scores_)
            aclrtFree(d_out_scores_);
        if (d_count_)
            aclrtFree(d_count_);
    }

    // Run aggregator: scan first `total_docs` of scores[]+filter[].
    // Returns matched count; fills out_doc_ids/out_scores (resized to count).
    double Run(const float* host_scores, const uint32_t* host_filter, uint32_t total_docs,
               std::vector<uint32_t>* out_doc_ids, std::vector<float>* out_scores, uint32_t* out_count) {
        constexpr uint32_t DOCS_PER_BLOCK_AGG = 16;
        constexpr uint32_t MAX_BLOCKS_PER_LAUNCH = 32;
        uint32_t total_blocks = (total_docs + DOCS_PER_BLOCK_AGG - 1) / DOCS_PER_BLOCK_AGG;

        out_doc_ids->assign(total_docs, 0);
        out_scores->assign(total_docs, 0.0f);
        h_counts_.assign(total_blocks * 16, 0);

        size_t scores_bytes = (size_t)total_docs * sizeof(float);
        size_t filter_bytes = (size_t)total_docs * sizeof(uint32_t);
        size_t count_bytes = (size_t)total_blocks * 16 * sizeof(uint32_t);

        if (!d_scores_ || scores_bytes > d_scores_cap_) {
            if (d_scores_)
                aclrtFree(d_scores_);
            d_scores_cap_ = scores_bytes;
            aclrtMalloc((void**)&d_scores_, d_scores_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        if (!d_filter_ || filter_bytes > d_filter_cap_) {
            if (d_filter_)
                aclrtFree(d_filter_);
            d_filter_cap_ = filter_bytes;
            aclrtMalloc((void**)&d_filter_, d_filter_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        if (!d_out_ids_ || scores_bytes > d_out_ids_cap_) {
            if (d_out_ids_)
                aclrtFree(d_out_ids_);
            d_out_ids_cap_ = scores_bytes;
            aclrtMalloc((void**)&d_out_ids_, d_out_ids_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        if (!d_out_scores_ || scores_bytes > d_out_scores_cap_) {
            if (d_out_scores_)
                aclrtFree(d_out_scores_);
            d_out_scores_cap_ = scores_bytes;
            aclrtMalloc((void**)&d_out_scores_, d_out_scores_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
        }
        if (!d_count_ || count_bytes > d_count_cap_) {
            if (d_count_)
                aclrtFree(d_count_);
            d_count_cap_ = count_bytes;
            aclrtMalloc((void**)&d_count_, d_count_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
        }

        Timer t;
        t.Start();

        aclError ret = aclrtMemcpy(d_scores_, scores_bytes, host_scores, scores_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS)
            throw std::runtime_error("agg H2D scores: " + std::to_string(ret));
        ret = aclrtMemcpy(d_filter_, filter_bytes, host_filter, filter_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS)
            throw std::runtime_error("agg H2D filter: " + std::to_string(ret));

        // Zero per-block counts.
        std::vector<uint32_t> zeros(total_blocks * 16, 0);
        ret = aclrtMemcpy(d_count_, count_bytes, zeros.data(), count_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS)
            throw std::runtime_error("agg zero counts: " + std::to_string(ret));

        // Chunked launch.
        for (uint32_t block_offset = 0; block_offset < total_blocks; block_offset += MAX_BLOCKS_PER_LAUNCH) {
            uint32_t blocks_this = std::min<uint32_t>(MAX_BLOCKS_PER_LAUNCH, total_blocks - block_offset);
            ACLRT_LAUNCH_KERNEL(kernel_aggregator)(blocks_this, sess_.Stream(), d_scores_, d_filter_, d_out_ids_,
                                                   d_out_scores_, d_count_, total_docs, block_offset);
            sess_.Sync();
        }

        // D2H counts.
        ret = aclrtMemcpy(h_counts_.data(), count_bytes, d_count_, count_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS)
            throw std::runtime_error("agg D2H counts: " + std::to_string(ret));

        // D2H full output slots.
        ret = aclrtMemcpy(out_doc_ids->data(), scores_bytes, d_out_ids_, scores_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS)
            throw std::runtime_error("agg D2H ids: " + std::to_string(ret));
        ret = aclrtMemcpy(out_scores->data(), scores_bytes, d_out_scores_, scores_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS)
            throw std::runtime_error("agg D2H scores: " + std::to_string(ret));

        // Host-side reduction: concatenate per-block compacted entries.
        std::vector<uint32_t> compact_ids;
        std::vector<float> compact_sc;
        compact_ids.reserve(total_docs);
        compact_sc.reserve(total_docs);
        for (uint32_t b = 0; b < total_blocks; ++b) {
            uint32_t n = h_counts_[b * 16];
            uint32_t slot_base = b * DOCS_PER_BLOCK_AGG;
            for (uint32_t i = 0; i < n; ++i) {
                compact_ids.push_back((*out_doc_ids)[slot_base + i]);
                compact_sc.push_back((*out_scores)[slot_base + i]);
            }
        }
        *out_count = (uint32_t)compact_ids.size();
        out_doc_ids->swap(compact_ids);
        out_scores->swap(compact_sc);

        return t.StopMs();
    }

   private:
    AclSession& sess_;
    std::vector<uint32_t> h_counts_;

    void* d_scores_ = nullptr;
    size_t d_scores_cap_ = 0;
    void* d_filter_ = nullptr;
    size_t d_filter_cap_ = 0;
    void* d_out_ids_ = nullptr;
    size_t d_out_ids_cap_ = 0;
    void* d_out_scores_ = nullptr;
    size_t d_out_scores_cap_ = 0;
    void* d_count_ = nullptr;
    size_t d_count_cap_ = 0;
};

}  // namespace full_npu
