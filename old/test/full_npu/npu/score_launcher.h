// npu/score_launcher.h — host launcher for kernel_score.
//
// Pipeline:
//   1. Convert FP32 query + FP32 docs → FP16 (host staging, aclFloatToFloat16)
//   2. H2D copy query + docs + alloc scores GM
//   3. Launch kernel_score(blockDim = ceil(M / 1024), stream, ...)
//   4. Sync + D2H scores
//
// Per-launch buffers held across calls (avoid alloc churn during sweep).

#pragma once

#include <acl/acl.h>
#include <acl/acl_base_rt.h>
#include <acl/acl_rt.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "aclrtlaunch_kernel_score.h"
#include "session.h"
#include "timer.h"

namespace full_npu {

constexpr uint32_t SCORE_DOCS_PER_BLOCK = 256;  // must match kernel_score.cpp

class ScoreLauncher {
   public:
    ScoreLauncher(AclSession& sess, uint32_t vector_dim) : sess_(sess), k_(vector_dim) {}

    ~ScoreLauncher() {
        if (d_query_)
            aclrtFree(d_query_);  // k_*16 fp16
        if (d_scores_)
            aclrtFree(d_scores_);
        ReallocDocs(0);
    }

    // Run scoring: query_f32 [k] · docs_f32 [M × k] → scores_f32 [M].
    // Returns per-launch latency in ms (H2D + kernel + D2H, NOT including FP32→FP16).
    double Run(const float* query_f32, const float* docs_f32, uint32_t M, std::vector<float>* scores_f32) {
        scores_f32->assign(M, 0.0f);

        // Query is broadcast to [k × 16] block layout for MMad (col 0 = query, cols 1-15 = 0).
        // After Mmad([m × k] · [k × 16]) → [m × 16], column 0 holds the dot products.
        constexpr uint32_t N_BLOCK = 16;
        h_query_fp16_.resize(k_ * N_BLOCK, 0);
        for (uint32_t i = 0; i < k_; ++i) {
            h_query_fp16_[i * N_BLOCK] = aclFloatToFloat16(query_f32[i]);
        }
        h_docs_fp16_.resize((size_t)M * k_);
        Fp32ToFp16(docs_f32, (size_t)M * k_, h_docs_fp16_.data());

        Timer t;
        t.Start();

        // Lazy-alloc persistent d_query_.
        if (!d_query_) {
            aclError ret = aclrtMalloc((void**)&d_query_, k_ * N_BLOCK * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclrtMalloc query failed: " + std::to_string(ret));
            }
        }

        aclError ret = aclrtMemcpy(d_query_, k_ * N_BLOCK * sizeof(aclFloat16), h_query_fp16_.data(),
                                   k_ * N_BLOCK * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtMemcpy query H2D failed: " + std::to_string(ret));
        }

        ReallocDocs(M);
        ret = aclrtMemcpy(d_docs_, (size_t)M * k_ * sizeof(aclFloat16), h_docs_fp16_.data(),
                          (size_t)M * k_ * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtMemcpy docs H2D failed: " + std::to_string(ret));
        }

        if (M > d_scores_cap_) {
            if (d_scores_)
                aclrtFree(d_scores_);
            d_scores_cap_ = ((M + 1023) / 1024) * 1024;
            ret = aclrtMalloc((void**)&d_scores_, d_scores_cap_ * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclrtMalloc scores failed: " + std::to_string(ret));
            }
        }

        uint32_t total_blocks = (M + SCORE_DOCS_PER_BLOCK - 1) / SCORE_DOCS_PER_BLOCK;
        if (total_blocks == 0)
            total_blocks = 1;

        // Chunked launch: AIC scheduler also has block-drop issues beyond ~32
        // blocks per launch on 910B3. Chunk to 32 + sync between chunks.
        // Kernel must take block_offset_blocks param.
        constexpr uint32_t MAX_BLOCKS_PER_LAUNCH = 32;
        for (uint32_t block_offset = 0; block_offset < total_blocks; block_offset += MAX_BLOCKS_PER_LAUNCH) {
            uint32_t blocks_this = std::min<uint32_t>(MAX_BLOCKS_PER_LAUNCH, total_blocks - block_offset);
            ACLRT_LAUNCH_KERNEL(kernel_score)(blocks_this, sess_.Stream(), d_query_, d_docs_, d_scores_, M, k_,
                                              block_offset);
            sess_.Sync();
        }

        ret =
            aclrtMemcpy(scores_f32->data(), M * sizeof(float), d_scores_, M * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtMemcpy scores D2H failed: " + std::to_string(ret));
        }

        sess_.Sync();
        return t.StopMs();
    }

   private:
    void ReallocDocs(uint32_t M) {
        size_t need = (size_t)M * k_ * sizeof(aclFloat16);
        if (need > d_docs_cap_) {
            if (d_docs_)
                aclrtFree(d_docs_);
            d_docs_cap_ = need;
            aclError ret = aclrtMalloc((void**)&d_docs_, d_docs_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclrtMalloc docs failed: " + std::to_string(ret));
            }
        } else if (M == 0 && d_docs_) {
            aclrtFree(d_docs_);
            d_docs_ = nullptr;
            d_docs_cap_ = 0;
        }
    }

    static void Fp32ToFp16(const float* src, size_t n, aclFloat16* dst) {
        for (size_t i = 0; i < n; ++i)
            dst[i] = aclFloatToFloat16(src[i]);
    }

    AclSession& sess_;
    uint32_t k_;

    void* d_query_ = nullptr;
    void* d_docs_ = nullptr;
    size_t d_docs_cap_ = 0;
    void* d_scores_ = nullptr;
    size_t d_scores_cap_ = 0;

    std::vector<aclFloat16> h_query_fp16_;
    std::vector<aclFloat16> h_docs_fp16_;
};

}  // namespace full_npu
