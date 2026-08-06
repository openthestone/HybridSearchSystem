// npu/filter_launcher.h — host launcher for kernel_filter.
//
// Pipeline:
//   1. Encode BooleanExpr RPN → uint32 token array (CPU)
//   2. H2D: rpn_u32 + (reuse existing) bitmap
//   3. Launch kernel_filter(blockDim = total_docs, stream, ...)
//   4. Sync + D2H result_u8
//
// Bitmap is read directly from dataset_HW.bin layout (no conversion).
// Result is per-doc uint8 match flag.

#pragma once

#include <acl/acl.h>
#include <acl/acl_rt.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "../common/expr.h"
#include "aclrtlaunch_kernel_filter.h"
#include "session.h"
#include "timer.h"

namespace full_npu {

class FilterLauncher {
   public:
    FilterLauncher(AclSession& sess) : sess_(sess) {}

    ~FilterLauncher() {
        if (d_rpn_)
            aclrtFree(d_rpn_);
        if (d_result_)
            aclrtFree(d_result_);
    }

    // Run filter on first `total_docs` docs. Bitmap is the host pointer to
    // the dataset's bitmap region (row-major docs × stride words).
    // Returns per-launch latency in ms (H2D rpn + launch + D2H result).
    // Bitmap is NOT copied — kernel reads it from host-pinned GM via H2D copy
    //   of the whole bitmap region in Phase A.2 (deferred optimization: stream
    //   chunks). For now we copy the whole doc_subset bitmap region.
    double Run(const BooleanExpr& expr, const uint64_t* host_bitmap, uint32_t stride_words, uint32_t total_docs,
               std::vector<uint8_t>* result_u8) {
        result_u8->assign(total_docs, 0);
        h_result_u32_.assign(total_docs, 0);

        // --- Encode RPN ---
        EncodeRpn(expr);

        // --- Copy whole bitmap region H2D ---
        size_t bitmap_bytes = (size_t)total_docs * stride_words * sizeof(uint64_t);

        Timer t;
        t.Start();

        if (!d_bitmap_ || bitmap_bytes > d_bitmap_cap_) {
            if (d_bitmap_)
                aclrtFree(d_bitmap_);
            d_bitmap_cap_ = bitmap_bytes;
            aclError ret = aclrtMalloc((void**)&d_bitmap_, d_bitmap_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclrtMalloc bitmap failed: " + std::to_string(ret));
            }
        }
        aclError ret = aclrtMemcpy(d_bitmap_, bitmap_bytes, host_bitmap, bitmap_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtMemcpy bitmap H2D failed: " + std::to_string(ret));
        }

        // --- RPN H2D ---
        size_t rpn_bytes = rpn_u32_.size() * sizeof(uint32_t);
        if (!d_rpn_ || rpn_bytes > d_rpn_cap_) {
            if (d_rpn_)
                aclrtFree(d_rpn_);
            d_rpn_cap_ = std::max<size_t>(rpn_bytes, 256);  // min alloc
            ret = aclrtMalloc((void**)&d_rpn_, d_rpn_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclrtMalloc rpn failed: " + std::to_string(ret));
            }
        }
        ret = aclrtMemcpy(d_rpn_, rpn_bytes, rpn_u32_.data(), rpn_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtMemcpy rpn H2D failed: " + std::to_string(ret));
        }

        // --- Result GM alloc/reuse (uint32 per doc) ---
        size_t result_bytes = (size_t)total_docs * sizeof(uint32_t);
        // Round up to 32KB page boundary + slack.
        size_t result_alloc = ((result_bytes + 32767) / 32768) * 32768;
        if (!d_result_ || result_alloc > d_result_cap_) {
            if (d_result_)
                aclrtFree(d_result_);
            d_result_cap_ = result_alloc;
            ret = aclrtMalloc((void**)&d_result_, d_result_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS) {
                throw std::runtime_error("aclrtMalloc result failed: " + std::to_string(ret));
            }
        }

        // --- Launch: chunk into <=32 blocks per launch (910B3 AIV cliff).
        //     Each launch passes block_offset_blocks so kernel can find global doc id.
        constexpr uint32_t DOCS_PER_BLOCK_FILTER = 16;
        constexpr uint32_t MAX_BLOCKS_PER_LAUNCH = 32;
        uint32_t total_blocks = (total_docs + DOCS_PER_BLOCK_FILTER - 1) / DOCS_PER_BLOCK_FILTER;

        for (uint32_t block_offset = 0; block_offset < total_blocks; block_offset += MAX_BLOCKS_PER_LAUNCH) {
            uint32_t blocks_this = std::min<uint32_t>(MAX_BLOCKS_PER_LAUNCH, total_blocks - block_offset);
            ACLRT_LAUNCH_KERNEL(kernel_filter)(blocks_this, sess_.Stream(), d_bitmap_, d_rpn_, d_result_, stride_words,
                                               (uint32_t)rpn_u32_.size(), total_docs, block_offset);
            sess_.Sync();
        }

        // --- D2H result (uint32 → uint8 view) ---
        ret = aclrtMemcpy(h_result_u32_.data(), result_bytes, d_result_, result_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS) {
            throw std::runtime_error("aclrtMemcpy result D2H failed: " + std::to_string(ret));
        }
        for (uint32_t i = 0; i < total_docs; ++i) {
            (*result_u8)[i] = h_result_u32_[i] ? 1 : 0;
        }

        sess_.Sync();
        return t.StopMs();
    }

    // Accessor for verification.
    const std::vector<uint32_t>& RpnU32() const {
        return rpn_u32_;
    }

   private:
    // Encode BooleanExpr (kind, arg) → uint32 (op<<28 | arg).
    void EncodeRpn(const BooleanExpr& expr) {
        rpn_u32_.clear();
        rpn_u32_.reserve(expr.rpn.size());
        for (const auto& tok : expr.rpn) {
            uint32_t op = static_cast<uint32_t>(tok.kind);  // 0..3 matches kernel
            uint32_t arg = tok.arg & 0x0FFFFFFFu;
            rpn_u32_.push_back((op << 28) | arg);
        }
    }

    AclSession& sess_;
    std::vector<uint32_t> rpn_u32_;
    std::vector<uint32_t> h_result_u32_;

    void* d_bitmap_ = nullptr;
    size_t d_bitmap_cap_ = 0;
    void* d_rpn_ = nullptr;
    size_t d_rpn_cap_ = 0;
    void* d_result_ = nullptr;
    size_t d_result_cap_ = 0;
};

}  // namespace full_npu
