// npu/fused_pipeline.h — device-side fused pipeline.
//
// Chains all 4 kernels with shared device buffers. Skips intermediate
// D2H/H2D between stages. Only H2D up front (query, docs, bitmap, rpn) and
// D2H at end (top-K result).
//
// vs chained launchers (--npu_full):
//   saves: scores D2H+H2D (M*4B × 2), filter D2H+H2D+u8→u32 (M × 2),
//          aggregator D2H+H2H compact (M*8B × 2)
//   cost:  same kernel launches + syncs

#pragma once

#include <acl/acl.h>
#include <acl/acl_base_rt.h>
#include <acl/acl_rt.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "../common/expr.h"
#include "aclrtlaunch_kernel_aggregator.h"
#include "aclrtlaunch_kernel_filter.h"
#include "aclrtlaunch_kernel_filter_inverted.h"
#include "aclrtlaunch_kernel_score.h"
#include "aclrtlaunch_kernel_topk.h"
#include "aclrtlaunch_kernel_topk_fused.h"
#include "aclrtlaunch_kernel_topk_fused_multi.h"
#include "aclrtlaunch_kernel_topk_merge.h"
#include "inverted_index_loader.h"
#include "session.h"
#include "timer.h"

// kernel_topk_fused is in same translation unit as kernel_topk.cpp but
// declared here because aclrtlaunch_*.h is auto-generated per kernel name.
extern "C" uint32_t aclrtlaunch_kernel_topk_fused(uint32_t blockDim, aclrtStream stream, void* scores, void* ids,
                                                  void* count_pblk, void* out_scores, void* out_ids, uint32_t k,
                                                  uint32_t total_blocks);

namespace full_npu {

constexpr uint32_t FUSED_SCORE_DOCS_PER_BLOCK = 256;
constexpr uint32_t FUSED_FILTER_DOCS_PER_BLOCK = 16;
constexpr uint32_t FUSED_AGG_DOCS_PER_BLOCK = 16;
constexpr uint32_t FUSED_AGG_COUNT_STRIDE = 16;
constexpr uint32_t FUSED_MAX_BLOCKS_PER_LAUNCH = 32;
constexpr uint32_t FUSED_N_BLOCK = 16;  // score MMad n-align
constexpr uint32_t FUSED_MAX_TOPK = 128;
constexpr uint32_t FUSED_TOPK_PARTIAL_STRIDE = ((FUSED_MAX_TOPK + 15) / 16) * 16;  // 128

class FusedPipeline {
   public:
    FusedPipeline(AclSession& sess, uint32_t vector_dim) : sess_(sess), k_(vector_dim) {}

    // Optional: enable inverted-index filter path (skips forward bitmap H2D).
    void SetInverted(const InvertedIndex* idx) {
        inv_idx_ = idx;
    }

    ~FusedPipeline() {
        if (d_query_)
            aclrtFree(d_query_);
        if (d_docs_)
            aclrtFree(d_docs_);
        if (d_scores_)
            aclrtFree(d_scores_);
        if (d_bitmap_)
            aclrtFree(d_bitmap_);
        if (d_rpn_)
            aclrtFree(d_rpn_);
        if (d_filter_)
            aclrtFree(d_filter_);
        if (d_postings_)
            aclrtFree(d_postings_);
        if (d_agg_ids_)
            aclrtFree(d_agg_ids_);
        if (d_agg_scores_)
            aclrtFree(d_agg_scores_);
        if (d_agg_count_)
            aclrtFree(d_agg_count_);
        if (d_partial_scores_)
            aclrtFree(d_partial_scores_);
        if (d_partial_ids_)
            aclrtFree(d_partial_ids_);
        if (d_partial_counts_)
            aclrtFree(d_partial_counts_);
        if (d_topk_scores_)
            aclrtFree(d_topk_scores_);
        if (d_topk_ids_)
            aclrtFree(d_topk_ids_);
    }

    struct Timing {
        double h2d_ms;
        double score_ms;
        double filter_ms;
        double agg_ms;
        double topk_ms;
        double d2h_ms;
        double total_ms;
    };

    // Run fused pipeline for one query.
    // docs_f32: flat FP32 docs [M * k]. bitmap: host bitmap region start.
    // Returns top-K via out_scores/out_ids (descending, size=topk).
    Timing Run(const float* query_f32, const float* docs_f32, const BooleanExpr& expr, const uint64_t* host_bitmap,
               uint32_t stride_words, uint32_t M, uint32_t topk, std::vector<float>* out_scores,
               std::vector<uint32_t>* out_ids, uint32_t* matched_count) {
        if (topk > FUSED_MAX_TOPK)
            throw std::runtime_error("fused: topk must be ≤ 128");
        out_scores->assign(topk, -1e30f);
        out_ids->assign(topk, 0xFFFFFFFFu);

        Timing tt;
        full_npu::Timer t_total;
        t_total.Start();

        // ====== H2D up-front ======
        full_npu::Timer t_h2d;
        t_h2d.Start();

        // Query: pad to [k × 16] FP16 (col 0 = query[i]).
        h_query_fp16_.assign(k_ * FUSED_N_BLOCK, 0);
        for (uint32_t i = 0; i < k_; ++i) {
            h_query_fp16_[i * FUSED_N_BLOCK] = aclFloatToFloat16(query_f32[i]);
        }
        if (!d_query_) {
            aclError ret =
                aclrtMalloc((void**)&d_query_, k_ * FUSED_N_BLOCK * sizeof(aclFloat16), ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("fused: alloc query");
        }
        aclError ret = aclrtMemcpy(d_query_, k_ * FUSED_N_BLOCK * sizeof(aclFloat16), h_query_fp16_.data(),
                                   k_ * FUSED_N_BLOCK * sizeof(aclFloat16), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS)
            throw std::runtime_error("fused: H2D query");

        // Docs FP32 → FP16 (cached across queries if same docs_f32 ptr + M).
        size_t docs_bytes = (size_t)M * k_ * sizeof(aclFloat16);
        if (docs_f32 == cached_docs_ptr_ && M == cached_docs_M_ && d_docs_ && docs_bytes <= d_docs_cap_) {
            // Reuse already-uploaded FP16 docs on device.
        } else {
            h_docs_fp16_.resize((size_t)M * k_);
            for (size_t i = 0; i < (size_t)M * k_; ++i) {
                h_docs_fp16_[i] = aclFloatToFloat16(docs_f32[i]);
            }
            if (!d_docs_ || docs_bytes > d_docs_cap_) {
                if (d_docs_)
                    aclrtFree(d_docs_);
                d_docs_cap_ = docs_bytes;
                ret = aclrtMalloc((void**)&d_docs_, d_docs_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS)
                    throw std::runtime_error("fused: alloc docs");
            }
            ret = aclrtMemcpy(d_docs_, docs_bytes, h_docs_fp16_.data(), docs_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("fused: H2D docs");
            cached_docs_ptr_ = docs_f32;
            cached_docs_M_ = M;
        }

        // Scores GM.
        size_t scores_bytes = (size_t)M * sizeof(float);
        if (!d_scores_ || scores_bytes > d_scores_cap_) {
            if (d_scores_)
                aclrtFree(d_scores_);
            d_scores_cap_ = ((M + 1023) / 1024) * 1024 * sizeof(float);
            ret = aclrtMalloc((void**)&d_scores_, d_scores_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("fused: alloc scores");
        }

        // Bitmap H2D (skipped when inverted-index path is enabled).
        if (inv_idx_) {
            // Inverted-index path: build compact postings + remapped RPN,
            // upload only expr's tags' segments (typically ~100 tags × segs × 2B).
            EncodeRpnInv(expr);
            uint32_t n_tags = (uint32_t)inv_tag_ids_.size();
            uint32_t seg_count = inv_idx_->Segments();
            size_t post_bytes = (size_t)n_tags * seg_count * sizeof(uint16_t);
            h_postings_.resize((size_t)n_tags * seg_count);
            for (uint32_t i = 0; i < n_tags; ++i) {
                inv_idx_->CopyTagSegments(inv_tag_ids_[i], h_postings_.data() + (size_t)i * seg_count);
            }
            if (!d_postings_ || post_bytes > d_postings_cap_) {
                if (d_postings_)
                    aclrtFree(d_postings_);
                d_postings_cap_ = std::max<size_t>(post_bytes, 4096);
                ret = aclrtMalloc((void**)&d_postings_, d_postings_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS)
                    throw std::runtime_error("fused: alloc postings");
            }
            if (post_bytes > 0) {
                ret = aclrtMemcpy(d_postings_, post_bytes, h_postings_.data(), post_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
                if (ret != ACL_SUCCESS)
                    throw std::runtime_error("fused: H2D postings");
            }
            inv_n_tags_ = n_tags;
            inv_seg_count_ = seg_count;
        } else {
            size_t bitmap_bytes = (size_t)M * stride_words * sizeof(uint64_t);
            if (!d_bitmap_ || bitmap_bytes > d_bitmap_cap_) {
                if (d_bitmap_)
                    aclrtFree(d_bitmap_);
                d_bitmap_cap_ = bitmap_bytes;
                ret = aclrtMalloc((void**)&d_bitmap_, d_bitmap_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
                if (ret != ACL_SUCCESS)
                    throw std::runtime_error("fused: alloc bitmap");
            }
            ret = aclrtMemcpy(d_bitmap_, bitmap_bytes, host_bitmap, bitmap_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("fused: H2D bitmap");
        }

        // RPN encode + H2D.
        if (inv_idx_) {
            EncodeRpnInv(expr);  // populates rpn_u32_ with tag_idx args
        } else {
            EncodeRpn(expr);
        }
        size_t rpn_bytes = rpn_u32_.size() * sizeof(uint32_t);
        if (!d_rpn_ || rpn_bytes > d_rpn_cap_) {
            if (d_rpn_)
                aclrtFree(d_rpn_);
            d_rpn_cap_ = std::max<size_t>(rpn_bytes, 256);
            ret = aclrtMalloc((void**)&d_rpn_, d_rpn_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("fused: alloc rpn");
        }
        ret = aclrtMemcpy(d_rpn_, rpn_bytes, rpn_u32_.data(), rpn_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS)
            throw std::runtime_error("fused: H2D rpn");

        // Filter GM (u32 per doc).
        size_t filter_bytes = (size_t)M * sizeof(uint32_t);
        if (!d_filter_ || filter_bytes > d_filter_cap_) {
            if (d_filter_)
                aclrtFree(d_filter_);
            size_t alloc = ((filter_bytes + 32767) / 32768) * 32768;
            d_filter_cap_ = alloc;
            ret = aclrtMalloc((void**)&d_filter_, d_filter_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("fused: alloc filter");
        }

        // Aggregator buffers.
        if (!d_agg_ids_ || scores_bytes > d_agg_ids_cap_) {
            if (d_agg_ids_)
                aclrtFree(d_agg_ids_);
            d_agg_ids_cap_ = scores_bytes;
            ret = aclrtMalloc((void**)&d_agg_ids_, d_agg_ids_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("fused: alloc agg_ids");
        }
        if (!d_agg_scores_ || scores_bytes > d_agg_scores_cap_) {
            if (d_agg_scores_)
                aclrtFree(d_agg_scores_);
            d_agg_scores_cap_ = scores_bytes;
            ret = aclrtMalloc((void**)&d_agg_scores_, d_agg_scores_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("fused: alloc agg_scores");
        }
        uint32_t total_agg_blocks = (M + FUSED_AGG_DOCS_PER_BLOCK - 1) / FUSED_AGG_DOCS_PER_BLOCK;
        size_t agg_count_bytes = (size_t)total_agg_blocks * FUSED_AGG_COUNT_STRIDE * sizeof(uint32_t);
        if (!d_agg_count_ || agg_count_bytes > d_agg_count_cap_) {
            if (d_agg_count_)
                aclrtFree(d_agg_count_);
            d_agg_count_cap_ = agg_count_bytes;
            ret = aclrtMalloc((void**)&d_agg_count_, d_agg_count_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("fused: alloc agg_count");
        }
        // Zero counts array.
        std::vector<uint32_t> zeros(total_agg_blocks * FUSED_AGG_COUNT_STRIDE, 0);
        ret = aclrtMemcpy(d_agg_count_, agg_count_bytes, zeros.data(), agg_count_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS)
            throw std::runtime_error("fused: zero agg_count");

        // TopK buffers. Reuse single-u32 count slot (must be u32* for kernel).
        if (!d_topk_count_) {
            ret = aclrtMalloc((void**)&d_topk_count_, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("fused: alloc topk_count");
        }
        size_t topk_bytes = (size_t)topk * sizeof(float);
        if (!d_topk_scores_ || topk_bytes > d_topk_scores_cap_) {
            if (d_topk_scores_)
                aclrtFree(d_topk_scores_);
            d_topk_scores_cap_ = topk_bytes;
            ret = aclrtMalloc((void**)&d_topk_scores_, d_topk_scores_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("fused: alloc topk_scores");
        }
        if (!d_topk_ids_ || topk_bytes > d_topk_ids_cap_) {
            if (d_topk_ids_)
                aclrtFree(d_topk_ids_);
            d_topk_ids_cap_ = topk_bytes;
            ret = aclrtMalloc((void**)&d_topk_ids_, d_topk_ids_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("fused: alloc topk_ids");
        }

        tt.h2d_ms = t_h2d.StopMs();

        // ====== Stage 1: score (chunked AIC) ======
        full_npu::Timer t_score;
        t_score.Start();
        uint32_t total_score_blocks = (M + FUSED_SCORE_DOCS_PER_BLOCK - 1) / FUSED_SCORE_DOCS_PER_BLOCK;
        if (total_score_blocks == 0)
            total_score_blocks = 1;
        for (uint32_t off = 0; off < total_score_blocks; off += FUSED_MAX_BLOCKS_PER_LAUNCH) {
            uint32_t b = std::min<uint32_t>(FUSED_MAX_BLOCKS_PER_LAUNCH, total_score_blocks - off);
            ACLRT_LAUNCH_KERNEL(kernel_score)(b, sess_.Stream(), d_query_, d_docs_, d_scores_, M, k_, off);
            sess_.Sync();
        }
        tt.score_ms = t_score.StopMs();

        // ====== Stage 2: filter (chunked AIV) ======
        full_npu::Timer t_filter;
        t_filter.Start();
        if (inv_idx_) {
            // Inverted-index filter: 1 block per segment.
            uint32_t seg_count = inv_seg_count_;
            for (uint32_t off = 0; off < seg_count; off += FUSED_MAX_BLOCKS_PER_LAUNCH) {
                uint32_t b = std::min<uint32_t>(FUSED_MAX_BLOCKS_PER_LAUNCH, seg_count - off);
                ACLRT_LAUNCH_KERNEL(kernel_filter_inverted)(b, sess_.Stream(), d_postings_, d_rpn_, d_filter_,
                                                            seg_count, (uint32_t)rpn_u32_.size(), M, off, inv_n_tags_);
                sess_.Sync();
            }
        } else {
            uint32_t total_filter_blocks = (M + FUSED_FILTER_DOCS_PER_BLOCK - 1) / FUSED_FILTER_DOCS_PER_BLOCK;
            for (uint32_t off = 0; off < total_filter_blocks; off += FUSED_MAX_BLOCKS_PER_LAUNCH) {
                uint32_t b = std::min<uint32_t>(FUSED_MAX_BLOCKS_PER_LAUNCH, total_filter_blocks - off);
                ACLRT_LAUNCH_KERNEL(kernel_filter)(b, sess_.Stream(), d_bitmap_, d_rpn_, d_filter_, stride_words,
                                                   (uint32_t)rpn_u32_.size(), M, off);
                sess_.Sync();
            }
        }
        tt.filter_ms = t_filter.StopMs();

        // ====== Stage 3: aggregator (chunked AIV) ======
        full_npu::Timer t_agg;
        t_agg.Start();
        for (uint32_t off = 0; off < total_agg_blocks; off += FUSED_MAX_BLOCKS_PER_LAUNCH) {
            uint32_t b = std::min<uint32_t>(FUSED_MAX_BLOCKS_PER_LAUNCH, total_agg_blocks - off);
            ACLRT_LAUNCH_KERNEL(kernel_aggregator)(b, sess_.Stream(), d_scores_, d_filter_, d_agg_ids_, d_agg_scores_,
                                                   d_agg_count_, M, off);
            sess_.Sync();
        }
        tt.agg_ms = t_agg.StopMs();

        // ====== Stage 4: multi-block topk (B partial heaps + 1 merge) ======
        // n_partial: ≤32 AIV blocks. Each block handles ceil(total/b) segments.
        full_npu::Timer t_topk;
        t_topk.Start();
        uint32_t n_partial = std::min<uint32_t>(FUSED_MAX_BLOCKS_PER_LAUNCH, std::max<uint32_t>(1, total_agg_blocks));
        uint32_t blocks_per_block = (total_agg_blocks + n_partial - 1) / n_partial;
        // Re-check after rounding: avoid launching blocks with zero work.
        uint32_t actual_n = (total_agg_blocks + blocks_per_block - 1) / blocks_per_block;

        size_t partial_scores_bytes = (size_t)actual_n * FUSED_TOPK_PARTIAL_STRIDE * sizeof(float);
        size_t partial_ids_bytes = (size_t)actual_n * FUSED_TOPK_PARTIAL_STRIDE * sizeof(uint32_t);
        size_t partial_counts_bytes = (size_t)actual_n * FUSED_AGG_COUNT_STRIDE * sizeof(uint32_t);
        if (partial_scores_bytes > d_partial_scores_cap_) {
            if (d_partial_scores_)
                aclrtFree(d_partial_scores_);
            ret = aclrtMalloc((void**)&d_partial_scores_, partial_scores_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("fused: alloc partial_scores");
            d_partial_scores_cap_ = partial_scores_bytes;
        }
        if (partial_ids_bytes > d_partial_ids_cap_) {
            if (d_partial_ids_)
                aclrtFree(d_partial_ids_);
            ret = aclrtMalloc((void**)&d_partial_ids_, partial_ids_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("fused: alloc partial_ids");
            d_partial_ids_cap_ = partial_ids_bytes;
        }
        if (partial_counts_bytes > d_partial_counts_cap_) {
            if (d_partial_counts_)
                aclrtFree(d_partial_counts_);
            ret = aclrtMalloc((void**)&d_partial_counts_, partial_counts_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("fused: alloc partial_counts");
            d_partial_counts_cap_ = partial_counts_bytes;
        }
        ACLRT_LAUNCH_KERNEL(kernel_topk_fused_multi)(actual_n, sess_.Stream(), d_agg_scores_, d_agg_ids_, d_agg_count_,
                                                     d_partial_scores_, d_partial_ids_, d_partial_counts_, topk,
                                                     total_agg_blocks, blocks_per_block);
        sess_.Sync();
        ACLRT_LAUNCH_KERNEL(kernel_topk_merge)(1, sess_.Stream(), d_partial_scores_, d_partial_ids_, d_partial_counts_,
                                               d_topk_scores_, d_topk_ids_, topk, actual_n);
        sess_.Sync();
        tt.topk_ms = t_topk.StopMs();

        // Matched count not strictly needed by host, but we still report it
        // via per-block count sum. Read total_blocks*16 u32s for stats.
        h_agg_count_.resize(total_agg_blocks * FUSED_AGG_COUNT_STRIDE);
        ret =
            aclrtMemcpy(h_agg_count_.data(), agg_count_bytes, d_agg_count_, agg_count_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS)
            throw std::runtime_error("fused: D2H agg_count (stats)");
        uint32_t compact_count = 0;
        for (uint32_t b = 0; b < total_agg_blocks; ++b) {
            compact_count += h_agg_count_[b * FUSED_AGG_COUNT_STRIDE];
        }

        // ====== D2H top-K ======
        full_npu::Timer t_d2h;
        t_d2h.Start();
        ret = aclrtMemcpy(out_scores->data(), topk * sizeof(float), d_topk_scores_, topk * sizeof(float),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS)
            throw std::runtime_error("fused: D2H topk_scores");
        ret = aclrtMemcpy(out_ids->data(), topk * sizeof(uint32_t), d_topk_ids_, topk * sizeof(uint32_t),
                          ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS)
            throw std::runtime_error("fused: D2H topk_ids");
        tt.d2h_ms = t_d2h.StopMs();

        *matched_count = compact_count;
        tt.total_ms = t_total.StopMs();
        return tt;
    }

   private:
    void EncodeRpn(const BooleanExpr& expr) {
        rpn_u32_.clear();
        rpn_u32_.reserve(expr.rpn.size());
        for (const auto& tok : expr.rpn) {
            uint32_t op = static_cast<uint32_t>(tok.kind);
            uint32_t arg = tok.arg & 0x0FFFFFFFu;
            rpn_u32_.push_back((op << 28) | arg);
        }
    }

    // Remap tag_id → compact tag_idx into inv_tag_ids_, pack RPN with indices.
    void EncodeRpnInv(const BooleanExpr& expr) {
        inv_tag_ids_.clear();
        std::unordered_map<uint32_t, uint32_t> tag_to_idx;
        for (const auto& tok : expr.rpn) {
            if (tok.kind == TokenKind::TAG || tok.kind == TokenKind::NOT_TAG) {
                if (tag_to_idx.find(tok.arg) == tag_to_idx.end()) {
                    tag_to_idx[tok.arg] = (uint32_t)inv_tag_ids_.size();
                    inv_tag_ids_.push_back(tok.arg);
                }
            }
        }
        rpn_u32_.clear();
        rpn_u32_.reserve(expr.rpn.size());
        for (const auto& tok : expr.rpn) {
            uint32_t op = static_cast<uint32_t>(tok.kind);
            uint32_t arg;
            if (tok.kind == TokenKind::TAG || tok.kind == TokenKind::NOT_TAG) {
                arg = tag_to_idx[tok.arg];
            } else {
                arg = tok.arg & 0x0FFFFFFFu;
            }
            rpn_u32_.push_back((op << 28) | arg);
        }
    }

    AclSession& sess_;
    uint32_t k_;

    std::vector<aclFloat16> h_query_fp16_;
    std::vector<aclFloat16> h_docs_fp16_;
    std::vector<uint32_t> rpn_u32_;
    std::vector<uint32_t> h_agg_count_;
    std::vector<uint32_t> h_agg_ids_;
    std::vector<float> h_agg_scores_;

    void* d_query_ = nullptr;
    void* d_docs_ = nullptr;
    size_t d_docs_cap_ = 0;
    void* d_scores_ = nullptr;
    size_t d_scores_cap_ = 0;
    void* d_bitmap_ = nullptr;
    size_t d_bitmap_cap_ = 0;
    void* d_rpn_ = nullptr;
    size_t d_rpn_cap_ = 0;
    void* d_filter_ = nullptr;
    size_t d_filter_cap_ = 0;
    void* d_agg_ids_ = nullptr;
    size_t d_agg_ids_cap_ = 0;
    void* d_agg_scores_ = nullptr;
    size_t d_agg_scores_cap_ = 0;
    void* d_agg_count_ = nullptr;
    size_t d_agg_count_cap_ = 0;
    void* d_topk_in_scores_ = nullptr;
    size_t d_topk_in_scores_cap_ = 0;
    void* d_topk_in_ids_ = nullptr;
    size_t d_topk_in_ids_cap_ = 0;
    void* d_topk_count_ = nullptr;
    void* d_topk_scores_ = nullptr;
    size_t d_topk_scores_cap_ = 0;
    void* d_topk_ids_ = nullptr;
    size_t d_topk_ids_cap_ = 0;
    void* d_partial_scores_ = nullptr;
    size_t d_partial_scores_cap_ = 0;
    void* d_partial_ids_ = nullptr;
    size_t d_partial_ids_cap_ = 0;
    void* d_partial_counts_ = nullptr;
    size_t d_partial_counts_cap_ = 0;

    // Inverted-index filter path.
    const InvertedIndex* inv_idx_ = nullptr;
    void* d_postings_ = nullptr;
    size_t d_postings_cap_ = 0;
    std::vector<uint16_t> h_postings_;
    std::vector<uint32_t> inv_tag_ids_;
    uint32_t inv_n_tags_ = 0;
    uint32_t inv_seg_count_ = 0;

    // Docs cache (avoid re-converting + re-uploading same docs across queries).
    const float* cached_docs_ptr_ = nullptr;
    uint32_t cached_docs_M_ = 0;
};

}  // namespace full_npu
