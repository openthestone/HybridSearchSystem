// npu/inverted_index_loader.h — tag-major inverted postings loader + filter.
//
// Loads dataset_HW_inv.bin (built by tools/build_inverted_index).
// Per query: picks only expr's tags, builds compact postings array, H2Ds,
// launches kernel_filter_inverted, D2Hs result.

#pragma once

#include <acl/acl.h>
#include <acl/acl_base_rt.h>
#include <acl/acl_rt.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "../common/expr.h"
#include "aclrtlaunch_kernel_filter_inverted.h"
#include "session.h"
#include "timer.h"

namespace full_npu {

#pragma pack(push, 1)
struct InvHeader {
    char magic[8];
    uint32_t version;
    uint32_t _pad0;
    uint64_t doc_num;
    uint32_t tag_num;
    uint32_t segments;
    uint32_t reserved;
    uint32_t _pad1;
};
#pragma pack(pop)
static_assert(sizeof(InvHeader) == 40, "inv header must be 40 bytes");

class InvertedIndex {
   public:
    bool Open(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f)
            return false;
        InvHeader h{};
        f.read(reinterpret_cast<char*>(&h), sizeof(h));
        if (std::memcmp(h.magic, "HYDINV02", 8) != 0) {
            std::fprintf(stderr, "[inv] bad magic: %.8s\n", h.magic);
            return false;
        }
        doc_num_ = h.doc_num;
        tag_num_ = h.tag_num;
        segments_ = h.segments;

        size_t post_count = (size_t)tag_num_ * segments_;
        postings_.resize(post_count);
        f.read(reinterpret_cast<char*>(postings_.data()), post_count * sizeof(uint16_t));
        if (!f) {
            std::fprintf(stderr, "[inv] short read\n");
            return false;
        }
        std::printf("[inv] loaded: doc_num=%llu tag_num=%u segments=%u (%.1f MB)\n", (unsigned long long)doc_num_,
                    tag_num_, segments_, (double)(post_count * 2) / 1048576.0);
        return true;
    }

    uint64_t DocNum() const {
        return doc_num_;
    }
    uint32_t TagNum() const {
        return tag_num_;
    }
    uint32_t Segments() const {
        return segments_;
    }

    // Copy one tag's segments into dst (segments × 2B).
    void CopyTagSegments(uint32_t tag_id, uint16_t* dst) const {
        if (tag_id >= tag_num_) {
            std::memset(dst, 0, segments_ * sizeof(uint16_t));
            return;
        }
        const uint16_t* src = postings_.data() + (size_t)tag_id * segments_;
        std::memcpy(dst, src, segments_ * sizeof(uint16_t));
    }

   private:
    uint64_t doc_num_ = 0;
    uint32_t tag_num_ = 0;
    uint32_t segments_ = 0;
    std::vector<uint16_t> postings_;
};

// Filter using inverted index. Returns u32 result per doc.
class InvertedFilterLauncher {
   public:
    InvertedFilterLauncher(AclSession& sess, const InvertedIndex& idx) : sess_(sess), idx_(idx) {}

    ~InvertedFilterLauncher() {
        if (d_postings_)
            aclrtFree(d_postings_);
        if (d_rpn_)
            aclrtFree(d_rpn_);
        if (d_result_)
            aclrtFree(d_result_);
    }

    // Run filter for expr over M docs. Writes result_u32[M].
    double Run(const BooleanExpr& expr, uint32_t M, std::vector<uint32_t>* result_u32) {
        if (M > idx_.DocNum())
            throw std::runtime_error("inv: M > idx doc_num");
        uint32_t seg_count = idx_.Segments();
        uint32_t docs_padded = seg_count * 16;
        result_u32->assign(docs_padded, 0);

        // 1) Collect unique tags from expr.rpn.
        std::unordered_map<uint32_t, uint32_t> tag_to_idx;
        std::vector<uint32_t> tag_ids;
        for (const auto& tok : expr.rpn) {
            if (tok.kind == TokenKind::TAG || tok.kind == TokenKind::NOT_TAG) {
                if (tag_to_idx.find(tok.arg) == tag_to_idx.end()) {
                    tag_to_idx[tok.arg] = (uint32_t)tag_ids.size();
                    tag_ids.push_back(tok.arg);
                }
            }
        }
        uint32_t n_tags = (uint32_t)tag_ids.size();

        Timer t_h2d;
        t_h2d.Start();

        // 2) Build compact postings: [n_tags * seg_count] uint16.
        size_t post_bytes = (size_t)n_tags * seg_count * sizeof(uint16_t);
        h_postings_.resize(n_tags * seg_count);
        for (uint32_t i = 0; i < n_tags; ++i) {
            idx_.CopyTagSegments(tag_ids[i], h_postings_.data() + (size_t)i * seg_count);
        }
        if (!d_postings_ || post_bytes > d_postings_cap_) {
            if (d_postings_)
                aclrtFree(d_postings_);
            d_postings_cap_ = std::max<size_t>(post_bytes, 4096);
            aclError ret = aclrtMalloc((void**)&d_postings_, d_postings_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("inv: alloc postings");
        }
        if (post_bytes > 0) {
            aclError ret =
                aclrtMemcpy(d_postings_, post_bytes, h_postings_.data(), post_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("inv: H2D postings");
        }

        // 3) Build remapped RPN (tag_id → compact tag_idx).
        rpn_u32_.clear();
        rpn_u32_.reserve(expr.rpn.size());
        for (const auto& tok : expr.rpn) {
            uint32_t op = static_cast<uint32_t>(tok.kind);
            uint32_t arg;
            if (tok.kind == TokenKind::TAG || tok.kind == TokenKind::NOT_TAG) {
                arg = tag_to_idx[tok.arg];  // remap to compact idx
            } else {
                arg = tok.arg;
            }
            rpn_u32_.push_back((op << 28) | (arg & 0x0FFFFFFFu));
        }
        size_t rpn_bytes = rpn_u32_.size() * sizeof(uint32_t);
        if (!d_rpn_ || rpn_bytes > d_rpn_cap_) {
            if (d_rpn_)
                aclrtFree(d_rpn_);
            d_rpn_cap_ = std::max<size_t>(rpn_bytes, 256);
            aclError ret = aclrtMalloc((void**)&d_rpn_, d_rpn_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("inv: alloc rpn");
        }
        if (rpn_bytes > 0) {
            aclError ret = aclrtMemcpy(d_rpn_, rpn_bytes, rpn_u32_.data(), rpn_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("inv: H2D rpn");
        }

        // 4) Alloc result GM.
        size_t result_bytes = (size_t)docs_padded * sizeof(uint32_t);
        if (!d_result_ || result_bytes > d_result_cap_) {
            if (d_result_)
                aclrtFree(d_result_);
            d_result_cap_ = ((result_bytes + 32767) / 32768) * 32768;
            if (d_result_cap_ == 0)
                d_result_cap_ = 32768;
            aclError ret = aclrtMalloc((void**)&d_result_, d_result_cap_, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("inv: alloc result");
        }

        // 5) Chunked launch (≤32 blocks per launch).
        constexpr uint32_t MAX_BLOCKS_PER_LAUNCH = 32;
        for (uint32_t off = 0; off < seg_count; off += MAX_BLOCKS_PER_LAUNCH) {
            uint32_t b = std::min<uint32_t>(MAX_BLOCKS_PER_LAUNCH, seg_count - off);
            ACLRT_LAUNCH_KERNEL(kernel_filter_inverted)(b, sess_.Stream(), d_postings_, d_rpn_, d_result_, seg_count,
                                                        (uint32_t)rpn_u32_.size(), M, off, n_tags);
            sess_.Sync();
        }

        // 6) D2H result.
        aclError ret =
            aclrtMemcpy(result_u32->data(), result_bytes, d_result_, result_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        if (ret != ACL_SUCCESS)
            throw std::runtime_error("inv: D2H result");

        result_u32->resize(M);  // trim padding
        return t_h2d.StopMs();
    }

   private:
    AclSession& sess_;
    const InvertedIndex& idx_;

    std::vector<uint16_t> h_postings_;
    std::vector<uint32_t> rpn_u32_;

    void* d_postings_ = nullptr;
    size_t d_postings_cap_ = 0;
    void* d_rpn_ = nullptr;
    size_t d_rpn_cap_ = 0;
    void* d_result_ = nullptr;
    size_t d_result_cap_ = 0;
};

}  // namespace full_npu
