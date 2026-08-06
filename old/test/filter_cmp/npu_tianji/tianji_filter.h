// Host launcher for bitmap_text_filter AIV kernel (Tianji port).
#pragma once

#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "../npu_common/acl_guard.h"
#include "../npu_common/npu_posting.h"
#include "../common/expr.h"

namespace filter_cmp {

class TianjiFilter {
   public:
    TianjiFilter(AclStream& stream, const InvertedPostings& posts, const ExprSpec& expr, uint32_t block_dim);
    ~TianjiFilter();

    TianjiFilter(const TianjiFilter&) = delete;
    TianjiFilter& operator=(const TianjiFilter&) = delete;

    std::vector<uint8_t> Run(const InvertedPostings& posts, const ExprSpec& expr);

   private:
    AclStream& stream_;
    uint32_t seg_u16_count_;
    uint32_t segments_num_;
    uint32_t doc_num_;
    uint32_t block_dim_;

    void* d_postings_ = nullptr;
    void* d_rpn_ = nullptr;
    void* d_result_ = nullptr;
    uint32_t rpn_capacity_ = 0;
    uint32_t rpn_len_ = 0;
};

}  // namespace filter_cmp
