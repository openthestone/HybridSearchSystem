// Host launcher for simple_filter AIV kernel.
#pragma once

#include <cstdint>
#include <vector>

#include "acl/acl.h"
#include "../npu_common/acl_guard.h"
#include "../npu_common/npu_posting.h"
#include "../common/expr.h"

namespace filter_cmp {

class SimpleFilter {
   public:
    // Allocates device buffers once for given posting layout + expr.
    // Reused across rounds to isolate per-query kernel cost.
    SimpleFilter(AclStream& stream, const InvertedPostings& posts, const ExprSpec& expr, uint32_t max_stack_depth);
    ~SimpleFilter();

    SimpleFilter(const SimpleFilter&) = delete;
    SimpleFilter& operator=(const SimpleFilter&) = delete;

    // Returns result as uint8_t[doc_num] (1 bit per doc, packed to bytes externally).
    // expr.rpn may differ in length from ctor expr — d_rpn_ sized for worst case.
    std::vector<uint8_t> Run(const InvertedPostings& posts, const ExprSpec& expr);

   private:
    AclStream& stream_;
    uint32_t seg_u16_count_;
    uint32_t segments_num_;
    uint32_t doc_num_;

    void* d_postings_ = nullptr;
    void* d_rpn_ = nullptr;
    void* d_result_ = nullptr;
    uint32_t rpn_capacity_ = 0;
    uint32_t rpn_len_ = 0;
};

}  // namespace filter_cmp
