#include "simple_filter.h"

#include "aclrtlaunch_simple_filter.h"

#include <cstring>
#include <stdexcept>

namespace filter_cmp {

SimpleFilter::SimpleFilter(AclStream& stream, const InvertedPostings& posts, const ExprSpec& expr,
                           uint32_t max_stack_depth)
    : stream_(stream) {
    seg_u16_count_ = posts.seg_u16_count;
    segments_num_ = posts.segments_num;
    doc_num_ = posts.doc_num;
    rpn_capacity_ = std::max<uint32_t>((uint32_t)expr.rpn.size(), 64u);
    rpn_len_ = (uint32_t)expr.rpn.size();
    (void)max_stack_depth;

    d_postings_ = DevAlloc<uint16_t>(posts.bits.size());
    H2D(d_postings_, posts.bits.data(), posts.bits.size() * sizeof(uint16_t), stream_.Get());

    d_rpn_ = DevAlloc<uint32_t>(rpn_capacity_);
    H2D(d_rpn_, expr.rpn.data(), rpn_len_ * sizeof(uint32_t), stream_.Get());

    d_result_ = DevAlloc<uint16_t>((size_t)segments_num_ * seg_u16_count_);
}

SimpleFilter::~SimpleFilter() {
    DevFree(d_postings_);
    DevFree(d_rpn_);
    DevFree(d_result_);
}

std::vector<uint8_t> SimpleFilter::Run(const InvertedPostings& posts, const ExprSpec& expr) {
    // Postings already uploaded in ctor (data is static across benchmark rounds).
    (void)posts;
    rpn_len_ = (uint32_t)expr.rpn.size();
    if (rpn_len_ > rpn_capacity_) {
        throw std::runtime_error("rpn_len exceeds capacity");
    }
    H2D(d_rpn_, expr.rpn.data(), rpn_len_ * sizeof(uint32_t), stream_.Get());

    auto ret = aclrtSetDevice(stream_.DeviceId());
    if (ret != ACL_SUCCESS)
        throw std::runtime_error("aclrtSetDevice failed");
    ACLRT_LAUNCH_KERNEL(simple_filter)(segments_num_, stream_.Get(), reinterpret_cast<uint8_t*>(d_postings_),
                                       reinterpret_cast<uint8_t*>(d_rpn_), reinterpret_cast<uint8_t*>(d_result_),
                                       seg_u16_count_, rpn_len_, segments_num_);
    ret = aclrtSynchronizeStream(stream_.Get());
    if (ret != ACL_SUCCESS)
        throw std::runtime_error("aclrtSynchronizeStream failed");

    std::vector<uint16_t> d2h_result((size_t)segments_num_ * seg_u16_count_, 0);
    D2H(d2h_result.data(), d_result_, d2h_result.size() * sizeof(uint16_t), stream_.Get());

    std::vector<uint8_t> out(doc_num_, 0);
    for (uint32_t d = 0; d < doc_num_; ++d) {
        uint32_t seg = d / posts.doc_num_per_seg;
        uint32_t local = d % posts.doc_num_per_seg;
        uint32_t unit = local >> 4;
        uint32_t bit = local & 15u;
        uint16_t v = d2h_result[(size_t)seg * seg_u16_count_ + unit];
        out[d] = (uint8_t)((v >> bit) & 1u);
    }
    return out;
}

}  // namespace filter_cmp
