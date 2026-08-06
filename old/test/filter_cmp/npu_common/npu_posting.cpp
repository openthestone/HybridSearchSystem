#include "npu_posting.h"

#include <stdexcept>

namespace filter_cmp {

InvertedPostings BuildInvertedPostings(const RawForwardBitmap& fwd, uint32_t segments_num) {
    if (segments_num == 0)
        throw std::runtime_error("segments_num == 0");
    if (fwd.doc_num % segments_num != 0) {
        // doc_num_per_seg must be integer; require even split (caller can round up).
        throw std::runtime_error("doc_num must be divisible by segments_num");
    }
    InvertedPostings p;
    p.tag_num = fwd.tag_num;
    p.segments_num = segments_num;
    p.doc_num_per_seg = fwd.doc_num / segments_num;
    p.doc_num = fwd.doc_num;
    p.seg_u16_count_real = (p.doc_num_per_seg + 15u) / 16u;
    // Round up to multiple of 16 for vector op alignment (one vec reg = 16 uint16).
    p.seg_u16_count = (p.seg_u16_count_real + 15u) & ~15u;
    if (p.seg_u16_count == 0)
        p.seg_u16_count = 16;
    p.seg_byte_size = p.seg_u16_count * 2u;
    p.bits.assign((size_t)fwd.tag_num * segments_num * p.seg_u16_count, 0);

    const uint32_t u16 = p.seg_u16_count;
    for (uint32_t d = 0; d < fwd.doc_num; ++d) {
        uint32_t seg = d / p.doc_num_per_seg;
        uint32_t local = d % p.doc_num_per_seg;
        uint32_t unit = local >> 4;
        uint32_t bit = local & 15u;
        const uint64_t* docw = &fwd.bits[(size_t)d * fwd.u64_per_doc];
        for (uint32_t w = 0; w < fwd.u64_per_doc; ++w) {
            uint64_t word = docw[w];
            if (!word)
                continue;
            uint32_t base_tag = w * 64u;
            while (word) {
                uint32_t off = (uint32_t)__builtin_ctzll(word);
                uint32_t tag = base_tag + off;
                word &= word - 1;
                p.bits[((size_t)tag * segments_num + seg) * u16 + unit] |= (uint16_t)(1u << bit);
            }
        }
    }
    return p;
}

}  // namespace filter_cmp
