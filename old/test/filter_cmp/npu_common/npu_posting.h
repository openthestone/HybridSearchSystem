// Posting encoder: convert forward bitmap to inverted per-(tag, segment) BIT_SET.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "raw_data.h"

namespace filter_cmp {

// BIT_SET layout per (tag, segment):
//   seg_u16_count = round_up(ceil(doc_num_per_seg / 16), 16)
//   bit i in unit u means doc (u*16 + i + seg*doc_num_per_seg) has tag
//   Bits beyond doc_num_per_seg are zero padding for vector op alignment.
//   seg_byte_size = seg_u16_count * 2
struct InvertedPostings {
    uint32_t tag_num = 0;
    uint32_t segments_num = 0;
    uint32_t doc_num_per_seg = 0;
    uint32_t seg_u16_count_real = 0;  // = ceil(doc_num_per_seg/16)
    uint32_t seg_u16_count = 0;       // = round_up(seg_u16_count_real, 16) — padded for vector ops
    uint32_t seg_byte_size = 0;       // = seg_u16_count * 2
    uint32_t doc_num = 0;
    std::vector<uint16_t> bits;  // size = tag_num * segments_num * seg_u16_count

    uint16_t Unit(uint32_t tag, uint32_t seg, uint32_t u) const {
        return bits[((size_t)tag * segments_num + seg) * seg_u16_count + u];
    }
    size_t ByteSize() const {
        return bits.size() * sizeof(uint16_t);
    }
};

InvertedPostings BuildInvertedPostings(const RawForwardBitmap& fwd, uint32_t segments_num);

}  // namespace filter_cmp
