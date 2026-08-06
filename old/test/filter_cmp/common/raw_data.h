// Synthetic raw data: 100K docs x 1024 tags, forward bitmap (sks_hw style).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace filter_cmp {

// Forward bitmap: doc i occupies bits [i*U64_PER_DOC, (i+1)*U64_PER_DOC).
// Layout matches sks_hw: uint64[ceil(num_tags/64)] per doc, row-major.
struct RawForwardBitmap {
    uint32_t doc_num = 0;
    uint32_t tag_num = 0;
    uint32_t u64_per_doc = 0;    // = ceil(tag_num/64)
    std::vector<uint64_t> bits;  // size = doc_num * u64_per_doc

    uint64_t DocWord(uint32_t doc_id, uint32_t word_idx) const {
        return bits[(size_t)doc_id * u64_per_doc + word_idx];
    }
    void SetBit(uint32_t doc_id, uint32_t tag_id) {
        uint32_t word = tag_id >> 6;
        uint32_t bit = tag_id & 63u;
        bits[(size_t)doc_id * u64_per_doc + word] |= (1ULL << bit);
    }
    bool GetBit(uint32_t doc_id, uint32_t tag_id) const {
        uint32_t word = tag_id >> 6;
        uint32_t bit = tag_id & 63u;
        return (bits[(size_t)doc_id * u64_per_doc + word] >> bit) & 1ULL;
    }
};

// Each (doc, tag) bit is set independently with prob density.
// RNG seeded for reproducibility across paths.
RawForwardBitmap BuildForwardBitmap(uint32_t doc_num, uint32_t tag_num, double density, uint64_t seed);

// Reference CPU filter (forward bitmap): returns boolean per-doc match.
// Used to verify correctness of all 3 paths against ground truth.
std::vector<uint8_t> ReferenceFilterCpu(const RawForwardBitmap& data, const struct ExprSpec& expr);

}  // namespace filter_cmp
