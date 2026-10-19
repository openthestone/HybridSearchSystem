#pragma once
#include <cstdint>

// Common definitions exposed to the host side; only pointers (GM ADDR) or built-in types.
// Set on a filter operand's address to mean "still the encoded sparse posting, not a converted
// 16KB bitset". Every address is aclrtMalloc'd, so bit 0 is otherwise always clear.
const uint64_t SPARSE_OPERAND_TAG = 1ULL;
// Written INSTEAD of an address when an operand's posting is all-zero: a sentinel, never
// dereferenced. Only OR operands are given this; the others must not see one.
const uint64_t EMPTY_OPERAND_TAG = 2ULL;
// The DMA granule. Shared, not device-only, because FinishAdd pads the posting region by one.
constexpr uint32_t BLOCK_SIZE = 32;
// PostingLayout::SPARSE_PACKED as it appears in the head word's top 4 bits. Device code cannot
// include the format header, so posting_field_data.cpp static_asserts the two against each other.
const uint32_t SPARSE_PACKED_LAYOUT_CODE = 4U;

struct TextFilterContextData {
    uint32_t postingNum;     // number of postings (= number of tokens)
    uint32_t segmentNum;     // number of segments per posting
    uint32_t segmentLength;  // number of bitset elements per segment (uint16_t elements)
    uint32_t tileNum;        // number of data tiles per segment
    uint32_t exprLen;        // length of the post-order operator-expression array
    uint32_t opNum;          // number of operators in the post-order expression array
    uint32_t flags;
};

// Bits 8..10 carry NPUR_OR_ABLATE: skip one step of FilterOrOp::ApplySparse to price it. Every
// non-zero value produces WRONG filter results by construction.
const uint32_t FILTER_ABLATE_SHIFT = 8;
const uint32_t FILTER_ABLATE_MASK = 0x7u << FILTER_ABLATE_SHIFT;
const uint32_t OR_ABLATE_PLACE = 1;  // everything but the placement loop
const uint32_t OR_ABLATE_COPY = 2;   // ... and not the input DataCopy either
const uint32_t OR_ABLATE_FLOOR = 3;  // return as soon as K is known, before any queue

// Bits 2..4 are free: OR_K_IN_POINTER, OR_BULK_TARGETS and OR_UNROLL_PLACE lived there.
enum FilterFlag : uint32_t {
    // tileNum is always 1, so ApplySparse's per-unit tile bound is provably dead -- but it
    // arrives at runtime, so the compiler cannot fold it away.
    FILTER_FLAG_OR_NO_TILE_CHECK = 1u << 0,
    // An all-zero posting is the identity for OR, so the host tags those pointers instead.
    FILTER_FLAG_OR_SKIP_EMPTY = 1u << 1,
    // Describes the index, not an experiment: TextFilter sets it from the load-time walk.
    FILTER_FLAG_SPARSE_PACKED = 1u << 5,
    // Round a sparse operand's CopyIn up to a whole 32-byte block and use a plain DataCopy.
    FILTER_FLAG_OR_ALIGNED_COPY = 1u << 6,
};
