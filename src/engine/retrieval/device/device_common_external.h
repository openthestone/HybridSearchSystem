#pragma once
#include <cstdint>

// Common definitions exposed to the host side; only pointers (GM ADDR) or
// C/C++ built-in types are supported.
struct TextFilterContextData {
    uint32_t postingNum;     // number of postings (= number of tokens)
    uint32_t segmentNum;     // number of segments per posting
    uint32_t segmentLength;  // number of bitset elements per segment (uint16_t elements)
    uint32_t tileNum;        // number of data tiles per segment
    uint32_t exprLen;        // length of the post-order operator-expression array
    uint32_t opNum;          // number of operators in the post-order expression array
};
