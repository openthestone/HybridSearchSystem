// Boolean filter expression + RPN encoding (shared by CPU and NPU paths).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace filter_cmp {

enum class NodeKind : uint8_t {
    LEAF = 0,
    AND = 1,
    OR = 2,
    NOT = 3,
};

struct ExprNode {
    NodeKind kind;
    uint32_t tag_id;  // only valid when kind == LEAF
};

struct ExprTree {
    std::vector<ExprNode> nodes;  // root is nodes.back(); LEAF children before parents
    uint32_t leaf_count = 0;
};

// RPN op codes for NPU kernel side (kept identical to Tianji BitmapTextFilter):
//   POSTING_TYPE_NODE      0  : push posting[tag_id] from posting list
//   AND_NODE               1  : pop a, pop b, push a & b
//   OR_NODE                2  : pop a, pop b, push a | b
//   NOT_NODE               3  : pop a, push ~a (XOR with ones)
//   CONJ_NODE              4  : reserved (Tianji supports a 3-ary conj, unused here)
enum RpnOp : uint32_t {
    POSTING_TYPE_NODE = 0,
    AND_NODE = 1,
    OR_NODE = 2,
    NOT_NODE = 3,
    CONJ_NODE = 4,
};

// Each RPN slot packs (op, tag_id):
//   - bit[31..28]: op
//   - bit[27..0]:  tag id (valid for POSTING_TYPE_NODE only)
inline uint32_t rpn_pack(RpnOp op, uint32_t tag_id) {
    return ((uint32_t)op << 28) | (tag_id & 0x0FFFFFFFu);
}
inline RpnOp rpn_op(uint32_t slot) {
    return (RpnOp)(slot >> 28);
}
inline uint32_t rpn_tag(uint32_t slot) {
    return slot & 0x0FFFFFFFu;
}

// Compile a random boolean tree into both ExprTree and RPN stream.
// Depth d controls leaf count: roughly 2^d leaves but with NOT/OR mix.
struct ExprSpec {
    ExprTree tree;
    std::vector<uint32_t> rpn;
    std::string pretty;
};

// Deterministic RNG so all three paths see the same expr.
ExprSpec BuildRandomExpr(uint32_t num_tags, uint32_t target_leaves, uint64_t seed);

}  // namespace filter_cmp
