#include "expr.h"

#include <random>
#include <sstream>
#include <stdexcept>

namespace filter_cmp {

ExprSpec BuildRandomExpr(uint32_t num_tags, uint32_t target_leaves, uint64_t seed) {
    if (num_tags == 0)
        throw std::runtime_error("num_tags == 0");
    if (target_leaves == 0)
        target_leaves = 1;
    ExprSpec s;
    std::mt19937_64 rng(seed);

    // Phase 1: emit target_leaves leaves
    for (uint32_t i = 0; i < target_leaves; ++i) {
        uint32_t tag = (uint32_t)(rng() % num_tags);
        s.rpn.push_back(rpn_pack(POSTING_TYPE_NODE, tag));
        s.tree.nodes.push_back({NodeKind::LEAF, tag});
    }
    s.tree.leaf_count = target_leaves;

    // Phase 2: emit target_leaves-1 binary ops, optionally preceded by NOT on top operand.
    // RPN validity: at any point, the number of operands on stack >= 1.
    // Binary op consumes 2 produces 1 (net -1). NOT consumes 1 produces 1 (net 0).
    // After all leaves emitted, operands = target_leaves. Each binary op reduces by 1.
    // NOT does not reduce, so it's safe to insert.
    for (uint32_t i = 1; i < target_leaves; ++i) {
        if ((rng() & 1u) && (rng() & 1u)) {  // ~25% chance: insert NOT before this combine
            s.rpn.push_back(rpn_pack(NOT_NODE, 0));
            s.tree.nodes.push_back({NodeKind::NOT, 0});
        }
        RpnOp op = (rng() & 1u) ? AND_NODE : OR_NODE;
        s.rpn.push_back(rpn_pack(op, 0));
        s.tree.nodes.push_back({op == AND_NODE ? NodeKind::AND : NodeKind::OR, 0});
    }
    // Maybe invert at root.
    if ((rng() & 1u)) {
        s.rpn.push_back(rpn_pack(NOT_NODE, 0));
        s.tree.nodes.push_back({NodeKind::NOT, 0});
    }

    std::ostringstream oss;
    oss << "[" << target_leaves << "-leaf RPN len=" << s.rpn.size() << "]";
    s.pretty = oss.str();
    return s;
}

}  // namespace filter_cmp
