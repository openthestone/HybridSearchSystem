// common/expr.h — random Boolean expr generator compatible with sks_hw
// filter_expr semantics. Generates TAG/NOT_TAG/AND/OR tree, compiles to
// flat RPN-like sequence for fast per-doc evaluation.
//
// Output structure mirrors sks_hw BucketPlan but stripped down:
//   - leaf: tag_id + polarity (TAG vs NOT_TAG)
//   - AND_GROUP: child_count
//   - OR_GROUP:  child_count
//
// RPN flat form for evaluator:
//   token = (kind, tag_id_or_child_count)
//   kinds: 0=TAG, 1=NOT_TAG, 2=AND_N, 3=OR_N

#pragma once

#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

namespace full_npu {

enum class TokenKind : uint8_t {
    TAG = 0,
    NOT_TAG = 1,
    AND_N = 2,
    OR_N = 3,
};

struct ExprToken {
    TokenKind kind;
    uint32_t arg;  // tag_id for TAG/NOT_TAG, child_count for AND_N/OR_N
};

struct BooleanExpr {
    std::vector<ExprToken> rpn;  // post-order, leaf-first
    uint32_t leaves;             // total leaf count
};

// Generate a random Boolean expr with exactly `target_leaves` leaves.
// Tree shape: alternate AND/OR at each depth, children count 2-4.
inline BooleanExpr GenRandomExpr(uint32_t tag_num, uint32_t target_leaves, uint64_t seed) {
    BooleanExpr expr;
    expr.leaves = 0;

    if (target_leaves == 0)
        return expr;
    if (target_leaves == 1) {
        // Single TAG leaf (skip root op).
        std::mt19937_64 rng(seed);
        uint32_t tag_id = static_cast<uint32_t>(rng() % tag_num);
        bool neg = (rng() & 1) != 0;
        expr.rpn.push_back({neg ? TokenKind::NOT_TAG : TokenKind::TAG, tag_id});
        expr.leaves = 1;
        return expr;
    }

    std::mt19937_64 rng(seed);
    // Recursive build, post-order emit.
    // depth 0 = OR root, depth odd = AND, depth even = OR (alternate).
    std::function<void(uint32_t, uint32_t, bool)> build = [&](uint32_t depth, uint32_t budget, bool top) {
        if (budget == 0)
            return;
        if (budget == 1 || depth >= 6) {
            // Emit a leaf.
            uint32_t tag_id = static_cast<uint32_t>(rng() % tag_num);
            bool neg = (rng() & 1) != 0;
            expr.rpn.push_back({neg ? TokenKind::NOT_TAG : TokenKind::TAG, tag_id});
            expr.leaves++;
            return;
        }
        // Decide group: alternate OR at even depth, AND at odd.
        bool is_or = (depth % 2 == 0);
        // Decide child count: 2-4, but cap by budget.
        uint32_t max_child = std::min<uint32_t>(4, budget);
        uint32_t min_child = 2;
        uint32_t child_count = min_child + static_cast<uint32_t>(rng() % (max_child - min_child + 1));
        // Distribute budget across children, ensure each ≥1.
        std::vector<uint32_t> budgets(child_count, 1);
        uint32_t remaining = budget - child_count;
        for (uint32_t i = 0; i < remaining; ++i) {
            budgets[i % child_count]++;
        }
        for (uint32_t c : budgets) {
            build(depth + 1, c, false);
        }
        expr.rpn.push_back({is_or ? TokenKind::OR_N : TokenKind::AND_N, child_count});
    };

    build(0, target_leaves, true);
    return expr;
}

// Evaluate expr against a doc's bitmap (uint64 array of stride words).
// Returns true if doc matches.
inline bool EvalExpr(const BooleanExpr& e, const uint64_t* bm) {
    if (e.rpn.empty())
        return true;
    // Stack machine — max depth small.
    std::vector<uint8_t> stack;
    stack.reserve(64);
    for (const auto& tok : e.rpn) {
        switch (tok.kind) {
            case TokenKind::TAG: {
                bool v = (bm[tok.arg >> 6] >> (tok.arg & 63)) & 1ULL;
                stack.push_back(v ? 1 : 0);
                break;
            }
            case TokenKind::NOT_TAG: {
                bool v = (bm[tok.arg >> 6] >> (tok.arg & 63)) & 1ULL;
                stack.push_back(v ? 0 : 1);
                break;
            }
            case TokenKind::AND_N: {
                uint8_t acc = 1;
                for (uint32_t i = 0; i < tok.arg; ++i) {
                    acc &= stack.back();
                    stack.pop_back();
                    if (!acc) {
                        // Drain remaining children to keep stack aligned.
                        for (uint32_t j = i + 1; j < tok.arg; ++j)
                            stack.pop_back();
                        break;
                    }
                }
                stack.push_back(acc);
                break;
            }
            case TokenKind::OR_N: {
                uint8_t acc = 0;
                for (uint32_t i = 0; i < tok.arg; ++i) {
                    acc |= stack.back();
                    stack.pop_back();
                    if (acc) {
                        for (uint32_t j = i + 1; j < tok.arg; ++j)
                            stack.pop_back();
                        break;
                    }
                }
                stack.push_back(acc);
                break;
            }
        }
    }
    return stack.size() == 1 && stack[0] == 1;
}

}  // namespace full_npu
