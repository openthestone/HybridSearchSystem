/*
 * filter_expr.h — parse old/'s filter expressions into a small boolean AST, and
 * evaluate them against a document's tag set (for the CPU brute-force recall
 * reference). Pure C++ (no NPU / engine/ deps) so it can be unit-tested locally.
 *
 * Grammar (from old/filter_expr_600.txt), precedence NOT > AND > OR:
 *   expr    := or_expr
 *   or_expr := and_expr ("OR"  and_expr)*
 *   and_expr:= not_expr ("AND" not_expr)*
 *   not_expr:= "NOT" not_expr | atom
 *   atom    := NUMBER | "(" expr ")"
 * NUMBER is a tag id. AND/OR are flattened into n-ary nodes to match engine/'s
 * AndNode/OrNode (which take multiple children).
 *
 * The harness turns a FilterNode into a NpuRetrieval::QueryNode tree (see
 * main.cpp) for the NPU path, and uses EvalFilter() for the CPU reference.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace npur_harness {

enum class FilterOp { Term, And, Or, Not };

struct FilterNode {
    FilterOp op;
    uint64_t tag = 0;  // valid when op == Term
    std::vector<std::unique_ptr<FilterNode>> children;
};

// ---- tokenizer -------------------------------------------------------------
struct FilterTokenizer {
    const std::string& s;
    size_t i = 0;
    explicit FilterTokenizer(const std::string& str) : s(str) {}

    void skipWs() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\r' || s[i] == '\n'))
            i++;
    }
    bool eof() {
        skipWs();
        return i >= s.size();
    }
    // Peek the next token without consuming: returns "" at eof, else "(", ")",
    // "AND", "OR", "NOT", or "#" for a number (value in numVal).
    std::string peek(uint64_t& numVal) {
        skipWs();
        if (i >= s.size())
            return "";
        char c = s[i];
        if (c == '(')
            return "(";
        if (c == ')')
            return ")";
        if (c >= '0' && c <= '9') {
            size_t j = i;
            uint64_t v = 0;
            while (j < s.size() && s[j] >= '0' && s[j] <= '9') {
                v = v * 10 + (s[j] - '0');
                j++;
            }
            numVal = v;
            return "#";
        }
        // keyword
        size_t j = i;
        while (j < s.size() && ((s[j] >= 'A' && s[j] <= 'Z') || (s[j] >= 'a' && s[j] <= 'z')))
            j++;
        return s.substr(i, j - i);
    }
    // Consume the token peek() returned.
    void consume(const std::string& tok) {
        skipWs();
        if (tok == "#") {
            while (i < s.size() && s[i] >= '0' && s[i] <= '9')
                i++;
        } else {
            i += tok.size();
        }
    }
};

// forward
inline std::unique_ptr<FilterNode> ParseOr(FilterTokenizer& tk, bool& ok);

inline std::unique_ptr<FilterNode> ParseAtom(FilterTokenizer& tk, bool& ok) {
    uint64_t num = 0;
    std::string t = tk.peek(num);
    if (t == "(") {
        tk.consume("(");
        auto node = ParseOr(tk, ok);
        std::string close = tk.peek(num);
        if (close != ")") {
            ok = false;
            return nullptr;
        }
        tk.consume(")");
        return node;
    }
    if (t == "#") {
        tk.consume("#");
        auto n = std::make_unique<FilterNode>();
        n->op = FilterOp::Term;
        n->tag = num;
        return n;
    }
    ok = false;
    return nullptr;
}

inline std::unique_ptr<FilterNode> ParseNot(FilterTokenizer& tk, bool& ok) {
    uint64_t num = 0;
    std::string t = tk.peek(num);
    if (t == "NOT") {
        tk.consume("NOT");
        auto child = ParseNot(tk, ok);
        if (!ok)
            return nullptr;
        auto n = std::make_unique<FilterNode>();
        n->op = FilterOp::Not;
        n->children.push_back(std::move(child));
        return n;
    }
    return ParseAtom(tk, ok);
}

inline std::unique_ptr<FilterNode> ParseAnd(FilterTokenizer& tk, bool& ok) {
    auto first = ParseNot(tk, ok);
    if (!ok)
        return nullptr;
    std::vector<std::unique_ptr<FilterNode>> parts;
    parts.push_back(std::move(first));
    uint64_t num = 0;
    while (tk.peek(num) == "AND") {
        tk.consume("AND");
        auto next = ParseNot(tk, ok);
        if (!ok)
            return nullptr;
        parts.push_back(std::move(next));
    }
    if (parts.size() == 1)
        return std::move(parts[0]);
    auto n = std::make_unique<FilterNode>();
    n->op = FilterOp::And;
    n->children = std::move(parts);
    return n;
}

inline std::unique_ptr<FilterNode> ParseOr(FilterTokenizer& tk, bool& ok) {
    auto first = ParseAnd(tk, ok);
    if (!ok)
        return nullptr;
    std::vector<std::unique_ptr<FilterNode>> parts;
    parts.push_back(std::move(first));
    uint64_t num = 0;
    while (tk.peek(num) == "OR") {
        tk.consume("OR");
        auto next = ParseAnd(tk, ok);
        if (!ok)
            return nullptr;
        parts.push_back(std::move(next));
    }
    if (parts.size() == 1)
        return std::move(parts[0]);
    auto n = std::make_unique<FilterNode>();
    n->op = FilterOp::Or;
    n->children = std::move(parts);
    return n;
}

// Parse a full expression. Returns nullptr on empty string (== match all) or on
// parse error (ok set to false).
inline std::unique_ptr<FilterNode> ParseFilter(const std::string& expr, bool& ok) {
    ok = true;
    FilterTokenizer tk(expr);
    if (tk.eof())
        return nullptr;  // empty -> match all
    auto node = ParseOr(tk, ok);
    if (!ok)
        return nullptr;
    if (!tk.eof()) {
        ok = false;
        return nullptr;
    }  // trailing garbage
    return node;
}

// Evaluate the filter against a document's tag set.
inline bool EvalFilter(const FilterNode* n, const std::unordered_set<uint32_t>& tags) {
    if (n == nullptr)
        return true;  // no filter -> match all
    switch (n->op) {
        case FilterOp::Term:
            return tags.count(static_cast<uint32_t>(n->tag)) > 0;
        case FilterOp::And:
            for (const auto& c : n->children)
                if (!EvalFilter(c.get(), tags))
                    return false;
            return true;
        case FilterOp::Or:
            for (const auto& c : n->children)
                if (EvalFilter(c.get(), tags))
                    return true;
            return false;
        case FilterOp::Not:
            return !EvalFilter(n->children[0].get(), tags);
    }
    return true;
}

// Is bit `tag` set in a doc's bitmap (`words` u64 words)?
inline bool BitSet(const uint64_t* bm, uint32_t words, uint64_t tag) {
    uint32_t w = static_cast<uint32_t>(tag >> 6);
    if (w >= words)
        return false;
    return (bm[w] >> (tag & 63u)) & 1ULL;
}

// Evaluate the filter directly against a doc's raw bitmap (no per-doc tag-set
// allocation) — fast enough to brute-force over the full corpus. Semantics match
// EvalFilter (an absent tag == term is false). NOTE: this treats a missing tag as
// false, whereas engine/ DROPS a term whose token is absent from the index; the two
// agree only when every referenced tag exists in the index (i.e. the full
// dataset). Use a full-corpus index for a valid recall check.
inline bool EvalFilterBitmap(const FilterNode* n, const uint64_t* bm, uint32_t words) {
    if (n == nullptr)
        return true;
    switch (n->op) {
        case FilterOp::Term:
            return BitSet(bm, words, n->tag);
        case FilterOp::And:
            for (const auto& c : n->children)
                if (!EvalFilterBitmap(c.get(), bm, words))
                    return false;
            return true;
        case FilterOp::Or:
            for (const auto& c : n->children)
                if (EvalFilterBitmap(c.get(), bm, words))
                    return true;
            return false;
        case FilterOp::Not:
            return !EvalFilterBitmap(n->children[0].get(), bm, words);
    }
    return true;
}

}  // namespace npur_harness
