#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "filter_expr.h"

namespace npur_harness {

struct FlatFilter {
    enum Kind : uint8_t { TermsOr, TermsAnd, And, Or, Not, True };
    Kind kind = True;
    std::vector<std::pair<uint32_t, uint64_t>> tests;
    std::vector<FlatFilter> kids;
};

namespace flat_detail {
inline void MergeTests(std::vector<std::pair<uint32_t, uint64_t>>& tests) {
    std::sort(tests.begin(), tests.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
    size_t w = 0;
    for (size_t i = 0; i < tests.size(); ++i) {
        if (w > 0 && tests[w - 1].first == tests[i].first)
            tests[w - 1].second |= tests[i].second;
        else
            tests[w++] = tests[i];
    }
    tests.resize(w);
}
}  // namespace flat_detail

inline FlatFilter FlattenFilter(const FilterNode* n, uint32_t words) {
    FlatFilter f;
    if (n == nullptr) {
        f.kind = FlatFilter::True;
        return f;
    }
    switch (n->op) {
        case FilterOp::Term: {
            f.kind = FlatFilter::TermsOr;
            const uint32_t w = static_cast<uint32_t>(n->tag >> 6);
            if (w < words)
                f.tests.emplace_back(w, 1ULL << (n->tag & 63u));
            return f;
        }
        case FilterOp::Not: {
            f.kind = FlatFilter::Not;
            f.kids.push_back(FlattenFilter(n->children[0].get(), words));
            return f;
        }
        case FilterOp::And:
        case FilterOp::Or: {
            const bool isAnd = n->op == FilterOp::And;
            std::vector<std::pair<uint32_t, uint64_t>> terms;
            std::vector<FlatFilter> subs;
            bool deadTerm = false;
            for (const auto& c : n->children) {
                if (c->op == FilterOp::Term) {
                    const uint32_t w = static_cast<uint32_t>(c->tag >> 6);
                    if (w < words)
                        terms.emplace_back(w, 1ULL << (c->tag & 63u));
                    else
                        deadTerm = true;
                } else {
                    subs.push_back(FlattenFilter(c.get(), words));
                }
            }
            flat_detail::MergeTests(terms);
            if (isAnd && deadTerm) {
                f.kind = FlatFilter::TermsAnd;
                f.tests.assign(1, {0u, ~0ULL});
                return f;
            }
            if (subs.empty()) {
                f.kind = isAnd ? FlatFilter::TermsAnd : FlatFilter::TermsOr;
                f.tests = std::move(terms);
                return f;
            }
            f.kind = isAnd ? FlatFilter::And : FlatFilter::Or;
            if (!terms.empty()) {
                FlatFilter t;
                t.kind = isAnd ? FlatFilter::TermsAnd : FlatFilter::TermsOr;
                t.tests = std::move(terms);
                f.kids.push_back(std::move(t));
            }
            for (FlatFilter& s : subs)
                f.kids.push_back(std::move(s));
            return f;
        }
    }
    f.kind = FlatFilter::True;
    return f;
}

inline bool EvalFlatFilter(const FlatFilter& f, const uint64_t* bm) {
    switch (f.kind) {
        case FlatFilter::True:
            return true;
        case FlatFilter::TermsOr:
            for (const auto& t : f.tests)
                if (bm[t.first] & t.second)
                    return true;
            return false;
        case FlatFilter::TermsAnd:
            for (const auto& t : f.tests)
                if ((bm[t.first] & t.second) != t.second)
                    return false;
            return true;
        case FlatFilter::Or:
            for (const FlatFilter& k : f.kids)
                if (EvalFlatFilter(k, bm))
                    return true;
            return false;
        case FlatFilter::And:
            for (const FlatFilter& k : f.kids)
                if (!EvalFlatFilter(k, bm))
                    return false;
            return true;
        case FlatFilter::Not:
            return !EvalFlatFilter(f.kids[0], bm);
    }
    return true;
}

inline size_t FlatFilterTests(const FlatFilter& f) {
    size_t n = f.tests.size();
    for (const FlatFilter& k : f.kids)
        n += FlatFilterTests(k);
    return n;
}

}  // namespace npur_harness
