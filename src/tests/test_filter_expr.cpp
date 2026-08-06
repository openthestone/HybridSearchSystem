// Tests for src/harness/filter_expr.h — the filter-expression parser and the
// two evaluators (tag-set and bitmap). Pure C++, no NPU/protobuf deps.
#include "src/harness/filter_expr.h"

#include <cstdio>
#include <unordered_set>
#include <vector>

using namespace npur_harness;

static int g_fail = 0, g_pass = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (cond) {                                                              \
            ++g_pass;                                                            \
        } else {                                                                 \
            ++g_fail;                                                            \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                        \
    } while (0)

// Evaluate helper: parse + eval against a tag set. -1 on parse error.
static int ev(const std::string& e, const std::unordered_set<uint32_t>& tags) {
    bool ok = true;
    auto n = ParseFilter(e, ok);
    if (!ok)
        return -1;
    return EvalFilter(n.get(), tags) ? 1 : 0;
}

// Build a bitmap from a tag set and eval both ways; assert they agree, return set-eval.
static int evBoth(const std::string& e, const std::unordered_set<uint32_t>& tags, uint32_t words = 560) {
    bool ok = true;
    auto n = ParseFilter(e, ok);
    if (!ok)
        return -1;
    std::vector<uint64_t> bm(words, 0);
    for (uint32_t t : tags)
        if (t / 64 < words)
            bm[t / 64] |= (1ULL << (t % 64));
    bool a = EvalFilter(n.get(), tags);
    bool b = EvalFilterBitmap(n.get(), bm.data(), words);
    CHECK(a == b);  // the two evaluators must always agree
    return a ? 1 : 0;
}

static bool parses(const std::string& e) {
    bool ok = true;
    auto n = ParseFilter(e, ok);
    (void)n;
    return ok;
}

int main() {
    // ---- basic term / and / or / not ----
    CHECK(ev("47", {47}) == 1);
    CHECK(ev("47", {48}) == 0);
    CHECK(ev("47 AND 48", {47, 48}) == 1);
    CHECK(ev("47 AND 48", {47}) == 0);
    CHECK(ev("47 OR 48", {48}) == 1);
    CHECK(ev("47 OR 48", {99}) == 0);
    CHECK(ev("NOT 47", {48}) == 1);
    CHECK(ev("NOT 47", {47}) == 0);

    // ---- precedence: NOT > AND > OR ----
    // "1 OR 2 AND 3" == "1 OR (2 AND 3)"
    CHECK(ev("1 OR 2 AND 3", {1}) == 1);
    CHECK(ev("1 OR 2 AND 3", {2}) == 0);
    CHECK(ev("1 OR 2 AND 3", {3}) == 0);
    CHECK(ev("1 OR 2 AND 3", {2, 3}) == 1);
    // "NOT 1 AND 2" == "(NOT 1) AND 2"
    CHECK(ev("NOT 1 AND 2", {2}) == 1);
    CHECK(ev("NOT 1 AND 2", {1, 2}) == 0);
    // parens override precedence
    CHECK(ev("(1 OR 2) AND 3", {1, 3}) == 1);
    CHECK(ev("(1 OR 2) AND 3", {1}) == 0);

    // ---- nested / NOT of a group / double negation ----
    CHECK(ev("(1 OR 2) AND NOT (3 OR 4)", {1}) == 1);
    CHECK(ev("(1 OR 2) AND NOT (3 OR 4)", {1, 3}) == 0);
    CHECK(ev("NOT NOT 5", {5}) == 1);
    CHECK(ev("NOT NOT 5", {6}) == 0);
    CHECK(ev("NOT (1 AND 2)", {1}) == 1);  // not both -> true
    CHECK(ev("NOT (1 AND 2)", {1, 2}) == 0);

    // ---- n-ary flattening (many AND/OR operands) ----
    CHECK(ev("1 AND 2 AND 3 AND 4", {1, 2, 3, 4}) == 1);
    CHECK(ev("1 AND 2 AND 3 AND 4", {1, 2, 4}) == 0);
    CHECK(ev("1 OR 2 OR 3 OR 4", {4}) == 1);
    CHECK(ev("1 OR 2 OR 3 OR 4", {9}) == 0);

    // ---- empty string == match all (nullptr AST) ----
    {
        bool ok = true;
        auto n = ParseFilter("", ok);
        CHECK(ok);
        CHECK(n == nullptr);
        CHECK(EvalFilter(n.get(), {}) == true);
        CHECK(EvalFilter(nullptr, {1, 2, 3}) == true);
    }
    CHECK(ev("   ", {}) == 1);  // whitespace-only == match all

    // ---- malformed expressions must fail to parse (ok=false) ----
    CHECK(!parses("(1 OR 2"));    // unbalanced (
    CHECK(!parses("1 OR 2)"));    // unbalanced )
    CHECK(!parses("()"));         // empty parens
    CHECK(!parses("AND 2"));      // leading operator
    CHECK(!parses("1 AND"));      // trailing operator
    CHECK(!parses("1 2"));        // two terms no operator
    CHECK(!parses("1 OR OR 2"));  // double operator
    CHECK(!parses("NOT"));        // NOT with no operand

    // ---- boundary tags (35840 tags -> 560 u64 words; tag 35839 is the last bit) ----
    CHECK(evBoth("35839", {35839}) == 1);
    CHECK(evBoth("35839", {47}) == 0);
    CHECK(evBoth("35839 AND 47", {35839, 47}) == 1);
    CHECK(evBoth("0", {0}) == 1);                        // tag 0 (word 0, bit 0)
    CHECK(evBoth("0 AND 63 AND 64", {0, 63, 64}) == 1);  // word/bit boundaries

    // ---- set-eval and bitmap-eval agree on a realistic expression ----
    const std::string real = "((47 AND (728 OR 31852 OR 730) AND (546 OR 11)) AND (NOT (25985 OR 15399)) AND 89)";
    CHECK(evBoth(real, {47, 728, 546, 89}) == 1);         // passes
    CHECK(evBoth(real, {47, 728, 546, 89, 25985}) == 0);  // fails NOT clause
    CHECK(evBoth(real, {47, 728, 89}) == 0);              // missing (546 OR 11)
    CHECK(evBoth(real, {728, 546, 89}) == 0);             // missing 47

    std::fprintf(stderr, "test_filter_expr: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
