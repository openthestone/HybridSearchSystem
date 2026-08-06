#include "raw_data.h"
#include "expr.h"

#include <random>
#include <stdexcept>

namespace filter_cmp {

RawForwardBitmap BuildForwardBitmap(uint32_t doc_num, uint32_t tag_num, double density, uint64_t seed) {
    if (density < 0.0)
        density = 0.0;
    if (density > 1.0)
        density = 1.0;

    RawForwardBitmap b;
    b.doc_num = doc_num;
    b.tag_num = tag_num;
    b.u64_per_doc = (tag_num + 63u) / 64u;
    b.bits.assign((size_t)doc_num * b.u64_per_doc, 0ULL);

    std::mt19937_64 rng(seed);
    std::bernoulli_distribution bd(density);
    for (uint32_t d = 0; d < doc_num; ++d) {
        for (uint32_t w = 0; w < b.u64_per_doc; ++w) {
            uint64_t word = 0;
            for (int i = 0; i < 64; ++i) {
                if (bd(rng))
                    word |= (1ULL << i);
            }
            b.bits[(size_t)d * b.u64_per_doc + w] = word;
        }
    }
    return b;
}

std::vector<uint8_t> ReferenceFilterCpu(const RawForwardBitmap& data, const ExprSpec& expr) {
    std::vector<uint8_t> out(data.doc_num, 0);
    const uint32_t u64 = data.u64_per_doc;
    if (u64 > 256)
        throw std::runtime_error("u64_per_doc > 256 unsupported");

    const auto& rpn = expr.rpn;
    std::vector<std::vector<uint64_t>> stack;
    stack.reserve(rpn.size());

    for (uint32_t d = 0; d < data.doc_num; ++d) {
        stack.clear();
        bool ok = true;
        for (uint32_t slot : rpn) {
            RpnOp op = rpn_op(slot);
            if (op == POSTING_TYPE_NODE) {
                uint32_t tag = rpn_tag(slot);
                uint32_t word = tag >> 6;
                uint32_t bit = tag & 63u;
                std::vector<uint64_t> v(u64, 0ULL);
                if (word < u64) {
                    v[word] = data.bits[(size_t)d * u64 + word] & (1ULL << bit);
                }
                stack.emplace_back(std::move(v));
            } else if (op == AND_NODE) {
                if (stack.size() < 2) {
                    ok = false;
                    break;
                }
                auto b = std::move(stack.back());
                stack.pop_back();
                auto& a = stack.back();
                for (uint32_t w = 0; w < u64; ++w)
                    a[w] &= b[w];
            } else if (op == OR_NODE) {
                if (stack.size() < 2) {
                    ok = false;
                    break;
                }
                auto b = std::move(stack.back());
                stack.pop_back();
                auto& a = stack.back();
                for (uint32_t w = 0; w < u64; ++w)
                    a[w] |= b[w];
            } else if (op == NOT_NODE) {
                if (stack.empty()) {
                    ok = false;
                    break;
                }
                auto& a = stack.back();
                for (uint32_t w = 0; w < u64; ++w)
                    a[w] = ~a[w];
            } else {
                ok = false;
                break;
            }
        }
        if (!ok || stack.size() != 1) {
            out[d] = 0;
            continue;
        }
        const auto& v = stack.back();
        bool match = false;
        for (uint32_t w = 0; w < u64; ++w)
            if (v[w]) {
                match = true;
                break;
            }
        out[d] = match ? 1 : 0;
    }
    return out;
}

}  // namespace filter_cmp
