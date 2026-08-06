#include "cpu_filter.h"

#include <thread>

namespace filter_cmp {

namespace {

void WorkerRange(const RawForwardBitmap& data, const ExprSpec& expr, uint32_t d_begin, uint32_t d_end, uint8_t* out) {
    const uint32_t u64 = data.u64_per_doc;
    const auto& rpn = expr.rpn;
    // Stack of bits (depth <= rpn.size()). Bit = 1 means "non-zero" match so far.
    uint8_t stack[64];
    for (uint32_t d = d_begin; d < d_end; ++d) {
        const uint64_t* doc = &data.bits[(size_t)d * u64];
        int sp = 0;
        for (uint32_t slot : rpn) {
            switch ((RpnOp)(slot >> 28)) {
                case POSTING_TYPE_NODE: {
                    uint32_t tag = slot & 0x0FFFFFFFu;
                    uint32_t word = tag >> 6;
                    uint32_t bit = tag & 63u;
                    uint8_t v = 0;
                    if (word < u64)
                        v = (uint8_t)((doc[word] >> bit) & 1ULL);
                    stack[sp++] = v;
                    break;
                }
                case AND_NODE: {
                    uint8_t b = stack[--sp];
                    uint8_t a = stack[--sp];
                    stack[sp++] = (uint8_t)(a & b);
                    break;
                }
                case OR_NODE: {
                    uint8_t b = stack[--sp];
                    uint8_t a = stack[--sp];
                    stack[sp++] = (uint8_t)(a | b);
                    break;
                }
                case NOT_NODE: {
                    uint8_t a = stack[--sp];
                    stack[sp++] = (uint8_t)(a ^ 1u);
                    break;
                }
                default:
                    // Skip unknown ops (CONJ_NODE etc.)
                    break;
            }
        }
        out[d] = (sp == 1) ? stack[0] : 0;
    }
}

}  // namespace

std::vector<uint8_t> CpuFilterForward(const RawForwardBitmap& data, const ExprSpec& expr, uint32_t num_threads) {
    if (num_threads == 0)
        num_threads = 1;
    std::vector<uint8_t> out(data.doc_num, 0);
    if (data.doc_num == 0)
        return out;

    if (num_threads == 1) {
        WorkerRange(data, expr, 0, data.doc_num, out.data());
        return out;
    }

    std::vector<std::thread> ths;
    ths.reserve(num_threads);
    uint32_t per = (data.doc_num + num_threads - 1) / num_threads;
    for (uint32_t t = 0; t < num_threads; ++t) {
        uint32_t lo = t * per;
        uint32_t hi = std::min<uint32_t>(lo + per, data.doc_num);
        if (lo >= hi)
            break;
        ths.emplace_back(WorkerRange, std::cref(data), std::cref(expr), lo, hi, out.data());
    }
    for (auto& th : ths)
        th.join();
    return out;
}

}  // namespace filter_cmp
