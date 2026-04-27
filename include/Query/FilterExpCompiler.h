#pragma once

#include "utils/DataReader.h"
#include "utils/MemoryEventLogger.h"
#include "Query/FilterExp.h"
#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <cctype>
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstring> 

// 负责将查询中原始的属性过滤表达式字符串优化并转化为 RPN序列
/*
    表达式字符串需要满足以下要求：
    允许有多余空格
    允许有冗余括号，例如((1 OR 2)) AND (3)
    操作符必须大写：AND, OR, NOT
    各Tag必须是纯数字，且在（0 ~ MAX_TAG_ID）范围内
*/
class FilterExpCompiler
{
private:
    std::string_view expr_;
    size_t pos_;

    enum class FrameType : uint8_t
    {
        Expr,
        Term,
        Factor
    };

    struct Frame
    {
        FrameType type;
        uint8_t stage;      // 0: enter, 1: after left, 2: after right
        bool inverted;
        uint8_t pending_op;
    };

    static std::vector<Frame> &ThreadLocalCompileStack()
    {
        static thread_local std::vector<Frame> stack;
        return stack;
    }

    void skipSpaces()
    {
        while (pos_ < expr_.size() && std::isspace(static_cast<unsigned char>(expr_[pos_])))
            pos_++;
    }

    char peek()
    {
        skipSpaces();
        return (pos_ == expr_.size()) ? 0 : expr_[pos_];
    }

    char get()
    {
        char c = peek();
        if (pos_ < expr_.size())
            pos_++;
        return c;
    }

    bool matchKeyword(const char *expected)
    {
        skipSpaces();
        size_t len = std::strlen(expected);
        if (pos_ + len <= expr_.size() && expr_.compare(pos_, len, expected) == 0)
        {
            char next = (pos_ + len < expr_.size()) ? expr_[pos_ + len] : 0;
            if (std::isalnum(static_cast<unsigned char>(next)))
                return false;
            pos_ += len;
            return true;
        }
        return false;
    }

    uint32_t parseTagId()
    {
        skipSpaces();
        if (!std::isdigit(static_cast<unsigned char>(peek())))
            throw std::runtime_error("Syntax Error: Expected Tag ID");

        uint64_t val = 0;
        while (std::isdigit(static_cast<unsigned char>(peek())))
        {
            val = val * 10 + static_cast<uint64_t>(get() - '0');
        }
        return static_cast<uint32_t>(val);
    }

    void emitTag(uint32_t tag_id, bool inverted, FilterExp &output)
    {
        if (inverted)
        {
            // Bucket_RPN: Tag(NEG)
            output.Bucket_RPN.push_back({false, tag_id, 1}); // flag=1 for NEG

            // BucketLevelIVF_RPN: Tag + LOAD_MISSING
            output.BucketLevelIVF_RPN.push_back({false, tag_id, 0});
            output.BucketLevelIVF_RPN.push_back({true, FilterOp8::OP_IVF_LOAD_MISSING, 0});
        }
        else
        {
            output.Bucket_RPN.push_back({false, tag_id, 0}); // flag=0 for POS

            output.BucketLevelIVF_RPN.push_back({false, tag_id, 0});
            output.BucketLevelIVF_RPN.push_back({true, FilterOp8::OP_IVF_LOAD_EXIST, 0});
        }
    }

    void emitBinary(uint8_t op, bool inverted, FilterExp &output)
    {
        // §2.2B: XOR-based branchless flip: inverted flips AND↔OR
        // Requires OP_AND=0, OP_OR=1. Then op^1 = NOT op.
        uint8_t final_op = inverted ? (op ^ 1) : op;
        output.Bucket_RPN.push_back({true, final_op, 0});
        output.BucketLevelIVF_RPN.push_back({true, final_op, 0});
    }

    void compileNoAST(FilterExp &output)
    {
        std::vector<Frame> &stack = ThreadLocalCompileStack();
        MemoryEventSession *session = GetActiveMemoryEventSession();
        const size_t stack_old_capacity = session == nullptr ? 0 : stack.capacity();
        stack.clear();
        if (stack.capacity() < static_cast<size_t>(query_compile_stack_reserve_items))
        {
            stack.reserve(static_cast<size_t>(query_compile_stack_reserve_items));
        }
        stack.push_back({FrameType::Expr, 0, false, 0});

        while (!stack.empty())
        {
            Frame &f = stack.back();

            if (f.type == FrameType::Expr)
            {
                if (f.stage == 0)
                {
                    f.stage = 1;
                    stack.push_back({FrameType::Term, 0, f.inverted, 0});
                    continue;
                }
                if (f.stage == 1)
                {
                    if (matchKeyword("OR"))
                    {
                        f.pending_op = FilterOp8::OP_OR;
                        f.stage = 2;
                        stack.push_back({FrameType::Term, 0, f.inverted, 0});
                        continue;
                    }
                    stack.pop_back();
                    continue;
                }
                if (f.stage == 2)
                {
                    emitBinary(f.pending_op, f.inverted, output);
                    f.stage = 1;
                    continue;
                }
            }
            else if (f.type == FrameType::Term)
            {
                if (f.stage == 0)
                {
                    f.stage = 1;
                    stack.push_back({FrameType::Factor, 0, f.inverted, 0});
                    continue;
                }
                if (f.stage == 1)
                {
                    if (matchKeyword("AND"))
                    {
                        f.pending_op = FilterOp8::OP_AND;
                        f.stage = 2;
                        stack.push_back({FrameType::Factor, 0, f.inverted, 0});
                        continue;
                    }
                    stack.pop_back();
                    continue;
                }
                if (f.stage == 2)
                {
                    emitBinary(f.pending_op, f.inverted, output);
                    f.stage = 1;
                    continue;
                }
            }
            else // Factor
            {
                if (f.stage == 0)
                {
                    if (matchKeyword("NOT"))
                    {
                        f.inverted = !f.inverted;
                        continue;
                    }

                    char c = peek();
                    if (c == '(')
                    {
                        get();
                        f.stage = 1;
                        stack.push_back({FrameType::Expr, 0, f.inverted, 0});
                        continue;
                    }
                    if (std::isdigit(static_cast<unsigned char>(c)))
                    {
                        uint32_t tag_id = parseTagId();
                        emitTag(tag_id, f.inverted, output);
                        stack.pop_back();
                        continue;
                    }

                    throw std::runtime_error("Syntax Error: Expected Tag ID or '('");
                }

                if (f.stage == 1)
                {
                    if (peek() != ')')
                        throw std::runtime_error("Syntax Error: Missing ')'");
                    get();
                    stack.pop_back();
                    continue;
                }
            }
        }

        skipSpaces();
        if (pos_ != expr_.size())
            throw std::runtime_error("Syntax Error: Unexpected trailing tokens");

        RecordCapacityGrowth<Frame>(session,
                                    "FilterExpCompiler.h:compile_stack",
                                    stack_old_capacity,
                                    stack.capacity());
    }

public:
    explicit FilterExpCompiler(std::string_view expression) : expr_(expression), pos_(0) {}

    static void WarmUpThreadLocalBuffers()
    {
        std::vector<Frame> &stack = ThreadLocalCompileStack();
        const size_t target = static_cast<size_t>(query_compile_stack_reserve_items);
        if (stack.capacity() < target)
        {
            stack.reserve(target);
        }
        if (target != 0 && stack.size() < target)
        {
            stack.resize(target);
        }
        stack.clear();
    }

    void compile(FilterExp &output)
    {
        MemoryEventSession *session = GetActiveMemoryEventSession();
        const size_t bucket_rpn_old_capacity = session == nullptr ? 0 : output.Bucket_RPN.capacity();
        const size_t bucket_level_ivf_rpn_old_capacity =
            session == nullptr ? 0 : output.BucketLevelIVF_RPN.capacity();

        output.Bucket_RPN.clear();
        output.BucketLevelIVF_RPN.clear();
        compileNoAST(output);

        RecordCapacityGrowth<RPNItem>(session,
                                      "FilterExp.h:Bucket_RPN",
                                      bucket_rpn_old_capacity,
                                      output.Bucket_RPN.capacity());
        RecordCapacityGrowth<RPNItem>(session,
                                      "FilterExp.h:BucketLevelIVF_RPN",
                                      bucket_level_ivf_rpn_old_capacity,
                                      output.BucketLevelIVF_RPN.capacity());
    }

    static void run(std::string_view expression, FilterExp &output)
    {
        FilterExpCompiler compiler(expression);
        compiler.compile(output);
    }
};

// FilterExp 构造函数实现
inline FilterExp::FilterExp(const std::string &query_filter)
{
    Reserve(static_cast<size_t>(query_bucket_rpn_reserve_items),
            static_cast<size_t>(query_bucket_level_ivf_rpn_reserve_items));
    CompileFrom(query_filter);
}

inline FilterExp::FilterExp(std::string_view query_filter)
{
    Reserve(static_cast<size_t>(query_bucket_rpn_reserve_items),
            static_cast<size_t>(query_bucket_level_ivf_rpn_reserve_items));
    CompileFrom(query_filter);
}

// ---- LSD Radix Sort for uint32_t ----
// 4-pass × 8-bit digit, O(n) comparison-free. Optimal for small N (< 1000).
inline void radix_sort_u32(uint32_t *data, uint32_t *temp, size_t n)
{
    if (n <= 1) return;

    uint32_t *src = data, *dst = temp;
    for (int shift = 0; shift < 32; shift += 8)
    {
        uint32_t count[256] = {};
        for (size_t i = 0; i < n; ++i)
            count[(src[i] >> shift) & 0xFF]++;

        // Prefix sum → starting offset
        uint32_t total = 0;
        for (int i = 0; i < 256; ++i)
        {
            uint32_t c = count[i];
            count[i] = total;
            total += c;
        }

        for (size_t i = 0; i < n; ++i)
        {
            uint32_t val = src[i];
            dst[count[(val >> shift) & 0xFF]++] = val;
        }

        std::swap(src, dst);
    }

    // If odd number of passes (4 = even), final result is in data[] already.
    // If result ended in temp, copy back.
    if (src != data)
        std::memcpy(data, temp, n * sizeof(uint32_t));
}

// Sort + dedup using radix sort (returns sorted unique count)
inline size_t radix_sort_dedup_u32(std::vector<uint32_t> &arr)
{
    const size_t n = arr.size();
    if (n <= 1) return n;

    std::vector<uint32_t> temp(n);
    radix_sort_u32(arr.data(), temp.data(), n);

    // Dedup in-place
    size_t out = 0;
    for (size_t i = 1; i < n; ++i)
    {
        if (arr[i] != arr[out])
            arr[++out] = arr[i];
    }
    arr.resize(out + 1);
    return out + 1;
}

inline void FilterExp::CompileFrom(std::string_view query_filter)
{
    Bucket_RPN.clear();
    BucketLevelIVF_RPN.clear();
    sorted_unique_tag_ids.clear();
    rpn_tag_to_sorted_idx.clear();
    if (query_filter.empty())
        return;
    FilterExpCompiler::run(query_filter, *this);

    // Build sorted unique tag IDs and RPN-to-sorted index mapping
    // Step 1: Collect all tag IDs from Bucket_RPN
    std::vector<uint32_t> all_tags;
    all_tags.reserve(Bucket_RPN.size());
    for (const auto &item : Bucket_RPN)
    {
        if (!item.is_op)
            all_tags.push_back(item.value);
    }

    // Step 2: Sort and dedup via radix sort (O(n), branch-free)
    radix_sort_dedup_u32(all_tags);
    sorted_unique_tag_ids = std::move(all_tags);

    // Step 3: Build RPN-to-sorted mapping using linear scan
    // (both arrays are sorted by tag_id, so merge-style scan is O(n))
    rpn_tag_to_sorted_idx.reserve(Bucket_RPN.size());
    {
        const uint32_t *sorted = sorted_unique_tag_ids.data();
        const size_t sorted_n = sorted_unique_tag_ids.size();
        size_t hint = 0; // monotonic hint — tags in RPN tend to repeat, so this helps
        for (const auto &item : Bucket_RPN)
        {
            if (!item.is_op)
            {
                // Linear scan from hint (monotonic since sorted tags are unique & ascending)
                while (hint < sorted_n && sorted[hint] < item.value)
                    ++hint;
                rpn_tag_to_sorted_idx.push_back(static_cast<uint32_t>(hint));
            }
            else
            {
                rpn_tag_to_sorted_idx.push_back(UINT32_MAX);
            }
        }
    }
}
