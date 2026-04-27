#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <cstdint>
#include <iostream>

struct RPNItem
{
    bool is_op;     // true if Operator, false if Operand (Tag)
    uint32_t value; // TagID or OpCode
    uint8_t flags;  // Flags (e.g., bit 0 = Inverted/NOT for Tag)

    // Helper for debugging
    bool is_inverted() const { return flags & 1; }
};

namespace FilterOp8
{
    // §2.2B: OP_AND=0, OP_OR=1 enables branchless XOR flip: op ^ inverted
    // 0 is used for OP_AND, 1 for OP_OR (XOR semantics)
    const uint8_t OP_AND = 0;
    const uint8_t OP_OR = 1;

    // IVF Specific Ops
    const uint8_t OP_IVF_LOAD_EXIST = 3;   // Load Tag
    const uint8_t OP_IVF_LOAD_MISSING = 4; // Load Not Tag

    // Note: OP_NOT is removed from execution stream, handled via RPNItem.flags
}

class FilterExp
{
public:
    // Bucket_RPN: 用于桶内属性过滤
    std::vector<RPNItem> Bucket_RPN;

    // BucketLevelIVF_RPN: 用于桶级属性过滤
    std::vector<RPNItem> BucketLevelIVF_RPN;

    // ---- Sorted unique tag IDs for merge-based batch lookup ----
    // 在 CompileFrom 中一次性计算，查询时复用
    std::vector<uint32_t> sorted_unique_tag_ids;   // 升序去重的 tag ID 列表
    std::vector<uint32_t> rpn_tag_to_sorted_idx;   // RPN 中每个 tag 在 sorted_unique_tag_ids 中的索引

    FilterExp() = default;

    // 构造函数
    explicit FilterExp(const std::string &query_filter);
    explicit FilterExp(std::string_view query_filter);

    void Reserve(size_t bucket_rpn_capacity, size_t bucket_level_ivf_rpn_capacity)
    {
        if (Bucket_RPN.capacity() < bucket_rpn_capacity)
        {
            Bucket_RPN.reserve(bucket_rpn_capacity);
            Bucket_RPN.resize(bucket_rpn_capacity);
            Bucket_RPN.clear();
        }
        if (BucketLevelIVF_RPN.capacity() < bucket_level_ivf_rpn_capacity)
        {
            BucketLevelIVF_RPN.reserve(bucket_level_ivf_rpn_capacity);
            BucketLevelIVF_RPN.resize(bucket_level_ivf_rpn_capacity);
            BucketLevelIVF_RPN.clear();
        }
    }

    void CompileFrom(std::string_view query_filter);

    void dump() const
    {
        std::cout << ">>> Bucket_RPN (No-AST): ";
        for (const auto &item : Bucket_RPN)
        {
            if (item.is_op)
            {
                if (item.value == FilterOp8::OP_AND)
                    std::cout << "AND ";
                else if (item.value == FilterOp8::OP_OR)
                    std::cout << "OR ";
                else
                    std::cout << "OP(" << item.value << ") ";
            }
            else
            {
                std::cout << "Tag" << item.value;
                if (item.flags & 1)
                    std::cout << "(NOT)";
                std::cout << " ";
            }
        }
        std::cout << "\n>>> BucketLevelIVF_RPN (No-AST): ";
        for (const auto &item : BucketLevelIVF_RPN)
        {
            if (item.is_op)
            {
                if (item.value == FilterOp8::OP_AND)
                    std::cout << "AND ";
                else if (item.value == FilterOp8::OP_OR)
                    std::cout << "OR ";
                else if (item.value == FilterOp8::OP_IVF_LOAD_EXIST)
                    std::cout << "LD_EXIST ";
                else if (item.value == FilterOp8::OP_IVF_LOAD_MISSING)
                    std::cout << "LD_MISSING ";
                else
                    std::cout << "OP(" << item.value << ") ";
            }
            else
            {
                std::cout << "Tag" << item.value << " ";
            }
        }
        std::cout << std::endl;
    }
};
