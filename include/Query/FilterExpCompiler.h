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

// 负责将查询中原始的属性过滤表达式字符串直接编译为 TmpNode 树
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

    static std::vector<uint32_t> &ThreadLocalValueStack()
    {
        static thread_local std::vector<uint32_t> vs;
        return vs;
    }

    void compileNoAST(std::vector<FilterTmpNode> &tmp, std::vector<uint32_t> &val_stack)
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
                        f.pending_op = 1; // OR marker
                        f.stage = 2;
                        stack.push_back({FrameType::Term, 0, f.inverted, 0});
                        continue;
                    }
                    stack.pop_back();
                    continue;
                }
                if (f.stage == 2)
                {
                    // Binary OR: pop right and left from val_stack, create GROUP
                    if (val_stack.size() < 2)
                        throw std::runtime_error("Compile Error: Value stack underflow");
                    uint32_t right_idx = val_stack.back(); val_stack.pop_back();
                    uint32_t left_idx = val_stack.back(); val_stack.pop_back();
                    PlanOp group_op = f.inverted ? PlanOp::AND_GROUP : PlanOp::OR_GROUP;
                    FilterTmpNode group;
                    group.type = FilterTmpNode::GROUP;
                    group.op = group_op;
                    group.children.push_back(left_idx);
                    group.children.push_back(right_idx);
                    tmp.push_back(std::move(group));
                    val_stack.push_back(static_cast<uint32_t>(tmp.size() - 1));
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
                        f.pending_op = 0; // AND marker
                        f.stage = 2;
                        stack.push_back({FrameType::Factor, 0, f.inverted, 0});
                        continue;
                    }
                    stack.pop_back();
                    continue;
                }
                if (f.stage == 2)
                {
                    // Binary AND: pop right and left from val_stack, create GROUP
                    if (val_stack.size() < 2)
                        throw std::runtime_error("Compile Error: Value stack underflow");
                    uint32_t right_idx = val_stack.back(); val_stack.pop_back();
                    uint32_t left_idx = val_stack.back(); val_stack.pop_back();
                    PlanOp group_op = f.inverted ? PlanOp::OR_GROUP : PlanOp::AND_GROUP;
                    FilterTmpNode group;
                    group.type = FilterTmpNode::GROUP;
                    group.op = group_op;
                    group.children.push_back(left_idx);
                    group.children.push_back(right_idx);
                    tmp.push_back(std::move(group));
                    val_stack.push_back(static_cast<uint32_t>(tmp.size() - 1));
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
                        FilterTmpNode leaf;
                        leaf.type = FilterTmpNode::LEAF;
                        leaf.tag_id = tag_id;
                        leaf.inverted = f.inverted;
                        tmp.push_back(std::move(leaf));
                        val_stack.push_back(static_cast<uint32_t>(tmp.size() - 1));
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

        std::vector<uint32_t> &vs = ThreadLocalValueStack();
        if (vs.capacity() < target)
        {
            vs.reserve(target);
        }
        vs.clear();
    }

    void compile(std::vector<FilterTmpNode> &tmp, uint32_t &root_idx)
    {
        tmp.clear();
        std::vector<uint32_t> &val_stack = ThreadLocalValueStack();
        val_stack.clear();
        compileNoAST(tmp, val_stack);
        if (val_stack.size() == 1)
        {
            root_idx = val_stack.back();
        }
        else
        {
            root_idx = UINT32_MAX;
        }
    }

    static void run(std::string_view expression, std::vector<FilterTmpNode> &tmp, uint32_t &root_idx)
    {
        FilterExpCompiler compiler(expression);
        compiler.compile(tmp, root_idx);
    }
};

// FilterExp 构造函数实现
inline FilterExp::FilterExp(const std::string &query_filter)
{
    CompileFrom(query_filter);
}

inline FilterExp::FilterExp(std::string_view query_filter)
{
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

// ---- BucketPlan Compilation ----

inline void BuildBucketPlan(FilterExp &output, std::vector<FilterTmpNode> &tmp, uint32_t root_idx)
{
    BucketPlan &plan = output.bucket_plan;
    plan = BucketPlan(); // reset

    if (tmp.empty() || root_idx == UINT32_MAX)
        return;

    // Helper: compute selectivity for a leaf
    auto leaf_selectivity = [](uint32_t tag_id, bool inverted) -> float {
        if (!g_global_tag_freq.empty() && tag_id < g_global_tag_freq.size())
        {
            float sel = g_global_tag_freq[tag_id];
            return inverted ? (1.0f - sel) : sel;
        }
        return 0.5f;
    };

    // Assign selectivity to leaf nodes
    for (auto &n : tmp)
    {
        if (n.type == FilterTmpNode::LEAF)
        {
            n.selectivity = leaf_selectivity(n.tag_id, n.inverted);
        }
    }

    // Helper: compute selectivity for a group
    std::function<float(uint32_t)> compute_sel = [&](uint32_t idx) -> float {
        const FilterTmpNode &n = tmp[idx];
        if (n.type == FilterTmpNode::LEAF || n.type == FilterTmpNode::CONSTANT)
            return n.selectivity;
        // GROUP
        if (n.children.empty())
            return n.op == PlanOp::AND_GROUP ? 1.0f : 0.0f;
        if (n.op == PlanOp::AND_GROUP)
        {
            float prod = 1.0f;
            for (uint32_t c : n.children)
                prod *= compute_sel(c);
            return prod;
        }
        else
        { // OR_GROUP
            float prod = 1.0f;
            for (uint32_t c : n.children)
                prod *= (1.0f - compute_sel(c));
            return 1.0f - prod;
        }
    };

    // Step 1: Compile-time constant folding (global_tag_freq)
    std::function<uint32_t(uint32_t)> fold_constants = [&](uint32_t idx) -> uint32_t {
        FilterTmpNode &n = tmp[idx];
        if (n.type == FilterTmpNode::CONSTANT) return idx;

        if (n.type == FilterTmpNode::LEAF)
        {
            if (!g_global_tag_freq.empty() && n.tag_id < g_global_tag_freq.size())
            {
                float freq = g_global_tag_freq[n.tag_id];
                if (freq == 0.0f)
                {
                    n.type = FilterTmpNode::CONSTANT;
                    n.const_value = n.inverted;
                    n.selectivity = n.const_value ? 1.0f : 0.0f;
                }
                else if (freq == 1.0f)
                {
                    n.type = FilterTmpNode::CONSTANT;
                    n.const_value = !n.inverted;
                    n.selectivity = n.const_value ? 1.0f : 0.0f;
                }
            }
            return idx;
        }

        // GROUP: recursively fold children
        std::vector<uint32_t> new_children;
        for (uint32_t c : n.children)
        {
            uint32_t fc = fold_constants(c);
            const FilterTmpNode &child = tmp[fc];
            if (child.type == FilterTmpNode::CONSTANT)
            {
                if (n.op == PlanOp::AND_GROUP && !child.const_value)
                {
                    n.type = FilterTmpNode::CONSTANT;
                    n.const_value = false;
                    n.selectivity = 0.0f;
                    return idx;
                }
                if (n.op == PlanOp::OR_GROUP && child.const_value)
                {
                    n.type = FilterTmpNode::CONSTANT;
                    n.const_value = true;
                    n.selectivity = 1.0f;
                    return idx;
                }
                continue;
            }
            new_children.push_back(fc);
        }
        n.children = std::move(new_children);

        if (n.children.empty())
        {
            n.type = FilterTmpNode::CONSTANT;
            n.const_value = (n.op == PlanOp::AND_GROUP);
            n.selectivity = n.const_value ? 1.0f : 0.0f;
            return idx;
        }
        if (n.children.size() == 1)
            return n.children[0];
        return idx;
    };
    root_idx = fold_constants(root_idx);

    if (tmp[root_idx].type == FilterTmpNode::CONSTANT)
    {
        if (tmp[root_idx].const_value)
        {
            plan.valid = false;
        }
        else
        {
            plan.always_false = true;
            plan.valid = true;
        }
        return;
    }

    // Step 2: Flatten homogeneous chains
    std::function<void(uint32_t)> flatten = [&](uint32_t idx) {
        FilterTmpNode &n = tmp[idx];
        if (n.type == FilterTmpNode::LEAF || n.type == FilterTmpNode::CONSTANT)
            return;
        // First, recursively flatten children
        for (uint32_t c : n.children)
            flatten(c);
        // Now absorb homogeneous grandchildren
        std::vector<uint32_t> flat_children;
        for (uint32_t c : n.children)
        {
            if (tmp[c].type == FilterTmpNode::GROUP && tmp[c].op == n.op)
            {
                // Same op type — absorb grandchildren
                for (uint32_t gc : tmp[c].children)
                    flat_children.push_back(gc);
            }
            else
            {
                flat_children.push_back(c);
            }
        }
        n.children = std::move(flat_children);
    };
    flatten(root_idx);

    // Step 3: Sort children by selectivity
    std::function<void(uint32_t)> sort_children = [&](uint32_t idx) {
        FilterTmpNode &n = tmp[idx];
        if (n.type == FilterTmpNode::LEAF || n.type == FilterTmpNode::CONSTANT)
            return;
        for (uint32_t c : n.children)
            sort_children(c);
        // Recompute selectivity after flattening
        n.selectivity = compute_sel(idx);
        if (n.op == PlanOp::AND_GROUP)
        {
            // Sort ascending — lowest selectivity first
            std::sort(n.children.begin(), n.children.end(),
                      [&](uint32_t a, uint32_t b) { return compute_sel(a) < compute_sel(b); });
        }
        else
        {
            // Sort descending — highest selectivity first
            std::sort(n.children.begin(), n.children.end(),
                      [&](uint32_t a, uint32_t b) { return compute_sel(a) > compute_sel(b); });
        }
    };
    sort_children(root_idx);

    // Step 4: Assign leaf_index and serialize
    uint16_t next_leaf_index = 0;
    std::function<void(uint32_t)> serialize = [&](uint32_t idx) {
        const FilterTmpNode &n = tmp[idx];
        size_t node_pos = plan.nodes.size();

        if (n.type == FilterTmpNode::LEAF)
        {
            PlanNode pn;
            pn.op = n.inverted ? PlanOp::NOT_TAG : PlanOp::TAG;
            pn.value = n.tag_id;
            pn.first_child = 0;
            pn.child_count = 0;
            pn.leaf_index = next_leaf_index++;
            pn.selectivity = n.selectivity;
            plan.nodes.push_back(pn);
        }
        else
        {
            // Reserve slot for this group node — will fill first_child after children are serialized
            PlanNode pn;
            pn.op = n.op;
            pn.value = 0;
            pn.first_child = 0;
            pn.child_count = static_cast<uint16_t>(n.children.size());
            pn.leaf_index = std::numeric_limits<uint16_t>::max();
            pn.selectivity = n.selectivity;
            plan.nodes.push_back(pn);

            // Serialize children
            uint32_t child_start = static_cast<uint32_t>(plan.children.size());
            for (size_t ci = 0; ci < n.children.size(); ++ci)
            {
                plan.children.push_back(0); // placeholder
            }

            // Recursively serialize each child, record its node position
            for (size_t i = 0; i < n.children.size(); ++i)
            {
                uint32_t child_node_pos = static_cast<uint32_t>(plan.nodes.size());
                plan.children[child_start + i] = child_node_pos;
                serialize(n.children[i]);
            }

            // Fix up group node's first_child
            plan.nodes[node_pos].first_child = child_start;
        }
    };
    serialize(root_idx);

    plan.root = 0; // root is always the first serialized node
    plan.leaf_count = next_leaf_index;
    plan.valid = (next_leaf_index >= 1);

    // Validate plan integrity once at compile time
    if (plan.valid)
    {
        plan.validated = true;
        if (plan.root >= plan.nodes.size()) { plan.validated = false; }
        else
        {
            const PlanNode &root = plan.nodes[plan.root];
            if (root.first_child + root.child_count > plan.children.size())
                plan.validated = false;
        }
        for (uint32_t ci = 0; ci < plan.children.size() && plan.validated; ++ci)
        {
            if (plan.children[ci] >= plan.nodes.size())
                plan.validated = false;
        }
        if (!plan.validated)
        {
            std::cerr << "[BUG] Plan validation failed: root=" << plan.root
                      << " nodes=" << plan.nodes.size()
                      << " children=" << plan.children.size() << std::endl;
            std::abort();
        }
    }

    // Build leaf_to_sorted_idx: map each leaf_index to position in sorted_unique_tag_ids
    plan.leaf_to_sorted_idx.resize(plan.leaf_count);
    for (uint32_t i = 0; i < plan.nodes.size(); ++i)
    {
        const PlanNode &node = plan.nodes[i];
        if (node.op != PlanOp::TAG && node.op != PlanOp::NOT_TAG) continue;
        if (node.leaf_index >= plan.leaf_count) continue;
        auto it = std::lower_bound(output.sorted_unique_tag_ids.begin(),
                                   output.sorted_unique_tag_ids.end(),
                                   node.value);
        if (it != output.sorted_unique_tag_ids.end() && *it == node.value)
            plan.leaf_to_sorted_idx[node.leaf_index] =
                static_cast<uint32_t>(it - output.sorted_unique_tag_ids.begin());
        else
            plan.leaf_to_sorted_idx[node.leaf_index] = UINT32_MAX;
    }

    // Build eval_order: post-order traversal so every group node comes after its children
    if (plan.valid && plan.root < plan.nodes.size())
    {
        plan.eval_order.clear();
        plan.eval_order.reserve(plan.nodes.size());
        // Iterative post-order using a small stack
        struct StackEntry { uint32_t node_idx; uint32_t child_pos; bool visited; };
        std::vector<StackEntry> stack;
        stack.push_back({plan.root, 0, false});
        while (!stack.empty())
        {
            StackEntry &top = stack.back();
            const PlanNode &node = plan.nodes[top.node_idx];
            if (node.op == PlanOp::TAG || node.op == PlanOp::NOT_TAG || top.visited)
            {
                plan.eval_order.push_back(top.node_idx);
                stack.pop_back();
                continue;
            }
            top.visited = true;
            const uint32_t child_begin = node.first_child;
            const uint32_t child_end = child_begin + static_cast<uint32_t>(node.child_count);
            // Push children in reverse so leftmost child is on top
            for (uint32_t c = child_end; c > child_begin; --c)
            {
                stack.push_back({plan.children[c - 1], 0, false});
            }
        }
    }
}


inline void FilterExp::CompileFrom(std::string_view query_filter)
{
    bucket_plan = BucketPlan();
    sorted_unique_tag_ids.clear();
    if (query_filter.empty())
        return;

    // Compile directly to TmpNode tree (no RPN intermediate)
    std::vector<FilterTmpNode> tmp;
    uint32_t root_idx = UINT32_MAX;
    FilterExpCompiler::run(query_filter, tmp, root_idx);

    if (root_idx == UINT32_MAX)
        return;

    // Build sorted unique tag IDs from TmpNode tree
    std::vector<uint32_t> all_tags;
    all_tags.reserve(tmp.size());
    for (const auto &node : tmp)
    {
        if (node.type == FilterTmpNode::LEAF)
            all_tags.push_back(node.tag_id);
    }

    radix_sort_dedup_u32(all_tags);
    sorted_unique_tag_ids = std::move(all_tags);

    // Build BucketPlan from TmpNode tree
    BuildBucketPlan(*this, tmp, root_idx);
}