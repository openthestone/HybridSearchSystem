#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <cstdint>
#include <iostream>
#include <limits>

// ---- BucketPlan: Linear Execution Plan with AND/OR Group Short-Circuit ----

enum class PlanOp : uint8_t {
    TAG,
    NOT_TAG,
    AND_GROUP,
    OR_GROUP,
};

struct PlanNode {
    PlanOp op;
    uint32_t value;        // TAG/NOT_TAG: tag_id
    uint32_t first_child;  // GROUP: children 起始下标（在 children 数组中）
    uint16_t child_count;  // GROUP: children 数量
    uint16_t leaf_index;   // TAG/NOT_TAG: leaf_base_ptrs 下标；GROUP 置 UINT16_MAX
    float selectivity;     // 编译期估计选择率
};

struct BucketPlan {
    std::vector<PlanNode> nodes;               // 所有节点（叶子 + group）连续存储
    std::vector<uint32_t> children;            // group 节点的 child 下标
    std::vector<uint32_t> leaf_to_sorted_idx;  // leaf_index -> index in sorted_unique_tag_ids
    std::vector<uint32_t> eval_order;          // 后序遍历求值顺序（叶子在前，group 在子节点之后）
    uint32_t root = UINT32_MAX;
    uint16_t leaf_count = 0;
    bool valid = false;
    bool validated = false;
    bool always_false = false;
};

// ---- Compilation tree node (intermediate, used by FilterExpCompiler) ----

struct FilterTmpNode {
    enum Type { LEAF, GROUP, CONSTANT } type;
    // LEAF
    uint32_t tag_id = 0;
    bool inverted = false;
    // GROUP
    PlanOp op = PlanOp::AND_GROUP;
    std::vector<uint32_t> children;  // indices into tmp vector
    // CONSTANT
    bool const_value = false;
    float selectivity = 0.5f;
};

class FilterExp {
   public:
    // ---- Sorted unique tag IDs for merge-based batch lookup ----
    std::vector<uint32_t> sorted_unique_tag_ids;

    // ---- BucketPlan: group short-circuit execution plan ----
    BucketPlan bucket_plan;

    FilterExp() = default;

    explicit FilterExp(const std::string& query_filter);
    explicit FilterExp(std::string_view query_filter);

    void CompileFrom(std::string_view query_filter);
};
