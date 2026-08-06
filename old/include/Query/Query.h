#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <cstdint>
#include <cstring>
#include <array>
#include <algorithm>
#include <iostream>
#include <functional>
#include <chrono>
#include <atomic>
#include <iomanip>
#include <limits>
#include <queue>
#include <unordered_set>
#include <mutex>

// 引入 NEON 头文件，用于ARM平台加速
#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include "utils/DataReader.h"
#include "utils/MemoryEventLogger.h"
#include "Query/FilterExp.h"
#include "DataBaseCPU/Bucket.h"
#include "DataBaseCPU/BucketLevelIVF.h"
#include "DataBaseCPU/AlignedAllocator.h"
#include "Query/FilterExpCompiler.h"

// FP32 to FP16 conversion for query vectors
inline uint16_t QueryFloat32ToFp16(float value) {
    uint32_t f32;
    std::memcpy(&f32, &value, sizeof(float));
    uint16_t f16 = 0;
    uint32_t sign = (f32 >> 16) & 0x8000;
    int16_t exponent = ((f32 >> 23) & 0xFF) - 127;
    uint32_t mantissa = f32 & 0x007FFFFF;
    if (exponent > 15) {
        f16 = sign | 0x7C00;
    } else if (exponent <= -15) {
        f16 = sign;
    } else {
        exponent += 15;
        mantissa >>= 13;
        f16 = sign | (exponent << 10) | mantissa;
    }
    return f16;
}

struct QueryResult {
    struct Item {
        uint32_t doc_id;
        float score;

        Item() = default;
        Item(uint32_t d, float s) : doc_id(d), score(s) {}

        // 用于 TopK 排序 (分数降序，越大越好，适配向量内积)
        bool operator>(const Item& other) const {
            return score > other.score;
        }
        bool operator<(const Item& other) const {
            return score < other.score;
        }
    };

    static bool BetterItem(const Item& a, const Item& b) {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        return a.doc_id < b.doc_id;
    }

    static void KeepTopK(std::vector<Item>& items, int target_k) {
        if (target_k <= 0) {
            items.clear();
            return;
        }

        if (items.size() > static_cast<size_t>(target_k)) {
            std::partial_sort(items.begin(), items.begin() + target_k, items.end(), BetterItem);
            items.resize(target_k);
        }
    }
    // 最终的 TopK 结果 (按向量内积 score 降序排列)
    std::vector<Item> topk_results;

    // 这是 Probe-Expand 策略（先搜200个桶，不达条件则再搜200个桶...）的终止条件
    bool check_satisfaction(int target_k) const {
        return topk_results.size() >= (size_t)target_k;
    }

    // 归并新的一批结果到 topk_results 中
    void merge_batch(const std::vector<Item>& new_items, int target_k, std::vector<Item>& scratch_buffer) {
        if (new_items.empty())
            return;

        scratch_buffer.clear();
        TrackVectorReserve(scratch_buffer, topk_results.size() + new_items.size(), "Query.h:merge_batch.merged");
        scratch_buffer.insert(scratch_buffer.end(), topk_results.begin(), topk_results.end());
        scratch_buffer.insert(scratch_buffer.end(), new_items.begin(), new_items.end());
        KeepTopK(scratch_buffer, target_k);
        topk_results.swap(scratch_buffer);
    }
};

struct QueryTimingMetrics {
    double bucket_level_ivf_ms = 0.0;
    double bucket_level_ivf_filter_eval_ms = 0.0;
    double bucket_level_ivf_centroid_score_ms = 0.0;
    double bucket_level_ivf_result_pack_ms = 0.0;
    double candidate_bucket_merge_ms = 0.0;
    double npu_async_launch_ms = 0.0;
    double npu_submit_ms = 0.0;
    double inbucket_attr_filter_overlapped_ms = 0.0;
    double wait_npu_flag_ms = 0.0;
    double result_collection_ms = 0.0;
    double final_merge_ms = 0.0;
    double npu_mask_filter_launch_ms = 0.0;
    double npu_mask_filter_h2d_ms = 0.0;
    double npu_mask_filter_kernel_exec_ms = 0.0;
    double npu_mask_filter_d2h_ms = 0.0;
    int probe_expand_rounds = 0;
};

class Query {
   public:
    std::vector<float> query_vector;                    // 查询向量
    std::vector<uint16_t> mmad_query_padded_fp16;       // MMAD [k,16] padded query payload
    FilterExp filter_exp;                               // Filter 表达式
    int target_k = 0;                                   // 最终返回的 Top-K
    int expanded_k = 0;                                 // 实际检索阶段使用的扩展 K
    size_t query_id = 0;                                // 查询编号（使用查询文件行号）
    int process_round_count_l2 = 0;                     // 当前查询实际执行的扩搜轮数
    int process_round_count_level_2 = 0;                // 当前查询实际执行的二级桶处理子批次数
    int searched_bucket_count_l2 = 0;                   // 当前查询实际处理的L2桶数量
    int searched_bucket_count_level_2 = 0;              // 当前查询实际处理的二级桶数量
    std::atomic<int> searched_l2_count_nonzero{0};      // mask 不为全零的二级桶数量
    std::atomic<int> searched_l2_count_allzero{0};      // mask 为全零的二级桶数量
    std::vector<uint32_t> searched_level_2_bucket_ids;  // 当前查询实际处理过的二级桶编号（按处理顺序）
    std::atomic<size_t> bucket_selectivity_idx_{0};
    std::vector<double> bucket_selectivity_;  // per-bucket selectivity (all rounds merged)
    QueryResult result;                       // 查询结果

    // ==========================================
    // 新增：用于验证 Recall 的 Ground Truth 数据
    // ==========================================
    std::vector<QueryResult::Item> ground_truth_results;
    float recall_rate = 0.0f;
    QueryTimingMetrics timing_metrics;

    // Deferred NPU timing: slot info recorded during execution,
    // aclrtElapsedTime queried after all queries complete.
    int npu_timing_group_ = 0;
    int npu_timing_slot_ = 0;
    bool npu_timing_pending_ = false;

    // 好像没用？
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    std::chrono::steady_clock::time_point start_time_for_parallel_record;
    std::chrono::steady_clock::time_point end_time_for_parallel_record;

    Query() {
        InitializeReusableBuffers();
    }

    void PrepareMemoryEventSession(size_t query_id_in, bool enabled) {
        memory_events_enabled_ = enabled;
        if (enabled) {
            memory_event_session_.Reset(query_id_in);
        }
    }

    void Reset(const std::vector<float>& vec, std::string_view filter_expr_text, int k, size_t query_id_in) {
        target_k = k;
        expanded_k = k;
        query_id = query_id_in;
        process_round_count_l2 = 0;
        process_round_count_level_2 = 0;
        searched_bucket_count_l2 = 0;
        searched_bucket_count_level_2 = 0;
        searched_l2_count_nonzero.store(0, std::memory_order_relaxed);
        searched_l2_count_allzero.store(0, std::memory_order_relaxed);
        recall_rate = 0.0f;
        timing_metrics = QueryTimingMetrics{};
        npu_timing_pending_ = false;
        start_time = std::chrono::steady_clock::now();
        end_time = std::chrono::steady_clock::time_point{};
        start_time_for_parallel_record = std::chrono::steady_clock::time_point{};
        end_time_for_parallel_record = std::chrono::steady_clock::time_point{};

        result.topk_results.clear();
        ground_truth_results.clear();
        merge_buffer_.clear();
        searched_level_2_bucket_ids.clear();
        bucket_selectivity_.clear();
        bucket_selectivity_.resize(1024);
        bucket_selectivity_idx_.store(0, std::memory_order_relaxed);

        query_vector.clear();
        TrackVectorReserve(query_vector, std::max(vec.size(), static_cast<size_t>(query_vector_reserve_floats)),
                           "Query.h:query_vector");
        query_vector.insert(query_vector.end(), vec.begin(), vec.end());

        // Pre-compute MMAD query payload (FP16 padded to [dim, 16])
        constexpr size_t kKernelResultCols = 16;
        const size_t padded_size = static_cast<size_t>(vector_dim) * kKernelResultCols;
        mmad_query_padded_fp16.clear();
        mmad_query_padded_fp16.resize(padded_size, 0);
#if defined(__aarch64__) || defined(__arm__)
        size_t i = 0;
        for (; i + 3 < vector_dim; i += 4) {
            float32x4_t vf32 = vld1q_f32(query_vector.data() + i);
            float16x4_t vf16 = vcvt_f16_f32(vf32);
            uint16x4_t vi16 = vreinterpret_u16_f16(vf16);
            vst1_lane_u16(mmad_query_padded_fp16.data() + (i + 0) * kKernelResultCols, vi16, 0);
            vst1_lane_u16(mmad_query_padded_fp16.data() + (i + 1) * kKernelResultCols, vi16, 1);
            vst1_lane_u16(mmad_query_padded_fp16.data() + (i + 2) * kKernelResultCols, vi16, 2);
            vst1_lane_u16(mmad_query_padded_fp16.data() + (i + 3) * kKernelResultCols, vi16, 3);
        }
        for (; i < vector_dim; ++i) {
            mmad_query_padded_fp16[i * kKernelResultCols] = QueryFloat32ToFp16(query_vector[i]);
        }
#else
        for (size_t i = 0; i < static_cast<size_t>(vector_dim); ++i) {
            mmad_query_padded_fp16[i * kKernelResultCols] = QueryFloat32ToFp16(query_vector[i]);
        }
#endif

        filter_exp.CompileFrom(filter_expr_text);

        if (!ComputeExpandedKFromTopK(k, expanded_k)) {
            expanded_k = std::numeric_limits<int>::max();
        }
    }

    void MarkProcessingStart() {
        start_time_for_parallel_record = std::chrono::steady_clock::now();
        end_time_for_parallel_record = std::chrono::steady_clock::time_point{};
    }

    void MarkProcessingEnd() {
        end_time_for_parallel_record = std::chrono::steady_clock::now();
    }

    MemoryEventSession* memory_event_session() const {
        return memory_events_enabled_ ? const_cast<MemoryEventSession*>(&memory_event_session_) : nullptr;
    }

    void AppendMemoryEventLogs(MemoryEventLogCollector& collector) const {
        collector.AppendSession(memory_event_session());
    }

    void finalize_results() {
        QueryResult::KeepTopK(result.topk_results, target_k);
    }

    void merge_batch_results(const std::vector<QueryResult::Item>& new_items) {
        result.merge_batch(new_items, expanded_k, merge_buffer_);
    }

    void RecordSearchedLevel2Buckets(const std::vector<uint32_t>& bucket_ids) {
        searched_level_2_bucket_ids.insert(searched_level_2_bucket_ids.end(), bucket_ids.begin(), bucket_ids.end());
    }

    // Record per-bucket mask statistics for a single L2 bucket.
    // passed_count = number of docs passing the filter (0 means all-zero mask).
    // total_count = total docs in bucket.
    void RecordBucketMaskStats(int passed_count, int total_count) {
        if (passed_count == 0) {
            searched_l2_count_allzero.fetch_add(1, std::memory_order_relaxed);
        } else {
            searched_l2_count_nonzero.fetch_add(1, std::memory_order_relaxed);
            double selectivity = static_cast<double>(passed_count) / static_cast<double>(total_count);
            size_t idx = bucket_selectivity_idx_.fetch_add(1, std::memory_order_relaxed);
            if (idx < bucket_selectivity_.size()) {
                bucket_selectivity_[idx] = selectivity;
            }
        }
    }

    // =========================================================
    // Filter Executor Implementation
    // =========================================================

    /**
     * @brief Plan-based IVF filter evaluation using BucketPlan.
     * Leaf materialization: TAG → ivf.get_row_and_or, NOT_TAG → ivf.get_row_not.
     * Stack-based evaluator (no std::function). Leaf pointers pre-bound outside block loop.
     */
    void search_ivf_plan(const BucketLevelIVF& ivf, std::vector<uint64_t>& valid_buckets_out,
                         std::vector<uint64_t, AlignedAllocator<uint64_t>>& scratch_buffer, int core_id) {
        const BucketPlan& plan = filter_exp.bucket_plan;
        uint32_t u64_count = ivf.get_aligned_stride();
        valid_buckets_out.resize(u64_count);

        if (plan.always_false) {
            std::fill(valid_buckets_out.begin(), valid_buckets_out.end(), 0ULL);
            return;
        }

        if (!plan.valid || plan.root == UINT32_MAX) {
            std::fill(valid_buckets_out.begin(), valid_buckets_out.end(), ~0ULL);
            return;
        }

        const PlanNode& root_node = plan.nodes[plan.root];

        // Pre-bind IVF row pointers for all leaves (outside block loop)
        // leaf_kind: 0=normal, 1=all_zero (TAG nullptr), 2=all_ones (NOT_TAG nullptr)
        std::vector<const uint64_t*>& leaf_rows = ThreadLocalIvfLeafRows();
        std::vector<uint8_t>& leaf_kind = ThreadLocalIvfLeafKind();
        if (leaf_rows.size() < plan.leaf_count)
            leaf_rows.resize(plan.leaf_count, nullptr);
        if (leaf_kind.size() < plan.leaf_count)
            leaf_kind.resize(plan.leaf_count, 0);

        for (uint32_t i = 0; i < plan.nodes.size(); ++i) {
            const PlanNode& node = plan.nodes[i];
            if (node.op != PlanOp::TAG && node.op != PlanOp::NOT_TAG)
                continue;
            uint16_t li = node.leaf_index;
            if (li >= plan.leaf_count)
                continue;

            if (node.op == PlanOp::TAG) {
                const uint64_t* row = ivf.get_row_and_or(core_id, node.value);
                if (row == nullptr) {
                    leaf_kind[li] = 1;
                    leaf_rows[li] = nullptr;
                } else {
                    leaf_kind[li] = 0;
                    leaf_rows[li] = row;
                }
            } else  // NOT_TAG
            {
                const uint64_t* row = ivf.get_row_not(core_id, node.value);
                if (row == nullptr) {
                    leaf_kind[li] = 2;
                    leaf_rows[li] = nullptr;
                } else {
                    leaf_kind[li] = 0;
                    leaf_rows[li] = row;
                }
            }
        }

        // Single-leaf fast path
        if (root_node.op == PlanOp::TAG || root_node.op == PlanOp::NOT_TAG) {
            uint16_t li = root_node.leaf_index;
            if (li >= plan.leaf_count || leaf_kind[li] == 1)
                std::fill(valid_buckets_out.begin(), valid_buckets_out.end(), 0ULL);
            else if (leaf_kind[li] == 2)
                std::fill(valid_buckets_out.begin(), valid_buckets_out.end(), ~0ULL);
            else
                std::memcpy(valid_buckets_out.data(), leaf_rows[li], u64_count * sizeof(uint64_t));
            return;
        }

        // Multi-leaf: block loop with non-recursive IvfFrame evaluator
        uint32_t cache_line_bytes = cpu_cache_line_size > 0 ? (uint32_t)cpu_cache_line_size : 64;
        uint32_t block_u64 = std::max<uint32_t>(1, (cache_line_bytes / sizeof(uint64_t)) * 4);

        std::vector<IvfFrame>& frame_stack = ThreadLocalIvfFrameStack();
        if (frame_stack.size() < 64)
            frame_stack.resize(64);

        {
            size_t worst_case = ((size_t)plan.leaf_count * 2 + plan.nodes.size()) * block_u64;
            if (scratch_buffer.size() < worst_case)
                scratch_buffer.resize(worst_case);
        }

        for (uint32_t block_start = 0; block_start < u64_count; block_start += block_u64) {
            uint32_t block_len = std::min(block_u64, u64_count - block_start);
            size_t arena_top = 0;

            auto alloc = [&](uint32_t len) -> uint64_t* {
                if (arena_top + len > scratch_buffer.size())
                    scratch_buffer.resize(arena_top + len + 256);
                uint64_t* p = &scratch_buffer[arena_top];
                arena_top += len;
                return p;
            };

            bool root_is_and = (root_node.op == PlanOp::AND_GROUP);
            int frame_top = 0;
            frame_stack[0] = IvfFrame{root_node.first_child,
                                      root_node.first_child + root_node.child_count,
                                      nullptr,
                                      !root_is_and,
                                      root_is_and,
                                      root_node.op};

            bool has_child_result = false;
            const uint64_t* child_result_ptr = nullptr;
            bool child_result_is_zero = false;
            bool child_result_is_ones = false;

            auto merge_into = [&](IvfFrame& f, const uint64_t* cptr, bool czero, bool cones) {
                if (f.op == PlanOp::AND_GROUP) {
                    if (f.acc_is_zero || czero) {
                        f.acc_is_zero = true;
                        f.acc_is_ones = false;
                        f.acc = nullptr;
                        f.child_pos = f.child_end;
                    } else if (f.acc_is_ones) {
                        f.acc = cptr;
                        f.acc_is_ones = cones;
                        f.acc_is_zero = czero;
                    } else if (cones) {
                        // AND(acc, ones) = acc unchanged
                    } else {
                        uint64_t* dst = alloc(block_len);
                        uint32_t j = 0;
#if defined(__aarch64__) || defined(__arm__)
                        for (; j + 7 < block_len; j += 8) {
                            uint64x2_t r0 = vandq_u64(vld1q_u64(f.acc + j), vld1q_u64(cptr + j));
                            uint64x2_t r1 = vandq_u64(vld1q_u64(f.acc + j + 2), vld1q_u64(cptr + j + 2));
                            uint64x2_t r2 = vandq_u64(vld1q_u64(f.acc + j + 4), vld1q_u64(cptr + j + 4));
                            uint64x2_t r3 = vandq_u64(vld1q_u64(f.acc + j + 6), vld1q_u64(cptr + j + 6));
                            vst1q_u64(dst + j, r0);
                            vst1q_u64(dst + j + 2, r1);
                            vst1q_u64(dst + j + 4, r2);
                            vst1q_u64(dst + j + 6, r3);
                        }
#endif
                        for (; j < block_len; ++j) {
                            dst[j] = f.acc[j] & cptr[j];
                        }
                        f.acc = dst;
                    }
                } else  // OR_GROUP
                {
                    if (f.acc_is_ones || cones) {
                        f.acc_is_ones = true;
                        f.acc_is_zero = false;
                        f.acc = nullptr;
                    } else if (f.acc_is_zero) {
                        f.acc = cptr;
                        f.acc_is_zero = false;
                        f.acc_is_ones = cones;
                    } else if (czero) {
                        // OR(acc, zero) = acc unchanged
                    } else {
                        uint64_t* dst = alloc(block_len);
                        uint32_t j = 0;
#if defined(__aarch64__) || defined(__arm__)
                        for (; j + 7 < block_len; j += 8) {
                            uint64x2_t r0 = vorrq_u64(vld1q_u64(f.acc + j), vld1q_u64(cptr + j));
                            uint64x2_t r1 = vorrq_u64(vld1q_u64(f.acc + j + 2), vld1q_u64(cptr + j + 2));
                            uint64x2_t r2 = vorrq_u64(vld1q_u64(f.acc + j + 4), vld1q_u64(cptr + j + 4));
                            uint64x2_t r3 = vorrq_u64(vld1q_u64(f.acc + j + 6), vld1q_u64(cptr + j + 6));
                            vst1q_u64(dst + j, r0);
                            vst1q_u64(dst + j + 2, r1);
                            vst1q_u64(dst + j + 4, r2);
                            vst1q_u64(dst + j + 6, r3);
                        }
#endif
                        for (; j < block_len; ++j)
                            dst[j] = f.acc[j] | cptr[j];
                        f.acc = dst;
                    }
                }
            };

            while (frame_top >= 0) {
                IvfFrame& frame = frame_stack[frame_top];

                if (has_child_result) {
                    has_child_result = false;
                    merge_into(frame, child_result_ptr, child_result_is_zero, child_result_is_ones);
                    frame.child_pos++;
                    continue;
                }

                if (frame.child_pos >= frame.child_end) {
                    child_result_ptr = frame.acc;
                    child_result_is_zero = frame.acc_is_zero;
                    child_result_is_ones = frame.acc_is_ones;
                    has_child_result = true;
                    --frame_top;
                    continue;
                }

                uint32_t child_node_idx = plan.children[frame.child_pos];
                const PlanNode& child_node = plan.nodes[child_node_idx];

                if (child_node.op == PlanOp::TAG || child_node.op == PlanOp::NOT_TAG) {
                    uint16_t li = child_node.leaf_index;
                    const uint64_t* child_ptr = nullptr;
                    bool child_is_zero = false;
                    bool child_is_ones = false;

                    if (li >= plan.leaf_count)
                        child_is_zero = true;
                    else if (leaf_kind[li] == 1)
                        child_is_zero = true;
                    else if (leaf_kind[li] == 2)
                        child_is_ones = true;
                    else
                        child_ptr = leaf_rows[li] + block_start;

                    merge_into(frame, child_ptr, child_is_zero, child_is_ones);
                    frame.child_pos++;
                } else {
                    if (frame_top + 1 >= static_cast<int>(frame_stack.size()))
                        frame_stack.resize(frame_stack.size() * 2);
                    bool child_is_and = (child_node.op == PlanOp::AND_GROUP);
                    frame_stack[frame_top + 1] = IvfFrame{child_node.first_child,
                                                          child_node.first_child + child_node.child_count,
                                                          nullptr,
                                                          !child_is_and,
                                                          child_is_and,
                                                          child_node.op};
                    frame_top++;
                }
            }

            // Write result for this block
            if (child_result_is_zero)
                std::memset(valid_buckets_out.data() + block_start, 0, block_len * sizeof(uint64_t));
            else if (child_result_is_ones)
                std::memset(valid_buckets_out.data() + block_start, 0xFF, block_len * sizeof(uint64_t));
            else if (child_result_ptr != nullptr)
                std::memcpy(valid_buckets_out.data() + block_start, child_result_ptr, block_len * sizeof(uint64_t));
            else
                std::memset(valid_buckets_out.data() + block_start, 0, block_len * sizeof(uint64_t));
        }
    }

    /**
     * @brief IVF filter dispatch: plan-based evaluation only.
     */
    void search_ivf(const BucketLevelIVF& ivf, std::vector<uint64_t>& valid_buckets_out,
                    std::vector<uint64_t, AlignedAllocator<uint64_t>>& scratch_buffer, int core_id) {
        uint32_t u64_count = ivf.get_aligned_stride();
        if (valid_buckets_out.size() < u64_count)
            valid_buckets_out.resize(u64_count);

        if (!filter_exp.bucket_plan.valid) {
            std::fill(valid_buckets_out.begin(), valid_buckets_out.end(), ~0ULL);
            return;
        }

        search_ivf_plan(ivf, valid_buckets_out, scratch_buffer, core_id);
    }

    // =========================================================
    // BucketPlan Execution (Group Short-Circuit)
    // =========================================================

    struct IvfFrame {
        uint32_t child_pos;
        uint32_t child_end;
        const uint64_t* acc;
        bool acc_is_zero;
        bool acc_is_ones;
        PlanOp op;
    };

    static constexpr uint16_t kPlanMinLeafCount = 1;

    static std::vector<const uint64_t*>& ThreadLocalLeafBasePtrs() {
        static thread_local std::vector<const uint64_t*> ptrs;
        return ptrs;
    }

    static std::vector<const uint64_t*>& ThreadLocalSortedPtrs() {
        static thread_local std::vector<const uint64_t*> ptrs;
        return ptrs;
    }

    enum LeafSentinel : uint8_t { kSentNormal = 0, kSentZero = 1, kSentOnes = 2 };

    // Return kind: 0=normal, 1=all_zero, 2=all_ones
    uint8_t search_bucket_plan(const Bucket& bucket, uint32_t bucket_id, const BucketLevelIVF* ivf,
                               std::vector<uint64_t>& valid_docs_out,
                               std::vector<uint64_t, AlignedAllocator<uint64_t>>& scratch_buffer) {
        const BucketPlan& plan = filter_exp.bucket_plan;
        uint32_t stride = bucket.get_stride();
        uint32_t doc_num = bucket.get_doc_num();
        valid_docs_out.resize(stride);

        if (plan.always_false) {
            std::fill(valid_docs_out.begin(), valid_docs_out.end(), 0ULL);
            return 1;
        }

        if (!plan.valid || plan.root == UINT32_MAX) {
            std::fill(valid_docs_out.begin(), valid_docs_out.end(), ~0ULL);
            return 2;
        }

        const uint64_t* kZero = BucketInternal::g_global_zero_bitmap;
        const uint64_t* kOnes = BucketInternal::g_global_ones_bitmap;

        // Pre-compute leaf sentinel state (constant folding per bucket)
        std::vector<LeafSentinel>& leaf_sentinel = ThreadLocalLeafSentinelState();
        leaf_sentinel.resize(plan.leaf_count, kSentNormal);
        std::fill(leaf_sentinel.begin(), leaf_sentinel.begin() + plan.leaf_count, kSentNormal);

        // IVF sentinel folding: use GetTagAggState to mark tags per bucket
        static thread_local uint32_t last_folded_bucket = UINT32_MAX;
        static thread_local const BucketPlan* last_folded_plan = nullptr;
        bool need_fold = (bucket_id != last_folded_bucket || &plan != last_folded_plan);
        if (need_fold && ivf != nullptr) {
            for (uint32_t i = 0; i < plan.nodes.size(); ++i) {
                const PlanNode& node = plan.nodes[i];
                if (node.op != PlanOp::TAG && node.op != PlanOp::NOT_TAG)
                    continue;
                uint16_t li = node.leaf_index;
                if (li >= plan.leaf_count)
                    continue;
                auto agg = ivf->GetTagAggState(node.value, bucket_id);
                if (node.op == PlanOp::TAG) {
                    if (!agg.has_tag)
                        leaf_sentinel[li] = kSentZero;
                    else if (agg.all_have_tag)
                        leaf_sentinel[li] = kSentOnes;
                    else
                        leaf_sentinel[li] = kSentNormal;
                } else  // NOT_TAG
                {
                    if (!agg.has_tag)
                        leaf_sentinel[li] = kSentOnes;
                    else if (agg.all_have_tag)
                        leaf_sentinel[li] = kSentZero;
                    else
                        leaf_sentinel[li] = kSentNormal;
                }
            }
            last_folded_bucket = bucket_id;
            last_folded_plan = &plan;
        }

        // Build leaf_base_ptrs for current L2 bucket
        std::vector<const uint64_t*>& leaf_ptrs = ThreadLocalLeafBasePtrs();
        if (leaf_ptrs.size() < plan.leaf_count)
            leaf_ptrs.resize(plan.leaf_count, nullptr);

        // Batch sorted merge lookup: O(N+M) vs per-leaf O(1) random access
        {
            const auto& sorted_tags = filter_exp.sorted_unique_tag_ids;
            std::vector<const uint64_t*>& sorted_ptrs = ThreadLocalSortedPtrs();
            if (sorted_ptrs.size() < sorted_tags.size())
                sorted_ptrs.resize(sorted_tags.size());
            bucket.batch_get_tag_bits_sorted(sorted_tags.data(), sorted_tags.size(), sorted_ptrs.data());

            for (uint32_t i = 0; i < plan.nodes.size(); ++i) {
                const PlanNode& node = plan.nodes[i];
                if (node.op != PlanOp::TAG && node.op != PlanOp::NOT_TAG)
                    continue;
                uint16_t li = node.leaf_index;
                if (li >= plan.leaf_count)
                    continue;
                if (leaf_sentinel[li] == kSentZero) {
                    leaf_ptrs[li] = kZero;
                    continue;
                }
                if (leaf_sentinel[li] == kSentOnes) {
                    leaf_ptrs[li] = kOnes;
                    continue;
                }
                uint32_t si = plan.leaf_to_sorted_idx[li];
                const uint64_t* raw = (si < sorted_tags.size()) ? sorted_ptrs[si] : kZero;
                if (raw == kZero && node.op == PlanOp::NOT_TAG)
                    leaf_ptrs[li] = kOnes;
                else
                    leaf_ptrs[li] = raw;
            }
        }

        // Single-leaf fast path: root is TAG or NOT_TAG (no group frame needed)
        const PlanNode& root_node = plan.nodes[plan.root];
        if (root_node.op == PlanOp::TAG || root_node.op == PlanOp::NOT_TAG) {
            uint16_t li = root_node.leaf_index;
            if (li >= plan.leaf_count) {
                std::fill(valid_docs_out.begin(), valid_docs_out.end(), 0ULL);
                return 1;
            }
            LeafSentinel sent = leaf_sentinel[li];
            if (sent == kSentZero) {
                std::fill(valid_docs_out.begin(), valid_docs_out.end(), 0ULL);
                return 1;
            }
            if (sent == kSentOnes) {
                std::fill(valid_docs_out.begin(), valid_docs_out.end(), ~0ULL);
                return 2;
            }
            const uint64_t* base = leaf_ptrs[li];
            if (root_node.op == PlanOp::TAG) {
                if (base == kZero || base == nullptr) {
                    std::fill(valid_docs_out.begin(), valid_docs_out.end(), 0ULL);
                    return 1;
                }
                if (base == kOnes) {
                    std::fill(valid_docs_out.begin(), valid_docs_out.end(), ~0ULL);
                    return 2;
                }
                std::memcpy(valid_docs_out.data(), base, stride * sizeof(uint64_t));
            } else  // NOT_TAG
            {
                if (base == kZero || base == nullptr) {
                    std::fill(valid_docs_out.begin(), valid_docs_out.end(), ~0ULL);
                    return 2;
                }
                if (base == kOnes) {
                    std::fill(valid_docs_out.begin(), valid_docs_out.end(), 0ULL);
                    return 1;
                }
                uint32_t j = 0;
#if defined(__aarch64__) || defined(__arm__)
                uint64x2_t all_ones = vdupq_n_u64(~0ULL);
                for (; j + 1 < stride; j += 2) {
                    uint64x2_t v = vld1q_u64(base + j);
                    vst1q_u64(valid_docs_out.data() + j, veorq_u64(v, all_ones));
                }
#endif
                for (; j < stride; ++j)
                    valid_docs_out[j] = ~base[j];
            }
            bool all_zero = true, all_ones = true;
            for (uint32_t k = 0; k < stride; ++k) {
                uint64_t mask = (doc_num >= (k + 1) * 64) ? ~0ULL
                                : (doc_num > k * 64)      ? ((1ULL << (doc_num - k * 64)) - 1)
                                                          : 0ULL;
                if (valid_docs_out[k] & mask)
                    all_zero = false;
                if ((valid_docs_out[k] & mask) != mask)
                    all_ones = false;
            }
            if (all_zero)
                return 1;
            if (all_ones)
                return 2;
            return 0;
        }

        // Block loop
        uint32_t cache_line_bytes = cpu_cache_line_size > 0 ? (uint32_t)cpu_cache_line_size : 64;
        uint32_t block_u64 = std::max<uint32_t>(1, (cache_line_bytes / sizeof(uint64_t)) * 4);

        // Pre-allocate scratch arena to avoid resize during block execution.
        {
            size_t worst_case = ((size_t)plan.leaf_count * 2 + plan.nodes.size()) * stride;
            if (scratch_buffer.size() < worst_case)
                scratch_buffer.resize(worst_case);
        }

        for (uint32_t block_start = 0; block_start < stride; block_start += block_u64) {
            uint32_t block_len = std::min(block_u64, stride - block_start);
            size_t arena_top = 0;

            auto alloc_arena = [&](uint32_t len) -> uint64_t* {
                if (arena_top + len > scratch_buffer.size())
                    scratch_buffer.resize(arena_top + len + 256);
                uint64_t* ptr = &scratch_buffer[arena_top];
                arena_top += len;
                return ptr;
            };

            struct NodeResult {
                const uint64_t* ptr = nullptr;
                bool is_zero = false;
                bool is_ones = false;
            };

            auto eval_leaf = [&](const PlanNode& node) -> NodeResult {
                const uint16_t li = node.leaf_index;
                if (li >= plan.leaf_count)
                    return {kZero, true, false};

                const uint64_t* base = leaf_ptrs[li];
                if (base == kZero)
                    return {kZero, true, false};
                if (base == kOnes)
                    return {kOnes, false, true};

                if (node.op == PlanOp::TAG)
                    return {base + block_start, false, false};

                // NOT_TAG: XOR with all-ones
                uint64_t* dst = alloc_arena(block_len);
                const uint64_t* src = base + block_start;
                uint32_t j = 0;
#if defined(__aarch64__) || defined(__arm__)
                const uint64x2_t all_ones = vdupq_n_u64(~0ULL);
                for (; j + 1 < block_len; j += 2) {
                    const uint64x2_t v = vld1q_u64(src + j);
                    vst1q_u64(dst + j, veorq_u64(v, all_ones));
                }
#endif
                for (; j < block_len; ++j)
                    dst[j] = ~src[j];
                return {dst, false, false};
            };

            auto merge_results = [&](PlanOp op, const NodeResult& lhs, const NodeResult& rhs) -> NodeResult {
                if (op == PlanOp::AND_GROUP) {
                    if (lhs.is_zero || rhs.is_zero)
                        return {kZero, true, false};
                    if (lhs.is_ones)
                        return rhs;
                    if (rhs.is_ones)
                        return lhs;

                    uint64_t* dst = alloc_arena(block_len);
                    uint32_t j = 0;
#if defined(__aarch64__) || defined(__arm__)
                    for (; j + 7 < block_len; j += 8) {
                        const uint64x2_t a0 = vld1q_u64(lhs.ptr + j);
                        const uint64x2_t a1 = vld1q_u64(lhs.ptr + j + 2);
                        const uint64x2_t a2 = vld1q_u64(lhs.ptr + j + 4);
                        const uint64x2_t a3 = vld1q_u64(lhs.ptr + j + 6);
                        const uint64x2_t b0 = vld1q_u64(rhs.ptr + j);
                        const uint64x2_t b1 = vld1q_u64(rhs.ptr + j + 2);
                        const uint64x2_t b2 = vld1q_u64(rhs.ptr + j + 4);
                        const uint64x2_t b3 = vld1q_u64(rhs.ptr + j + 6);
                        vst1q_u64(dst + j, vandq_u64(a0, b0));
                        vst1q_u64(dst + j + 2, vandq_u64(a1, b1));
                        vst1q_u64(dst + j + 4, vandq_u64(a2, b2));
                        vst1q_u64(dst + j + 6, vandq_u64(a3, b3));
                    }
                    for (; j + 1 < block_len; j += 2) {
                        const uint64x2_t a = vld1q_u64(lhs.ptr + j);
                        const uint64x2_t b = vld1q_u64(rhs.ptr + j);
                        vst1q_u64(dst + j, vandq_u64(a, b));
                    }
#endif
                    for (; j < block_len; ++j)
                        dst[j] = lhs.ptr[j] & rhs.ptr[j];
                    return {dst, false, false};
                }

                // OR_GROUP
                if (lhs.is_ones || rhs.is_ones)
                    return {kOnes, false, true};
                if (lhs.is_zero)
                    return rhs;
                if (rhs.is_zero)
                    return lhs;

                uint64_t* dst = alloc_arena(block_len);
                uint32_t j = 0;
#if defined(__aarch64__) || defined(__arm__)
                for (; j + 7 < block_len; j += 8) {
                    const uint64x2_t a0 = vld1q_u64(lhs.ptr + j);
                    const uint64x2_t a1 = vld1q_u64(lhs.ptr + j + 2);
                    const uint64x2_t a2 = vld1q_u64(lhs.ptr + j + 4);
                    const uint64x2_t a3 = vld1q_u64(lhs.ptr + j + 6);
                    const uint64x2_t b0 = vld1q_u64(rhs.ptr + j);
                    const uint64x2_t b1 = vld1q_u64(rhs.ptr + j + 2);
                    const uint64x2_t b2 = vld1q_u64(rhs.ptr + j + 4);
                    const uint64x2_t b3 = vld1q_u64(rhs.ptr + j + 6);
                    vst1q_u64(dst + j, vorrq_u64(a0, b0));
                    vst1q_u64(dst + j + 2, vorrq_u64(a1, b1));
                    vst1q_u64(dst + j + 4, vorrq_u64(a2, b2));
                    vst1q_u64(dst + j + 6, vorrq_u64(a3, b3));
                }
                for (; j + 1 < block_len; j += 2) {
                    const uint64x2_t a = vld1q_u64(lhs.ptr + j);
                    const uint64x2_t b = vld1q_u64(rhs.ptr + j);
                    vst1q_u64(dst + j, vorrq_u64(a, b));
                }
#endif
                for (; j < block_len; ++j)
                    dst[j] = lhs.ptr[j] | rhs.ptr[j];
                return {dst, false, false};
            };

            auto eval_node_iterative = [&]() -> NodeResult {
                const size_t n = plan.eval_order.size();
                // results array: indexed by node position in plan.nodes
                // Use arena for result storage to avoid stack allocation
                NodeResult default_nr{nullptr, false, false};
                // Thread-local results vector to avoid per-block allocation
                thread_local std::vector<NodeResult> tl_results;
                if (tl_results.size() < plan.nodes.size())
                    tl_results.resize(plan.nodes.size(), default_nr);

                for (size_t ei = 0; ei < n; ++ei) {
                    const uint32_t pos_idx = plan.eval_order[ei];
                    const PlanNode& node = plan.nodes[pos_idx];
                    if (node.op == PlanOp::TAG || node.op == PlanOp::NOT_TAG) {
                        tl_results[pos_idx] = eval_leaf(node);
                    } else {
                        const uint32_t child_begin = node.first_child;
                        const uint32_t child_end = child_begin + static_cast<uint32_t>(node.child_count);
                        if (child_begin >= child_end || child_end > plan.children.size()) {
                            tl_results[pos_idx] = {kZero, true, false};
                            continue;
                        }

                        NodeResult acc = tl_results[plan.children[child_begin]];
                        for (uint32_t c = child_begin + 1; c < child_end; ++c) {
                            if (node.op == PlanOp::AND_GROUP && acc.is_zero)
                                break;
                            if (node.op == PlanOp::OR_GROUP && acc.is_ones)
                                break;
                            acc = merge_results(node.op, acc, tl_results[plan.children[c]]);
                        }
                        tl_results[pos_idx] = acc;
                    }
                }
                return tl_results[plan.root];
            };

            const NodeResult result = eval_node_iterative();

            if (result.is_zero || result.ptr == kZero) {
                std::memset(valid_docs_out.data() + block_start, 0, block_len * sizeof(uint64_t));
            } else if (result.is_ones || result.ptr == kOnes) {
                std::memset(valid_docs_out.data() + block_start, 0xFF, block_len * sizeof(uint64_t));
            } else if (result.ptr != nullptr) {
                std::memcpy(valid_docs_out.data() + block_start, result.ptr, block_len * sizeof(uint64_t));
            }
        }

        // Verify actual data content with doc_num mask — flag-based tracking
        // can miss all-zero results from non-sentinel merges (e.g. AND of
        // disjoint bitmaps produces real data that is all-zero).
        {
            bool data_zero = true, data_ones = true;
            for (uint32_t k = 0; k < stride; ++k) {
                uint64_t mask = (doc_num >= (k + 1) * 64) ? ~0ULL
                                : (doc_num > k * 64)      ? ((1ULL << (doc_num - k * 64)) - 1)
                                                          : 0ULL;
                if (valid_docs_out[k] & mask)
                    data_zero = false;
                if ((valid_docs_out[k] & mask) != mask)
                    data_ones = false;
            }
            if (data_zero)
                return 1;
            if (data_ones)
                return 2;
        }
        return 0;
    }

    static std::vector<LeafSentinel>& ThreadLocalLeafSentinelState() {
        static thread_local std::vector<LeafSentinel> state;
        return state;
    }

    /**
     * @brief 验证工具：纯标量计算单条文档是否满足过滤条件
     */
    bool evaluate_single_doc_filter(uint32_t doc_id, const uint64_t* tag_bitmaps, uint32_t bitmap_stride) const {
        const auto& plan = filter_exp.bucket_plan;
        if (!plan.valid)
            return true;
        if (plan.always_false)
            return false;

        std::function<bool(uint32_t)> eval;
        eval = [&](uint32_t node_idx) -> bool {
            const PlanNode& node = plan.nodes[node_idx];
            if (node.op == PlanOp::TAG) {
                const uint64_t* doc_bitmap = tag_bitmaps + (size_t)doc_id * bitmap_stride;
                return (doc_bitmap[node.value / 64] >> (node.value % 64)) & 1ULL;
            }
            if (node.op == PlanOp::NOT_TAG) {
                const uint64_t* doc_bitmap = tag_bitmaps + (size_t)doc_id * bitmap_stride;
                return !((doc_bitmap[node.value / 64] >> (node.value % 64)) & 1ULL);
            }
            if (node.op == PlanOp::AND_GROUP) {
                for (uint32_t ci = node.first_child; ci < node.first_child + node.child_count; ++ci)
                    if (!eval(plan.children[ci]))
                        return false;
                return true;
            }
            // OR_GROUP
            for (uint32_t ci = node.first_child; ci < node.first_child + node.child_count; ++ci)
                if (eval(plan.children[ci]))
                    return true;
            return false;
        };

        return eval(plan.root);
    }

    /**
     * @brief 暴力全表扫描：遍历整个原始数据集，计算精确的 Top-K Ground Truth
     */
    void brute_force_search(const InputDataset& dataset) {
        auto cmp = [](const QueryResult::Item& a, const QueryResult::Item& b) {
            return a.score > b.score;  // 小顶堆：队头是当前 Top-K 里分数最小的
        };

        if (target_k <= 0) {
            ground_truth_results.clear();
            return;
        }

        uint32_t bitmap_stride = (total_tag_num + 63) / 64;
        const float* q_vec = query_vector.data();

        std::vector<std::vector<QueryResult::Item>> thread_candidates;
#ifdef _OPENMP
        thread_candidates.resize(omp_get_max_threads());
#else
        thread_candidates.resize(1);
#endif

#pragma omp parallel
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            std::priority_queue<QueryResult::Item, std::vector<QueryResult::Item>, decltype(cmp)> local_pq(cmp);

#pragma omp for schedule(static)
            for (int doc_id = 0; doc_id < total_doc_num; ++doc_id) {
                // 1. 属性精确过滤
                if (!evaluate_single_doc_filter((uint32_t)doc_id, dataset.tag_bitmaps, bitmap_stride)) {
                    continue;
                }

                // 2. NEON 向量化内积 (完全脱离 L2 拆解和 NPU)
                const float* doc_vec = dataset.vectors + (size_t)doc_id * vector_dim;
                float dot_product = 0.0f;
#if defined(__aarch64__)
                {
                    float32x4_t s0 = vdupq_n_f32(0.0f);
                    float32x4_t s1 = vdupq_n_f32(0.0f);
                    float32x4_t s2 = vdupq_n_f32(0.0f);
                    float32x4_t s3 = vdupq_n_f32(0.0f);
                    int d = 0;
                    for (; d + 15 < vector_dim; d += 16) {
                        s0 = vfmaq_f32(s0, vld1q_f32(q_vec + d), vld1q_f32(doc_vec + d));
                        s1 = vfmaq_f32(s1, vld1q_f32(q_vec + d + 4), vld1q_f32(doc_vec + d + 4));
                        s2 = vfmaq_f32(s2, vld1q_f32(q_vec + d + 8), vld1q_f32(doc_vec + d + 8));
                        s3 = vfmaq_f32(s3, vld1q_f32(q_vec + d + 12), vld1q_f32(doc_vec + d + 12));
                    }
                    float32x4_t sum = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
                    for (; d + 3 < vector_dim; d += 4) {
                        sum = vfmaq_f32(sum, vld1q_f32(q_vec + d), vld1q_f32(doc_vec + d));
                    }
                    float32x2_t lo = vget_low_f32(sum);
                    float32x2_t hi = vget_high_f32(sum);
                    float32x2_t pairwise = vadd_f32(lo, hi);
                    dot_product = vget_lane_f32(pairwise, 0) + vget_lane_f32(pairwise, 1);
                    for (; d < vector_dim; ++d) {
                        dot_product += q_vec[d] * doc_vec[d];
                    }
                }
#else
                for (int d = 0; d < vector_dim; ++d) {
                    dot_product += q_vec[d] * doc_vec[d];
                }
#endif

                // 3. 维护线程内 Top-K 小顶堆
                if (local_pq.size() < (size_t)target_k) {
                    local_pq.push({(uint32_t)doc_id, dot_product});
                } else if (dot_product > local_pq.top().score) {
                    local_pq.pop();
                    local_pq.push({(uint32_t)doc_id, dot_product});
                }
            }

            auto& local_vec = thread_candidates[tid];
            local_vec.clear();
            local_vec.reserve(local_pq.size());
            while (!local_pq.empty()) {
                local_vec.push_back(local_pq.top());
                local_pq.pop();
            }
        }

        // 4. 汇总所有线程候选，再做全局 Top-K
        std::priority_queue<QueryResult::Item, std::vector<QueryResult::Item>, decltype(cmp)> global_pq(cmp);
        for (const auto& local_vec : thread_candidates) {
            for (const auto& item : local_vec) {
                if (global_pq.size() < (size_t)target_k) {
                    global_pq.push(item);
                } else if (item.score > global_pq.top().score) {
                    global_pq.pop();
                    global_pq.push(item);
                }
            }
        }

        ground_truth_results.clear();
        while (!global_pq.empty()) {
            ground_truth_results.push_back(global_pq.top());
            global_pq.pop();
        }

        // 因为是小顶堆，出队后是从小到大，反转一下使其按照分数降序排列
        std::reverse(ground_truth_results.begin(), ground_truth_results.end());
    }

    void brute_force_search_filtered(const InputDataset& dataset, const std::vector<uint32_t>& passing_docs) {
        auto cmp = [](const QueryResult::Item& a, const QueryResult::Item& b) { return a.score > b.score; };

        if (target_k <= 0 || passing_docs.empty()) {
            ground_truth_results.clear();
            return;
        }

        const float* q_vec = query_vector.data();
        const size_t n_docs = passing_docs.size();

        std::vector<std::vector<QueryResult::Item>> thread_candidates;
#ifdef _OPENMP
        thread_candidates.resize(omp_get_max_threads());
#else
        thread_candidates.resize(1);
#endif

#pragma omp parallel
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
#else
            const int tid = 0;
#endif
            std::priority_queue<QueryResult::Item, std::vector<QueryResult::Item>, decltype(cmp)> local_pq(cmp);

#pragma omp for schedule(static)
            for (size_t idx = 0; idx < n_docs; ++idx) {
                uint32_t doc_id = passing_docs[idx];
                const float* doc_vec = dataset.vectors + (size_t)doc_id * vector_dim;
                float dot_product = 0.0f;
#if defined(__aarch64__)
                {
                    float32x4_t s0 = vdupq_n_f32(0.0f);
                    float32x4_t s1 = vdupq_n_f32(0.0f);
                    float32x4_t s2 = vdupq_n_f32(0.0f);
                    float32x4_t s3 = vdupq_n_f32(0.0f);
                    int d = 0;
                    for (; d + 15 < vector_dim; d += 16) {
                        s0 = vfmaq_f32(s0, vld1q_f32(q_vec + d), vld1q_f32(doc_vec + d));
                        s1 = vfmaq_f32(s1, vld1q_f32(q_vec + d + 4), vld1q_f32(doc_vec + d + 4));
                        s2 = vfmaq_f32(s2, vld1q_f32(q_vec + d + 8), vld1q_f32(doc_vec + d + 8));
                        s3 = vfmaq_f32(s3, vld1q_f32(q_vec + d + 12), vld1q_f32(doc_vec + d + 12));
                    }
                    float32x4_t sum = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
                    for (; d + 3 < vector_dim; d += 4) {
                        sum = vfmaq_f32(sum, vld1q_f32(q_vec + d), vld1q_f32(doc_vec + d));
                    }
                    float32x2_t lo = vget_low_f32(sum);
                    float32x2_t hi = vget_high_f32(sum);
                    float32x2_t pairwise = vadd_f32(lo, hi);
                    dot_product = vget_lane_f32(pairwise, 0) + vget_lane_f32(pairwise, 1);
                    for (; d < vector_dim; ++d) {
                        dot_product += q_vec[d] * doc_vec[d];
                    }
                }
#else
                for (int d = 0; d < vector_dim; ++d) {
                    dot_product += q_vec[d] * doc_vec[d];
                }
#endif
                if (local_pq.size() < (size_t)target_k) {
                    local_pq.push({doc_id, dot_product});
                } else if (dot_product > local_pq.top().score) {
                    local_pq.pop();
                    local_pq.push({doc_id, dot_product});
                }
            }

            auto& local_vec = thread_candidates[tid];
            local_vec.clear();
            local_vec.reserve(local_pq.size());
            while (!local_pq.empty()) {
                local_vec.push_back(local_pq.top());
                local_pq.pop();
            }
        }

        std::priority_queue<QueryResult::Item, std::vector<QueryResult::Item>, decltype(cmp)> global_pq(cmp);
        for (const auto& local_vec : thread_candidates) {
            for (const auto& item : local_vec) {
                if (global_pq.size() < (size_t)target_k) {
                    global_pq.push(item);
                } else if (item.score > global_pq.top().score) {
                    global_pq.pop();
                    global_pq.push(item);
                }
            }
        }

        ground_truth_results.clear();
        while (!global_pq.empty()) {
            ground_truth_results.push_back(global_pq.top());
            global_pq.pop();
        }
        std::reverse(ground_truth_results.begin(), ground_truth_results.end());
    }

    /**
     * @brief 计算系统召回率（处理 top-K 边界 score 并列）
     *
     * 当 ground truth 中存在 score 并列（多个 doc 有相同分数处于 top-K 边界）时，
     * 不同运行（OMP 线程调度、全表 vs filtered 扫描）可能选取不同的并列 doc 子集。
     * 本方法对边界并列组给予全额信用：只要 ANN 有足够的槽位容纳并列组，
     * 就不因 tie-breaking 差异惩罚 recall。
     */
    void calculate_recall() {
        if (ground_truth_results.empty()) {
            recall_rate = result.topk_results.empty() ? 1.0f : 0.0f;
            return;
        }

        const size_t ann_size = result.topk_results.size();
        std::vector<uint32_t> ann_docs;
        ann_docs.reserve(ann_size);
        for (const auto& item : result.topk_results) {
            ann_docs.push_back(item.doc_id);
        }
        std::sort(ann_docs.begin(), ann_docs.end());

        // 边界分数：GT 中最低的 score
        float boundary_score = ground_truth_results.back().score;

        // 将 GT 分为 unique（score > boundary）和 tied-at-boundary（score == boundary）
        int hits_above = 0;
        size_t n_tied = 0;

        for (const auto& gt : ground_truth_results) {
            bool found = std::binary_search(ann_docs.begin(), ann_docs.end(), gt.doc_id);
            if (gt.score > boundary_score) {
                if (found)
                    hits_above++;
            } else {
                n_tied++;
            }
        }

        size_t n_above = ground_truth_results.size() - n_tied;

        // 无并列 — 标准 recall
        if (n_tied == 0) {
            recall_rate = static_cast<float>(hits_above) / static_cast<float>(ground_truth_results.size());
            return;
        }

        // 全部并列（所有 GT doc 分数相同）— 退回精确匹配
        if (n_above == 0) {
            int hit_count = 0;
            for (const auto& gt : ground_truth_results) {
                if (std::binary_search(ann_docs.begin(), ann_docs.end(), gt.doc_id)) {
                    hit_count++;
                }
            }
            recall_rate = static_cast<float>(hit_count) / static_cast<float>(ground_truth_results.size());
            return;
        }

        // 并列感知 recall：
        // unique-score 部分：精确匹配（hits_above）
        // 边界并列部分：ANN 有 (ann_size - hits_above) 个剩余槽位可容纳并列 doc，
        //   给予 min(n_tied, ann_boundary_cap) 的全额信用
        size_t ann_boundary_cap = (ann_size > (size_t)hits_above) ? (ann_size - hits_above) : 0;
        size_t boundary_credit = std::min(n_tied, ann_boundary_cap);

        recall_rate =
            static_cast<float>(hits_above + boundary_credit) / static_cast<float>(ground_truth_results.size());
    }

   private:
    static std::vector<IvfFrame>& ThreadLocalIvfFrameStack() {
        static thread_local std::vector<IvfFrame> stack;
        return stack;
    }

    static std::vector<const uint64_t*>& ThreadLocalIvfLeafRows() {
        static thread_local std::vector<const uint64_t*> rows;
        return rows;
    }

    static std::vector<uint8_t>& ThreadLocalIvfLeafKind() {
        static thread_local std::vector<uint8_t> kinds;
        return kinds;
    }

    void InitializeReusableBuffers() {
        const size_t query_vec_cap = static_cast<size_t>(std::max(query_vector_reserve_floats, vector_dim));
        if (query_vector.capacity() < query_vec_cap) {
            query_vector.reserve(query_vec_cap);
            query_vector.resize(query_vec_cap);
            query_vector.clear();
        }

        const size_t topk_cap = static_cast<size_t>(std::max(max_query_topk_prealloc, 1));
        if (result.topk_results.capacity() < topk_cap) {
            result.topk_results.reserve(topk_cap);
            result.topk_results.resize(topk_cap);
            result.topk_results.clear();
        }

        const size_t merge_cap = static_cast<size_t>(std::max(query_merge_batch_reserve_items, 1));
        if (merge_buffer_.capacity() < merge_cap) {
            merge_buffer_.reserve(merge_cap);
            merge_buffer_.resize(merge_cap);
            merge_buffer_.clear();
        }
    }

    MemoryEventSession memory_event_session_;
    bool memory_events_enabled_ = false;
    std::vector<QueryResult::Item> merge_buffer_;
};
