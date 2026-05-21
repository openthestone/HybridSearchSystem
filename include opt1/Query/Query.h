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
inline uint16_t QueryFloat32ToFp16(float value)
{
    uint32_t f32;
    std::memcpy(&f32, &value, sizeof(float));
    uint16_t f16 = 0;
    uint32_t sign = (f32 >> 16) & 0x8000;
    int16_t exponent = ((f32 >> 23) & 0xFF) - 127;
    uint32_t mantissa = f32 & 0x007FFFFF;
    if (exponent > 15)
    {
        f16 = sign | 0x7C00;
    }
    else if (exponent <= -15)
    {
        f16 = sign;
    }
    else
    {
        exponent += 15;
        mantissa >>= 13;
        f16 = sign | (exponent << 10) | mantissa;
    }
    return f16;
}

struct QueryResult
{
    struct Item
    {
        uint32_t doc_id;
        float score;

        Item() = default;
        Item(uint32_t d, float s) : doc_id(d), score(s) {}

        // 用于 TopK 排序 (分数降序，越大越好，适配向量内积)
        bool operator>(const Item &other) const { return score > other.score; }
        bool operator<(const Item &other) const { return score < other.score; }
    };

    static bool BetterItem(const Item &a, const Item &b)
    {
        if (a.score != b.score) {
            return a.score > b.score;
        }
        return a.doc_id < b.doc_id;
    }

    static void KeepTopK(std::vector<Item> &items, int target_k)
    {
        if (target_k <= 0) {
            items.clear();
            return;
        }

        if (items.size() > static_cast<size_t>(target_k))
        {
            std::partial_sort(items.begin(),
                             items.begin() + target_k,
                             items.end(),
                             BetterItem);
            items.resize(target_k);
        }
    }
    // 最终的 TopK 结果 (按向量内积 score 降序排列)
    std::vector<Item> topk_results;

    // 这是 Probe-Expand 策略（先搜200个桶，不达条件则再搜200个桶...）的终止条件
    bool check_satisfaction(int target_k) const
    {
        return topk_results.size() >= (size_t)target_k;
    }

    // 归并新的一批结果到 topk_results 中
    void merge_batch(const std::vector<Item> &new_items,
                     int target_k,
                     std::vector<Item> &scratch_buffer)
    {
        if (new_items.empty())
            return;

        scratch_buffer.clear();
        TrackVectorReserve(scratch_buffer,
                           topk_results.size() + new_items.size(),
                           "Query.h:merge_batch.merged");
        scratch_buffer.insert(scratch_buffer.end(), topk_results.begin(), topk_results.end());
        scratch_buffer.insert(scratch_buffer.end(), new_items.begin(), new_items.end());
        KeepTopK(scratch_buffer, target_k);
        topk_results.swap(scratch_buffer);
    }
};

struct QueryTimingMetrics
{
    double bucket_level_ivf_ms = 0.0;
    double bucket_level_ivf_filter_eval_ms = 0.0;
    double bucket_level_ivf_centroid_score_ms = 0.0;
    double bucket_level_ivf_result_pack_ms = 0.0;
    double candidate_bucket_merge_ms = 0.0;
    double npu_async_launch_ms = 0.0;
    double npu_submit_ms = 0.0;
    double npu_kernel_exec_ms = 0.0;
    double npu_d2d_gather_ms = 0.0;
    double npu_d2h_transfer_ms = 0.0;
    double npu_sync_overhead_ms = 0.0;
    double inbucket_attr_filter_overlapped_ms = 0.0;
    double wait_npu_flag_ms = 0.0;
    double result_collection_ms = 0.0;
    double final_merge_ms = 0.0;
    int probe_expand_rounds = 0;
};

class Query
{
public:
    struct EvalStackItem
    {
        const uint64_t *ptr = nullptr;
        bool is_temp = false;
    };

    // search_bucket diagnostic counters removed (atomic overhead was too high)

    std::vector<float> query_vector; // 查询向量
    std::vector<uint16_t> mmad_query_padded_fp16; // MMAD [k,16] padded query payload
    FilterExp filter_exp;            // Filter 表达式
    int target_k = 0;                // 最终返回的 Top-K
    int expanded_k = 0;              // 实际检索阶段使用的扩展 K
    size_t query_id = 0;             // 查询编号（使用查询文件行号）
    int process_round_count_level_1 = 0; // 当前查询实际执行的一级桶扩搜轮数
    int process_round_count_level_2 = 0; // 当前查询实际执行的二级桶处理子批次数
    int searched_bucket_count_level_1 = 0; // 当前查询实际处理的一级桶数量
    int searched_bucket_count_level_2 = 0; // 当前查询实际处理的二级桶数量
    std::vector<uint32_t> searched_level_2_bucket_ids; // 当前查询实际处理过的二级桶编号（按处理顺序）
    QueryResult result;              // 查询结果

    // ==========================================
    // 新增：用于验证 Recall 的 Ground Truth 数据
    // ==========================================
    std::vector<QueryResult::Item> ground_truth_results;
    float recall_rate = 0.0f;
    QueryTimingMetrics timing_metrics;

    // 好像没用？
    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point end_time;
    std::chrono::steady_clock::time_point start_time_for_parallel_record;
    std::chrono::steady_clock::time_point end_time_for_parallel_record;

    Query()
    {
        InitializeReusableBuffers();
    }

    void PrepareMemoryEventSession(size_t query_id_in, bool enabled)
    {
        memory_events_enabled_ = enabled;
        if (enabled)
        {
            memory_event_session_.Reset(query_id_in);
        }
    }

    void Reset(const std::vector<float> &vec,
               std::string_view filter_expr_text,
               int k,
               size_t query_id_in)
    {
        target_k = k;
        expanded_k = k;
        query_id = query_id_in;
        process_round_count_level_1 = 0;
        process_round_count_level_2 = 0;
        searched_bucket_count_level_1 = 0;
        searched_bucket_count_level_2 = 0;
        recall_rate = 0.0f;
        timing_metrics = QueryTimingMetrics{};
        start_time = std::chrono::steady_clock::now();
        end_time = std::chrono::steady_clock::time_point{};
        start_time_for_parallel_record = std::chrono::steady_clock::time_point{};
        end_time_for_parallel_record = std::chrono::steady_clock::time_point{};

        result.topk_results.clear();
        ground_truth_results.clear();
        merge_buffer_.clear();
        searched_level_2_bucket_ids.clear();

        query_vector.clear();
        TrackVectorReserve(query_vector,
                           std::max(vec.size(), static_cast<size_t>(query_vector_reserve_floats)),
                           "Query.h:query_vector");
        query_vector.insert(query_vector.end(), vec.begin(), vec.end());

        // Pre-compute MMAD query payload (FP16 padded to [dim, 16])
        constexpr size_t kKernelResultCols = 16;
        const size_t padded_size = static_cast<size_t>(vector_dim) * kKernelResultCols;
        mmad_query_padded_fp16.clear();
        mmad_query_padded_fp16.resize(padded_size, 0);
#if defined(__aarch64__) || defined(__arm__)
        size_t i = 0;
        for (; i + 3 < vector_dim; i += 4)
        {
            float32x4_t vf32 = vld1q_f32(query_vector.data() + i);
            float16x4_t vf16 = vcvt_f16_f32(vf32);
            uint16x4_t vi16 = vreinterpret_u16_f16(vf16);
            vst1_lane_u16(mmad_query_padded_fp16.data() + (i + 0) * kKernelResultCols, vi16, 0);
            vst1_lane_u16(mmad_query_padded_fp16.data() + (i + 1) * kKernelResultCols, vi16, 1);
            vst1_lane_u16(mmad_query_padded_fp16.data() + (i + 2) * kKernelResultCols, vi16, 2);
            vst1_lane_u16(mmad_query_padded_fp16.data() + (i + 3) * kKernelResultCols, vi16, 3);
        }
        for (; i < vector_dim; ++i)
        {
            mmad_query_padded_fp16[i * kKernelResultCols] = QueryFloat32ToFp16(query_vector[i]);
        }
#else
        for (size_t i = 0; i < static_cast<size_t>(vector_dim); ++i)
        {
            mmad_query_padded_fp16[i * kKernelResultCols] = QueryFloat32ToFp16(query_vector[i]);
        }
#endif

        filter_exp.CompileFrom(filter_expr_text);

        if (!ComputeExpandedKFromTopK(k, expanded_k))
        {
            expanded_k = std::numeric_limits<int>::max();
        }
    }

    void MarkProcessingStart()
    {
        start_time_for_parallel_record = std::chrono::steady_clock::now();
        end_time_for_parallel_record = std::chrono::steady_clock::time_point{};
    }

    void MarkProcessingEnd()
    {
        end_time_for_parallel_record = std::chrono::steady_clock::now();
    }

    MemoryEventSession *memory_event_session() const
    {
        return memory_events_enabled_ ? const_cast<MemoryEventSession *>(&memory_event_session_) : nullptr;
    }

    void AppendMemoryEventLogs(MemoryEventLogCollector &collector) const
    {
        collector.AppendSession(memory_event_session());
    }

    void finalize_results()
    {
        QueryResult::KeepTopK(result.topk_results, target_k);
    }

    void merge_batch_results(const std::vector<QueryResult::Item> &new_items)
    {
        result.merge_batch(new_items, expanded_k, merge_buffer_);
    }

    static void WarmUpThreadLocalExecutionBuffers()
    {
        WarmUpEvalStack(ThreadLocalIvfEvalStack());
        WarmUpEvalStack(ThreadLocalBucketEvalStack());
    }

    void RecordSearchedLevel2Buckets(const std::vector<uint32_t> &bucket_ids)
    {
        searched_level_2_bucket_ids.insert(searched_level_2_bucket_ids.end(),
                                           bucket_ids.begin(),
                                           bucket_ids.end());
    }

    // =========================================================
    // Filter Executor Implementation
    // =========================================================

    /**
     * @brief 使用 BucketLevelIVF_RPN 筛选候选桶 (支持物理分片并行 + NEON SIMD)
     * @param ivf 倒排索引
     * @param valid_buckets_out 输出，用于存放有效的 Bucket Mask (局部Mask)
     * @param scratch_buffer 线程私有的草稿内存，由调用者传入
     * @param core_id 当前执行的 Core ID (用于定位 Shard)
     * @return void
     */
    void search_ivf(const BucketLevelIVF &ivf,
                    std::vector<uint64_t> &valid_buckets_out,
                    std::vector<uint64_t, AlignedAllocator<uint64_t>> &scratch_buffer,
                    int core_id)
    {
        // 获取该 Core 负责的数据宽度 (aligned stride)
        uint32_t u64_count = ivf.get_aligned_stride();

        // 如果没有 Filter，则该范围内所有桶都有效 (全1)
        if (filter_exp.BucketLevelIVF_RPN.empty())
        {
            if (valid_buckets_out.size() < u64_count)
                valid_buckets_out.resize(u64_count);
            std::fill(valid_buckets_out.begin(), valid_buckets_out.end(), ~0ULL);
            return;
        }

        // 确保输出 buffer 大小足够 (只存当前分片的 Mask)
        if (valid_buckets_out.size() < u64_count)
            valid_buckets_out.resize(u64_count);

        const auto &rpn = filter_exp.BucketLevelIVF_RPN;
        uint32_t cache_line_bytes = cpu_cache_line_size > 0 ? (uint32_t)cpu_cache_line_size : 64;
        uint32_t block_u64 = std::max<uint32_t>(1, (cache_line_bytes / sizeof(uint64_t)) * 4);
        std::vector<EvalStackItem> &stack = ThreadLocalIvfEvalStack();
        MemoryEventSession *session = GetActiveMemoryEventSession();
        const size_t stack_old_capacity = session == nullptr ? 0 : stack.capacity();

        for (uint32_t block_start = 0; block_start < u64_count; block_start += block_u64)
        {
            uint32_t block_len = std::min(block_u64, u64_count - block_start);
            size_t current_top = 0;
            size_t needed_size = rpn.size() * block_len;
            if (scratch_buffer.size() < needed_size)
            {
                // 仅为了 Debug，实际生产环境应预分配足够大
                // scratch_buffer.resize(needed_size);
            }

            auto alloc_scratch = [&](uint32_t len) -> uint64_t *
            {
                if (current_top + len > scratch_buffer.size())
                {
                    scratch_buffer.resize(current_top + len + 1024);
                }
                uint64_t *ptr = &scratch_buffer[current_top];
                current_top += len;
                return ptr;
            };

            stack.clear();

            size_t i = 0;
            size_t n = rpn.size();
            while (i < n)
            {
                const auto &item = rpn[i];

                if (!item.is_op)
                {
                    uint32_t tag_id = item.value;
                    i++;
                    if (i >= n)
                        break;
                    const auto &op_item = rpn[i];

                    if (op_item.value == FilterOp8::OP_IVF_LOAD_EXIST)
                    {
                        const uint64_t *row = ivf.get_row_and_or(core_id, tag_id);
                        stack.push_back({row + block_start, false});
                    }
                    else
                    {
                        const uint64_t *row = ivf.get_row_not(core_id, tag_id);
                        stack.push_back({row + block_start, false});
                    }
                }
                else
                {
                    if (stack.size() < 2)
                        break;

                    auto right = stack.back();
                    stack.pop_back();
                    auto left = stack.back();
                    stack.pop_back();

                    uint64_t *res_ptr = alloc_scratch(block_len);

                    if (item.value == FilterOp8::OP_AND)
                    {
                        uint32_t j = 0;
#if defined(__aarch64__) || defined(__arm__)
                        for (; j + 7 < block_len; j += 8)
                        {
                            uint64x2_t v1_0 = vld1q_u64(left.ptr + j);
                            uint64x2_t v1_1 = vld1q_u64(left.ptr + j + 2);
                            uint64x2_t v1_2 = vld1q_u64(left.ptr + j + 4);
                            uint64x2_t v1_3 = vld1q_u64(left.ptr + j + 6);
                            uint64x2_t v2_0 = vld1q_u64(right.ptr + j);
                            uint64x2_t v2_1 = vld1q_u64(right.ptr + j + 2);
                            uint64x2_t v2_2 = vld1q_u64(right.ptr + j + 4);
                            uint64x2_t v2_3 = vld1q_u64(right.ptr + j + 6);
                            vst1q_u64(res_ptr + j,     vandq_u64(v1_0, v2_0));
                            vst1q_u64(res_ptr + j + 2, vandq_u64(v1_1, v2_1));
                            vst1q_u64(res_ptr + j + 4, vandq_u64(v1_2, v2_2));
                            vst1q_u64(res_ptr + j + 6, vandq_u64(v1_3, v2_3));
                        }
                        for (; j + 3 < block_len; j += 4)
                        {
                            uint64x2_t v1_0 = vld1q_u64(left.ptr + j);
                            uint64x2_t v1_1 = vld1q_u64(left.ptr + j + 2);
                            uint64x2_t v2_0 = vld1q_u64(right.ptr + j);
                            uint64x2_t v2_1 = vld1q_u64(right.ptr + j + 2);
                            vst1q_u64(res_ptr + j,     vandq_u64(v1_0, v2_0));
                            vst1q_u64(res_ptr + j + 2, vandq_u64(v1_1, v2_1));
                        }
#endif
                        for (; j < block_len; ++j)
                            res_ptr[j] = left.ptr[j] & right.ptr[j];
                    }
                    else
                    {
                        uint32_t j = 0;
#if defined(__aarch64__) || defined(__arm__)
                        for (; j + 7 < block_len; j += 8)
                        {
                            uint64x2_t v1_0 = vld1q_u64(left.ptr + j);
                            uint64x2_t v1_1 = vld1q_u64(left.ptr + j + 2);
                            uint64x2_t v1_2 = vld1q_u64(left.ptr + j + 4);
                            uint64x2_t v1_3 = vld1q_u64(left.ptr + j + 6);
                            uint64x2_t v2_0 = vld1q_u64(right.ptr + j);
                            uint64x2_t v2_1 = vld1q_u64(right.ptr + j + 2);
                            uint64x2_t v2_2 = vld1q_u64(right.ptr + j + 4);
                            uint64x2_t v2_3 = vld1q_u64(right.ptr + j + 6);
                            vst1q_u64(res_ptr + j,     vorrq_u64(v1_0, v2_0));
                            vst1q_u64(res_ptr + j + 2, vorrq_u64(v1_1, v2_1));
                            vst1q_u64(res_ptr + j + 4, vorrq_u64(v1_2, v2_2));
                            vst1q_u64(res_ptr + j + 6, vorrq_u64(v1_3, v2_3));
                        }
                        for (; j + 3 < block_len; j += 4)
                        {
                            uint64x2_t v1_0 = vld1q_u64(left.ptr + j);
                            uint64x2_t v1_1 = vld1q_u64(left.ptr + j + 2);
                            uint64x2_t v2_0 = vld1q_u64(right.ptr + j);
                            uint64x2_t v2_1 = vld1q_u64(right.ptr + j + 2);
                            vst1q_u64(res_ptr + j,     vorrq_u64(v1_0, v2_0));
                            vst1q_u64(res_ptr + j + 2, vorrq_u64(v1_1, v2_1));
                        }
#endif
                        for (; j < block_len; ++j)
                            res_ptr[j] = left.ptr[j] | right.ptr[j];
                    }

                    stack.push_back({res_ptr, true});
                }
                i++;
            }

            if (stack.empty())
                return;

            const uint64_t *final_res = stack.back().ptr;
            std::memcpy(valid_buckets_out.data() + block_start, final_res, block_len * sizeof(uint64_t));
        }

        RecordCapacityGrowth<EvalStackItem>(session,
                                        "Query.h:search_ivf.stack",
                                        stack_old_capacity,
                                        stack.capacity());
    }

    /**
     * @brief 使用 Bucket_RPN 筛选桶内文档 (NEON SIMD 加速)
     * @param bucket 当前处理的桶
     * @param valid_docs_out 输出 bitset
     * @param scratch_buffer 线程私有的草稿内存，由调用者传入
     */
    void search_bucket(const Bucket &bucket,
                       std::vector<uint64_t> &valid_docs_out,
                       std::vector<uint64_t, AlignedAllocator<uint64_t>> &scratch_buffer)
    {
        uint32_t stride = bucket.get_stride();
        // 输出长度必须与当前桶 stride 一致，避免复用较大桶留下的旧尾部。
        valid_docs_out.resize(stride);

        if (filter_exp.Bucket_RPN.empty())
        {
            // 无 Filter，全保留
            std::fill(valid_docs_out.begin(), valid_docs_out.end(), ~0ULL);
            return;
        }

        const auto &rpn = filter_exp.Bucket_RPN;
        uint32_t cache_line_bytes = cpu_cache_line_size > 0 ? (uint32_t)cpu_cache_line_size : 64;
        uint32_t block_u64 = std::max<uint32_t>(1, (cache_line_bytes / sizeof(uint64_t)) * 4);
        std::vector<EvalStackItem> &stack = ThreadLocalBucketEvalStack();
        MemoryEventSession *session = GetActiveMemoryEventSession();
        const size_t stack_old_capacity = session == nullptr ? 0 : stack.capacity();

        // Sentinel pointers for short-circuit optimization
        const uint64_t *kZero = BucketInternal::g_global_zero_bitmap;
        const uint64_t *kOnes = BucketInternal::g_global_ones_bitmap;

        // ---- OPT: Pre-cache tag bitmap pointers outside block loop ----
        const size_t rpn_size = rpn.size();
        std::vector<const uint64_t *> tag_base_ptrs;
        tag_base_ptrs.reserve(rpn_size);
        for (size_t i = 0; i < rpn_size; ++i)
        {
            if (!rpn[i].is_op)
            {
                const uint64_t *base = bucket.get_tag_bits(rpn[i].value);
                if ((rpn[i].flags & 1) && base == kZero)
                    tag_base_ptrs.push_back(kOnes);
                else
                    tag_base_ptrs.push_back(base);
            }
            else
            {
                tag_base_ptrs.push_back(nullptr);
            }
        }

        for (uint32_t block_start = 0; block_start < stride; block_start += block_u64)
        {
            uint32_t block_len = std::min(block_u64, stride - block_start);
            size_t current_top = 0;
            size_t needed_size = rpn.size() * block_len;
            if (scratch_buffer.size() < needed_size)
            {
                scratch_buffer.resize(needed_size + 1024);
            }

            auto alloc_scratch = [&](uint32_t len) -> uint64_t *
            {
                if (current_top + len > scratch_buffer.size())
                {
                    scratch_buffer.resize(current_top + len + 1024);
                }
                uint64_t *ptr = &scratch_buffer[current_top];
                current_top += len;
                return ptr;
            };

            stack.clear();

            for (size_t rpn_idx = 0; rpn_idx < rpn.size(); ++rpn_idx)
            {
                const auto &item = rpn[rpn_idx];
                if (!item.is_op)
                {
                    const uint64_t *tag_base = tag_base_ptrs[rpn_idx];

                    // Prefetch next few tag bitmaps (lookahead 4 in RPN order)
                    {
                        size_t look = rpn_idx + 1;
                        int issued = 0;
                        while (look < rpn.size() && issued < 4)
                        {
                            if (!rpn[look].is_op)
                            {
                                const uint64_t *look_base = tag_base_ptrs[look];
                                if (look_base != kZero && look_base != kOnes)
                                    __builtin_prefetch(look_base + block_start, 0, 1);
                                ++issued;
                            }
                            ++look;
                        }
                    }

                    if (item.flags & 1)
                    {
                        if (tag_base == kOnes)
                        {
                            stack.push_back({kOnes, false});
                        }
                        else
                        {
                            const uint64_t *tag_bits = tag_base + block_start;
                            uint64_t *res_ptr = alloc_scratch(block_len);
                            uint32_t i = 0;
#if defined(__aarch64__) || defined(__arm__)
                            uint64x2_t all_ones = vdupq_n_u64(~0ULL);
                            for (; i + 7 < block_len; i += 8)
                            {
                                uint64x2_t v_0 = vld1q_u64(tag_bits + i);
                                uint64x2_t v_1 = vld1q_u64(tag_bits + i + 2);
                                uint64x2_t v_2 = vld1q_u64(tag_bits + i + 4);
                                uint64x2_t v_3 = vld1q_u64(tag_bits + i + 6);
                                vst1q_u64(res_ptr + i,     veorq_u64(v_0, all_ones));
                                vst1q_u64(res_ptr + i + 2, veorq_u64(v_1, all_ones));
                                vst1q_u64(res_ptr + i + 4, veorq_u64(v_2, all_ones));
                                vst1q_u64(res_ptr + i + 6, veorq_u64(v_3, all_ones));
                            }
                            for (; i + 3 < block_len; i += 4)
                            {
                                uint64x2_t v_0 = vld1q_u64(tag_bits + i);
                                uint64x2_t v_1 = vld1q_u64(tag_bits + i + 2);
                                vst1q_u64(res_ptr + i, veorq_u64(v_0, all_ones));
                                vst1q_u64(res_ptr + i + 2, veorq_u64(v_1, all_ones));
                            }
#endif
                            for (; i < block_len; ++i)
                                res_ptr[i] = ~tag_bits[i];
                            stack.push_back({res_ptr, true});
                        }
                    }
                    else
                    {
                        if (tag_base == kZero)
                        {
                            stack.push_back({kZero, false});
                        }
                        else
                        {
                            const uint64_t *tag_bits = tag_base + block_start;
                            stack.push_back({tag_bits, false});
                        }
                    }
                }
                else
                {
                    auto right = stack.back();
                    stack.pop_back();
                    auto left = stack.back();
                    stack.pop_back();

                    if (item.value == FilterOp8::OP_AND)
                    {
                        // ---- Short-circuit for AND ----
                        if (left.ptr == kZero || right.ptr == kZero)
                        {
                            // 0 AND X = 0
                            stack.push_back({kZero, false});
                        }
                        else if (left.ptr == kOnes)
                        {
                            // all-ones AND X = X
                            stack.push_back(right);
                        }
                        else if (right.ptr == kOnes)
                        {
                            // X AND all-ones = X
                            stack.push_back(left);
                        }
                        else if (left.is_temp)
                        {
                            // In-place AND into left's temp buffer (saves alloc + write)
                            uint64_t *dst = const_cast<uint64_t*>(left.ptr);
                            uint32_t i = 0;
#if defined(__aarch64__) || defined(__arm__)
                            for (; i + 7 < block_len; i += 8)
                            {
                                uint64x2_t vl_0 = vld1q_u64(dst + i);
                                uint64x2_t vl_1 = vld1q_u64(dst + i + 2);
                                uint64x2_t vl_2 = vld1q_u64(dst + i + 4);
                                uint64x2_t vl_3 = vld1q_u64(dst + i + 6);
                                uint64x2_t vr_0 = vld1q_u64(right.ptr + i);
                                uint64x2_t vr_1 = vld1q_u64(right.ptr + i + 2);
                                uint64x2_t vr_2 = vld1q_u64(right.ptr + i + 4);
                                uint64x2_t vr_3 = vld1q_u64(right.ptr + i + 6);
                                vst1q_u64(dst + i,     vandq_u64(vl_0, vr_0));
                                vst1q_u64(dst + i + 2, vandq_u64(vl_1, vr_1));
                                vst1q_u64(dst + i + 4, vandq_u64(vl_2, vr_2));
                                vst1q_u64(dst + i + 6, vandq_u64(vl_3, vr_3));
                            }
                            for (; i + 3 < block_len; i += 4)
                            {
                                uint64x2_t vl_0 = vld1q_u64(dst + i);
                                uint64x2_t vl_1 = vld1q_u64(dst + i + 2);
                                uint64x2_t vr_0 = vld1q_u64(right.ptr + i);
                                uint64x2_t vr_1 = vld1q_u64(right.ptr + i + 2);
                                vst1q_u64(dst + i, vandq_u64(vl_0, vr_0));
                                vst1q_u64(dst + i + 2, vandq_u64(vl_1, vr_1));
                            }
#endif
                            for (; i < block_len; ++i)
                                dst[i] &= right.ptr[i];
                            stack.push_back({left.ptr, true});
                        }
                        else if (right.is_temp)
                        {
                            // In-place AND into right's temp buffer
                            uint64_t *dst = const_cast<uint64_t*>(right.ptr);
                            uint32_t i = 0;
#if defined(__aarch64__) || defined(__arm__)
                            for (; i + 7 < block_len; i += 8)
                            {
                                uint64x2_t vl_0 = vld1q_u64(left.ptr + i);
                                uint64x2_t vl_1 = vld1q_u64(left.ptr + i + 2);
                                uint64x2_t vl_2 = vld1q_u64(left.ptr + i + 4);
                                uint64x2_t vl_3 = vld1q_u64(left.ptr + i + 6);
                                uint64x2_t vr_0 = vld1q_u64(dst + i);
                                uint64x2_t vr_1 = vld1q_u64(dst + i + 2);
                                uint64x2_t vr_2 = vld1q_u64(dst + i + 4);
                                uint64x2_t vr_3 = vld1q_u64(dst + i + 6);
                                vst1q_u64(dst + i,     vandq_u64(vl_0, vr_0));
                                vst1q_u64(dst + i + 2, vandq_u64(vl_1, vr_1));
                                vst1q_u64(dst + i + 4, vandq_u64(vl_2, vr_2));
                                vst1q_u64(dst + i + 6, vandq_u64(vl_3, vr_3));
                            }
                            for (; i + 3 < block_len; i += 4)
                            {
                                uint64x2_t vl_0 = vld1q_u64(left.ptr + i);
                                uint64x2_t vl_1 = vld1q_u64(left.ptr + i + 2);
                                uint64x2_t vr_0 = vld1q_u64(dst + i);
                                uint64x2_t vr_1 = vld1q_u64(dst + i + 2);
                                vst1q_u64(dst + i, vandq_u64(vl_0, vr_0));
                                vst1q_u64(dst + i + 2, vandq_u64(vl_1, vr_1));
                            }
#endif
                            for (; i < block_len; ++i)
                                dst[i] &= left.ptr[i];
                            stack.push_back({right.ptr, true});
                        }
                        else
                        {
                            // Neither is temp: allocate scratch
                            uint64_t *res_ptr = alloc_scratch(block_len);
                            uint32_t i = 0;
#if defined(__aarch64__) || defined(__arm__)
                            for (; i + 7 < block_len; i += 8)
                            {
                                uint64x2_t vl_0 = vld1q_u64(left.ptr + i);
                                uint64x2_t vl_1 = vld1q_u64(left.ptr + i + 2);
                                uint64x2_t vl_2 = vld1q_u64(left.ptr + i + 4);
                                uint64x2_t vl_3 = vld1q_u64(left.ptr + i + 6);
                                uint64x2_t vr_0 = vld1q_u64(right.ptr + i);
                                uint64x2_t vr_1 = vld1q_u64(right.ptr + i + 2);
                                uint64x2_t vr_2 = vld1q_u64(right.ptr + i + 4);
                                uint64x2_t vr_3 = vld1q_u64(right.ptr + i + 6);
                                vst1q_u64(res_ptr + i,     vandq_u64(vl_0, vr_0));
                                vst1q_u64(res_ptr + i + 2, vandq_u64(vl_1, vr_1));
                                vst1q_u64(res_ptr + i + 4, vandq_u64(vl_2, vr_2));
                                vst1q_u64(res_ptr + i + 6, vandq_u64(vl_3, vr_3));
                            }
                            for (; i + 3 < block_len; i += 4)
                            {
                                uint64x2_t vl_0 = vld1q_u64(left.ptr + i);
                                uint64x2_t vl_1 = vld1q_u64(left.ptr + i + 2);
                                uint64x2_t vr_0 = vld1q_u64(right.ptr + i);
                                uint64x2_t vr_1 = vld1q_u64(right.ptr + i + 2);
                                vst1q_u64(res_ptr + i, vandq_u64(vl_0, vr_0));
                                vst1q_u64(res_ptr + i + 2, vandq_u64(vl_1, vr_1));
                            }
#endif
                            for (; i < block_len; ++i)
                                res_ptr[i] = left.ptr[i] & right.ptr[i];
                            stack.push_back({res_ptr, true});
                        }
                    }
                    else  // OP_OR
                    {
                        // ---- Short-circuit for OR ----
                        if (left.ptr == kOnes || right.ptr == kOnes)
                        {
                            // all-ones OR X = all-ones
                            stack.push_back({kOnes, false});
                        }
                        else if (left.ptr == kZero)
                        {
                            // 0 OR X = X
                            stack.push_back(right);
                        }
                        else if (right.ptr == kZero)
                        {
                            // X OR 0 = X
                            stack.push_back(left);
                        }
                        else if (left.is_temp)
                        {
                            // In-place OR into left's temp buffer
                            uint64_t *dst = const_cast<uint64_t*>(left.ptr);
                            uint32_t i = 0;
#if defined(__aarch64__) || defined(__arm__)
                            for (; i + 7 < block_len; i += 8)
                            {
                                uint64x2_t vl_0 = vld1q_u64(dst + i);
                                uint64x2_t vl_1 = vld1q_u64(dst + i + 2);
                                uint64x2_t vl_2 = vld1q_u64(dst + i + 4);
                                uint64x2_t vl_3 = vld1q_u64(dst + i + 6);
                                uint64x2_t vr_0 = vld1q_u64(right.ptr + i);
                                uint64x2_t vr_1 = vld1q_u64(right.ptr + i + 2);
                                uint64x2_t vr_2 = vld1q_u64(right.ptr + i + 4);
                                uint64x2_t vr_3 = vld1q_u64(right.ptr + i + 6);
                                vst1q_u64(dst + i,     vorrq_u64(vl_0, vr_0));
                                vst1q_u64(dst + i + 2, vorrq_u64(vl_1, vr_1));
                                vst1q_u64(dst + i + 4, vorrq_u64(vl_2, vr_2));
                                vst1q_u64(dst + i + 6, vorrq_u64(vl_3, vr_3));
                            }
                            for (; i + 3 < block_len; i += 4)
                            {
                                uint64x2_t vl_0 = vld1q_u64(dst + i);
                                uint64x2_t vl_1 = vld1q_u64(dst + i + 2);
                                uint64x2_t vr_0 = vld1q_u64(right.ptr + i);
                                uint64x2_t vr_1 = vld1q_u64(right.ptr + i + 2);
                                vst1q_u64(dst + i, vorrq_u64(vl_0, vr_0));
                                vst1q_u64(dst + i + 2, vorrq_u64(vl_1, vr_1));
                            }
#endif
                            for (; i < block_len; ++i)
                                dst[i] |= right.ptr[i];
                            stack.push_back({left.ptr, true});
                        }
                        else if (right.is_temp)
                        {
                            // In-place OR into right's temp buffer
                            uint64_t *dst = const_cast<uint64_t*>(right.ptr);
                            uint32_t i = 0;
#if defined(__aarch64__) || defined(__arm__)
                            for (; i + 7 < block_len; i += 8)
                            {
                                uint64x2_t vl_0 = vld1q_u64(left.ptr + i);
                                uint64x2_t vl_1 = vld1q_u64(left.ptr + i + 2);
                                uint64x2_t vl_2 = vld1q_u64(left.ptr + i + 4);
                                uint64x2_t vl_3 = vld1q_u64(left.ptr + i + 6);
                                uint64x2_t vr_0 = vld1q_u64(dst + i);
                                uint64x2_t vr_1 = vld1q_u64(dst + i + 2);
                                uint64x2_t vr_2 = vld1q_u64(dst + i + 4);
                                uint64x2_t vr_3 = vld1q_u64(dst + i + 6);
                                vst1q_u64(dst + i,     vorrq_u64(vl_0, vr_0));
                                vst1q_u64(dst + i + 2, vorrq_u64(vl_1, vr_1));
                                vst1q_u64(dst + i + 4, vorrq_u64(vl_2, vr_2));
                                vst1q_u64(dst + i + 6, vorrq_u64(vl_3, vr_3));
                            }
                            for (; i + 3 < block_len; i += 4)
                            {
                                uint64x2_t vl_0 = vld1q_u64(left.ptr + i);
                                uint64x2_t vl_1 = vld1q_u64(left.ptr + i + 2);
                                uint64x2_t vr_0 = vld1q_u64(dst + i);
                                uint64x2_t vr_1 = vld1q_u64(dst + i + 2);
                                vst1q_u64(dst + i, vorrq_u64(vl_0, vr_0));
                                vst1q_u64(dst + i + 2, vorrq_u64(vl_1, vr_1));
                            }
#endif
                            for (; i < block_len; ++i)
                                dst[i] |= left.ptr[i];
                            stack.push_back({right.ptr, true});
                        }
                        else
                        {
                            // Neither is temp: allocate scratch
                            uint64_t *res_ptr = alloc_scratch(block_len);
                            uint32_t i = 0;
#if defined(__aarch64__) || defined(__arm__)
                            for (; i + 7 < block_len; i += 8)
                            {
                                uint64x2_t vl_0 = vld1q_u64(left.ptr + i);
                                uint64x2_t vl_1 = vld1q_u64(left.ptr + i + 2);
                                uint64x2_t vl_2 = vld1q_u64(left.ptr + i + 4);
                                uint64x2_t vl_3 = vld1q_u64(left.ptr + i + 6);
                                uint64x2_t vr_0 = vld1q_u64(right.ptr + i);
                                uint64x2_t vr_1 = vld1q_u64(right.ptr + i + 2);
                                uint64x2_t vr_2 = vld1q_u64(right.ptr + i + 4);
                                uint64x2_t vr_3 = vld1q_u64(right.ptr + i + 6);
                                vst1q_u64(res_ptr + i,     vorrq_u64(vl_0, vr_0));
                                vst1q_u64(res_ptr + i + 2, vorrq_u64(vl_1, vr_1));
                                vst1q_u64(res_ptr + i + 4, vorrq_u64(vl_2, vr_2));
                                vst1q_u64(res_ptr + i + 6, vorrq_u64(vl_3, vr_3));
                            }
                            for (; i + 3 < block_len; i += 4)
                            {
                                uint64x2_t vl_0 = vld1q_u64(left.ptr + i);
                                uint64x2_t vl_1 = vld1q_u64(left.ptr + i + 2);
                                uint64x2_t vr_0 = vld1q_u64(right.ptr + i);
                                uint64x2_t vr_1 = vld1q_u64(right.ptr + i + 2);
                                vst1q_u64(res_ptr + i, vorrq_u64(vl_0, vr_0));
                                vst1q_u64(res_ptr + i + 2, vorrq_u64(vl_1, vr_1));
                            }
#endif
                            for (; i < block_len; ++i)
                                res_ptr[i] = left.ptr[i] | right.ptr[i];
                            stack.push_back({res_ptr, true});
                        }
                    }
                }
            }

            if (stack.empty())
                return;

            const uint64_t *final_res = stack.back().ptr;
            // Handle sentinel pointers in final memcpy
            if (final_res == kZero)
            {
                std::memset(valid_docs_out.data() + block_start, 0, block_len * sizeof(uint64_t));
            }
            else if (final_res == kOnes)
            {
                std::memset(valid_docs_out.data() + block_start, 0xFF, block_len * sizeof(uint64_t));
            }
            else
            {
                std::memcpy(valid_docs_out.data() + block_start, final_res, block_len * sizeof(uint64_t));
            }
        }

        RecordCapacityGrowth<EvalStackItem>(session,
                                        "Query.h:search_bucket.stack",
                                        stack_old_capacity,
                                        stack.capacity());
    }

    // =========================================================
    // Brute Force & Validation Implementation
    // =========================================================

    /**
     * @brief 验证工具：纯标量计算单条文档是否满足过滤条件
     */
    bool evaluate_single_doc_filter(uint32_t doc_id, const uint64_t* tag_bitmaps, uint32_t bitmap_stride) const
    {
        if (filter_exp.Bucket_RPN.empty()) return true;

        const uint64_t* doc_bitmap = tag_bitmaps + (size_t)doc_id * bitmap_stride;
        
        static thread_local std::vector<bool> bool_stack;
        bool_stack.clear();

        for (const auto& item : filter_exp.Bucket_RPN)
        {
            if (!item.is_op)
            {
                uint32_t tag_id = item.value;
                // 从文档的 bitmap 中提取对应的 bit
                bool has_tag = (doc_bitmap[tag_id / 64] >> (tag_id % 64)) & 1ULL;
                // 处理 NOT 逻辑 (flags 的最低位代表是否 inverted)
                if (item.flags & 1) {
                    has_tag = !has_tag;
                }
                bool_stack.push_back(has_tag);
            }
            else
            {
                if (bool_stack.size() < 2) return false; // 防止异常 RPN
                
                bool right = bool_stack.back(); bool_stack.pop_back();
                bool left = bool_stack.back(); bool_stack.pop_back();

                if (item.value == FilterOp8::OP_AND) {
                    bool_stack.push_back(left && right);
                } else if (item.value == FilterOp8::OP_OR) {
                    bool_stack.push_back(left || right);
                }
            }
        }
        
        return bool_stack.empty() ? true : bool_stack.back();
    }

    /**
     * @brief 暴力全表扫描：遍历整个原始数据集，计算精确的 Top-K Ground Truth
     */
    void brute_force_search(const InputDataset& dataset)
    {
        auto cmp = [](const QueryResult::Item& a, const QueryResult::Item& b) {
            return a.score > b.score; // 小顶堆：队头是当前 Top-K 里分数最小的
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
            for (int doc_id = 0; doc_id < total_doc_num; ++doc_id)
            {
                // 1. 属性精确过滤
                if (!evaluate_single_doc_filter((uint32_t)doc_id, dataset.tag_bitmaps, bitmap_stride)) {
                    continue;
                }

                // 2. 纯标量内积计算 (完全脱离 L2 拆解和 NPU)
                float dot_product = 0.0f;
                const float* doc_vec = dataset.vectors + (size_t)doc_id * vector_dim;
                for (int d = 0; d < vector_dim; ++d) {
                    dot_product += q_vec[d] * doc_vec[d];
                }

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

    /**
     * @brief 计算系统召回率
     */
    void calculate_recall()
    {
        if (ground_truth_results.empty()) {
            // 如果连 GT 都是空，且引擎返回的也是空，则 100% 准确
            recall_rate = result.topk_results.empty() ? 1.0f : 0.0f;
            return;
        }

        // 将 ANN 返回的文档 ID 存入哈希集合，加速 O(1) 查找
        std::unordered_set<uint32_t> ann_docs;
        for (const auto& item : result.topk_results) {
            ann_docs.insert(item.doc_id);
        }

        int hit_count = 0;
#pragma omp parallel for reduction(+:hit_count) schedule(static)
        for (int i = 0; i < (int)ground_truth_results.size(); ++i) {
            if (ann_docs.count(ground_truth_results[(size_t)i].doc_id)) {
                hit_count++;
            }
        }

        recall_rate = static_cast<float>(hit_count) / ground_truth_results.size();
    }

private:
    static std::vector<EvalStackItem> &ThreadLocalIvfEvalStack()
    {
        static thread_local std::vector<EvalStackItem> stack;
        return stack;
    }

    static std::vector<EvalStackItem> &ThreadLocalBucketEvalStack()
    {
        static thread_local std::vector<EvalStackItem> stack;
        return stack;
    }

    static void WarmUpEvalStack(std::vector<EvalStackItem> &stack)
    {
        const size_t target = static_cast<size_t>(query_exec_stack_reserve_items);
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

    void InitializeReusableBuffers()
    {
        const size_t query_vec_cap = static_cast<size_t>(std::max(query_vector_reserve_floats, vector_dim));
        if (query_vector.capacity() < query_vec_cap)
        {
            query_vector.reserve(query_vec_cap);
            query_vector.resize(query_vec_cap);
            query_vector.clear();
        }

        filter_exp.Reserve(static_cast<size_t>(query_bucket_rpn_reserve_items),
                           static_cast<size_t>(query_bucket_level_ivf_rpn_reserve_items));

        const size_t topk_cap = static_cast<size_t>(std::max(max_query_topk_prealloc, 1));
        if (result.topk_results.capacity() < topk_cap)
        {
            result.topk_results.reserve(topk_cap);
            result.topk_results.resize(topk_cap);
            result.topk_results.clear();
        }

        const size_t merge_cap = static_cast<size_t>(std::max(query_merge_batch_reserve_items, 1));
        if (merge_buffer_.capacity() < merge_cap)
        {
            merge_buffer_.reserve(merge_cap);
            merge_buffer_.resize(merge_cap);
            merge_buffer_.clear();
        }
    }

    MemoryEventSession memory_event_session_;
    bool memory_events_enabled_ = false;
    std::vector<QueryResult::Item> merge_buffer_;
};
