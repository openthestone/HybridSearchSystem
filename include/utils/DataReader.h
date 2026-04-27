#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataBaseCPU/InputDataset.h"
#include "nlohmann/json.hpp"
#include "utils/ProtoParser.h"

#ifdef _OPENMP
#include <omp.h>
#endif

inline int GetNonQueryOpenMpThreadCount()
{
#ifdef _OPENMP
    const int thread_count = omp_get_num_procs();
    return thread_count > 0 ? thread_count : 1;
#else
    return 1;
#endif
}

// --- 全局参数定义 (Header-Only) ---
// 说明：默认值仅用于兜底，正常流程必须由 LoadParams(config.txt) 覆盖。
inline int total_doc_num = 0;
inline int total_tag_num = 0;
inline int total_bucket_num_level_1 = 10000;
inline int total_bucket_num_level_2 = 10000; // 由一级桶切分结果推导得到
inline int vector_dim = 0;
inline int max_doc_per_bucket_level_2 = 1024; // 该参数必须是16的倍数

inline int cpu_core_count = 24;
inline int cores_per_group = 8;
inline int group_count = 3;

inline int valid_bucket_num_base_level_1 = 200;
inline int valid_bucket_num_incremental_level_1 = 200;
inline int max_probe_l1_bucket_num_enable = 0;
inline int max_probe_l1_bucket_num = 64;
inline int max_process_bucket_num_level_2 = 200;
inline int analyze_t1 = 5;
inline double k_expand_param = 1.0;
inline std::vector<double> k_expand_param_test_set;
inline std::vector<int> max_probe_l1_bucket_num_test_set;
inline double target_qps_parallel = 2000.0;
inline std::string dataset_cache_file = "../../dataset.bin";
inline std::string tag_map_cache_file = "../../tag_map.bin";
inline std::string query_file = "../../QueryData_10000.txt";
inline std::string query_result_root = "../../result/";
inline std::string ground_truth_cache_file = "../../ground_truth_cache.bin";
inline std::string bucket_ivf_index_file = "../../bucket_ivf_index.bin";
inline std::string bucket_index_file = "../../bucket_index.bin";
inline std::string raw_vector_file = "/mnt/paas/kubernetes/kubelet/DataManager/0/relevance_que2search";
inline std::string raw_attr_dir = "/mnt/paas/kubernetes/kubelet/DataManager/0/inverted_union";
inline std::string raw_query_file = "/mnt/paas/kubernetes/kubelet/DataManager/0/raw_query.txt";
inline std::string raw_query_result_num_key = "result_num";
inline std::string raw_query_main_tier_key = "main_tier";
inline std::string raw_query_json_query_key = "json_query";
inline std::string raw_query_vector_node_key = "vector";
inline std::string raw_query_vector_value_key = "relevance_learning2rank";
inline std::string raw_query_syntax_filter_key = "syntax_filter";

inline int npu_device_id_start = 0;
inline int npu_device_count = 1;
inline int groups_per_device = 3;

// Asymmetric group-to-device mapping for balanced NPU utilization.
// Populated from config key "device_group_split" (comma-separated list,
// one count per device).  When set, overrides the symmetric groups_per_device
// allocation.  Example: "4,2" gives device 0 four groups and device 1 two.
// Fallback: when empty, groups are assigned symmetrically (groups_per_device
// per device).
inline std::vector<int> g_device_group_split; // per-device group count

// Derived mapping tables (built by BuildDeviceGroupMapping).
inline std::vector<int> g_group_to_device; // group_id -> device index
inline std::vector<int> g_group_to_stream; // group_id -> stream index within device
inline std::vector<std::vector<int>> g_device_to_groups; // device_idx -> list of group_ids

inline void BuildDeviceGroupMapping()
{
    const int total_groups = cpu_core_count / cores_per_group;
    g_group_to_device.resize(static_cast<size_t>(total_groups));
    g_group_to_stream.resize(static_cast<size_t>(total_groups));
    g_device_to_groups.resize(static_cast<size_t>(npu_device_count));

    if (g_device_group_split.empty())
    {
        // Symmetric fallback: contiguous groups_per_device per device
        for (int g = 0; g < total_groups; ++g)
        {
            int dev = g / groups_per_device;
            int strm = g % groups_per_device;
            g_group_to_device[g] = dev;
            g_group_to_stream[g] = strm;
            g_device_to_groups[dev].push_back(g);
        }
    }
    else
    {
        // Asymmetric: assign groups sequentially to each device per its count
        int g = 0;
        for (int d = 0; d < npu_device_count; ++d)
        {
            int count = (d < static_cast<int>(g_device_group_split.size()))
                            ? g_device_group_split[d]
                            : 0;
            for (int s = 0; s < count; ++s)
            {
                g_group_to_device[g] = d;
                g_group_to_stream[g] = s;
                g_device_to_groups[d].push_back(g);
                ++g;
            }
        }
    }
}

inline int GroupCountForDevice(int device_idx)
{
    return static_cast<int>(g_device_to_groups[device_idx].size());
}

inline int GroupIdForDeviceStream(int device_idx, int stream_idx)
{
    return g_device_to_groups[device_idx][stream_idx];
}

inline int max_query_tags = 2000;
inline int max_rpn_length = 6000;
inline int max_query_topk_prealloc = 32768;
inline int query_pool_capacity = 64;

inline int query_vector_reserve_floats = 0;
inline int query_bucket_rpn_reserve_items = 0;
inline int query_bucket_level_ivf_rpn_reserve_items = 0;
inline int query_compile_stack_reserve_items = 0;
inline int query_exec_stack_reserve_items = 0;
inline int query_merge_batch_reserve_items = 0;

inline int group_tls_scratch_pool_reserve_u64 = 0;
inline int group_tls_ivf_mask_reserve_u64 = 0;
inline int group_tls_batch_mask_reserve_u64 = 0;
inline int group_local_pq_reserve_items_per_rank = 0;
inline int group_thread_bucket_results_reserve_items_per_rank = 0;
inline int group_candidate_merge_reserve_items = 0;
inline int group_sorted_buckets_reserve_items = 0;
inline int group_batch_bucket_ids_reserve_items = 0;
inline int group_temp_doc_mask_reserve_u64_per_rank = 0;
inline int group_thread_doc_results_reserve_items_per_rank = 0;
inline int group_all_items_reserve_items = 0;

inline int cpu_cache_line_size = 128;

inline int memory_events_log_enable = 0;

inline int npu_debug_verify = 0;
inline int npu_debug_verify_max_report = 5;
inline float npu_debug_verify_abs_tol = 0.1f;
inline float npu_debug_verify_rel_tol = 0.02f;

inline bool g_params_loaded = false;
inline bool g_runtime_params_finalized = false;
inline constexpr int kSupportedVectorDimMultiple = 16;
inline constexpr int kMaxSupportedVectorDim = 1024;

enum class ResourceConfigProfile {
    Serial,
    Parallel
};

inline const char *ResourceConfigProfileName(ResourceConfigProfile profile) {
    switch (profile) {
        case ResourceConfigProfile::Serial:
            return "serial";
        case ResourceConfigProfile::Parallel:
            return "parallel";
    }
    return "unknown";
}

inline std::string ParamTrim(const std::string &s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) {
        ++b;
    }
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
        --e;
    }
    return s.substr(b, e - b);
}

inline void StripConfigComment(std::string &line) {
    const size_t sharp = line.find('#');
    const size_t slash = line.find("//");
    size_t pos = std::string::npos;
    if (sharp != std::string::npos) {
        pos = sharp;
    }
    if (slash != std::string::npos) {
        pos = (pos == std::string::npos) ? slash : std::min(pos, slash);
    }
    if (pos != std::string::npos) {
        line = line.substr(0, pos);
    }
}

inline bool ParseConfigInt(const std::string &raw, int &out) {
    std::string v = ParamTrim(raw);
    if (!v.empty() && v.back() == ';') {
        v.pop_back();
        v = ParamTrim(v);
    }
    if (v.empty()) {
        return false;
    }
    size_t pos = 0;
    long long parsed = 0;
    try {
        parsed = std::stoll(v, &pos, 10);
    } catch (...) {
        return false;
    }
    if (pos != v.size()) {
        return false;
    }
    if (parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) {
        return false;
    }
    out = static_cast<int>(parsed);
    return true;
}

inline bool ParseConfigFloat(const std::string &raw, float &out) {
    std::string v = ParamTrim(raw);
    if (!v.empty() && v.back() == ';') {
        v.pop_back();
        v = ParamTrim(v);
    }
    if (v.empty()) {
        return false;
    }
    size_t pos = 0;
    float parsed = 0.0f;
    try {
        parsed = std::stof(v, &pos);
    } catch (...) {
        return false;
    }
    if (pos != v.size()) {
        return false;
    }
    out = parsed;
    return true;
}

inline bool ParseConfigDouble(const std::string &raw, double &out) {
    std::string v = ParamTrim(raw);
    if (!v.empty() && v.back() == ';') {
        v.pop_back();
        v = ParamTrim(v);
    }
    if (v.empty()) {
        return false;
    }
    size_t pos = 0;
    double parsed = 0.0;
    try {
        parsed = std::stod(v, &pos);
    } catch (...) {
        return false;
    }
    if (pos != v.size() || !std::isfinite(parsed)) {
        return false;
    }
    out = parsed;
    return true;
}

inline bool ParseConfigDoubleList(const std::string &raw, std::vector<double> &out) {
    std::string v = ParamTrim(raw);
    if (!v.empty() && v.back() == ';') {
        v.pop_back();
        v = ParamTrim(v);
    }
    if (v.size() < 2 || v.front() != '[' || v.back() != ']') {
        return false;
    }

    std::string body = ParamTrim(v.substr(1, v.size() - 2));
    out.clear();
    if (body.empty()) {
        return true;
    }

    size_t begin = 0;
    while (begin <= body.size()) {
        const size_t comma = body.find(',', begin);
        const size_t end = (comma == std::string::npos) ? body.size() : comma;
        const std::string token = ParamTrim(body.substr(begin, end - begin));
        double value = 0.0;
        if (token.empty() || !ParseConfigDouble(token, value)) {
            return false;
        }
        out.push_back(value);

        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    return true;
}

inline bool ParseConfigIntList(const std::string &raw, std::vector<int> &out) {
    std::string v = ParamTrim(raw);
    if (!v.empty() && v.back() == ';') {
        v.pop_back();
        v = ParamTrim(v);
    }
    if (v.size() < 2 || v.front() != '[' || v.back() != ']') {
        return false;
    }

    std::string body = ParamTrim(v.substr(1, v.size() - 2));
    out.clear();
    if (body.empty()) {
        return true;
    }

    size_t begin = 0;
    while (begin <= body.size()) {
        const size_t comma = body.find(',', begin);
        const size_t end = (comma == std::string::npos) ? body.size() : comma;
        const std::string token = ParamTrim(body.substr(begin, end - begin));
        int value = 0;
        if (token.empty() || !ParseConfigInt(token, value)) {
            return false;
        }
        out.push_back(value);

        if (comma == std::string::npos) {
            break;
        }
        begin = comma + 1;
    }
    return true;
}

inline bool ComputeExpandedKFromTopK(int top_k, int &expanded_k) {
    const int safe_top_k = std::max(top_k, 0);
    if (safe_top_k == 0) {
        expanded_k = 0;
        return true;
    }

    const double expanded =
        std::ceil(static_cast<double>(safe_top_k) * k_expand_param);
    if (!std::isfinite(expanded) ||
        expanded > static_cast<double>(std::numeric_limits<int>::max())) {
        return false;
    }

    expanded_k = std::max(safe_top_k, static_cast<int>(expanded));
    return true;
}

inline bool ParseConfigString(const std::string &raw, std::string &out) {
    std::string v = ParamTrim(raw);
    if (!v.empty() && v.back() == ';') {
        v.pop_back();
        v = ParamTrim(v);
    }
    if (v.empty()) {
        return false;
    }

    if (v.size() >= 2) {
        const char first = v.front();
        const char last = v.back();
        if ((first == '"' && last == '"') || (first == '\'' && last == '\'')) {
            out = v.substr(1, v.size() - 2);
            return true;
        }
    }

    out = v;
    return true;
}

inline size_t CeilDivSizeT(size_t a, size_t b) {
    return b == 0 ? 0 : (a + b - 1) / b;
}

inline size_t RoundUpPow2SizeT(size_t v) {
    if (v <= 1) {
        return 1;
    }
    size_t out = 1;
    while (out < v && out <= (std::numeric_limits<size_t>::max() / 2)) {
        out <<= 1;
    }
    return out < v ? v : out;
}

inline int ClampSizeTToInt(size_t v) {
    return (v > static_cast<size_t>(std::numeric_limits<int>::max()))
               ? std::numeric_limits<int>::max()
               : static_cast<int>(v);
}

inline uint64_t MixBucketLayoutHash(uint64_t state, uint64_t value) {
    state ^= value + 0x9e3779b97f4a7c15ULL + (state << 6) + (state >> 2);
    return state;
}

inline uint64_t ComputeBucketDocTableHash(const BucketDocTable &bucket_doc_table) {
    uint64_t hash = 1469598103934665603ULL;
    hash = MixBucketLayoutHash(hash, static_cast<uint64_t>(bucket_doc_table.size()));
    for (const auto &bucket_docs : bucket_doc_table) {
        hash = MixBucketLayoutHash(hash, static_cast<uint64_t>(bucket_docs.size()));
        for (uint32_t doc_id : bucket_docs) {
            hash = MixBucketLayoutHash(hash, static_cast<uint64_t>(doc_id) + 1ULL);
        }
    }
    return hash;
}

inline void EnsurePreallocAtLeast(const char *key, int &actual, size_t required) {
    const int required_int = ClampSizeTToInt(required);
    if (actual < required_int) {
        std::cout << "[Config] Warning: " << key << "=" << actual
                  << " is below estimated minimum " << required_int
                  << "; auto-adjusting to " << required_int << std::endl;
        actual = required_int;
    }
}

inline int GetGroupId(int core_id) {
    return core_id / cores_per_group;
}

inline int GetInGroupRank(int core_id) {
    return core_id % cores_per_group;
}

inline bool IsLeader(int core_id) {
    return GetInGroupRank(core_id) == 0;
}

inline bool FinalizePreallocationParams() {
    const size_t buckets_per_core =
        CeilDivSizeT(static_cast<size_t>(total_bucket_num_level_1), static_cast<size_t>(cores_per_group));
    const size_t ivf_raw_stride = CeilDivSizeT(buckets_per_core, 64);
    const size_t bucket_stride = CeilDivSizeT(static_cast<size_t>(max_doc_per_bucket_level_2), 64);
    const size_t buckets_per_core_in_batch =
        CeilDivSizeT(static_cast<size_t>(max_process_bucket_num_level_2), static_cast<size_t>(cores_per_group));
    const size_t cache_line_u64 =
        std::max<size_t>(1, static_cast<size_t>(cpu_cache_line_size > 0 ? cpu_cache_line_size : 64) / sizeof(uint64_t));
    const size_t ivf_storage_aligned_stride =
        CeilDivSizeT(ivf_raw_stride, cache_line_u64) * cache_line_u64;
    const size_t eval_block_u64 = std::max<size_t>(1, cache_line_u64 * 4);
    // search_ivf/search_bucket 的 scratch 峰值取决于单次 block_len，而不是原始逻辑 stride。
    const size_t ivf_eval_width = std::min(ivf_storage_aligned_stride, eval_block_u64);
    const size_t bucket_eval_width = std::min(bucket_stride, eval_block_u64);
    const size_t max_simd_width = std::max(ivf_eval_width, bucket_eval_width);
    const size_t required_scratch_u64 =
        static_cast<size_t>(max_rpn_length) * max_simd_width + 128;
    const size_t required_ivf_mask_u64 = ivf_storage_aligned_stride + 8;
    const size_t required_batch_mask_u64 = buckets_per_core_in_batch * bucket_stride + 64;
    const size_t required_local_pq_items = buckets_per_core;
    const size_t required_bucket_results_items = buckets_per_core;
    const size_t required_candidate_merge_items = static_cast<size_t>(total_bucket_num_level_1);
    const size_t required_sorted_bucket_items = static_cast<size_t>(total_bucket_num_level_1);
    const size_t required_batch_bucket_ids_items = static_cast<size_t>(max_process_bucket_num_level_2);
    const size_t required_temp_doc_mask_u64 = bucket_stride;
    const size_t required_thread_doc_results_items =
        buckets_per_core_in_batch * static_cast<size_t>(max_doc_per_bucket_level_2);
    const size_t required_all_items =
        std::min(static_cast<size_t>(max_process_bucket_num_level_2) * static_cast<size_t>(max_doc_per_bucket_level_2),
                 static_cast<size_t>(cores_per_group) * static_cast<size_t>(max_query_topk_prealloc));
    const size_t required_merge_batch_items =
        static_cast<size_t>(std::max(max_query_topk_prealloc, 0)) * 2;

    if (query_vector_reserve_floats <= 0) {
        query_vector_reserve_floats = vector_dim;
    }
    if (query_bucket_rpn_reserve_items <= 0) {
        query_bucket_rpn_reserve_items = ClampSizeTToInt(
            std::max(static_cast<size_t>(max_rpn_length), RoundUpPow2SizeT(static_cast<size_t>(max_rpn_length))));
    }
    if (query_bucket_level_ivf_rpn_reserve_items <= 0) {
        query_bucket_level_ivf_rpn_reserve_items = ClampSizeTToInt(
            std::max(static_cast<size_t>(max_rpn_length), RoundUpPow2SizeT(static_cast<size_t>(max_rpn_length))));
    }
    if (query_compile_stack_reserve_items <= 0) {
        query_compile_stack_reserve_items =
            ClampSizeTToInt(RoundUpPow2SizeT(std::max<size_t>(32, static_cast<size_t>(max_query_tags) / 2)));
    }
    if (query_exec_stack_reserve_items <= 0) {
        query_exec_stack_reserve_items =
            ClampSizeTToInt(RoundUpPow2SizeT(std::max<size_t>(32, static_cast<size_t>(max_query_tags))));
    }
    if (query_merge_batch_reserve_items <= 0) {
        query_merge_batch_reserve_items =
            ClampSizeTToInt(RoundUpPow2SizeT(std::max<size_t>(64, required_merge_batch_items)));
    }
    if (group_tls_scratch_pool_reserve_u64 <= 0) {
        group_tls_scratch_pool_reserve_u64 = ClampSizeTToInt(RoundUpPow2SizeT(required_scratch_u64));
    }
    if (group_tls_ivf_mask_reserve_u64 <= 0) {
        group_tls_ivf_mask_reserve_u64 = ClampSizeTToInt(RoundUpPow2SizeT(required_ivf_mask_u64));
    }
    if (group_tls_batch_mask_reserve_u64 <= 0) {
        group_tls_batch_mask_reserve_u64 = ClampSizeTToInt(RoundUpPow2SizeT(required_batch_mask_u64));
    }
    if (group_local_pq_reserve_items_per_rank <= 0) {
        group_local_pq_reserve_items_per_rank =
            ClampSizeTToInt(RoundUpPow2SizeT(required_local_pq_items));
    }
    if (group_thread_bucket_results_reserve_items_per_rank <= 0) {
        group_thread_bucket_results_reserve_items_per_rank =
            ClampSizeTToInt(RoundUpPow2SizeT(required_bucket_results_items));
    }
    if (group_candidate_merge_reserve_items <= 0) {
        group_candidate_merge_reserve_items =
            ClampSizeTToInt(RoundUpPow2SizeT(required_candidate_merge_items));
    }
    if (group_sorted_buckets_reserve_items <= 0) {
        group_sorted_buckets_reserve_items =
            ClampSizeTToInt(RoundUpPow2SizeT(required_sorted_bucket_items));
    }
    if (group_batch_bucket_ids_reserve_items <= 0) {
        group_batch_bucket_ids_reserve_items =
            ClampSizeTToInt(RoundUpPow2SizeT(required_batch_bucket_ids_items));
    }
    if (group_temp_doc_mask_reserve_u64_per_rank <= 0) {
        group_temp_doc_mask_reserve_u64_per_rank = ClampSizeTToInt(
            RoundUpPow2SizeT(std::max<size_t>(required_temp_doc_mask_u64, 32)));
    }
    if (group_thread_doc_results_reserve_items_per_rank <= 0) {
        group_thread_doc_results_reserve_items_per_rank =
            ClampSizeTToInt(RoundUpPow2SizeT(required_thread_doc_results_items));
    }
    if (group_all_items_reserve_items <= 0) {
        group_all_items_reserve_items =
            ClampSizeTToInt(RoundUpPow2SizeT(std::max<size_t>(64, required_all_items)));
    }
    if (query_pool_capacity <= 0) {
        query_pool_capacity = std::max(group_count * 2, 64);
    }

    if (max_query_topk_prealloc <= 0) {
        std::cerr << "[Config] Error: max_query_topk_prealloc must be > 0" << std::endl;
        return false;
    }
    if (query_pool_capacity <= 0) {
        std::cerr << "[Config] Error: query_pool_capacity must be > 0" << std::endl;
        return false;
    }
    EnsurePreallocAtLeast("query_vector_reserve_floats",
                          query_vector_reserve_floats,
                          static_cast<size_t>(vector_dim));
    EnsurePreallocAtLeast("query_bucket_rpn_reserve_items",
                          query_bucket_rpn_reserve_items,
                          static_cast<size_t>(max_rpn_length));
    EnsurePreallocAtLeast("query_bucket_level_ivf_rpn_reserve_items",
                          query_bucket_level_ivf_rpn_reserve_items,
                          static_cast<size_t>(max_rpn_length));
    EnsurePreallocAtLeast("query_merge_batch_reserve_items",
                          query_merge_batch_reserve_items,
                          required_merge_batch_items);
    EnsurePreallocAtLeast("group_tls_scratch_pool_reserve_u64",
                          group_tls_scratch_pool_reserve_u64,
                          required_scratch_u64);
    EnsurePreallocAtLeast("group_tls_ivf_mask_reserve_u64",
                          group_tls_ivf_mask_reserve_u64,
                          required_ivf_mask_u64);
    EnsurePreallocAtLeast("group_tls_batch_mask_reserve_u64",
                          group_tls_batch_mask_reserve_u64,
                          required_batch_mask_u64);
    EnsurePreallocAtLeast("group_local_pq_reserve_items_per_rank",
                          group_local_pq_reserve_items_per_rank,
                          required_local_pq_items);
    EnsurePreallocAtLeast("group_thread_bucket_results_reserve_items_per_rank",
                          group_thread_bucket_results_reserve_items_per_rank,
                          required_bucket_results_items);
    EnsurePreallocAtLeast("group_candidate_merge_reserve_items",
                          group_candidate_merge_reserve_items,
                          required_candidate_merge_items);
    EnsurePreallocAtLeast("group_sorted_buckets_reserve_items",
                          group_sorted_buckets_reserve_items,
                          required_sorted_bucket_items);
    EnsurePreallocAtLeast("group_batch_bucket_ids_reserve_items",
                          group_batch_bucket_ids_reserve_items,
                          required_batch_bucket_ids_items);
    EnsurePreallocAtLeast("group_temp_doc_mask_reserve_u64_per_rank",
                          group_temp_doc_mask_reserve_u64_per_rank,
                          required_temp_doc_mask_u64);
    EnsurePreallocAtLeast("group_thread_doc_results_reserve_items_per_rank",
                          group_thread_doc_results_reserve_items_per_rank,
                          required_thread_doc_results_items);
    EnsurePreallocAtLeast("group_all_items_reserve_items",
                          group_all_items_reserve_items,
                          required_all_items);

    return true;
}

inline bool LoadParams(const std::string &config_file, ResourceConfigProfile profile) {
    g_params_loaded = false;

    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "[Config] Error: Cannot open config file: " << config_file << std::endl;
        return false;
    }

    bool seen_total_bucket_num_level_1 = false;
    bool seen_max_doc_per_bucket_level_2 = false;
    bool seen_cores_per_group = false;
    bool seen_valid_bucket_num_base_level_1 = false;
    bool seen_valid_bucket_num_incremental_level_1 = false;
    bool seen_max_probe_l1_bucket_num_enable = false;
    bool seen_max_probe_l1_bucket_num = false;
    bool seen_max_process_bucket_num_level_2 = false;
    bool seen_k_expand_param = false;
    bool seen_bucket_ivf_index_file = false;
    bool seen_bucket_index_file = false;
    bool seen_npu_device_id_start = false;
    bool seen_max_query_tags = false;
    bool seen_max_rpn_length = false;
    bool seen_cpu_cache_line_size = false;
    bool seen_target_qps_parallel = false;

    int cpu_core_count_serial_value = cpu_core_count;
    int npu_device_count_serial_value = npu_device_count;
    int groups_per_device_serial_value = groups_per_device;
    int cpu_core_count_parallel_value = cpu_core_count;
    int npu_device_count_parallel_value = npu_device_count;
    int groups_per_device_parallel_value = groups_per_device;
    bool seen_cpu_core_count_serial = false;
    bool seen_npu_device_count_serial = false;
    bool seen_groups_per_device_serial = false;
    bool seen_cpu_core_count_parallel = false;
    bool seen_npu_device_count_parallel = false;
    bool seen_groups_per_device_parallel = false;

    std::string line;
    while (std::getline(file, line)) {
        StripConfigComment(line);
        line = ParamTrim(line);
        if (line.empty()) {
            continue;
        }

        const size_t equal_pos = line.find('=');
        if (equal_pos == std::string::npos) {
            continue;
        }

        const std::string key = ParamTrim(line.substr(0, equal_pos));
        const std::string value_str = ParamTrim(line.substr(equal_pos + 1));

        if (key.empty() || value_str.empty()) {
            continue;
        }

        if (key == "dataset_cache_file") {
            std::string value;
            if (!ParseConfigString(value_str, value)) {
                std::cerr << "[Config] Error parsing value for key: " << key
                          << ", raw value: " << value_str << std::endl;
                return false;
            }
            dataset_cache_file = value;
            continue;
        } else if (key == "tag_map_cache_file") {
            std::string value;
            if (!ParseConfigString(value_str, value)) {
                std::cerr << "[Config] Error parsing value for key: " << key
                          << ", raw value: " << value_str << std::endl;
                return false;
            }
            tag_map_cache_file = value;
            continue;
        } else if (key == "raw_vector_file") {
            std::string value;
            if (!ParseConfigString(value_str, value)) {
                std::cerr << "[Config] Error parsing value for key: " << key
                          << ", raw value: " << value_str << std::endl;
                return false;
            }
            raw_vector_file = value;
            continue;
        } else if (key == "raw_attr_dir") {
            std::string value;
            if (!ParseConfigString(value_str, value)) {
                std::cerr << "[Config] Error parsing value for key: " << key
                          << ", raw value: " << value_str << std::endl;
                return false;
            }
            raw_attr_dir = value;
            continue;
        } else if (key == "raw_query_file") {
            std::string value;
            if (!ParseConfigString(value_str, value)) {
                std::cerr << "[Config] Error parsing value for key: " << key
                          << ", raw value: " << value_str << std::endl;
                return false;
            }
            raw_query_file = value;
            continue;
        } else if (key == "raw_query_result_num_key") {
            std::string value;
            if (!ParseConfigString(value_str, value)) {
                std::cerr << "[Config] Error parsing value for key: " << key
                          << ", raw value: " << value_str << std::endl;
                return false;
            }
            raw_query_result_num_key = value;
            continue;
        } else if (key == "raw_query_main_tier_key") {
            std::string value;
            if (!ParseConfigString(value_str, value)) {
                std::cerr << "[Config] Error parsing value for key: " << key
                          << ", raw value: " << value_str << std::endl;
                return false;
            }
            raw_query_main_tier_key = value;
            continue;
        } else if (key == "raw_query_json_query_key") {
            std::string value;
            if (!ParseConfigString(value_str, value)) {
                std::cerr << "[Config] Error parsing value for key: " << key
                          << ", raw value: " << value_str << std::endl;
                return false;
            }
            raw_query_json_query_key = value;
            continue;
        } else if (key == "raw_query_vector_node_key") {
            std::string value;
            if (!ParseConfigString(value_str, value)) {
                std::cerr << "[Config] Error parsing value for key: " << key
                          << ", raw value: " << value_str << std::endl;
                return false;
            }
            raw_query_vector_node_key = value;
            continue;
        } else if (key == "raw_query_vector_value_key") {
            std::string value;
            if (!ParseConfigString(value_str, value)) {
                std::cerr << "[Config] Error parsing value for key: " << key
                          << ", raw value: " << value_str << std::endl;
                return false;
            }
            raw_query_vector_value_key = value;
            continue;
        } else if (key == "raw_query_syntax_filter_key") {
            std::string value;
            if (!ParseConfigString(value_str, value)) {
                std::cerr << "[Config] Error parsing value for key: " << key
                          << ", raw value: " << value_str << std::endl;
                return false;
            }
            raw_query_syntax_filter_key = value;
            continue;
        } else if (key == "query_file") {
            std::string value;
            if (!ParseConfigString(value_str, value)) {
                std::cerr << "[Config] Error parsing value for key: " << key
                          << ", raw value: " << value_str << std::endl;
                return false;
            }
            query_file = value;
            continue;
        } else if (key == "query_result_root") {
            std::string value;
            if (!ParseConfigString(value_str, value)) {
                std::cerr << "[Config] Error parsing value for key: " << key
                          << ", raw value: " << value_str << std::endl;
                return false;
            }
            query_result_root = value;
            continue;
        } else if (key == "ground_truth_cache_file") {
            std::string value;
            if (!ParseConfigString(value_str, value)) {
                std::cerr << "[Config] Error parsing value for key: " << key
                          << ", raw value: " << value_str << std::endl;
                return false;
            }
            ground_truth_cache_file = value;
            continue;
        } else if (key == "bucket_ivf_index_file") {
            std::string value;
            if (!ParseConfigString(value_str, value)) {
                std::cerr << "[Config] Error parsing value for key: " << key
                          << ", raw value: " << value_str << std::endl;
                return false;
            }
            bucket_ivf_index_file = value;
            seen_bucket_ivf_index_file = true;
            continue;
        } else if (key == "bucket_index_file") {
            std::string value;
            if (!ParseConfigString(value_str, value)) {
                std::cerr << "[Config] Error parsing value for key: " << key
                          << ", raw value: " << value_str << std::endl;
                return false;
            }
            bucket_index_file = value;
            seen_bucket_index_file = true;
            continue;
        } else if (key == "npu_debug_verify_abs_tol") {
            float value = 0.0f;
            if (!ParseConfigFloat(value_str, value)) {
                std::cerr << "[Config] Error parsing value for key: " << key
                          << ", raw value: " << value_str << std::endl;
                return false;
            }
            npu_debug_verify_abs_tol = value;
            continue;
        } else if (key == "npu_debug_verify_rel_tol") {
            float value = 0.0f;
            if (!ParseConfigFloat(value_str, value)) {
                std::cerr << "[Config] Error parsing value for key: " << key
                          << ", raw value: " << value_str << std::endl;
                return false;
            }
            npu_debug_verify_rel_tol = value;
            continue;
        } else if (key == "k_expand_param") {
            double value = 0.0;
            if (!ParseConfigDouble(value_str, value)) {
                std::cerr << "[Config] Error parsing value for key: " << key
                          << ", raw value: " << value_str << std::endl;
                return false;
            }
            k_expand_param = value;
            seen_k_expand_param = true;
            continue;
        } else if (key == "target_qps_parallel") {
            double value = 0.0;
            if (!ParseConfigDouble(value_str, value)) {
                std::cerr << "[Config] Error parsing value for key: " << key
                          << ", raw value: " << value_str << std::endl;
                return false;
            }
            target_qps_parallel = value;
            seen_target_qps_parallel = true;
            continue;
        } else if (key == "k_expand_param_test_set") {
            continue;
        } else if (key == "max_probe_l1_bucket_num_test_set") {
            continue;
        } else if (key == "device_group_split") {
            // Parse comma-separated per-device group counts (e.g. "4,2")
            g_device_group_split.clear();
            std::istringstream dgs_ss(value_str);
            std::string dgs_token;
            while (std::getline(dgs_ss, dgs_token, ',')) {
                size_t dgs_start = dgs_token.find_first_not_of(" \t");
                size_t dgs_end = dgs_token.find_last_not_of(" \t");
                if (dgs_start != std::string::npos) {
                    g_device_group_split.push_back(std::stoi(dgs_token.substr(dgs_start, dgs_end - dgs_start + 1)));
                }
            }
            continue;
        }

        int value = 0;
        if (!ParseConfigInt(value_str, value)) {
            std::cerr << "[Config] Error parsing value for key: " << key
                      << ", raw value: " << value_str << std::endl;
            return false;
        }

        if (key == "total_doc_num") {
            std::cout << "[Config] Warning: total_doc_num is ignored; it is read from dataset.bin or raw data.\n";
        } else if (key == "total_tag_num") {
            std::cout << "[Config] Warning: total_tag_num is ignored; it is read from dataset.bin or raw data.\n";
        } else if (key == "total_bucket_num_level_1") {
            total_bucket_num_level_1 = value;
            total_bucket_num_level_2 = value;
            seen_total_bucket_num_level_1 = true;
        } else if (key == "vector_dim") {
            std::cout << "[Config] Warning: vector_dim is ignored; it is read from dataset.bin or raw data.\n";
        } else if (key == "max_doc_per_bucket_level_2") {
            max_doc_per_bucket_level_2 = value;
            seen_max_doc_per_bucket_level_2 = true;
        } else if (key == "cpu_core_count_serial") {
            cpu_core_count_serial_value = value;
            seen_cpu_core_count_serial = true;
        } else if (key == "cpu_core_count_parallel") {
            cpu_core_count_parallel_value = value;
            seen_cpu_core_count_parallel = true;
        } else if (key == "cores_per_group") {
            cores_per_group = value;
            seen_cores_per_group = true;
        } else if (key == "valid_bucket_num_base_level_1") {
            valid_bucket_num_base_level_1 = value;
            seen_valid_bucket_num_base_level_1 = true;
        } else if (key == "valid_bucket_num_incremental_level_1") {
            valid_bucket_num_incremental_level_1 = value;
            seen_valid_bucket_num_incremental_level_1 = true;
        } else if (key == "max_probe_l1_bucket_num_enable") {
            max_probe_l1_bucket_num_enable = value;
            seen_max_probe_l1_bucket_num_enable = true;
        } else if (key == "max_probe_l1_bucket_num") {
            max_probe_l1_bucket_num = value;
            seen_max_probe_l1_bucket_num = true;
        } else if (key == "max_process_bucket_num_level_2") {
            max_process_bucket_num_level_2 = value;
            seen_max_process_bucket_num_level_2 = true;
        } else if (key == "analyze_t1") {
            analyze_t1 = value;
        } else if (key == "npu_device_id_start") {
            npu_device_id_start = value;
            seen_npu_device_id_start = true;
        } else if (key == "npu_device_count_serial") {
            npu_device_count_serial_value = value;
            seen_npu_device_count_serial = true;
        } else if (key == "groups_per_device_serial") {
            groups_per_device_serial_value = value;
            seen_groups_per_device_serial = true;
        } else if (key == "npu_device_count_parallel") {
            npu_device_count_parallel_value = value;
            seen_npu_device_count_parallel = true;
        } else if (key == "groups_per_device_parallel") {
            groups_per_device_parallel_value = value;
            seen_groups_per_device_parallel = true;
        } else if (key == "max_query_tags") {
            max_query_tags = value;
            seen_max_query_tags = true;
            if (!seen_max_rpn_length) {
                max_rpn_length = max_query_tags * 3;
            }
        } else if (key == "max_rpn_length") {
            max_rpn_length = value;
            seen_max_rpn_length = true;
        } else if (key == "max_query_topk_prealloc") {
            max_query_topk_prealloc = value;
        } else if (key == "query_pool_capacity") {
            query_pool_capacity = value;
        } else if (key == "query_vector_reserve_floats") {
            query_vector_reserve_floats = value;
        } else if (key == "query_bucket_rpn_reserve_items") {
            query_bucket_rpn_reserve_items = value;
        } else if (key == "query_bucket_level_ivf_rpn_reserve_items") {
            query_bucket_level_ivf_rpn_reserve_items = value;
        } else if (key == "query_compile_stack_reserve_items") {
            query_compile_stack_reserve_items = value;
        } else if (key == "query_exec_stack_reserve_items") {
            query_exec_stack_reserve_items = value;
        } else if (key == "query_merge_batch_reserve_items") {
            query_merge_batch_reserve_items = value;
        } else if (key == "group_tls_scratch_pool_reserve_u64") {
            group_tls_scratch_pool_reserve_u64 = value;
        } else if (key == "group_tls_ivf_mask_reserve_u64") {
            group_tls_ivf_mask_reserve_u64 = value;
        } else if (key == "group_tls_batch_mask_reserve_u64") {
            group_tls_batch_mask_reserve_u64 = value;
        } else if (key == "group_local_pq_reserve_items_per_rank") {
            group_local_pq_reserve_items_per_rank = value;
        } else if (key == "group_thread_bucket_results_reserve_items_per_rank") {
            group_thread_bucket_results_reserve_items_per_rank = value;
        } else if (key == "group_candidate_merge_reserve_items") {
            group_candidate_merge_reserve_items = value;
        } else if (key == "group_sorted_buckets_reserve_items") {
            group_sorted_buckets_reserve_items = value;
        } else if (key == "group_batch_bucket_ids_reserve_items") {
            group_batch_bucket_ids_reserve_items = value;
        } else if (key == "group_temp_doc_mask_reserve_u64_per_rank") {
            group_temp_doc_mask_reserve_u64_per_rank = value;
        } else if (key == "group_thread_doc_results_reserve_items_per_rank") {
            group_thread_doc_results_reserve_items_per_rank = value;
        } else if (key == "group_all_items_reserve_items") {
            group_all_items_reserve_items = value;
        } else if (key == "cpu_cache_line_size") {
            cpu_cache_line_size = value;
            seen_cpu_cache_line_size = true;
        } else if (key == "memory_events_log_enable") {
            memory_events_log_enable = value;
        } else if (key == "npu_debug_verify") {
            npu_debug_verify = value;
        } else if (key == "npu_debug_verify_max_report") {
            npu_debug_verify_max_report = value;
        } else {
            std::cout << "[Config] Warning: Unknown key: " << key << std::endl;
        }
    }

    std::vector<std::string> missing_keys;
    if (!seen_total_bucket_num_level_1) missing_keys.emplace_back("total_bucket_num_level_1");
    if (!seen_max_doc_per_bucket_level_2) missing_keys.emplace_back("max_doc_per_bucket_level_2");
    if (!seen_cpu_core_count_serial) missing_keys.emplace_back("cpu_core_count_serial");
    if (!seen_npu_device_count_serial) missing_keys.emplace_back("npu_device_count_serial");
    if (!seen_groups_per_device_serial) missing_keys.emplace_back("groups_per_device_serial");
    if (!seen_cpu_core_count_parallel) missing_keys.emplace_back("cpu_core_count_parallel");
    if (!seen_npu_device_count_parallel) missing_keys.emplace_back("npu_device_count_parallel");
    if (!seen_groups_per_device_parallel) missing_keys.emplace_back("groups_per_device_parallel");
    if (!seen_cores_per_group) missing_keys.emplace_back("cores_per_group");
    if (!seen_valid_bucket_num_base_level_1) missing_keys.emplace_back("valid_bucket_num_base_level_1");
    if (!seen_valid_bucket_num_incremental_level_1) missing_keys.emplace_back("valid_bucket_num_incremental_level_1");
    if (!seen_max_probe_l1_bucket_num_enable) missing_keys.emplace_back("max_probe_l1_bucket_num_enable");
    if (!seen_max_probe_l1_bucket_num) missing_keys.emplace_back("max_probe_l1_bucket_num");
    if (!seen_max_process_bucket_num_level_2) missing_keys.emplace_back("max_process_bucket_num_level_2");
    if (!seen_k_expand_param) missing_keys.emplace_back("k_expand_param");
    if (!seen_bucket_ivf_index_file) missing_keys.emplace_back("bucket_ivf_index_file");
    if (!seen_bucket_index_file) missing_keys.emplace_back("bucket_index_file");
    if (!seen_npu_device_id_start) missing_keys.emplace_back("npu_device_id_start");
    if (!seen_max_query_tags) missing_keys.emplace_back("max_query_tags");
    if (!seen_cpu_cache_line_size) missing_keys.emplace_back("cpu_cache_line_size");
    if (!seen_target_qps_parallel) missing_keys.emplace_back("target_qps_parallel");

    if (!missing_keys.empty()) {
        std::cerr << "[Config] Error: Missing required keys:";
        for (const auto &k : missing_keys) {
            std::cerr << " " << k;
        }
        std::cerr << std::endl;
        return false;
    }

    switch (profile) {
        case ResourceConfigProfile::Serial:
            cpu_core_count = cpu_core_count_serial_value;
            npu_device_count = npu_device_count_serial_value;
            groups_per_device = groups_per_device_serial_value;
            break;
        case ResourceConfigProfile::Parallel:
            cpu_core_count = cpu_core_count_parallel_value;
            npu_device_count = npu_device_count_parallel_value;
            groups_per_device = groups_per_device_parallel_value;
            break;
    }

    if (cpu_core_count <= 0 || cores_per_group <= 0) {
        std::cerr << "[Config] Error: cpu_core_count_" << ResourceConfigProfileName(profile)
                  << " and cores_per_group must be > 0" << std::endl;
        return false;
    }
    if (npu_device_count <= 0 || groups_per_device <= 0) {
        std::cerr << "[Config] Error: npu_device_count_" << ResourceConfigProfileName(profile)
                  << " and groups_per_device_" << ResourceConfigProfileName(profile)
                  << " must be > 0" << std::endl;
        return false;
    }
    if (total_bucket_num_level_1 <= 0) {
        std::cerr << "[Config] Error: total_bucket_num_level_1 must be > 0" << std::endl;
        return false;
    }
    if (max_doc_per_bucket_level_2 <= 0 || (max_doc_per_bucket_level_2 % 16) != 0) {
        std::cerr << "[Config] Error: max_doc_per_bucket_level_2 must be > 0 and divisible by 16" << std::endl;
        return false;
    }
    if (valid_bucket_num_base_level_1 <= 0 || valid_bucket_num_incremental_level_1 <= 0) {
        std::cerr << "[Config] Error: valid_bucket_num_base_level_1 and valid_bucket_num_incremental_level_1 must be > 0" << std::endl;
        return false;
    }
    if (max_probe_l1_bucket_num_enable != 0 && max_probe_l1_bucket_num_enable != 1) {
        std::cerr << "[Config] Error: max_probe_l1_bucket_num_enable must be 0 or 1" << std::endl;
        return false;
    }
    if (max_probe_l1_bucket_num <= 0) {
        std::cerr << "[Config] Error: max_probe_l1_bucket_num must be > 0" << std::endl;
        return false;
    }
    if (max_process_bucket_num_level_2 <= 0) {
        std::cerr << "[Config] Error: max_process_bucket_num_level_2 must be > 0" << std::endl;
        return false;
    }
    if (!std::isfinite(k_expand_param) || k_expand_param <= 0.0) {
        std::cerr << "[Config] Error: k_expand_param must be a finite value > 0" << std::endl;
        return false;
    }
    if (analyze_t1 <= 0) {
        std::cerr << "[Config] Error: analyze_t1 must be > 0" << std::endl;
        return false;
    }
    if (valid_bucket_num_incremental_level_1 > valid_bucket_num_base_level_1) {
        std::cerr << "[Config] Error: valid_bucket_num_incremental_level_1 must be <= valid_bucket_num_base_level_1" << std::endl;
        return false;
    }

    group_count = cpu_core_count / cores_per_group;
    if (group_count <= 0) {
        std::cerr << "[Config] Error: invalid group_count=" << group_count << std::endl;
        return false;
    }
    if ((cpu_core_count % cores_per_group) != 0) {
        std::cout << "[Config] Warning: cpu_core_count is not divisible by cores_per_group, tail cores are ignored" << std::endl;
    }

    // Validate device_group_split: if provided, its sum must equal group_count.
    // If not provided, fall back to symmetric groups_per_device.
    if (!g_device_group_split.empty())
    {
        if (static_cast<int>(g_device_group_split.size()) != npu_device_count) {
            std::cerr << "[Config] Error: device_group_split has "
                      << g_device_group_split.size() << " entries but npu_device_count="
                      << npu_device_count << std::endl;
            return false;
        }
        int split_sum = 0;
        for (int v : g_device_group_split) split_sum += v;
        if (split_sum != group_count) {
            std::cerr << "[Config] Error: device_group_split sum=" << split_sum
                      << " does not match group_count=" << group_count << std::endl;
            return false;
        }
        // Override groups_per_device with the maximum entry for stream allocation
        groups_per_device = *std::max_element(g_device_group_split.begin(), g_device_group_split.end());
    }
    else
    {
        const int expected_groups = npu_device_count * groups_per_device;
        if (group_count != expected_groups) {
            std::cerr << "[Config] Error: group_count(cpu_core_count/cores_per_group)=" << group_count
                      << " does not match npu_device_count_" << ResourceConfigProfileName(profile)
                      << "*groups_per_device_" << ResourceConfigProfileName(profile)
                      << "=" << expected_groups << std::endl;
            return false;
        }
    }

    // Build the group-to-device mapping tables.
    BuildDeviceGroupMapping();
    if (!g_device_group_split.empty()) {
        std::cout << "[Config] Asymmetric device_group_split:";
        for (int d = 0; d < npu_device_count; ++d) {
            std::cout << " dev" << d << "=" << g_device_group_split[d] << " groups";
        }
        std::cout << std::endl;
    }

    if (!seen_max_rpn_length) {
        max_rpn_length = max_query_tags * 3;
    }
    if (max_query_tags <= 0 || max_rpn_length <= 0) {
        std::cerr << "[Config] Error: max_query_tags and max_rpn_length must be > 0" << std::endl;
        return false;
    }
    if (cpu_cache_line_size <= 0 || (cpu_cache_line_size % 8) != 0) {
        std::cerr << "[Config] Error: cpu_cache_line_size must be > 0 and divisible by 8" << std::endl;
        return false;
    }
    if (memory_events_log_enable != 0 && memory_events_log_enable != 1) {
        std::cerr << "[Config] Error: memory_events_log_enable must be 0 or 1" << std::endl;
        return false;
    }
    if (npu_debug_verify_max_report < 0) {
        std::cerr << "[Config] Error: npu_debug_verify_max_report must be >= 0" << std::endl;
        return false;
    }
    if (npu_debug_verify_abs_tol < 0.0f || npu_debug_verify_rel_tol < 0.0f) {
        std::cerr << "[Config] Error: npu_debug_verify_abs_tol and npu_debug_verify_rel_tol must be >= 0" << std::endl;
        return false;
    }

    g_params_loaded = true;
    g_runtime_params_finalized = false;
    return true;
}

inline bool LoadKExpandParamTestSet(const std::string &config_file) {
    k_expand_param_test_set.clear();

    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "[Config] Error: Cannot open config file: " << config_file << std::endl;
        return false;
    }

    bool seen_k_expand_param_test_set = false;
    std::string line;
    while (std::getline(file, line)) {
        StripConfigComment(line);
        line = ParamTrim(line);
        if (line.empty()) {
            continue;
        }

        const size_t equal_pos = line.find('=');
        if (equal_pos == std::string::npos) {
            continue;
        }

        const std::string key = ParamTrim(line.substr(0, equal_pos));
        const std::string value_str = ParamTrim(line.substr(equal_pos + 1));
        if (key != "k_expand_param_test_set") {
            continue;
        }

        if (!ParseConfigDoubleList(value_str, k_expand_param_test_set)) {
            std::cerr << "[Config] Error parsing value for key: " << key
                      << ", raw value: " << value_str << std::endl;
            return false;
        }
        seen_k_expand_param_test_set = true;
    }

    if (!seen_k_expand_param_test_set || k_expand_param_test_set.empty()) {
        std::cerr << "[Config] Error: k_expand_param_test_set must contain at least one value." << std::endl;
        return false;
    }

    for (double test_value : k_expand_param_test_set) {
        if (!std::isfinite(test_value) || test_value <= 0.0) {
            std::cerr << "[Config] Error: k_expand_param_test_set values must be finite and > 0" << std::endl;
            return false;
        }
    }
    return true;
}

inline bool LoadMaxProbeL1BucketNumTestSet(const std::string &config_file) {
    max_probe_l1_bucket_num_test_set.clear();

    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "[Config] Error: Cannot open config file: " << config_file << std::endl;
        return false;
    }

    bool seen_max_probe_l1_bucket_num_test_set = false;
    std::string line;
    while (std::getline(file, line)) {
        StripConfigComment(line);
        line = ParamTrim(line);
        if (line.empty()) {
            continue;
        }

        const size_t equal_pos = line.find('=');
        if (equal_pos == std::string::npos) {
            continue;
        }

        const std::string key = ParamTrim(line.substr(0, equal_pos));
        const std::string value_str = ParamTrim(line.substr(equal_pos + 1));
        if (key != "max_probe_l1_bucket_num_test_set") {
            continue;
        }

        if (!ParseConfigIntList(value_str, max_probe_l1_bucket_num_test_set)) {
            std::cerr << "[Config] Error parsing value for key: " << key
                      << ", raw value: " << value_str << std::endl;
            return false;
        }
        seen_max_probe_l1_bucket_num_test_set = true;
    }

    if (!seen_max_probe_l1_bucket_num_test_set || max_probe_l1_bucket_num_test_set.empty()) {
        std::cerr << "[Config] Error: max_probe_l1_bucket_num_test_set must contain at least one value." << std::endl;
        return false;
    }

    for (int test_value : max_probe_l1_bucket_num_test_set) {
        if (test_value <= 0) {
            std::cerr << "[Config] Error: max_probe_l1_bucket_num_test_set values must be > 0" << std::endl;
            return false;
        }
    }
    return true;
}

inline bool LoadParams(const std::string &config_file) {
    return LoadParams(config_file, ResourceConfigProfile::Serial);
}

inline bool FinalizeRuntimeParamsAfterDatasetLoad() {
    if (total_doc_num <= 0 || total_tag_num <= 0 || vector_dim <= 0) {
        std::cerr << "[Config] Error: dataset-derived total_doc_num/total_tag_num/vector_dim must be > 0"
                  << std::endl;
        return false;
    }
    if ((vector_dim % kSupportedVectorDimMultiple) != 0 || vector_dim > kMaxSupportedVectorDim) {
        std::cerr << "[Config] Error: dataset vector_dim must be divisible by "
                  << kSupportedVectorDimMultiple
                  << " and <= " << kMaxSupportedVectorDim << std::endl;
        return false;
    }
    if (!FinalizePreallocationParams()) {
        return false;
    }
    g_runtime_params_finalized = true;
    std::cout << "[Config] Runtime data parameters: total_doc_num=" << total_doc_num
              << ", total_tag_num=" << total_tag_num
              << ", vector_dim=" << vector_dim << std::endl;
    return true;
}

namespace DataReader {
namespace fs = std::filesystem;
using json = nlohmann::json;

enum class LoadFailureReason {
    None = 0,
    MissingFile,
    OpenFailed,
    InvalidFormat,
    HeaderMismatch,
    VectorDimMismatch,
};

inline void PrintVectorDimMismatchWarning(const std::string &source,
                                          int actual_dim,
                                          int expected_dim,
                                          size_t line_no = 0) {
    std::cerr << "要求的向量维度和数据向量不匹配" << std::endl;
    std::cerr << "[Warn] " << source;
    if (line_no > 0) {
        std::cerr << " 第" << line_no << "行";
    }
    std::cerr << " 的向量维度为 " << actual_dim
              << "，运行时 dataset vector_dim 为 " << expected_dim << std::endl;
}

struct BucketIvfIndexFileHeaderDisk {
    char magic[8];
    uint32_t version = 0;
    uint32_t total_tag_num = 0;
    uint32_t total_bucket_num = 0;
    uint32_t cores_per_group = 0;
    uint32_t cpu_cache_line_size = 0;
    uint32_t buckets_per_core = 0;
    uint32_t aligned_stride = 0;
    uint64_t layout_hash = 0;
};

struct BucketIndexFileHeaderDisk {
    char magic[8];
    uint32_t version = 0;
    uint32_t total_doc_num = 0;
    uint32_t total_tag_num = 0;
    uint32_t total_bucket_num = 0;
    uint32_t vector_dim = 0;
    uint32_t max_doc_per_bucket = 0;
    uint64_t layout_hash = 0;
};

struct BucketIndexEntryDisk {
    uint64_t offset = 0;
    uint64_t size = 0;
};

inline bool ReadBinaryExact(std::ifstream &in, void *buffer, size_t bytes) {
    if (bytes == 0) {
        return true;
    }

    in.read(reinterpret_cast<char *>(buffer), static_cast<std::streamsize>(bytes));
    return in.good();
}

inline bool WriteBinaryExact(std::ostream &out, const void *buffer, size_t bytes) {
    if (bytes == 0) {
        return true;
    }

    out.write(reinterpret_cast<const char *>(buffer), static_cast<std::streamsize>(bytes));
    return out.good();
}

template <typename T, typename Alloc>
inline bool WriteBinaryVector(std::ostream &out, const std::vector<T, Alloc> &values) {
    const uint64_t count = static_cast<uint64_t>(values.size());
    return WriteBinaryExact(out, &count, sizeof(count)) &&
           WriteBinaryExact(out, values.data(), static_cast<size_t>(count) * sizeof(T));
}

template <typename T, typename Alloc>
inline bool ReadBinaryVector(std::ifstream &in, std::vector<T, Alloc> &values) {
    uint64_t count = 0;
    if (!ReadBinaryExact(in, &count, sizeof(count))) {
        return false;
    }

    if (count > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(T))) {
        return false;
    }

    values.resize(static_cast<size_t>(count));
    return ReadBinaryExact(in, values.data(), static_cast<size_t>(count) * sizeof(T));
}

inline bool EnsureBinaryParentDirectory(const fs::path &path) {
    std::error_code ec;
    const fs::path parent = path.parent_path();
    if (parent.empty()) {
        return true;
    }

    fs::create_directories(parent, ec);
    return !ec;
}

inline fs::path BuildBinaryTempPath(const fs::path &path) {
    fs::path temp_path = path;
    temp_path += ".tmp";
    return temp_path;
}

inline bool ReplaceBinaryFileAtomically(const fs::path &temp_path, const fs::path &target_path) {
    std::error_code ec;
    fs::rename(temp_path, target_path, ec);
    if (!ec) {
        return true;
    }

    ec.clear();
    fs::remove(target_path, ec);
    ec.clear();
    fs::rename(temp_path, target_path, ec);
    if (!ec) {
        return true;
    }

    ec.clear();
    fs::remove(temp_path, ec);
    return false;
}

struct PreparedQuery {
    std::vector<float> query_vec;
    std::string filter_expr;
    int top_k = 0;
    size_t line_no = 0;
};

struct QueryLoadDiagnostics {
    std::vector<size_t> missing_syntax_filter_queries;
    std::vector<size_t> invalid_syntax_filter_queries;
    std::vector<size_t> missing_vector_queries;

    void Clear() {
        missing_syntax_filter_queries.clear();
        invalid_syntax_filter_queries.clear();
        missing_vector_queries.clear();
    }
};

enum class QueryParseIssue {
    None = 0,
    MissingVector,
    InvalidSyntaxFilter,
    VectorDimMismatch,
};

struct ExprResult {
    bool is_valid = true;
    bool is_const = false;
    bool val = false;
    std::string expr;

    static ExprResult Const(bool v) {
        ExprResult r;
        r.is_valid = true;
        r.is_const = true;
        r.val = v;
        return r;
    }

    static ExprResult Expr(const std::string &e) {
        ExprResult r;
        r.is_valid = true;
        r.is_const = false;
        r.expr = e;
        return r;
    }

    static ExprResult Invalid() {
        ExprResult r;
        r.is_valid = false;
        return r;
    }
};

struct DatasetBuffers {
    std::vector<float> vectors;
    std::vector<uint64_t> bitmaps;
    // Runtime-only: filled by clustering cache or clustering execution, not stored in dataset.bin.
    std::vector<float> centroids;
};

struct GroundTruthResultItem {
    uint32_t doc_id = 0;
    float score = 0.0f;
};

class GroundTruthCache
{
public:
    explicit GroundTruthCache(fs::path cache_path)
        : cache_path_(std::move(cache_path))
    {
    }

    bool Load()
    {
        entries_.clear();

        if (cache_path_.empty()) {
            return true;
        }

        if (!fs::exists(cache_path_)) {
            return true;
        }

        std::ifstream in(cache_path_, std::ios::binary);
        if (!in.is_open()) {
            std::cerr << "[GroundTruthCache] Failed to open cache file: " << cache_path_ << std::endl;
            return false;
        }

        in.seekg(0, std::ios::end);
        const std::streamoff file_size = in.tellg();
        in.seekg(0, std::ios::beg);
        if (file_size <= 0) {
            return true;
        }

        FileHeaderDisk header{};
        if (!ReadExact(in, &header, sizeof(header))) {
            std::cerr << "[GroundTruthCache] Failed to read cache header: " << cache_path_ << std::endl;
            return false;
        }
        if (!IsValidHeader(header)) {
            std::cerr << "[GroundTruthCache] Invalid cache header: " << cache_path_ << std::endl;
            return false;
        }

        while (true) {
            EntryHeaderDisk entry_header{};
            if (!in.read(reinterpret_cast<char *>(&entry_header), sizeof(entry_header))) {
                if (in.eof()) {
                    break;
                }
                std::cerr << "[GroundTruthCache] Failed to read cache entry header: "
                          << cache_path_ << std::endl;
                return false;
            }

            if (entry_header.vector_dim == 0) {
                std::cerr << "[GroundTruthCache] Invalid vector_dim=0 in cache entry." << std::endl;
                return false;
            }
            if (entry_header.filter_size > kMaxFilterSize) {
                std::cerr << "[GroundTruthCache] filter_size exceeds limit in cache entry." << std::endl;
                return false;
            }
            if (entry_header.result_count > kMaxResultCount) {
                std::cerr << "[GroundTruthCache] result_count exceeds limit in cache entry." << std::endl;
                return false;
            }

            std::vector<float> query_vector(entry_header.vector_dim);
            if (!ReadExact(in,
                           query_vector.data(),
                           static_cast<size_t>(entry_header.vector_dim) * sizeof(float))) {
                std::cerr << "[GroundTruthCache] Failed to read query vector from cache entry."
                          << std::endl;
                return false;
            }

            std::string filter_expr(entry_header.filter_size, '\0');
            if (entry_header.filter_size > 0 &&
                !ReadExact(in, filter_expr.data(), static_cast<size_t>(entry_header.filter_size))) {
                std::cerr << "[GroundTruthCache] Failed to read filter expression from cache entry."
                          << std::endl;
                return false;
            }

            std::vector<ResultItemDisk> disk_items(entry_header.result_count);
            if (entry_header.result_count > 0 &&
                !ReadExact(in,
                           disk_items.data(),
                           static_cast<size_t>(entry_header.result_count) * sizeof(ResultItemDisk))) {
                std::cerr << "[GroundTruthCache] Failed to read result items from cache entry."
                          << std::endl;
                return false;
            }

            std::vector<GroundTruthResultItem> results;
            results.reserve(entry_header.result_count);
            for (const auto &disk_item : disk_items) {
                results.push_back({disk_item.doc_id, disk_item.score});
            }

            entries_[BuildKey(query_vector, filter_expr, entry_header.top_k)] = std::move(results);
        }

        return true;
    }

    bool Lookup(const std::vector<float> &query_vector,
                const std::string &filter_expr,
                int top_k,
                std::vector<GroundTruthResultItem> &results) const
    {
        const auto it = entries_.find(BuildKey(query_vector, filter_expr, top_k));
        if (it == entries_.end()) {
            return false;
        }

        results = it->second;
        return true;
    }

    bool Store(const std::vector<float> &query_vector,
               const std::string &filter_expr,
               int top_k,
               const std::vector<GroundTruthResultItem> &results)
    {
        std::string key = BuildKey(query_vector, filter_expr, top_k);
        const auto found = entries_.find(key);
        if (found != entries_.end()) {
            return true;
        }

        if (!AppendEntry(query_vector, filter_expr, top_k, results)) {
            return false;
        }

        entries_.emplace(std::move(key), results);
        return true;
    }

    size_t EntryCount() const
    {
        return entries_.size();
    }

    const fs::path &path() const
    {
        return cache_path_;
    }

private:
    struct FileHeaderDisk
    {
        char magic[8];
        uint32_t version;
        uint32_t reserved;
    };

    struct EntryHeaderDisk
    {
        uint32_t vector_dim;
        int32_t top_k;
        uint32_t filter_size;
        uint32_t result_count;
    };

    struct ResultItemDisk
    {
        uint32_t doc_id;
        float score;
    };

    static constexpr uint32_t kCacheVersion = 1;
    static constexpr uint32_t kMaxFilterSize = 1u << 20;
    static constexpr uint32_t kMaxResultCount = 1u << 20;

    static FileHeaderDisk MakeHeader()
    {
        FileHeaderDisk header{};
        header.magic[0] = 'G';
        header.magic[1] = 'T';
        header.magic[2] = 'C';
        header.magic[3] = 'A';
        header.magic[4] = 'C';
        header.magic[5] = 'H';
        header.magic[6] = 'E';
        header.magic[7] = '\0';
        header.version = kCacheVersion;
        header.reserved = 0;
        return header;
    }

    static bool IsValidHeader(const FileHeaderDisk &header)
    {
        const FileHeaderDisk expected = MakeHeader();
        return std::memcmp(header.magic, expected.magic, sizeof(header.magic)) == 0 &&
               header.version == expected.version;
    }

    static bool ReadExact(std::ifstream &in, void *buffer, size_t bytes)
    {
        if (bytes == 0) {
            return true;
        }

        in.read(reinterpret_cast<char *>(buffer), static_cast<std::streamsize>(bytes));
        return in.good();
    }

    static bool WriteExact(std::ofstream &out, const void *buffer, size_t bytes)
    {
        if (bytes == 0) {
            return true;
        }

        out.write(reinterpret_cast<const char *>(buffer), static_cast<std::streamsize>(bytes));
        return out.good();
    }

    static void AppendBytes(std::string &buffer, const void *data, size_t bytes)
    {
        buffer.append(reinterpret_cast<const char *>(data), bytes);
    }

    static std::string BuildKey(const std::vector<float> &query_vector,
                                const std::string &filter_expr,
                                int top_k)
    {
        const uint32_t vector_dim = static_cast<uint32_t>(query_vector.size());
        const uint32_t filter_size = static_cast<uint32_t>(filter_expr.size());

        std::string key;
        key.reserve(sizeof(vector_dim) + sizeof(top_k) + sizeof(filter_size) +
                    static_cast<size_t>(vector_dim) * sizeof(float) + filter_expr.size());
        AppendBytes(key, &vector_dim, sizeof(vector_dim));
        AppendBytes(key, &top_k, sizeof(top_k));
        AppendBytes(key, &filter_size, sizeof(filter_size));
        AppendBytes(key, query_vector.data(), static_cast<size_t>(vector_dim) * sizeof(float));
        key.append(filter_expr);
        return key;
    }

    bool EnsureCacheFileExists()
    {
        if (cache_path_.empty()) {
            return false;
        }

        std::error_code ec;
        const auto parent = cache_path_.parent_path();
        if (!parent.empty()) {
            fs::create_directories(parent, ec);
            if (ec) {
                std::cerr << "[GroundTruthCache] Failed to create parent directory for "
                          << cache_path_ << ", error=" << ec.message() << std::endl;
                return false;
            }
        }

        if (fs::exists(cache_path_)) {
            const auto size = fs::file_size(cache_path_, ec);
            if (!ec && size > 0) {
                return true;
            }
        }

        std::ofstream out(cache_path_, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            std::cerr << "[GroundTruthCache] Failed to create cache file: " << cache_path_ << std::endl;
            return false;
        }

        const FileHeaderDisk header = MakeHeader();
        if (!WriteExact(out, &header, sizeof(header))) {
            std::cerr << "[GroundTruthCache] Failed to write cache header: " << cache_path_
                      << std::endl;
            return false;
        }

        return true;
    }

    bool AppendEntry(const std::vector<float> &query_vector,
                     const std::string &filter_expr,
                     int top_k,
                     const std::vector<GroundTruthResultItem> &results)
    {
        if (!EnsureCacheFileExists()) {
            return false;
        }

        std::ofstream out(cache_path_, std::ios::binary | std::ios::app);
        if (!out.is_open()) {
            std::cerr << "[GroundTruthCache] Failed to append cache file: " << cache_path_
                      << std::endl;
            return false;
        }

        EntryHeaderDisk entry_header{};
        entry_header.vector_dim = static_cast<uint32_t>(query_vector.size());
        entry_header.top_k = top_k;
        entry_header.filter_size = static_cast<uint32_t>(filter_expr.size());
        entry_header.result_count = static_cast<uint32_t>(results.size());

        if (!WriteExact(out, &entry_header, sizeof(entry_header)) ||
            !WriteExact(out,
                        query_vector.data(),
                        static_cast<size_t>(entry_header.vector_dim) * sizeof(float)) ||
            !WriteExact(out, filter_expr.data(), filter_expr.size())) {
            std::cerr << "[GroundTruthCache] Failed to append cache entry payload: "
                      << cache_path_ << std::endl;
            return false;
        }

        std::vector<ResultItemDisk> disk_items;
        disk_items.reserve(results.size());
        for (const auto &item : results) {
            disk_items.push_back({item.doc_id, item.score});
        }

        if (!WriteExact(out, disk_items.data(), disk_items.size() * sizeof(ResultItemDisk))) {
            std::cerr << "[GroundTruthCache] Failed to append cache entry results: "
                      << cache_path_ << std::endl;
            return false;
        }

        out.flush();
        return out.good();
    }

    fs::path cache_path_;
    std::unordered_map<std::string, std::vector<GroundTruthResultItem>> entries_;
};

inline std::string Trim(const std::string &s) {
    size_t b = 0;
    while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) {
        ++b;
    }
    size_t e = s.size();
    while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) {
        --e;
    }
    return s.substr(b, e - b);
}

inline std::string ToLowerCopy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

inline bool IsLogicalOp(const std::string &op) {
    std::string low = ToLowerCopy(op);
    return low == "and" || low == "or";
}

inline std::string JsonValueToString(const json &v) {
    if (v.is_string()) {
        return v.get<std::string>();
    }
    if (v.is_number_integer()) {
        return std::to_string(v.get<long long>());
    }
    if (v.is_number_unsigned()) {
        return std::to_string(v.get<unsigned long long>());
    }
    if (v.is_number_float()) {
        std::ostringstream oss;
        oss << v.get<double>();
        return oss.str();
    }
    if (v.is_boolean()) {
        return v.get<bool>() ? "true" : "false";
    }
    return "";
}

inline ExprResult CombineExprs(const std::vector<ExprResult> &exprs, const std::string &op_type) {
    const std::string low = ToLowerCopy(op_type);
    if (low != "and" && low != "or") {
        return ExprResult::Invalid();
    }
    const bool is_and = (low == "and");
    std::vector<std::string> valid_exprs;

    for (const auto &e : exprs) {
        if (!e.is_valid) {
            return ExprResult::Invalid();
        }
        if (e.is_const) {
            if (is_and && !e.val) {
                return ExprResult::Const(false);
            }
            if (!is_and && e.val) {
                return ExprResult::Const(true);
            }
            continue;
        }
        valid_exprs.push_back(e.expr);
    }

    if (valid_exprs.empty()) {
        return ExprResult::Const(is_and ? true : false);
    }
    if (valid_exprs.size() == 1) {
        return ExprResult::Expr(valid_exprs.front());
    }

    std::string out = "(" + valid_exprs[0];
    const std::string joiner = is_and ? " AND " : " OR ";
    for (size_t i = 1; i < valid_exprs.size(); ++i) {
        out += joiner;
        out += valid_exprs[i];
    }
    out += ")";
    return ExprResult::Expr(out);
}

inline ExprResult ParseSyntaxFilterToExpr(const json &node, const std::unordered_map<std::string, int> &tag_map) {
    if (!node.is_object()) {
        return ExprResult::Invalid();
    }

    std::vector<std::string> keys;
    keys.reserve(node.size());
    for (auto it = node.begin(); it != node.end(); ++it) {
        keys.push_back(it.key());
    }

    if (keys.size() != 1) {
        return ExprResult::Invalid();
    }

    const std::string &k = keys[0];

    if (k == "term") {
        const auto &term_node = node["term"];
        if (!term_node.is_object() || term_node.empty()) {
            return ExprResult::Invalid();
        }

        std::vector<ExprResult> term_exprs;
        term_exprs.reserve(term_node.size());
        for (auto it = term_node.begin(); it != term_node.end(); ++it) {
            const std::string tag = it.key() + "#" + JsonValueToString(it.value());
            if (tag.back() == '#') {
                return ExprResult::Invalid();
            }
            auto m = tag_map.find(tag);
            if (m == tag_map.end()) {
                term_exprs.push_back(ExprResult::Const(false));
            } else {
                term_exprs.push_back(ExprResult::Expr(std::to_string(m->second)));
            }
        }
        return CombineExprs(term_exprs, "and");
    }

    if (k == "terms") {
        const auto &terms_node = node["terms"];
        if (!terms_node.is_object()) {
            return ExprResult::Invalid();
        }

        std::string join_type = "or";
        std::string inner_join = "or";

        if (terms_node.contains("join_type")) {
            if (!terms_node["join_type"].is_string()) {
                return ExprResult::Invalid();
            }
            join_type = terms_node["join_type"].get<std::string>();
        }

        if (terms_node.contains("inner_section_join_type")) {
            if (!terms_node["inner_section_join_type"].is_string()) {
                return ExprResult::Invalid();
            }
            inner_join = terms_node["inner_section_join_type"].get<std::string>();
        }

        if (!IsLogicalOp(join_type) || !IsLogicalOp(inner_join)) {
            return ExprResult::Invalid();
        }

        std::vector<ExprResult> section_exprs;

        for (auto it = terms_node.begin(); it != terms_node.end(); ++it) {
            const std::string sec = it.key();
            if (sec == "join_type" || sec == "inner_section_join_type") {
                continue;
            }

            const json &vals = it.value();
            std::vector<ExprResult> inner_exprs;

            if (vals.is_array()) {
                for (const auto &v : vals) {
                    const std::string val_str = JsonValueToString(v);
                    if (val_str.empty()) {
                        return ExprResult::Invalid();
                    }
                    const std::string tag = sec + "#" + val_str;
                    auto m = tag_map.find(tag);
                    if (m == tag_map.end()) {
                        inner_exprs.push_back(ExprResult::Const(false));
                    } else {
                        inner_exprs.push_back(ExprResult::Expr(std::to_string(m->second)));
                    }
                }
            } else {
                const std::string val_str = JsonValueToString(vals);
                if (val_str.empty()) {
                    return ExprResult::Invalid();
                }
                const std::string tag = sec + "#" + val_str;
                auto m = tag_map.find(tag);
                if (m == tag_map.end()) {
                    inner_exprs.push_back(ExprResult::Const(false));
                } else {
                    inner_exprs.push_back(ExprResult::Expr(std::to_string(m->second)));
                }
            }

            if (inner_exprs.empty()) {
                return ExprResult::Invalid();
            }

            const ExprResult section_expr = CombineExprs(inner_exprs, inner_join);
            if (!section_expr.is_valid) {
                return ExprResult::Invalid();
            }
            section_exprs.push_back(section_expr);
        }

        if (section_exprs.empty()) {
            return ExprResult::Invalid();
        }

        return CombineExprs(section_exprs, join_type);
    }

    if (k == "and" || k == "or") {
        const auto &arr = node[k];
        if (!arr.is_array()) {
            return ExprResult::Invalid();
        }

        std::vector<ExprResult> sub_exprs;
        sub_exprs.reserve(arr.size());
        for (const auto &sub : arr) {
            const ExprResult parsed = ParseSyntaxFilterToExpr(sub, tag_map);
            if (!parsed.is_valid) {
                return ExprResult::Invalid();
            }
            sub_exprs.push_back(parsed);
        }
        return CombineExprs(sub_exprs, k);
    }

    if (k == "not") {
        const ExprResult sub = ParseSyntaxFilterToExpr(node["not"], tag_map);
        if (!sub.is_valid) {
            return ExprResult::Invalid();
        }
        if (sub.is_const) {
            return ExprResult::Const(!sub.val);
        }
        return ExprResult::Expr("(NOT " + sub.expr + ")");
    }

    return ExprResult::Invalid();
}

inline std::string FinalizeFilterExpr(const ExprResult &r) {
    if (r.is_const) {
        return r.val ? "" : "0 AND (NOT 0)";
    }
    return r.expr;
}

struct DatasetCacheHeaderDisk {
    char magic[8];
    uint32_t version = 0;
    uint64_t doc_num = 0;
    uint32_t vector_dim = 0;
    uint32_t tag_num = 0;
    uint32_t reserved = 0;
};

inline DatasetCacheHeaderDisk MakeDatasetCacheHeader(uint64_t doc_num,
                                                     uint32_t vec_dim,
                                                     uint32_t tag_num) {
    DatasetCacheHeaderDisk header{};
    header.magic[0] = 'H';
    header.magic[1] = 'Y';
    header.magic[2] = 'D';
    header.magic[3] = 'S';
    header.magic[4] = 'E';
    header.magic[5] = 'T';
    header.magic[6] = '2';
    header.magic[7] = '\0';
    header.version = 2;
    header.doc_num = doc_num;
    header.vector_dim = vec_dim;
    header.tag_num = tag_num;
    header.reserved = 0;
    return header;
}

inline bool IsValidDatasetCacheHeader(const DatasetCacheHeaderDisk &header) {
    const DatasetCacheHeaderDisk expected = MakeDatasetCacheHeader(1, 1, 1);
    return std::memcmp(header.magic, expected.magic, sizeof(header.magic)) == 0 &&
           header.version == 2 &&
           header.doc_num > 0 &&
           header.doc_num <= static_cast<uint64_t>(std::numeric_limits<int>::max()) &&
           header.vector_dim > 0 &&
           header.vector_dim <= static_cast<uint32_t>(std::numeric_limits<int>::max()) &&
           header.tag_num > 0 &&
           header.tag_num <= static_cast<uint32_t>(std::numeric_limits<int>::max());
}

inline bool ReadDatasetCacheHeader(std::ifstream &in, DatasetCacheHeaderDisk &header) {
    if (!ReadBinaryExact(in, &header, sizeof(header))) {
        return false;
    }
    return IsValidDatasetCacheHeader(header);
}

inline bool SaveDatasetCache(const std::string &path,
                             const DatasetBuffers &buffers,
                             LoadFailureReason *failure_reason = nullptr) {
    if (failure_reason != nullptr) {
        *failure_reason = LoadFailureReason::None;
    }
    if (total_doc_num <= 0 || vector_dim <= 0 || total_tag_num <= 0) {
        if (failure_reason != nullptr) {
            *failure_reason = LoadFailureReason::InvalidFormat;
        }
        return false;
    }

    const size_t expected_vector_count =
        static_cast<size_t>(total_doc_num) * static_cast<size_t>(vector_dim);
    const size_t expected_bitmap_count =
        static_cast<size_t>(total_doc_num) *
        static_cast<size_t>((total_tag_num + 63) / 64);
    if (buffers.vectors.size() != expected_vector_count ||
        buffers.bitmaps.size() != expected_bitmap_count) {
        if (failure_reason != nullptr) {
            *failure_reason = LoadFailureReason::InvalidFormat;
        }
        return false;
    }

    const fs::path target(path);
    if (!EnsureBinaryParentDirectory(target)) {
        if (failure_reason != nullptr) {
            *failure_reason = LoadFailureReason::OpenFailed;
        }
        return false;
    }

    const fs::path temp_path = BuildBinaryTempPath(target);
    std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        if (failure_reason != nullptr) {
            *failure_reason = LoadFailureReason::OpenFailed;
        }
        return false;
    }

    const DatasetCacheHeaderDisk header = MakeDatasetCacheHeader(
        static_cast<uint64_t>(total_doc_num),
        static_cast<uint32_t>(vector_dim),
        static_cast<uint32_t>(total_tag_num));

    const bool ok =
        WriteBinaryExact(out, &header, sizeof(header)) &&
        WriteBinaryExact(out,
                         buffers.vectors.data(),
                         buffers.vectors.size() * sizeof(float)) &&
        WriteBinaryExact(out,
                         buffers.bitmaps.data(),
                         buffers.bitmaps.size() * sizeof(uint64_t));
    out.close();
    if (!ok || !out) {
        std::error_code ec;
        fs::remove(temp_path, ec);
        if (failure_reason != nullptr) {
            *failure_reason = LoadFailureReason::InvalidFormat;
        }
        return false;
    }

    if (!ReplaceBinaryFileAtomically(temp_path, target)) {
        if (failure_reason != nullptr) {
            *failure_reason = LoadFailureReason::OpenFailed;
        }
        return false;
    }
    return true;
}

inline bool LoadDatasetCache(const std::string &path,
                             DatasetBuffers &buffers,
                             LoadFailureReason *failure_reason = nullptr) {
    if (failure_reason != nullptr) {
        *failure_reason = LoadFailureReason::None;
    }

    if (!fs::exists(path)) {
        if (failure_reason != nullptr) {
            *failure_reason = LoadFailureReason::MissingFile;
        }
        return false;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (failure_reason != nullptr) {
            *failure_reason = LoadFailureReason::OpenFailed;
        }
        return false;
    }

    DatasetCacheHeaderDisk header{};
    if (!ReadDatasetCacheHeader(in, header)) {
        if (failure_reason != nullptr) {
            *failure_reason = LoadFailureReason::InvalidFormat;
        }
        return false;
    }

    total_doc_num = static_cast<int>(header.doc_num);
    vector_dim = static_cast<int>(header.vector_dim);
    total_tag_num = static_cast<int>(header.tag_num);

    const uint32_t bitmap_stride = static_cast<uint32_t>((total_tag_num + 63) / 64);

    buffers.vectors.resize(static_cast<size_t>(total_doc_num) * static_cast<size_t>(vector_dim));
    buffers.bitmaps.resize(static_cast<size_t>(total_doc_num) * bitmap_stride);
    buffers.centroids.clear();

    in.read(reinterpret_cast<char *>(buffers.vectors.data()),
            static_cast<std::streamsize>(buffers.vectors.size() * sizeof(float)));
    in.read(reinterpret_cast<char *>(buffers.bitmaps.data()),
            static_cast<std::streamsize>(buffers.bitmaps.size() * sizeof(uint64_t)));

    if (!in.good()) {
        if (failure_reason != nullptr) {
            *failure_reason = LoadFailureReason::InvalidFormat;
        }
        return false;
    }

    return true;
}

inline bool LoadTagMapCache(const std::string &path, std::unordered_map<std::string, int> &tag_map) {
    if (!fs::exists(path)) {
        return false;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    size_t map_size = 0;
    in.read(reinterpret_cast<char *>(&map_size), sizeof(size_t));
    if (!in.good() || map_size == 0) {
        return false;
    }

    tag_map.clear();
    tag_map.reserve(map_size);

    for (size_t i = 0; i < map_size; ++i) {
        uint32_t len = 0;
        in.read(reinterpret_cast<char *>(&len), sizeof(uint32_t));
        if (!in.good() || len == 0) {
            return false;
        }

        std::string term(len, '\0');
        in.read(&term[0], static_cast<std::streamsize>(len));

        int id = 0;
        in.read(reinterpret_cast<char *>(&id), sizeof(int));

        if (!in.good()) {
            return false;
        }

        tag_map[term] = id;
    }

    return true;
}

inline bool SaveTagMapCache(const std::string &path,
                            const std::unordered_map<std::string, int> &tag_map) {
    if (tag_map.empty()) {
        return false;
    }

    const fs::path target(path);
    if (!EnsureBinaryParentDirectory(target)) {
        return false;
    }

    const fs::path temp_path = BuildBinaryTempPath(target);
    std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        return false;
    }

    const size_t map_size = tag_map.size();
    if (!WriteBinaryExact(out, &map_size, sizeof(map_size))) {
        return false;
    }

    for (const auto &item : tag_map) {
        const uint32_t len = static_cast<uint32_t>(item.first.size());
        if (len == 0 ||
            !WriteBinaryExact(out, &len, sizeof(len)) ||
            !WriteBinaryExact(out, item.first.data(), len) ||
            !WriteBinaryExact(out, &item.second, sizeof(item.second))) {
            return false;
        }
    }

    out.close();
    if (!out) {
        return false;
    }

    return ReplaceBinaryFileAtomically(temp_path, target);
}

inline bool RequireAbsoluteRawPath(const std::string &path, const char *key_name) {
    if (!fs::path(path).is_absolute()) {
        std::cerr << "[Config] Error: " << key_name
                  << " must be an absolute path, got: " << path << std::endl;
        return false;
    }
    return true;
}

inline bool ReadRawRecordFile(const fs::path &path,
                              const std::function<bool(uint64_t, const std::vector<uint8_t> &)> &callback,
                              uint32_t *record_count_out = nullptr) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::cerr << "[RawData] Error: failed to open raw file: " << path << std::endl;
        return false;
    }

    uint32_t count = 0;
    if (!ReadBinaryExact(in, &count, sizeof(count))) {
        std::cerr << "[RawData] Error: failed to read record count: " << path << std::endl;
        return false;
    }
    if (record_count_out != nullptr) {
        *record_count_out = count;
    }

    for (uint32_t i = 0; i < count; ++i) {
        uint32_t len = 0;
        if (!ReadBinaryExact(in, &len, sizeof(len)) || len < sizeof(uint64_t)) {
            std::cerr << "[RawData] Error: invalid record header in " << path
                      << ", record_index=" << i << std::endl;
            return false;
        }

        uint64_t gid = 0;
        if (!ReadBinaryExact(in, &gid, sizeof(gid))) {
            std::cerr << "[RawData] Error: failed to read gid in " << path
                      << ", record_index=" << i << std::endl;
            return false;
        }

        const uint32_t payload_len = len - static_cast<uint32_t>(sizeof(uint64_t));
        std::vector<uint8_t> payload(payload_len);
        if (payload_len > 0 &&
            !ReadBinaryExact(in, payload.data(), static_cast<size_t>(payload_len))) {
            std::cerr << "[RawData] Error: failed to read payload in " << path
                      << ", record_index=" << i << std::endl;
            return false;
        }

        if (!callback(gid, payload)) {
            return false;
        }
    }

    return true;
}

inline bool ParseSectionFirstFloatEmbedding(const std::vector<uint8_t> &payload,
                                            std::vector<float> &embedding) {
    return ProtoParser::ParseFirstFloatEmbedding(payload.data(), payload.size(), embedding);
}

template <typename Func>
struct SectionTermCallbackBridge {
    static void Call(const char *term_data, size_t term_size, void *user_data) {
        auto *callback = static_cast<Func *>(user_data);
        (*callback)(std::string(term_data, term_size));
    }
};

template <typename Func>
inline bool ForEachSectionTerm(const std::vector<uint8_t> &payload, Func callback) {
    return ProtoParser::ForEachTerm(payload.data(),
                                    payload.size(),
                                    &SectionTermCallbackBridge<Func>::Call,
                                    &callback);
}

inline bool CollectRawAttributeFiles(std::vector<fs::path> &attr_files) {
    attr_files.clear();
    const fs::path dir(raw_attr_dir);
    std::error_code ec;
    if (!fs::exists(dir, ec) || ec || !fs::is_directory(dir, ec)) {
        std::cerr << "[RawData] Error: raw_attr_dir is not a readable directory: "
                  << dir << std::endl;
        return false;
    }

    for (const auto &entry : fs::directory_iterator(dir, ec)) {
        if (ec) {
            std::cerr << "[RawData] Error while scanning raw_attr_dir: "
                      << ec.message() << std::endl;
            return false;
        }
        if (!entry.is_regular_file(ec) || ec) {
            ec.clear();
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.find("section.inverted_union") != std::string::npos) {
            attr_files.push_back(entry.path());
        }
    }

    std::sort(attr_files.begin(), attr_files.end());
    if (attr_files.empty()) {
        std::cerr << "[RawData] Error: no section.inverted_union* files found under "
                  << dir << std::endl;
        return false;
    }
    return true;
}

inline bool BuildDatasetAndTagMapCachesFromRaw(DatasetBuffers &buffers,
                                               std::unordered_map<std::string, int> &tag_map) {
    if (!RequireAbsoluteRawPath(raw_vector_file, "raw_vector_file") ||
        !RequireAbsoluteRawPath(raw_attr_dir, "raw_attr_dir")) {
        return false;
    }

    std::cout << "[RawData] Cache miss. Building dataset.bin/tag_map.bin from raw data.\n"
              << "[RawData] raw_vector_file=" << raw_vector_file << "\n"
              << "[RawData] raw_attr_dir=" << raw_attr_dir << std::endl;

    buffers = DatasetBuffers{};
    tag_map.clear();
    total_doc_num = 0;
    total_tag_num = 0;
    vector_dim = 0;
    std::unordered_map<uint64_t, uint32_t> gid_to_lid;
    gid_to_lid.reserve(1024 * 1024);

    uint64_t raw_vector_records = 0;
    uint64_t valid_vectors = 0;
    uint64_t skipped_vectors = 0;
    std::vector<float> parsed_vector;
    uint32_t declared_vector_records = 0;
    if (!ReadRawRecordFile(raw_vector_file,
                           [&](uint64_t gid, const std::vector<uint8_t> &payload) {
                               ++raw_vector_records;
                               if (!ParseSectionFirstFloatEmbedding(payload, parsed_vector) || parsed_vector.empty()) {
                                   ++skipped_vectors;
                                   return true;
                               }
                               if (vector_dim == 0) {
                                   vector_dim = static_cast<int>(parsed_vector.size());
                                   std::cout << "[RawData] Detected vector_dim=" << vector_dim
                                             << " from first valid vector.\n";
                               }
                               if (static_cast<int>(parsed_vector.size()) != vector_dim) {
                                   ++skipped_vectors;
                                   return true;
                               }
                               const uint32_t local_id = static_cast<uint32_t>(valid_vectors);
                               if (!gid_to_lid.emplace(gid, local_id).second) {
                                   ++skipped_vectors;
                                   return true;
                               }
                               buffers.vectors.insert(buffers.vectors.end(),
                                                      parsed_vector.begin(),
                                                      parsed_vector.end());
                               ++valid_vectors;
                               if ((valid_vectors % 1000000ULL) == 0ULL) {
                                   std::cout << "[RawData] Vector progress: valid_docs="
                                             << valid_vectors << "\n";
                               }
                               return true;
                           },
                           &declared_vector_records)) {
        return false;
    }

    if (valid_vectors == 0 || vector_dim <= 0 ||
        valid_vectors > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        std::cerr << "[RawData] Error: no valid vectors were found in raw_vector_file.\n";
        return false;
    }
    total_doc_num = static_cast<int>(valid_vectors);
    std::cout << "[RawData] Vector file processed. declared_records=" << declared_vector_records
              << ", scanned_records=" << raw_vector_records
              << ", valid_docs=" << total_doc_num
              << ", skipped_records=" << skipped_vectors << std::endl;

    std::vector<fs::path> attr_files;
    if (!CollectRawAttributeFiles(attr_files)) {
        return false;
    }

    std::cout << "[RawData] First attribute pass: collecting tag map from "
              << attr_files.size() << " files.\n";
    uint64_t attr_records_seen = 0;
    uint64_t attr_records_matched = 0;
    for (size_t file_idx = 0; file_idx < attr_files.size(); ++file_idx) {
        if (!ReadRawRecordFile(attr_files[file_idx],
                               [&](uint64_t gid, const std::vector<uint8_t> &payload) {
                                   ++attr_records_seen;
                                   if (gid_to_lid.find(gid) == gid_to_lid.end()) {
                                       return true;
                                   }
                                   ++attr_records_matched;
                                   return ForEachSectionTerm(payload, [&](const std::string &term) {
                                       if (tag_map.find(term) == tag_map.end()) {
                                           const int tag_id = static_cast<int>(tag_map.size());
                                           tag_map.emplace(term, tag_id);
                                       }
                                   });
                               })) {
            return false;
        }
        std::cout << "[RawData] Tag-map pass progress: " << (file_idx + 1)
                  << "/" << attr_files.size()
                  << ", tags=" << tag_map.size() << "\n";
    }

    if (tag_map.empty() || tag_map.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::cerr << "[RawData] Error: no valid tags were found in raw_attr_dir.\n";
        return false;
    }
    total_tag_num = static_cast<int>(tag_map.size());
    const uint32_t bitmap_stride = static_cast<uint32_t>((total_tag_num + 63) / 64);
    buffers.bitmaps.assign(static_cast<size_t>(total_doc_num) * bitmap_stride, 0ULL);

    std::cout << "[RawData] Tag map collected. total_tag_num=" << total_tag_num
              << ", attr_records_seen=" << attr_records_seen
              << ", attr_records_matched_to_vectors=" << attr_records_matched
              << ", bitmap_stride_u64=" << bitmap_stride << std::endl;

    std::cout << "[RawData] Second attribute pass: filling document bitmaps.\n";
    for (size_t file_idx = 0; file_idx < attr_files.size(); ++file_idx) {
        if (!ReadRawRecordFile(attr_files[file_idx],
                               [&](uint64_t gid, const std::vector<uint8_t> &payload) {
                                   const auto doc_it = gid_to_lid.find(gid);
                                   if (doc_it == gid_to_lid.end()) {
                                       return true;
                                   }
                                   const uint32_t local_id = doc_it->second;
                                   uint64_t *doc_bitmap =
                                       buffers.bitmaps.data() +
                                       static_cast<size_t>(local_id) * bitmap_stride;
                                   return ForEachSectionTerm(payload, [&](const std::string &term) {
                                       const auto tag_it = tag_map.find(term);
                                       if (tag_it == tag_map.end()) {
                                           return;
                                       }
                                       const int tag_id = tag_it->second;
                                       doc_bitmap[static_cast<size_t>(tag_id / 64)] |=
                                           (1ULL << static_cast<unsigned>(tag_id % 64));
                                   });
                               })) {
            return false;
        }
        std::cout << "[RawData] Bitmap pass progress: " << (file_idx + 1)
                  << "/" << attr_files.size() << "\n";
    }

    std::cout << "[RawData] Saving dataset cache to " << dataset_cache_file << std::endl;
    if (!SaveDatasetCache(dataset_cache_file, buffers)) {
        std::cerr << "[RawData] Error: failed to save dataset cache: "
                  << dataset_cache_file << std::endl;
        return false;
    }
    std::cout << "[RawData] Saving tag map cache to " << tag_map_cache_file << std::endl;
    if (!SaveTagMapCache(tag_map_cache_file, tag_map)) {
        std::cerr << "[RawData] Error: failed to save tag map cache: "
                  << tag_map_cache_file << std::endl;
        return false;
    }

    std::cout << "[RawData] Raw dataset conversion completed. total_doc_num="
              << total_doc_num << ", total_tag_num=" << total_tag_num
              << ", vector_dim=" << vector_dim << std::endl;
    return true;
}

inline bool EnsureDatasetAndTagMapCaches(DatasetBuffers &buffers,
                                         std::unordered_map<std::string, int> &tag_map) {
    std::cout << "[Cache] Checking dataset cache: " << dataset_cache_file << "\n"
              << "[Cache] Checking tag map cache: " << tag_map_cache_file << std::endl;

    LoadFailureReason dataset_failure = LoadFailureReason::None;
    const bool dataset_loaded = LoadDatasetCache(dataset_cache_file, buffers, &dataset_failure);
    const bool tag_map_loaded = LoadTagMapCache(tag_map_cache_file, tag_map);

    if (dataset_loaded && tag_map_loaded &&
        tag_map.size() == static_cast<size_t>(total_tag_num)) {
        std::cout << "[Cache] dataset.bin and tag_map.bin loaded; raw dataset conversion skipped.\n";
        return FinalizeRuntimeParamsAfterDatasetLoad();
    }

    std::cout << "[Cache] Dataset/tag-map cache is unavailable or invalid. "
              << "dataset_loaded=" << (dataset_loaded ? "yes" : "no")
              << ", tag_map_loaded=" << (tag_map_loaded ? "yes" : "no")
              << ", tag_map_size=" << tag_map.size()
              << ", dataset_total_tag_num=" << total_tag_num
              << ". Rebuilding both caches from raw data.\n";
    if (!BuildDatasetAndTagMapCachesFromRaw(buffers, tag_map)) {
        return false;
    }
    return FinalizeRuntimeParamsAfterDatasetLoad();
}

inline bool RawQueryLineHasConfiguredKeys(const std::string &line) {
    json root;
    try {
        root = json::parse(line);
    } catch (...) {
        return false;
    }

    if (!root.contains(raw_query_result_num_key) ||
        !root.contains(raw_query_main_tier_key) ||
        !root[raw_query_main_tier_key].is_object()) {
        return false;
    }

    const auto &main_tier = root[raw_query_main_tier_key];
    if (!main_tier.contains(raw_query_json_query_key) ||
        !main_tier[raw_query_json_query_key].is_string()) {
        return false;
    }

    json inner;
    try {
        inner = json::parse(main_tier[raw_query_json_query_key].get<std::string>());
    } catch (...) {
        return false;
    }

    if (!inner.contains(raw_query_vector_node_key) ||
        !inner[raw_query_vector_node_key].is_object()) {
        return false;
    }

    const auto &vector_node = inner[raw_query_vector_node_key];
    return vector_node.contains(raw_query_vector_value_key) &&
           vector_node[raw_query_vector_value_key].is_array() &&
           vector_node.contains(raw_query_syntax_filter_key);
}

inline bool GenerateQueryCacheFromRaw() {
    if (!RequireAbsoluteRawPath(raw_query_file, "raw_query_file")) {
        return false;
    }

    std::ifstream in(raw_query_file);
    if (!in) {
        std::cerr << "[RawQuery] Error: failed to open raw_query_file: "
                  << raw_query_file << std::endl;
        return false;
    }

    const std::string raw_content((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());

    std::vector<std::string> non_empty_lines;
    const std::string trimmed_content = ParamTrim(raw_content);
    if (!trimmed_content.empty()) {
        try {
            const json parsed = json::parse(trimmed_content);
            if (parsed.is_array()) {
                for (const auto &item : parsed) {
                    if (item.is_object()) {
                        non_empty_lines.push_back(item.dump());
                    }
                }
            } else if (parsed.is_object()) {
                non_empty_lines.push_back(parsed.dump());
            }
        } catch (...) {
            std::istringstream line_stream(raw_content);
            std::string line;
            while (std::getline(line_stream, line)) {
                const std::string trimmed = ParamTrim(line);
                if (!trimmed.empty()) {
                    non_empty_lines.push_back(trimmed);
                }
            }
        }
    }

    if (non_empty_lines.empty()) {
        std::cerr << "[RawQuery] Error: raw_query_file has no non-empty query lines: "
                  << raw_query_file << std::endl;
        return false;
    }

    size_t key_covered = 0;
    for (const auto &query_line : non_empty_lines) {
        if (RawQueryLineHasConfiguredKeys(query_line)) {
            ++key_covered;
        }
    }

    std::cout << "[RawQuery] Configured JSON key coverage: "
              << key_covered << "/" << non_empty_lines.size()
              << " queries contain keys {"
              << raw_query_result_num_key << ", "
              << raw_query_main_tier_key << ", "
              << raw_query_json_query_key << ", "
              << raw_query_vector_node_key << ", "
              << raw_query_vector_value_key << ", "
              << raw_query_syntax_filter_key << "}.\n";

    if (key_covered * 2 <= non_empty_lines.size()) {
        std::cerr << "[RawQuery] Error: half or fewer raw queries contain the configured JSON keys. "
                  << "Please check config raw_query_*_key values before converting queries.\n";
        return false;
    }

    const fs::path target(query_file);
    if (!EnsureBinaryParentDirectory(target)) {
        std::cerr << "[RawQuery] Error: failed to create parent directory for "
                  << target << std::endl;
        return false;
    }
    const fs::path temp_path = BuildBinaryTempPath(target);
    std::ofstream out(temp_path, std::ios::out | std::ios::trunc);
    if (!out) {
        std::cerr << "[RawQuery] Error: failed to open query cache temp file: "
                  << temp_path << std::endl;
        return false;
    }
    for (const auto &query_line : non_empty_lines) {
        out << query_line << '\n';
    }
    out.close();
    if (!out || !ReplaceBinaryFileAtomically(temp_path, target)) {
        std::cerr << "[RawQuery] Error: failed to publish generated query file: "
                  << target << std::endl;
        return false;
    }

    std::cout << "[RawQuery] Query file generated from raw_query_file. output="
              << query_file << ", query_count=" << non_empty_lines.size() << std::endl;
    return true;
}

inline bool EnsurePreparedQueryFile() {
    std::ifstream existing(query_file);
    if (existing.good()) {
        std::cout << "[Cache] Query file exists; raw query conversion skipped: "
                  << query_file << std::endl;
        return true;
    }

    std::cout << "[Cache] Query file not readable: " << query_file
              << ". Building it from raw_query_file.\n";
    return GenerateQueryCacheFromRaw();
}

inline bool ParseRootQueryLine(const std::string &line,
                               size_t line_no,
                               PreparedQuery &prepared,
                               const std::unordered_map<std::string, int> &tag_map,
                               int expected_dim,
                               std::string &error,
                               QueryParseIssue *issue = nullptr,
                               bool *missing_syntax_filter = nullptr,
                               int *actual_vector_dim = nullptr) {
    if (issue != nullptr) {
        *issue = QueryParseIssue::None;
    }
    if (missing_syntax_filter != nullptr) {
        *missing_syntax_filter = false;
    }
    if (actual_vector_dim != nullptr) {
        *actual_vector_dim = 0;
    }

    json root;
    try {
        root = json::parse(line);
    } catch (const std::exception &e) {
        error = std::string("invalid root json: ") + e.what();
        return false;
    }

    if (!root.contains(raw_query_result_num_key) ||
        !(root[raw_query_result_num_key].is_number_integer() ||
          root[raw_query_result_num_key].is_number_unsigned())) {
        error = "missing or invalid root." + raw_query_result_num_key;
        return false;
    }
    const int top_k = root[raw_query_result_num_key].get<int>();
    if (top_k <= 0) {
        error = "root." + raw_query_result_num_key + " must be > 0";
        return false;
    }

    if (!root.contains(raw_query_main_tier_key) || !root[raw_query_main_tier_key].is_object()) {
        error = "missing " + raw_query_main_tier_key + " object";
        return false;
    }
    const auto &main_tier = root[raw_query_main_tier_key];

    if (!main_tier.contains(raw_query_json_query_key) ||
        !main_tier[raw_query_json_query_key].is_string()) {
        error = "missing " + raw_query_main_tier_key + "." +
                raw_query_json_query_key + " string";
        return false;
    }

    json inner;
    try {
        inner = json::parse(main_tier[raw_query_json_query_key].get<std::string>());
    } catch (const std::exception &e) {
        error = std::string("invalid ") + raw_query_main_tier_key + "." +
                raw_query_json_query_key + " json: " + e.what();
        return false;
    }

    if (!inner.contains(raw_query_vector_node_key) ||
        !inner[raw_query_vector_node_key].is_object()) {
        error = "missing " + raw_query_vector_node_key + " object in inner json_query";
        if (issue != nullptr) {
            *issue = QueryParseIssue::MissingVector;
        }
        return false;
    }

    const auto &vector_node = inner[raw_query_vector_node_key];
    if (!vector_node.contains(raw_query_vector_value_key) ||
        !vector_node[raw_query_vector_value_key].is_array()) {
        error = "missing " + raw_query_vector_node_key + "." +
                raw_query_vector_value_key + " array";
        if (issue != nullptr) {
            *issue = QueryParseIssue::MissingVector;
        }
        return false;
    }

    const auto &vec_json = vector_node[raw_query_vector_value_key];
    if (static_cast<int>(vec_json.size()) != expected_dim) {
        error = "vector dimension mismatch with config/dataset";
        if (issue != nullptr) {
            *issue = QueryParseIssue::VectorDimMismatch;
        }
        if (actual_vector_dim != nullptr) {
            *actual_vector_dim = static_cast<int>(vec_json.size());
        }
        return false;
    }

    std::vector<float> query_vec;
    query_vec.reserve(vec_json.size());
    for (const auto &v : vec_json) {
        if (!v.is_number()) {
            error = "vector contains non-numeric value";
            return false;
        }
        query_vec.push_back(v.get<float>());
    }

    std::string filter_expr;
    if (vector_node.contains(raw_query_syntax_filter_key)) {
        const ExprResult parsed = ParseSyntaxFilterToExpr(vector_node[raw_query_syntax_filter_key], tag_map);
        if (!parsed.is_valid) {
            error = "invalid syntax_filter node";
            if (issue != nullptr) {
                *issue = QueryParseIssue::InvalidSyntaxFilter;
            }
            return false;
        }
        filter_expr = FinalizeFilterExpr(parsed);
    } else {
        if (missing_syntax_filter != nullptr) {
            *missing_syntax_filter = true;
        }
        filter_expr = "";
    }

    prepared.query_vec = std::move(query_vec);
    prepared.filter_expr = std::move(filter_expr);
    prepared.top_k = top_k;
    prepared.line_no = line_no;

    return true;
}

inline bool LoadPreparedQueries(const std::string &query_file,
                                const std::unordered_map<std::string, int> &tag_map,
                                int expected_dim,
                                std::vector<PreparedQuery> &queries,
                                std::vector<std::string> &warnings,
                                QueryLoadDiagnostics *diagnostics = nullptr,
                                LoadFailureReason *failure_reason = nullptr) {
    if (failure_reason != nullptr) {
        *failure_reason = LoadFailureReason::None;
    }

    std::ifstream in(query_file);
    if (!in) {
        if (failure_reason != nullptr) {
            *failure_reason = LoadFailureReason::OpenFailed;
        }
        return false;
    }

    queries.clear();
    warnings.clear();
    if (diagnostics != nullptr) {
        diagnostics->Clear();
    }

    std::string line;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        const std::string trimmed = Trim(line);
        if (trimmed.empty()) {
            continue;
        }

        PreparedQuery q;
        std::string err;
        QueryParseIssue issue = QueryParseIssue::None;
        bool missing_syntax_filter = false;
        int actual_vector_dim = 0;
        if (!ParseRootQueryLine(trimmed,
                                line_no,
                                q,
                                tag_map,
                                expected_dim,
                                err,
                                &issue,
                                &missing_syntax_filter,
                                &actual_vector_dim)) {
            if (issue == QueryParseIssue::VectorDimMismatch) {
                PrintVectorDimMismatchWarning(query_file, actual_vector_dim, expected_dim, line_no);
                if (failure_reason != nullptr) {
                    *failure_reason = LoadFailureReason::VectorDimMismatch;
                }
                return false;
            }
            if (diagnostics != nullptr) {
                if (issue == QueryParseIssue::MissingVector) {
                    diagnostics->missing_vector_queries.push_back(line_no);
                } else if (issue == QueryParseIssue::InvalidSyntaxFilter) {
                    diagnostics->invalid_syntax_filter_queries.push_back(line_no);
                }
            }
            warnings.push_back("line " + std::to_string(line_no) + " skipped: " + err);
            continue;
        }

        if (diagnostics != nullptr && missing_syntax_filter) {
            diagnostics->missing_syntax_filter_queries.push_back(line_no);
        }
        queries.push_back(std::move(q));
    }

    return true;
}

inline InputDataset BuildInputDataset(const DatasetBuffers &buffers) {
    InputDataset dataset;
    dataset.vectors = buffers.vectors.data();
    dataset.tag_bitmaps = buffers.bitmaps.data();
    dataset.bucket_centroids = buffers.centroids.data();
    return dataset;
}

} // namespace DataReader
