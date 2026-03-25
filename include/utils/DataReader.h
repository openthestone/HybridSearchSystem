#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
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

// --- 全局参数定义 (Header-Only) ---
// 说明：默认值仅用于兜底，正常流程必须由 LoadParams(config.txt) 覆盖。
inline int total_doc_num = 10000000;
inline int total_tag_num = 30000;
inline int total_bucket_num = 10000;
inline int vector_dim = 32;
inline int max_doc_per_bucket = 1024; // 该参数必须是16的倍数

inline int cpu_core_count = 24;
inline int cores_per_group = 8;
inline int group_count = 3;

inline int probe_initial_bucket_num = 200; // config: valid_bucket_num_base
inline int valid_bucket_num_base = 200;    // single-round workspace capacity, must equal config valid_bucket_num_base
inline int valid_bucket_num_incremental = 200;
inline int k_expand_param = 1;
inline std::string query_file = "../../QueryData_10000.txt";
inline std::string query_result_root = "../../result/";
inline std::string ground_truth_cache_file = "../../ground_truth_cache.bin";
inline std::string bucket_ivf_index_file = "../../bucket_ivf_index.bin";
inline std::string bucket_index_file = "../../bucket_index.bin";

inline int npu_device_id_start = 0;
inline int npu_device_count = 1;
inline int groups_per_device = 3;

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

inline bool EnsurePreallocAtLeast(const char *key, int actual, size_t required) {
    const int required_int = ClampSizeTToInt(required);
    if (actual < required_int) {
        std::cerr << "[Config] Error: " << key << "=" << actual
                  << " is below required minimum " << required_int << std::endl;
        return false;
    }
    return true;
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
        CeilDivSizeT(static_cast<size_t>(total_bucket_num), static_cast<size_t>(cores_per_group));
    const size_t ivf_raw_stride = CeilDivSizeT(buckets_per_core, 64);
    const size_t bucket_stride = CeilDivSizeT(static_cast<size_t>(max_doc_per_bucket), 64);
    const size_t buckets_per_core_in_batch =
        CeilDivSizeT(static_cast<size_t>(valid_bucket_num_base), static_cast<size_t>(cores_per_group));
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
    const size_t required_candidate_merge_items = static_cast<size_t>(total_bucket_num);
    const size_t required_sorted_bucket_items = static_cast<size_t>(total_bucket_num);
    const size_t required_batch_bucket_ids_items = static_cast<size_t>(valid_bucket_num_base);
    const size_t required_temp_doc_mask_u64 = bucket_stride;
    const size_t required_thread_doc_results_items =
        buckets_per_core_in_batch * static_cast<size_t>(max_doc_per_bucket);
    const size_t required_all_items =
        std::min(static_cast<size_t>(valid_bucket_num_base) * static_cast<size_t>(max_doc_per_bucket),
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
    if (!EnsurePreallocAtLeast("query_vector_reserve_floats",
                               query_vector_reserve_floats,
                               static_cast<size_t>(vector_dim))) {
        return false;
    }
    if (!EnsurePreallocAtLeast("query_bucket_rpn_reserve_items",
                               query_bucket_rpn_reserve_items,
                               static_cast<size_t>(max_rpn_length))) {
        return false;
    }
    if (!EnsurePreallocAtLeast("query_bucket_level_ivf_rpn_reserve_items",
                               query_bucket_level_ivf_rpn_reserve_items,
                               static_cast<size_t>(max_rpn_length))) {
        return false;
    }
    if (!EnsurePreallocAtLeast("query_merge_batch_reserve_items",
                               query_merge_batch_reserve_items,
                               required_merge_batch_items)) {
        return false;
    }
    if (!EnsurePreallocAtLeast("group_tls_scratch_pool_reserve_u64",
                               group_tls_scratch_pool_reserve_u64,
                               required_scratch_u64)) {
        return false;
    }
    if (!EnsurePreallocAtLeast("group_tls_ivf_mask_reserve_u64",
                               group_tls_ivf_mask_reserve_u64,
                               required_ivf_mask_u64)) {
        return false;
    }
    if (!EnsurePreallocAtLeast("group_tls_batch_mask_reserve_u64",
                               group_tls_batch_mask_reserve_u64,
                               required_batch_mask_u64)) {
        return false;
    }
    if (!EnsurePreallocAtLeast("group_local_pq_reserve_items_per_rank",
                               group_local_pq_reserve_items_per_rank,
                               required_local_pq_items)) {
        return false;
    }
    if (!EnsurePreallocAtLeast("group_thread_bucket_results_reserve_items_per_rank",
                               group_thread_bucket_results_reserve_items_per_rank,
                               required_bucket_results_items)) {
        return false;
    }
    if (!EnsurePreallocAtLeast("group_candidate_merge_reserve_items",
                               group_candidate_merge_reserve_items,
                               required_candidate_merge_items)) {
        return false;
    }
    if (!EnsurePreallocAtLeast("group_sorted_buckets_reserve_items",
                               group_sorted_buckets_reserve_items,
                               required_sorted_bucket_items)) {
        return false;
    }
    if (!EnsurePreallocAtLeast("group_batch_bucket_ids_reserve_items",
                               group_batch_bucket_ids_reserve_items,
                               required_batch_bucket_ids_items)) {
        return false;
    }
    if (!EnsurePreallocAtLeast("group_temp_doc_mask_reserve_u64_per_rank",
                               group_temp_doc_mask_reserve_u64_per_rank,
                               required_temp_doc_mask_u64)) {
        return false;
    }
    if (!EnsurePreallocAtLeast("group_thread_doc_results_reserve_items_per_rank",
                               group_thread_doc_results_reserve_items_per_rank,
                               required_thread_doc_results_items)) {
        return false;
    }
    if (!EnsurePreallocAtLeast("group_all_items_reserve_items",
                               group_all_items_reserve_items,
                               required_all_items)) {
        return false;
    }

    return true;
}

inline bool LoadParams(const std::string &config_file, ResourceConfigProfile profile) {
    g_params_loaded = false;

    std::ifstream file(config_file);
    if (!file.is_open()) {
        std::cerr << "[Config] Error: Cannot open config file: " << config_file << std::endl;
        return false;
    }

    bool seen_total_doc_num = false;
    bool seen_total_tag_num = false;
    bool seen_total_bucket_num = false;
    bool seen_vector_dim = false;
    bool seen_max_doc_per_bucket = false;
    bool seen_cores_per_group = false;
    bool seen_valid_bucket_num_base = false;
    bool seen_valid_bucket_num_incremental = false;
    bool seen_k_expand_param = false;
    bool seen_bucket_ivf_index_file = false;
    bool seen_bucket_index_file = false;
    bool seen_npu_device_id_start = false;
    bool seen_max_query_tags = false;
    bool seen_max_rpn_length = false;
    bool seen_cpu_cache_line_size = false;

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

        if (key == "query_file") {
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
        }

        int value = 0;
        if (!ParseConfigInt(value_str, value)) {
            std::cerr << "[Config] Error parsing value for key: " << key
                      << ", raw value: " << value_str << std::endl;
            return false;
        }

        if (key == "total_doc_num") {
            total_doc_num = value;
            seen_total_doc_num = true;
        } else if (key == "total_tag_num") {
            total_tag_num = value;
            seen_total_tag_num = true;
        } else if (key == "total_bucket_num") {
            total_bucket_num = value;
            seen_total_bucket_num = true;
        } else if (key == "vector_dim") {
            vector_dim = value;
            seen_vector_dim = true;
        } else if (key == "max_doc_per_bucket") {
            max_doc_per_bucket = value;
            seen_max_doc_per_bucket = true;
        } else if (key == "cpu_core_count_serial") {
            cpu_core_count_serial_value = value;
            seen_cpu_core_count_serial = true;
        } else if (key == "cpu_core_count_parallel") {
            cpu_core_count_parallel_value = value;
            seen_cpu_core_count_parallel = true;
        } else if (key == "cores_per_group") {
            cores_per_group = value;
            seen_cores_per_group = true;
        } else if (key == "valid_bucket_num_base") {
            probe_initial_bucket_num = value;
            seen_valid_bucket_num_base = true;
        } else if (key == "valid_bucket_num_incremental") {
            valid_bucket_num_incremental = value;
            seen_valid_bucket_num_incremental = true;
        } else if (key == "k_expand_param") {
            k_expand_param = value;
            seen_k_expand_param = true;
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
    if (!seen_total_doc_num) missing_keys.emplace_back("total_doc_num");
    if (!seen_total_tag_num) missing_keys.emplace_back("total_tag_num");
    if (!seen_total_bucket_num) missing_keys.emplace_back("total_bucket_num");
    if (!seen_vector_dim) missing_keys.emplace_back("vector_dim");
    if (!seen_max_doc_per_bucket) missing_keys.emplace_back("max_doc_per_bucket");
    if (!seen_cpu_core_count_serial) missing_keys.emplace_back("cpu_core_count_serial");
    if (!seen_npu_device_count_serial) missing_keys.emplace_back("npu_device_count_serial");
    if (!seen_groups_per_device_serial) missing_keys.emplace_back("groups_per_device_serial");
    if (!seen_cpu_core_count_parallel) missing_keys.emplace_back("cpu_core_count_parallel");
    if (!seen_npu_device_count_parallel) missing_keys.emplace_back("npu_device_count_parallel");
    if (!seen_groups_per_device_parallel) missing_keys.emplace_back("groups_per_device_parallel");
    if (!seen_cores_per_group) missing_keys.emplace_back("cores_per_group");
    if (!seen_valid_bucket_num_base) missing_keys.emplace_back("valid_bucket_num_base");
    if (!seen_valid_bucket_num_incremental) missing_keys.emplace_back("valid_bucket_num_incremental");
    if (!seen_k_expand_param) missing_keys.emplace_back("k_expand_param");
    if (!seen_bucket_ivf_index_file) missing_keys.emplace_back("bucket_ivf_index_file");
    if (!seen_bucket_index_file) missing_keys.emplace_back("bucket_index_file");
    if (!seen_npu_device_id_start) missing_keys.emplace_back("npu_device_id_start");
    if (!seen_max_query_tags) missing_keys.emplace_back("max_query_tags");
    if (!seen_cpu_cache_line_size) missing_keys.emplace_back("cpu_cache_line_size");

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
    if (vector_dim <= 0 || total_doc_num <= 0 || total_tag_num <= 0 || total_bucket_num <= 0) {
        std::cerr << "[Config] Error: total_doc_num/total_tag_num/total_bucket_num/vector_dim must be > 0" << std::endl;
        return false;
    }
    if (max_doc_per_bucket <= 0 || (max_doc_per_bucket % 16) != 0) {
        std::cerr << "[Config] Error: max_doc_per_bucket must be > 0 and divisible by 16" << std::endl;
        return false;
    }
    if (probe_initial_bucket_num <= 0 || valid_bucket_num_incremental <= 0) {
        std::cerr << "[Config] Error: valid_bucket_num_base and valid_bucket_num_incremental must be > 0" << std::endl;
        return false;
    }
    if (k_expand_param <= 0) {
        std::cerr << "[Config] Error: k_expand_param must be > 0" << std::endl;
        return false;
    }
    if (valid_bucket_num_incremental > probe_initial_bucket_num) {
        std::cerr << "[Config] Error: valid_bucket_num_incremental must be <= valid_bucket_num_base" << std::endl;
        return false;
    }

    valid_bucket_num_base = probe_initial_bucket_num;

    group_count = cpu_core_count / cores_per_group;
    if (group_count <= 0) {
        std::cerr << "[Config] Error: invalid group_count=" << group_count << std::endl;
        return false;
    }
    if ((cpu_core_count % cores_per_group) != 0) {
        std::cout << "[Config] Warning: cpu_core_count is not divisible by cores_per_group, tail cores are ignored" << std::endl;
    }

    const int expected_groups = npu_device_count * groups_per_device;
    if (group_count != expected_groups) {
        std::cerr << "[Config] Error: group_count(cpu_core_count/cores_per_group)=" << group_count
                  << " does not match npu_device_count_" << ResourceConfigProfileName(profile)
                  << "*groups_per_device_" << ResourceConfigProfileName(profile)
                  << "=" << expected_groups << std::endl;
        return false;
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

    if (!FinalizePreallocationParams()) {
        return false;
    }

    g_params_loaded = true;
    return true;
}

inline bool LoadParams(const std::string &config_file) {
    return LoadParams(config_file, ResourceConfigProfile::Serial);
}

namespace DataReader {
namespace fs = std::filesystem;
using json = nlohmann::json;

struct BucketIvfIndexFileHeaderDisk {
    char magic[8];
    uint32_t version = 0;
    uint32_t total_tag_num = 0;
    uint32_t total_bucket_num = 0;
    uint32_t cores_per_group = 0;
    uint32_t cpu_cache_line_size = 0;
    uint32_t buckets_per_core = 0;
    uint32_t aligned_stride = 0;
};

struct BucketIndexFileHeaderDisk {
    char magic[8];
    uint32_t version = 0;
    uint32_t total_doc_num = 0;
    uint32_t total_tag_num = 0;
    uint32_t total_bucket_num = 0;
    uint32_t vector_dim = 0;
    uint32_t max_doc_per_bucket = 0;
};

inline bool ReadBinaryExact(std::ifstream &in, void *buffer, size_t bytes) {
    if (bytes == 0) {
        return true;
    }

    in.read(reinterpret_cast<char *>(buffer), static_cast<std::streamsize>(bytes));
    return in.good();
}

inline bool WriteBinaryExact(std::ofstream &out, const void *buffer, size_t bytes) {
    if (bytes == 0) {
        return true;
    }

    out.write(reinterpret_cast<const char *>(buffer), static_cast<std::streamsize>(bytes));
    return out.good();
}

template <typename T, typename Alloc>
inline bool WriteBinaryVector(std::ofstream &out, const std::vector<T, Alloc> &values) {
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
    std::vector<uint32_t> bucket_ids;
    std::vector<uint64_t> bitmaps;
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

inline bool LoadDatasetCache(const std::string &path, DatasetBuffers &buffers) {
    if (!fs::exists(path)) {
        return false;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    int doc_num = 0;
    int vec_dim = 0;
    int tag_num = 0;
    int bucket_num = 0;

    in.read(reinterpret_cast<char *>(&doc_num), sizeof(int));
    in.read(reinterpret_cast<char *>(&vec_dim), sizeof(int));
    in.read(reinterpret_cast<char *>(&tag_num), sizeof(int));
    in.read(reinterpret_cast<char *>(&bucket_num), sizeof(int));

    if (!in.good() || doc_num <= 0 || vec_dim <= 0 || tag_num <= 0 || bucket_num <= 0) {
        return false;
    }

    const uint32_t bitmap_stride = static_cast<uint32_t>((tag_num + 63) / 64);

    buffers.vectors.resize(static_cast<size_t>(doc_num) * static_cast<size_t>(vec_dim));
    buffers.bucket_ids.resize(static_cast<size_t>(doc_num));
    buffers.bitmaps.resize(static_cast<size_t>(doc_num) * bitmap_stride);
    buffers.centroids.resize(static_cast<size_t>(bucket_num) * static_cast<size_t>(vec_dim));

    in.read(reinterpret_cast<char *>(buffers.vectors.data()), static_cast<std::streamsize>(buffers.vectors.size() * sizeof(float)));
    in.read(reinterpret_cast<char *>(buffers.bucket_ids.data()), static_cast<std::streamsize>(buffers.bucket_ids.size() * sizeof(uint32_t)));
    in.read(reinterpret_cast<char *>(buffers.bitmaps.data()), static_cast<std::streamsize>(buffers.bitmaps.size() * sizeof(uint64_t)));
    in.read(reinterpret_cast<char *>(buffers.centroids.data()), static_cast<std::streamsize>(buffers.centroids.size() * sizeof(float)));

    if (!in.good()) {
        return false;
    }

    if (g_params_loaded) {
        if (total_doc_num != doc_num || vector_dim != vec_dim ||
            total_tag_num != tag_num) {
            std::cerr << "[Config] Error: dataset.bin header does not match loaded config values.\n"
                      << "  config:  docs=" << total_doc_num
                      << ", dim=" << vector_dim
                      << ", tags=" << total_tag_num
                      << ", buckets=" << total_bucket_num << "\n"
                      << "  dataset: docs=" << doc_num
                      << ", dim=" << vec_dim
                      << ", tags=" << tag_num
                      << ", buckets=" << bucket_num << std::endl;
            return false;
        }

        if (total_bucket_num != bucket_num) {
            std::cout << "[Config] Warning: total_bucket_num=" << total_bucket_num
                      << " differs from dataset.bin header bucket_num=" << bucket_num
                      << ". Using dataset bucket count until clustering results override it."
                      << std::endl;
            total_bucket_num = bucket_num;
        }
    } else {
        // 仅在未加载 config 的兜底路径下，使用 dataset 头更新全局参数。
        total_doc_num = doc_num;
        vector_dim = vec_dim;
        total_tag_num = tag_num;
        total_bucket_num = bucket_num;
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

inline bool ParseRootQueryLine(const std::string &line,
                               size_t line_no,
                               PreparedQuery &prepared,
                               const std::unordered_map<std::string, int> &tag_map,
                               int expected_dim,
                               std::string &error) {
    json root;
    try {
        root = json::parse(line);
    } catch (const std::exception &e) {
        error = std::string("invalid root json: ") + e.what();
        return false;
    }

    if (!root.contains("result_num") ||
        !(root["result_num"].is_number_integer() || root["result_num"].is_number_unsigned())) {
        error = "missing or invalid root.result_num";
        return false;
    }
    const int top_k = root["result_num"].get<int>();
    if (top_k <= 0) {
        error = "root.result_num must be > 0";
        return false;
    }

    if (!root.contains("main_tier") || !root["main_tier"].is_object()) {
        error = "missing main_tier object";
        return false;
    }
    const auto &main_tier = root["main_tier"];

    if (!main_tier.contains("json_query") || !main_tier["json_query"].is_string()) {
        error = "missing main_tier.json_query string";
        return false;
    }

    json inner;
    try {
        inner = json::parse(main_tier["json_query"].get<std::string>());
    } catch (const std::exception &e) {
        error = std::string("invalid main_tier.json_query json: ") + e.what();
        return false;
    }

    if (!inner.contains("vector") || !inner["vector"].is_object()) {
        error = "missing vector object in inner json_query";
        return false;
    }

    const auto &vector_node = inner["vector"];
    constexpr const char *kVectorKey = "relevance_learning2rank";
    if (!vector_node.contains(kVectorKey) || !vector_node[kVectorKey].is_array()) {
        error = "missing vector.relevance_learning2rank array";
        return false;
    }

    const auto &vec_json = vector_node[kVectorKey];
    if (static_cast<int>(vec_json.size()) != expected_dim) {
        error = "vector dimension mismatch with config/dataset";
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
    if (vector_node.contains("syntax_filter")) {
        const ExprResult parsed = ParseSyntaxFilterToExpr(vector_node["syntax_filter"], tag_map);
        if (!parsed.is_valid) {
            error = "invalid syntax_filter node";
            return false;
        }
        filter_expr = FinalizeFilterExpr(parsed);
    } else {
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
                                std::vector<std::string> &warnings) {
    std::ifstream in(query_file);
    if (!in) {
        return false;
    }

    queries.clear();
    warnings.clear();

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
        if (!ParseRootQueryLine(trimmed, line_no, q, tag_map, expected_dim, err)) {
            warnings.push_back("line " + std::to_string(line_no) + " skipped: " + err);
            continue;
        }

        queries.push_back(std::move(q));
    }

    return true;
}

inline InputDataset BuildInputDataset(const DatasetBuffers &buffers) {
    InputDataset dataset;
    dataset.vectors = buffers.vectors.data();
    dataset.doc_bucket_ids = buffers.bucket_ids.data();
    dataset.tag_bitmaps = buffers.bitmaps.data();
    dataset.bucket_centroids = buffers.centroids.data();
    return dataset;
}

} // namespace DataReader
