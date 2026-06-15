#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "utils/RunSupport.h"
#include "utils/DataReader.h"
#include "DataBaseCPU/DataBaseCPU.h"
#include "Query/Query.h"
#include "Schedule/Scheduler.h"
#include "Schedule/WorkerGroup.h"
#include "Clustering/run_clustering.h"

namespace fs = RunSupport::fs;

namespace
{
bool LoadPreparedQueriesFromFvec(const std::vector<std::string> &query_paths,
                                 int expected_dim,
                                 int default_top_k,
                                 std::vector<DataReader::PreparedQuery> &queries,
                                 std::string &loaded_path)
{
    queries.clear();
    loaded_path.clear();

    if (expected_dim < 64 || default_top_k <= 0)
    {
        return false;
    }

    auto append_query = [&](const float *raw, int source_dim, size_t line_no) {
        DataReader::PreparedQuery q;
        q.query_vec.assign(static_cast<size_t>(expected_dim), 0.0f);
        const int copy_dim = std::min<int>(64, std::min<int>(expected_dim, source_dim));
        for (int d = 0; d < copy_dim; ++d)
        {
            q.query_vec[static_cast<size_t>(d)] = raw[static_cast<size_t>(d)];
        }
        q.filter_expr = "";
        q.top_k = default_top_k;
        q.line_no = line_no;
        queries.push_back(std::move(q));
    };

    for (const auto &path : query_paths)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
        {
            continue;
        }

        in.seekg(0, std::ios::end);
        const std::streamoff file_size = in.tellg();
        in.seekg(0, std::ios::beg);
        if (file_size < static_cast<std::streamoff>(sizeof(int32_t) * 2))
        {
            continue;
        }

        int32_t first = 0;
        int32_t second = 0;
        in.read(reinterpret_cast<char *>(&first), sizeof(int32_t));
        in.read(reinterpret_cast<char *>(&second), sizeof(int32_t));
        if (!in)
        {
            continue;
        }

        const std::streamoff headered_bytes = static_cast<std::streamoff>(sizeof(int32_t) * 2) +
                                              static_cast<std::streamoff>(first) * static_cast<std::streamoff>(second) *
                                                  static_cast<std::streamoff>(sizeof(float));

        if (first > 0 && second >= 64 && file_size == headered_bytes)
        {
            const int32_t rows = first;
            const int32_t source_dim = second;
            std::vector<float> raw(static_cast<size_t>(source_dim));
            queries.clear();
            queries.reserve(static_cast<size_t>(rows));
            for (int32_t i = 0; i < rows; ++i)
            {
                in.read(reinterpret_cast<char *>(raw.data()),
                        static_cast<std::streamsize>(raw.size() * sizeof(float)));
                if (!in)
                {
                    queries.clear();
                    break;
                }
                append_query(raw.data(), source_dim, static_cast<size_t>(i + 1));
            }
            if (!queries.empty())
            {
                loaded_path = path;
                return true;
            }
            continue;
        }

        in.clear();
        in.seekg(0, std::ios::beg);
        queries.clear();
        size_t line_no = 0;
        while (true)
        {
            int32_t dim = 0;
            in.read(reinterpret_cast<char *>(&dim), sizeof(int32_t));
            if (!in)
            {
                break;
            }
            if (dim < 64)
            {
                queries.clear();
                break;
            }

            std::vector<float> raw(static_cast<size_t>(dim));
            in.read(reinterpret_cast<char *>(raw.data()),
                    static_cast<std::streamsize>(raw.size() * sizeof(float)));
            if (!in)
            {
                queries.clear();
                break;
            }

            ++line_no;
            append_query(raw.data(), dim, line_no);
        }

        if (!queries.empty())
        {
            loaded_path = path;
            return true;
        }
    }

    return false;
}

bool WriteProcessRoundCountFile(const fs::path &output_file,
                                const std::vector<size_t> &round_counts,
                                const std::vector<size_t> &level_2_counts)
{
    if (round_counts.size() != level_2_counts.size())
    {
        return false;
    }

    std::ofstream out(output_file, std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
        return false;
    }

    out << "process_round_count\tprocess_round_count_level_2\n";
    for (size_t i = 0; i < round_counts.size(); ++i)
    {
        out << round_counts[i] << '\t' << level_2_counts[i] << '\n';
    }

    return out.good();
}

bool WriteSelectivityFile(const fs::path &output_file,
                          const std::vector<std::vector<double>> &per_query_selectivity)
{
    std::ofstream out(output_file, std::ios::out | std::ios::trunc);
    if (!out.is_open()) return false;
    out << std::fixed << std::setprecision(6);
    for (const auto &sels : per_query_selectivity)
    {
        for (size_t j = 0; j < sels.size(); ++j)
        {
            if (j > 0) out << ' ';
            out << sels[j];
        }
        out << '\n';
    }
    return out.good();
}

bool WriteL2BucketCountFile(const fs::path &output_file,
                             const std::vector<size_t> &nonzero_counts,
                             const std::vector<size_t> &allzero_counts)
{
    std::ofstream out(output_file, std::ios::out | std::ios::trunc);
    if (!out.is_open()) return false;
    out << "l2_nonzero_mask\tl2_allzero_mask\n";
    for (size_t i = 0; i < nonzero_counts.size(); ++i)
    {
        out << nonzero_counts[i] << '\t' << allzero_counts[i] << '\n';
    }
    return out.good();
}

class QueryPool
{
public:
    explicit QueryPool(size_t capacity)
    {
        objects_.reserve(capacity);
        free_list_.reserve(capacity);
        for (size_t i = 0; i < capacity; ++i)
        {
            objects_.push_back(std::make_unique<Query>());
            free_list_.push_back(objects_.back().get());
        }
    }

    Query *Acquire()
    {
        if (free_list_.empty())
        {
            return nullptr;
        }
        Query *q = free_list_.back();
        free_list_.pop_back();
        return q;
    }

    void Release(Query *q)
    {
        if (q != nullptr)
        {
            free_list_.push_back(q);
        }
    }

private:
    std::vector<std::unique_ptr<Query>> objects_;
    std::vector<Query *> free_list_;
};

struct DeferredMetricFiles
{
    explicit DeferredMetricFiles(size_t expected_queries)
    {
        Resize(expected_queries);
    }

    void Resize(size_t expected_queries)
    {
        query_construction_ms.resize(expected_queries);
        total_end_to_end_ms.resize(expected_queries);
        ground_truth_count.resize(expected_queries);
        engine_result_count.resize(expected_queries);
        process_round_count.resize(expected_queries);
        process_round_count_level_2.resize(expected_queries);
        searched_l2_bucket_count.resize(expected_queries);
        searched_l2_nonzero_count.resize(expected_queries);
        searched_l2_allzero_count.resize(expected_queries);
        bucket_selectivity_per_query.resize(expected_queries);
        recall_rate_percent.resize(expected_queries);
        bucket_level_ivf_ms.resize(expected_queries);
        bucket_level_ivf_filter_eval_ms.resize(expected_queries);
        bucket_level_ivf_centroid_score_ms.resize(expected_queries);
        bucket_level_ivf_result_pack_ms.resize(expected_queries);
        candidate_bucket_merge_ms.resize(expected_queries);
        npu_async_launch_ms.resize(expected_queries);
        npu_submit_ms.resize(expected_queries);
        inbucket_attr_filter_overlapped_ms.resize(expected_queries);
        wait_npu_flag_ms.resize(expected_queries);
        result_collection_ms.resize(expected_queries);
        final_merge_ms.resize(expected_queries);
        npu_mask_filter_launch_ms.resize(expected_queries);
        npu_mask_filter_h2d_ms.resize(expected_queries);
        npu_mask_filter_kernel_exec_ms.resize(expected_queries);
        npu_mask_filter_d2h_ms.resize(expected_queries);
    }

    void Record(size_t query_index,
                const Query &query,
                double construct_ms,
                double total_ms)
    {
        query_construction_ms[query_index] = construct_ms;
        total_end_to_end_ms[query_index] = total_ms;
        process_round_count[query_index] =
            static_cast<size_t>(query.process_round_count_l2);
        process_round_count_level_2[query_index] =
            static_cast<size_t>(query.process_round_count_level_2);
        searched_l2_bucket_count[query_index] =
            static_cast<size_t>(query.searched_bucket_count_level_2);
        searched_l2_nonzero_count[query_index] =
            static_cast<size_t>(query.searched_l2_count_nonzero.load());
        searched_l2_allzero_count[query_index] =
            static_cast<size_t>(query.searched_l2_count_allzero.load());
        {
            size_t sel_count = query.bucket_selectivity_idx_.load(std::memory_order_relaxed);
            bucket_selectivity_per_query[query_index].assign(
                query.bucket_selectivity_.begin(),
                query.bucket_selectivity_.begin() + std::min(sel_count, query.bucket_selectivity_.size()));
        }
        bucket_level_ivf_ms[query_index] = query.timing_metrics.bucket_level_ivf_ms;
        bucket_level_ivf_filter_eval_ms[query_index] = query.timing_metrics.bucket_level_ivf_filter_eval_ms;
        bucket_level_ivf_centroid_score_ms[query_index] = query.timing_metrics.bucket_level_ivf_centroid_score_ms;
        bucket_level_ivf_result_pack_ms[query_index] = query.timing_metrics.bucket_level_ivf_result_pack_ms;
        candidate_bucket_merge_ms[query_index] = query.timing_metrics.candidate_bucket_merge_ms;
        npu_async_launch_ms[query_index] = query.timing_metrics.npu_async_launch_ms;
        npu_submit_ms[query_index] = query.timing_metrics.npu_submit_ms;
        inbucket_attr_filter_overlapped_ms[query_index] = query.timing_metrics.inbucket_attr_filter_overlapped_ms;
        wait_npu_flag_ms[query_index] = query.timing_metrics.wait_npu_flag_ms;
        result_collection_ms[query_index] = query.timing_metrics.result_collection_ms;
        final_merge_ms[query_index] = query.timing_metrics.final_merge_ms;
        npu_mask_filter_launch_ms[query_index] = query.timing_metrics.npu_mask_filter_launch_ms;
        npu_mask_filter_h2d_ms[query_index] = query.timing_metrics.npu_mask_filter_h2d_ms;
        npu_mask_filter_kernel_exec_ms[query_index] = query.timing_metrics.npu_mask_filter_kernel_exec_ms;
        npu_mask_filter_d2h_ms[query_index] = query.timing_metrics.npu_mask_filter_d2h_ms;
    }

    void RecordRecall(size_t query_index, const Query &query)
    {
        ground_truth_count[query_index] = query.ground_truth_results.size();
        engine_result_count[query_index] = query.result.topk_results.size();
        recall_rate_percent[query_index] = static_cast<double>(query.recall_rate) * 100.0;
    }

    bool Flush(const fs::path &result_root) const
    {
        const fs::path recall_dir = result_root / "recall";
        const fs::path latency_dir = result_root / "latency";
        const fs::path log_dir = result_root / "log";
        if (!RunSupport::EnsureOutputDirectory(recall_dir) ||
            !RunSupport::EnsureOutputDirectory(latency_dir) ||
            !RunSupport::EnsureOutputDirectory(log_dir))
        {
            return false;
        }

        auto f_recall = std::async(std::launch::async, [&] {
            return RunSupport::WriteMetricFile(recall_dir, "recall_rate_percent.txt", recall_rate_percent) &&
                   RunSupport::WriteMetricFile(recall_dir, "ground_truth_count.txt", ground_truth_count) &&
                   RunSupport::WriteMetricFile(recall_dir, "engine_result_count.txt", engine_result_count);
        });
        auto f_latency = std::async(std::launch::async, [&] {
            return RunSupport::WriteMetricFile(latency_dir, "total_end_to_end_ms.txt", total_end_to_end_ms) &&
                   RunSupport::WriteMetricFile(latency_dir, "query_construction_ms.txt", query_construction_ms) &&
                   RunSupport::WriteMetricFile(latency_dir, "bucket_level_ivf_ms.txt", bucket_level_ivf_ms) &&
                   RunSupport::WriteMetricFile(latency_dir, "bucket_level_ivf_filter_eval_ms.txt", bucket_level_ivf_filter_eval_ms) &&
                   RunSupport::WriteMetricFile(latency_dir, "bucket_level_ivf_centroid_score_ms.txt", bucket_level_ivf_centroid_score_ms) &&
                   RunSupport::WriteMetricFile(latency_dir, "bucket_level_ivf_result_pack_ms.txt", bucket_level_ivf_result_pack_ms) &&
                   RunSupport::WriteMetricFile(latency_dir, "candidate_bucket_merge_ms.txt", candidate_bucket_merge_ms) &&
                   RunSupport::WriteMetricFile(latency_dir, "npu_async_launch_ms.txt", npu_async_launch_ms) &&
                   RunSupport::WriteMetricFile(latency_dir, "npu_submit_ms.txt", npu_submit_ms) &&
                   RunSupport::WriteMetricFile(latency_dir, "inbucket_attr_filter_overlapped_ms.txt",
                                               inbucket_attr_filter_overlapped_ms) &&
                   RunSupport::WriteMetricFile(latency_dir, "wait_npu_flag_ms.txt", wait_npu_flag_ms) &&
                   RunSupport::WriteMetricFile(latency_dir, "result_collection_ms.txt", result_collection_ms) &&
                   RunSupport::WriteMetricFile(latency_dir, "final_merge_ms.txt", final_merge_ms) &&
                   RunSupport::WriteMetricFile(latency_dir, "npu_mask_filter_launch_ms.txt", npu_mask_filter_launch_ms) &&
                   RunSupport::WriteMetricFile(latency_dir, "npu_mask_filter_h2d_ms.txt", npu_mask_filter_h2d_ms) &&
                   RunSupport::WriteMetricFile(latency_dir, "npu_mask_filter_kernel_exec_ms.txt", npu_mask_filter_kernel_exec_ms) &&
                   RunSupport::WriteMetricFile(latency_dir, "npu_mask_filter_d2h_ms.txt", npu_mask_filter_d2h_ms);
        });
        auto f_log = std::async(std::launch::async, [&] {
            return WriteProcessRoundCountFile(log_dir / "process_round_count.txt",
                                              process_round_count,
                                              process_round_count_level_2) &&
                   WriteL2BucketCountFile(log_dir / "searched_l2_bucket_count.txt",
                                          searched_l2_nonzero_count,
                                          searched_l2_allzero_count) &&
                   WriteSelectivityFile(log_dir / "l2_bucket_selectivity.txt", bucket_selectivity_per_query);
        });

        return f_recall.get() && f_latency.get() && f_log.get();
    }

    std::vector<double> query_construction_ms;
    std::vector<double> total_end_to_end_ms;
    std::vector<size_t> ground_truth_count;
    std::vector<size_t> engine_result_count;
    std::vector<size_t> process_round_count;
    std::vector<size_t> process_round_count_level_2;
    std::vector<double> recall_rate_percent;
    std::vector<double> bucket_level_ivf_ms;
    std::vector<double> bucket_level_ivf_filter_eval_ms;
    std::vector<double> bucket_level_ivf_centroid_score_ms;
    std::vector<double> bucket_level_ivf_result_pack_ms;
    std::vector<double> candidate_bucket_merge_ms;
    std::vector<double> npu_async_launch_ms;
    std::vector<double> npu_submit_ms;
    std::vector<double> inbucket_attr_filter_overlapped_ms;
    std::vector<double> wait_npu_flag_ms;
    std::vector<double> result_collection_ms;
    std::vector<double> final_merge_ms;
    std::vector<double> npu_mask_filter_launch_ms;
    std::vector<double> npu_mask_filter_h2d_ms;
    std::vector<double> npu_mask_filter_kernel_exec_ms;
    std::vector<double> npu_mask_filter_d2h_ms;
    std::vector<size_t> searched_l2_bucket_count;
    std::vector<size_t> searched_l2_nonzero_count;
    std::vector<size_t> searched_l2_allzero_count;
    std::vector<std::vector<double>> bucket_selectivity_per_query;
};
} // namespace

int main() {
    constexpr int kDefaultTopK = 100;
    constexpr int QueryNum = 10000;
    constexpr const char *kDatasetFile = "../../dataset_HW.bin";
    const std::vector<std::string> kQueryPaths = {
        "datasets/hw_queries.fvecs",
        "../datasets/hw_queries.fvecs",
        "../../datasets/hw_queries.fvecs",
        "../../../datasets/hw_queries.fvecs"
    };
    // constexpr const char *kDatasetFile = "../../dataset_DEEP.bin";
    // const std::vector<std::string> kQueryPaths = {
    //     "queries/deep1B_queries.fvecs",
    //     "../queries/deep1B_queries.fvecs",
    //     "../../queries/deep1B_queries.fvecs",
    //     "../../../queries/deep1B_queries.fvecs"
    // };

    const fs::path config_path = RunSupport::ResolveConfigPath();
    std::cout << "[System] Loading config from " << config_path << "...\n";
    if (!LoadParams(config_path.string(), ResourceConfigProfile::Serial)) {
        std::cerr << "[Error] Failed to load config: " << config_path << "\n";
        return -1;
    }
    const fs::path query_result_root_dir =
        query_result_root.empty() ? fs::path(".") : fs::path(query_result_root);
    const fs::path memory_log_file = query_result_root_dir / "log" / "memory_logs.txt";

    std::cout << "[Loader] Loading dataset cache...\n";
    DataReader::DatasetBuffers dataset_buffers;
    if (!DataReader::LoadDatasetCache(kDatasetFile, dataset_buffers)) {
        std::cerr << "[Fatal] Failed to load dataset cache: " << kDatasetFile << "\n";
        return -1;
    }
    std::cout << "[Loader] Dataset loaded. Docs=" << total_doc_num
              << ", Dim=" << vector_dim
              << ", Tags=" << total_tag_num
              << ", Buckets(Level1)=" << total_bucket_num_level_1 << "\n";

    std::cout << "[Loader] Loading queries...\n";
    std::vector<DataReader::PreparedQuery> queries;
    std::string loaded_query_path;
    if (!LoadPreparedQueriesFromFvec(kQueryPaths,
                                     vector_dim,
                                     kDefaultTopK,
                                     queries,
                                     loaded_query_path)) {
        std::cerr << "[Fatal] Failed to read queries from deep1B_queries.fvecs\n";
        return -1;
    }
    std::cout << "[Loader] Queries loaded from " << loaded_query_path << "\n";

    if (queries.empty()) {
        std::cout << "[Info] No valid query found in " << loaded_query_path << ".\n";
        return 0;
    }

    // Load filter expressions, pick one for the single query
    {
        std::vector<std::string> filter_exprs;
        const fs::path filter_expr_path = config_path.has_parent_path()
                                              ? (config_path.parent_path() / "filter_expr_600.txt")
                                              : fs::path("filter_expr_600.txt");
        std::ifstream fexpr_file(filter_expr_path);
        std::string line;
        while (std::getline(fexpr_file, line)) {
            if (!line.empty()) {
                filter_exprs.push_back(line);
            }
        }
        if (!filter_exprs.empty()) {
            queries[0].filter_expr = filter_exprs[0];
        } else {
            std::cerr << "[Warn] filter_expr_600.txt not found or empty. Using empty filter.\n";
        }
    }

    // 1-query mode: duplicate first query 10000 times
    {
        DataReader::PreparedQuery first_q = std::move(queries[0]);
        queries.clear();
        queries.resize(10000, first_q);
        for (size_t i = 0; i < queries.size(); ++i) {
            queries[i].line_no = i + 1;
        }
    }
    std::cout << "[Loader] 1-query mode: first query duplicated to " << queries.size() << "\n";

    if (!RunSupport::ValidatePreparedQueriesAgainstPrealloc(queries)) {
        return -1;
    }
    FilterExpCompiler::WarmUpThreadLocalBuffers();

    auto deferred_metric_files = std::make_unique<DeferredMetricFiles>(queries.size());
    MemoryEventLogCollector memory_log_collector(memory_events_log_enable != 0);
    if (memory_events_log_enable != 0) {
        memory_log_collector.Reserve(queries.size() * 8);
    }

    std::unique_ptr<DataReader::GroundTruthCache> ground_truth_cache;
    if (!ground_truth_cache_file.empty()) {
        std::cout << "[Loader] Loading ground-truth cache...\n";
        ground_truth_cache = std::make_unique<DataReader::GroundTruthCache>(ground_truth_cache_file);
        if (!ground_truth_cache->Load()) {
            std::cerr << "[Warn] Failed to load ground-truth cache from "
                      << ground_truth_cache->path()
                      << ". Continue without cache.\n";
            ground_truth_cache.reset();
        } else {
            std::cout << "[Loader] Ground-truth cache loaded from "
                      << ground_truth_cache->path()
                      << ". entries=" << ground_truth_cache->EntryCount() << "\n";
        }
    } else {
        std::cout << "[Loader] Ground-truth cache disabled by config.\n";
    }

    std::cout << "[Clustering] Syncing clustering context...\n";
    std::vector<uint64_t> bucket_doc_offsets;
    std::vector<uint32_t> bucket_doc_ids;
    int sync_result = SyncClusteringContextFromMain(dataset_buffers.vectors,
                                                    dataset_buffers.bitmaps,
                                                    bucket_doc_offsets,
                                                    bucket_doc_ids,
                                                    dataset_buffers.centroids,
                                                    total_bucket_num_level_1,
                                                    total_tag_num,
                                                    vector_dim,
                                                    total_doc_num,
                                                    max_doc_per_bucket_level_2);
    if (sync_result != 0)
    {
        std::cerr << "[Fatal] Failed to run or load clustering outputs (centroids.bin/buckets).\n";
        return -1;
    }

    if (bucket_doc_offsets.size() < 2)
    {
        std::cerr << "[Fatal] Invalid clustering output: bucket_doc_offsets is empty.\n";
        return -1;
    }

    const size_t loaded_l2_count = bucket_doc_offsets.size() - 1;
    total_bucket_num_level_2 = static_cast<int>(loaded_l2_count);

    if (dataset_buffers.centroids.size() != loaded_l2_count * static_cast<size_t>(vector_dim))
    {
        std::cerr << "[Fatal] Invalid clustering output: centroids size mismatch. Expected "
                  << loaded_l2_count * static_cast<size_t>(vector_dim) << " floats, got "
                  << dataset_buffers.centroids.size() << "\n";
        return -1;
    }

    BucketDocTable l2_bucket_doc_table;
    try
    {
        l2_bucket_doc_table =
            RunSupport::BuildBucketDocTableFromOffsets(bucket_doc_offsets, bucket_doc_ids);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Fatal] Failed to build L2 BucketDocTable: " << e.what() << "\n";
        return -1;
    }

    TwoLevelBucketLayout bucket_layout;
    bucket_layout.level_1_bucket_doc_table.resize(1);
    bucket_layout.level_1_bucket_doc_table[0].reserve(loaded_l2_count);
    for (size_t i = 0; i < loaded_l2_count; ++i) {
        for (uint32_t doc_id : l2_bucket_doc_table[i]) {
            bucket_layout.level_1_bucket_doc_table[0].push_back(doc_id);
        }
    }
    bucket_layout.level_2_bucket_doc_table = std::move(l2_bucket_doc_table);
    bucket_layout.level_1_to_level_2_offsets = {0, static_cast<uint32_t>(loaded_l2_count)};

    if (!FinalizePreallocationParams())
    {
        return -1;
    }

    std::cout << "[Loader] L2 bucket layout loaded directly. L2BucketCount="
              << bucket_layout.Level2BucketCount() << "\n";

    InputDataset dataset = DataReader::BuildInputDataset(dataset_buffers);
    if (!g_l2_to_l0_map.empty() && static_cast<int>(g_l2_to_l0_map.size()) == total_bucket_num_level_2)
    {
        dataset.l1_to_l0_map = g_l2_to_l0_map.data();
    }
    else if (!g_l2_to_l0_map.empty())
    {
        std::cerr << "[Warn] l2_to_l0_map size (" << g_l2_to_l0_map.size()
                  << ") != total_bucket_num_level_2 (" << total_bucket_num_level_2
                  << "). L0 scoring disabled.\n";
    }

    std::cout << "[System] Initializing DataBaseCPU...\n";

    // Scope: all NPU-dependent objects must destruct before aclFinalize.
    {
    DataBaseCPU db(dataset, std::move(bucket_layout));

    std::cout << "[System] Launching worker threads...\n";

    Scheduler scheduler;
    std::vector<std::unique_ptr<WorkerGroup>> groups;
    for (int i = 0; i < group_count; ++i) {
        groups.push_back(std::make_unique<WorkerGroup>(i, &db, &scheduler));
    }
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(cpu_core_count));
    for (int i = 0; i < cpu_core_count; ++i) {
        const int gid = GetGroupId(i);
        threads.emplace_back(&WorkerGroup::Run, groups[gid].get(), i);
    }
    QueryPool query_pool(std::max(static_cast<size_t>(query_pool_capacity), queries.size()));

    // Lazy cache: filter expression → pre-computed passing doc IDs
    std::unordered_map<std::string, std::vector<uint32_t>> filter_passing_docs_cache;

    // Warmup: push queries through the pipeline to pre-load NPU kernels,
    // warm HBM caches, and initialize thread-local buffers.
    {
        const size_t warmup_count = 32;
        const auto &proto = queries[0];
        for (size_t i = 0; i < warmup_count; ++i)
        {
            Query *wq = query_pool.Acquire();
            wq->Reset(proto.query_vec, proto.filter_expr, proto.top_k, 0);
            scheduler.Push(wq);
            Query *done = scheduler.PopResult();
            query_pool.Release(done);
        }
        std::cout << "[Warmup] " << warmup_count << " warmup queries completed.\n";
    }

    std::vector<Query*> completed_queries(queries.size(), nullptr);

    for (size_t idx = 0; idx < queries.size(); ++idx) {
        const auto &prepared = queries[idx];

        auto start_construct = std::chrono::high_resolution_clock::now();
        Query *q = query_pool.Acquire();
        if (q == nullptr) {
            std::cerr << "[Fatal] QueryPool exhausted. capacity=" << query_pool_capacity << "\n";
            return -1;
        }
        q->PrepareMemoryEventSession(prepared.line_no, memory_events_log_enable != 0);
        {
            ScopedMemoryEventSession memory_event_scope(q->memory_event_session());
            q->Reset(prepared.query_vec, prepared.filter_expr, prepared.top_k, prepared.line_no);
        }
        auto end_construct = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> construct_ms = end_construct - start_construct;

        auto start_loop = std::chrono::high_resolution_clock::now();
        scheduler.Push(q);

        Query *completed_q = scheduler.PopResult();  // 阻塞等待，无 CPU 浪费

        auto end_loop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> loop_ms = end_loop - start_loop;

        const double total_ms = construct_ms.count() + loop_ms.count();
        deferred_metric_files->Record(idx,
                                      *completed_q,
                                      construct_ms.count(),
                                      total_ms);
        completed_q->AppendMemoryEventLogs(memory_log_collector);

        completed_queries[idx] = completed_q;
        if ((idx + 1) % 100 == 0 || idx + 1 == queries.size()) {
            std::cout << (idx + 1) << "/" << queries.size() << std::endl;
        }
    }

    // Deferred recall: brute-force + calculate_recall after all queries complete
    {
        const size_t n = queries.size();
        std::vector<bool> needs_brute_force(n, false);

        // Phase 1: Check cache
#pragma omp parallel for schedule(dynamic, 8)
        for (size_t idx = 0; idx < n; ++idx) {
            const auto &prepared = queries[idx];
            std::vector<DataReader::GroundTruthResultItem> cached_ground_truth;
            if (ground_truth_cache &&
                ground_truth_cache->Lookup(prepared.query_vec,
                                           prepared.filter_expr,
                                           prepared.top_k,
                                           cached_ground_truth)) {
                completed_queries[idx]->ground_truth_results = RunSupport::ConvertCacheItemsToQueryResults(cached_ground_truth);
            } else {
                needs_brute_force[idx] = true;
            }
        }

        // Phase 2: Compute passing docs per filter expression (OpenMP parallel)
        struct FilterGroup {
            std::vector<uint32_t> passing_docs;
            std::vector<size_t> query_indices;
        };
        std::vector<FilterGroup> groups;
        std::unordered_map<std::string, size_t> expr_to_group;

        for (size_t idx = 0; idx < n; ++idx) {
            if (!needs_brute_force[idx]) continue;
            const auto &expr = queries[idx].filter_expr;
            auto [it, inserted] = expr_to_group.emplace(expr, groups.size());
            if (inserted) {
                groups.emplace_back();
                Query &ref = *completed_queries[idx];
                uint32_t stride = (total_tag_num + 63) / 64;
                auto &passing = groups.back().passing_docs;
#pragma omp parallel
                {
                    std::vector<uint32_t> local;
#pragma omp for schedule(static) nowait
                    for (int doc_id = 0; doc_id < total_doc_num; ++doc_id)
                    {
                        if (ref.evaluate_single_doc_filter(doc_id, dataset.tag_bitmaps, stride))
                        {
                            local.push_back(doc_id);
                        }
                    }
#pragma omp critical
                    {
                        passing.insert(passing.end(), local.begin(), local.end());
                    }
                }
                std::sort(passing.begin(), passing.end());
            }
            groups[it->second].query_indices.push_back(idx);
        }

        // Phase 3: Brute-force search (OpenMP parallel)
        if (!groups.empty()) {
            struct Task { size_t query_idx; size_t group_idx; };
            std::vector<Task> tasks;
            for (size_t g = 0; g < groups.size(); ++g)
                for (size_t qi : groups[g].query_indices)
                    tasks.push_back({qi, g});

#pragma omp parallel for schedule(dynamic, 8)
            for (size_t t = 0; t < tasks.size(); ++t) {
                completed_queries[tasks[t].query_idx]->brute_force_search_filtered(
                    dataset, groups[tasks[t].group_idx].passing_docs);
            }
        }

        // Phase 4: Calculate recall (OpenMP parallel)
#pragma omp parallel for schedule(static)
        for (size_t idx = 0; idx < n; ++idx) {
            completed_queries[idx]->calculate_recall();
        }

        // Phase 4.5: Record recall metrics
        for (size_t idx = 0; idx < n; ++idx) {
            deferred_metric_files->RecordRecall(idx, *completed_queries[idx]);
        }

        // Phase 5: Deferred cache stores
        if (ground_truth_cache) {
            for (size_t idx = 0; idx < n; ++idx) {
                if (!needs_brute_force[idx]) continue;
                const auto &prepared = queries[idx];
                ground_truth_cache->Store(prepared.query_vec,
                                          prepared.filter_expr,
                                          prepared.top_k,
                                          RunSupport::ConvertQueryResultsToCacheItems(
                                              completed_queries[idx]->ground_truth_results));
            }
        }

        // Release all queries back to pool
        for (auto *q : completed_queries) {
            query_pool.Release(q);
        }
    }

    for (auto &g : groups) {
        g->Stop();
    }
    for (auto &t : threads) {
        if (t.joinable()) {
            t.join();
        }
    }

    if (!deferred_metric_files->Flush(query_result_root_dir)) {
        std::cerr << "[Fatal] Failed to write deferred metric txt files.\n";
        return -1;
    }
    if (!memory_log_collector.Flush(memory_log_file)) {
        std::cerr << "[Fatal] Failed to write " << memory_log_file << ".\n";
        return -1;
    }

    } // end NPU scope: groups, scheduler, db destruct here (releases pinned memory)

    std::cout << "[System] NPU scope ended. Calling aclFinalize...\n";
    npuAPI::Finalize();
    return 0;
}
