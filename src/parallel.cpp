#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "utils/RunSupport.h"
#include "Clustering/run_clustering.h"
#include "DataBaseCPU/DataBaseCPU.h"
#include "utils/DataReader.h"
#include "Query/Query.h"
#include "Schedule/Scheduler.h"
#include "Schedule/WorkerGroup.h"
#include "Schedule/ThreadUtils.h"

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

bool CollectParallelLatencyMetrics(const std::vector<std::unique_ptr<Query>> &queries,
                                   std::vector<double> &total_end_to_end_ms,
                                   std::vector<double> &bucket_level_ivf_ms,
                                   std::vector<double> &candidate_bucket_merge_ms,
                                   std::vector<double> &npu_async_launch_ms,
                                   std::vector<double> &inbucket_attr_filter_overlapped_ms,
                                   std::vector<double> &wait_npu_flag_ms,
                                   std::vector<double> &result_collection_ms,
                                   std::vector<double> &final_merge_ms,
                                   std::vector<size_t> &process_round_count_level_1,
                                   std::vector<size_t> &process_round_count_level_2)
{
    const size_t n = queries.size();
    total_end_to_end_ms.resize(n);
    bucket_level_ivf_ms.resize(n);
    candidate_bucket_merge_ms.resize(n);
    npu_async_launch_ms.resize(n);
    inbucket_attr_filter_overlapped_ms.resize(n);
    wait_npu_flag_ms.resize(n);
    result_collection_ms.resize(n);
    final_merge_ms.resize(n);
    process_round_count_level_1.resize(n);
    process_round_count_level_2.resize(n);

    for (size_t i = 0; i < n; ++i)
    {
        const Query &query = *queries[i];
        if (query.start_time_for_parallel_record == std::chrono::steady_clock::time_point{})
        {
            std::cerr << "[Parallel] Missing start_time_for_parallel_record for query line "
                      << query.query_id << "\n";
            return false;
        }
        if (query.end_time_for_parallel_record == std::chrono::steady_clock::time_point{})
        {
            std::cerr << "[Parallel] Missing end_time_for_parallel_record for query line "
                      << query.query_id << "\n";
            return false;
        }
        if (query.end_time_for_parallel_record < query.start_time_for_parallel_record)
        {
            std::cerr << "[Parallel] Invalid parallel timing order for query line "
                      << query.query_id << "\n";
            return false;
        }

        total_end_to_end_ms[i] =
            std::chrono::duration<double, std::milli>(query.end_time_for_parallel_record -
                                                      query.start_time_for_parallel_record)
                .count();
        bucket_level_ivf_ms[i] = query.timing_metrics.bucket_level_ivf_ms;
        candidate_bucket_merge_ms[i] = query.timing_metrics.candidate_bucket_merge_ms;
        npu_async_launch_ms[i] = query.timing_metrics.npu_async_launch_ms;
        inbucket_attr_filter_overlapped_ms[i] = query.timing_metrics.inbucket_attr_filter_overlapped_ms;
        wait_npu_flag_ms[i] = query.timing_metrics.wait_npu_flag_ms;
        result_collection_ms[i] = query.timing_metrics.result_collection_ms;
        final_merge_ms[i] = query.timing_metrics.final_merge_ms;
        process_round_count_level_1[i] = static_cast<size_t>(query.process_round_count_level_1);
        process_round_count_level_2[i] = static_cast<size_t>(query.process_round_count_level_2);
    }

    return true;
}

bool CollectParallelRecallMetrics(const InputDataset &dataset,
                                  const std::vector<DataReader::PreparedQuery> &prepared_queries,
                                  std::vector<std::unique_ptr<Query>> &queries,
                                  DataReader::GroundTruthCache *ground_truth_cache,
                                  std::vector<double> &recall_rate_percent)
{
    if (queries.size() != prepared_queries.size())
    {
        std::cerr << "[Parallel] Query count mismatch when collecting recall metrics. prepared="
                  << prepared_queries.size() << ", runtime=" << queries.size() << "\n";
        return false;
    }

    recall_rate_percent.resize(queries.size());
    for (size_t i = 0; i < queries.size(); ++i)
    {
        Query &query = *queries[i];
        const auto &prepared = prepared_queries[i];

        std::vector<DataReader::GroundTruthResultItem> cached_ground_truth;
        if (ground_truth_cache &&
            ground_truth_cache->Lookup(prepared.query_vec,
                                       prepared.filter_expr,
                                       prepared.top_k,
                                       cached_ground_truth))
        {
            query.ground_truth_results = RunSupport::ConvertCacheItemsToQueryResults(cached_ground_truth);
        }
        else
        {
            query.brute_force_search(dataset);
            if (ground_truth_cache &&
                !ground_truth_cache->Store(prepared.query_vec,
                                           prepared.filter_expr,
                                           prepared.top_k,
                                           RunSupport::ConvertQueryResultsToCacheItems(query.ground_truth_results)))
            {
                std::cerr << "[Warn] Failed to append ground-truth cache entry for query line "
                          << prepared.line_no << ".\n";
            }
        }

        query.calculate_recall();
        recall_rate_percent[i] = static_cast<double>(query.recall_rate) * 100.0;
    }

    return true;
}

bool WriteProcessRoundCountFile(const fs::path &output_file,
                                const std::vector<size_t> &level_1_counts,
                                const std::vector<size_t> &level_2_counts)
{
    if (level_1_counts.size() != level_2_counts.size())
    {
        return false;
    }

    std::ofstream out(output_file, std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
        return false;
    }

    out << "process_round_count_level_1\tprocess_round_count_level_2\n";
    for (size_t i = 0; i < level_1_counts.size(); ++i)
    {
        out << level_1_counts[i] << '\t' << level_2_counts[i] << '\n';
    }

    return out.good();
}

bool FlushParallelOutputs(const fs::path &result_root,
                          const std::vector<double> &total_end_to_end_ms,
                          const std::vector<double> &recall_rate_percent,
                          const std::vector<double> &bucket_level_ivf_ms,
                          const std::vector<double> &candidate_bucket_merge_ms,
                          const std::vector<double> &npu_async_launch_ms,
                          const std::vector<double> &inbucket_attr_filter_overlapped_ms,
                          const std::vector<double> &wait_npu_flag_ms,
                          const std::vector<double> &result_collection_ms,
                          const std::vector<double> &final_merge_ms,
                          const std::vector<size_t> &process_round_count_level_1,
                          const std::vector<size_t> &process_round_count_level_2,
                          double overall_qps)
{
    const fs::path recall_dir = result_root / "recall";
    const fs::path latency_dir = result_root / "latency";
    const fs::path log_dir = result_root / "log";
    const std::vector<double> qps_values{overall_qps};
    return RunSupport::EnsureOutputDirectory(recall_dir) &&
           RunSupport::EnsureOutputDirectory(latency_dir) &&
           RunSupport::EnsureOutputDirectory(log_dir) &&
           RunSupport::WriteMetricFile(recall_dir, "recall_rate_percent.txt", recall_rate_percent) &&
           RunSupport::WriteMetricFile(latency_dir, "total_end_to_end_ms.txt", total_end_to_end_ms) &&
           RunSupport::WriteMetricFile(latency_dir, "bucket_level_ivf_ms.txt", bucket_level_ivf_ms) &&
           RunSupport::WriteMetricFile(latency_dir, "candidate_bucket_merge_ms.txt", candidate_bucket_merge_ms) &&
           RunSupport::WriteMetricFile(latency_dir, "npu_async_launch_ms.txt", npu_async_launch_ms) &&
           RunSupport::WriteMetricFile(latency_dir, "inbucket_attr_filter_overlapped_ms.txt",
                                       inbucket_attr_filter_overlapped_ms) &&
           RunSupport::WriteMetricFile(latency_dir, "wait_npu_flag_ms.txt", wait_npu_flag_ms) &&
           RunSupport::WriteMetricFile(latency_dir, "result_collection_ms.txt", result_collection_ms) &&
           RunSupport::WriteMetricFile(latency_dir, "final_merge_ms.txt", final_merge_ms) &&
           WriteProcessRoundCountFile(log_dir / "process_round_count.txt",
                                      process_round_count_level_1,
                                      process_round_count_level_2) &&
           RunSupport::WriteMetricFile(log_dir, "QPS.txt", qps_values);
}

void PrintParallelProgress(size_t completed_count,
                           size_t total_count)
{
    std::cout << "[Progress] " << completed_count << "/" << total_count << std::endl;
}
} // namespace

int main()
{
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
    const double target_qps = 10000.0;
    
    const fs::path config_path = RunSupport::ResolveConfigPath();
    std::cout << "[System] Loading config from " << config_path << "...\n";
    if (!LoadParams(config_path.string(), ResourceConfigProfile::Parallel))
    {
        std::cerr << "[Error] Failed to load config: " << config_path << "\n";
        return -1;
    }
    const fs::path query_result_root_dir =
        query_result_root.empty() ? fs::path(".") : fs::path(query_result_root);
    const fs::path memory_log_file = query_result_root_dir / "log" / "memory_logs.txt";

    DataReader::DatasetBuffers dataset_buffers;
    if (!DataReader::LoadDatasetCache(kDatasetFile, dataset_buffers))
    {
        std::cerr << "[Fatal] Failed to load dataset cache: " << kDatasetFile << "\n";
        return -1;
    }
    std::cout << "[Loader] Dataset loaded. Docs=" << total_doc_num
              << ", Dim=" << vector_dim
              << ", Tags=" << total_tag_num
              << ", Buckets(Level1)=" << total_bucket_num_level_1 << "\n";

    std::vector<DataReader::PreparedQuery> prepared_queries;
    std::string loaded_query_path;
    if (!LoadPreparedQueriesFromFvec(kQueryPaths,
                                     vector_dim,
                                     kDefaultTopK,
                                     prepared_queries,
                                     loaded_query_path))
    {
        std::cerr << "[Fatal] Failed to read queries from deep1B_queries.fvecs\n";
        return -1;
    }
    std::cout << "[Loader] Queries loaded from " << loaded_query_path << "\n";

    if (prepared_queries.empty())
    {
        std::cout << "[Info] No valid query found in " << loaded_query_path << ". Exit with code 0.\n";
        return 0;
    }

    std::cout << "[Loader] Prepared query count: " << prepared_queries.size() << "\n";

    // Step 1: Ensure queries vector has exactly QueryNum items
    if (prepared_queries.size() > QueryNum) {
        prepared_queries.resize(QueryNum);
        std::cout << "[Loader] Truncated queries: " << prepared_queries.size() << "\n";
    } else if (prepared_queries.size() < QueryNum) {
        const size_t original_count = prepared_queries.size();
        prepared_queries.reserve(QueryNum);
        for (size_t i = original_count; i < QueryNum; ++i) {
            prepared_queries.push_back(prepared_queries[i % original_count]);
        }
        std::cout << "[Loader] Expanded queries: " << original_count << " -> " << prepared_queries.size() << "\n";
    }

    // Step 2: Load filter expressions: read 10 from file, replicate to QueryNum
    {
        std::vector<std::string> filter_exprs_10;
        const fs::path filter_expr_path = config_path.has_parent_path()
                                              ? (config_path.parent_path() / "filter_expr_600.txt")
                                              : fs::path("filter_expr_600.txt");
        {
            std::ifstream fexpr_file(filter_expr_path);
            std::string line;
            while (std::getline(fexpr_file, line)) {
                if (!line.empty()) {
                    filter_exprs_10.push_back(line);
                }
            }
        }
        if (filter_exprs_10.empty()) {
            std::cerr << "[Warn] filter_expr.txt not found or empty at " << filter_expr_path
                      << ". Using empty filters for all queries.\n";
        } else if (filter_exprs_10.size() < 10) {
            std::cerr << "[Warn] filter_expr.txt has fewer than 10 lines. Using available expressions.\n";
        }

        std::vector<std::string> filter_exprs_N;
        filter_exprs_N.reserve(QueryNum);
        if (!filter_exprs_10.empty()) {
            std::mt19937 rng(42);
            std::uniform_int_distribution<size_t> dist(0, filter_exprs_10.size() - 1);
            for (size_t i = 0; i < QueryNum; ++i) {
                filter_exprs_N.push_back(filter_exprs_10[dist(rng)]);
            }
        } else {
            filter_exprs_N.assign(QueryNum, std::string());
        }

        for (size_t i = 0; i < prepared_queries.size(); ++i) {
            prepared_queries[i].filter_expr = filter_exprs_N[i];
            // prepared_queries[i].filter_expr = "";
        }
        std::cout << "[Loader] Assigned filter expressions to all " << prepared_queries.size() << " queries\n";
    }

    if (!RunSupport::ValidatePreparedQueriesAgainstPrealloc(prepared_queries))
    {
        return -1;
    }
    FilterExpCompiler::WarmUpThreadLocalBuffers();

    MemoryEventLogCollector memory_log_collector(memory_events_log_enable != 0);
    if (memory_events_log_enable != 0)
    {
        memory_log_collector.Reserve(prepared_queries.size() * 8);
    }

    std::unique_ptr<DataReader::GroundTruthCache> ground_truth_cache;
    if (!ground_truth_cache_file.empty())
    {
        std::cout << "[Loader] Loading ground-truth cache...\n";
        ground_truth_cache = std::make_unique<DataReader::GroundTruthCache>(ground_truth_cache_file);
        if (!ground_truth_cache->Load())
        {
            std::cerr << "[Warn] Failed to load ground-truth cache from "
                      << ground_truth_cache->path()
                      << ". Continue without cache.\n";
            ground_truth_cache.reset();
        }
        else
        {
            std::cout << "[Loader] Ground-truth cache loaded from "
                      << ground_truth_cache->path()
                      << ". entries=" << ground_truth_cache->EntryCount() << "\n";
        }
    }
    else
    {
        std::cout << "[Loader] Ground-truth cache disabled by config.\n";
    }

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

    const size_t clustered_bucket_count = bucket_doc_offsets.size() - 1;
    if (clustered_bucket_count != static_cast<size_t>(total_bucket_num_level_1))
    {
        std::cerr << "[Fatal] Invalid clustering output: config total_bucket_num_level_1="
                  << total_bucket_num_level_1
                  << ", but clustering produced "
                  << clustered_bucket_count
                  << " level-1 buckets.\n";
        return -1;
    }

    if (dataset_buffers.centroids.size() !=
        static_cast<size_t>(total_bucket_num_level_1) * static_cast<size_t>(vector_dim))
    {
        std::cerr << "[Fatal] Invalid clustering output: centroids size mismatch.\n";
        return -1;
    }

    InputDataset dataset = DataReader::BuildInputDataset(dataset_buffers);
    BucketDocTable level_1_bucket_doc_table;
    try
    {
        level_1_bucket_doc_table =
            RunSupport::BuildBucketDocTableFromOffsets(bucket_doc_offsets, bucket_doc_ids);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Fatal] Failed to build BucketDocTable from clustering output: "
                  << e.what() << "\n";
        return -1;
    }

    TwoLevelBucketLayout bucket_layout;
    try
    {
        bucket_layout = RunSupport::BuildTwoLevelBucketLayout(std::move(level_1_bucket_doc_table),
                                                              max_doc_per_bucket_level_2);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Fatal] Failed to adapt clustering output into two-level buckets: "
                  << e.what() << "\n";
        return -1;
    }

    total_bucket_num_level_2 = static_cast<int>(bucket_layout.Level2BucketCount());
    if (!FinalizePreallocationParams())
    {
        return -1;
    }

    std::cout << "[Loader] Two-level bucket layout built. StoredDocRefs="
              << RunSupport::CountStoredDocRefs(bucket_layout.level_1_bucket_doc_table)
              << ", Level1BucketCount=" << bucket_layout.Level1BucketCount()
              << ", Level2BucketCount=" << bucket_layout.Level2BucketCount() << "\n";

    std::cout << "[System] Initializing DataBaseCPU...\n";
    DataBaseCPU db(dataset, std::move(bucket_layout));

    const size_t scheduler_capacity = prepared_queries.size() + 10000;
    std::cout << "[System] Initializing Scheduler. capacity=" << scheduler_capacity << "\n";
    Scheduler scheduler(scheduler_capacity);

    std::cout << "[System] Launching Worker Threads...\n";

    std::cout << "[NUMA] Using default sequential cores 0.." << (cpu_core_count - 1) << "\n";

    {
        cpu_set_t main_cpuset;
        CPU_ZERO(&main_cpuset);
        CPU_SET(48, &main_cpuset);
        int rc = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &main_cpuset);
        if (rc != 0)
        {
            std::cerr << "[NUMA] Warning: failed to bind main thread to core 48: " << rc << "\n";
        }
        else
        {
            std::cout << "[NUMA] Main thread pinned to core 48 (outside worker range 0-47)\n";
        }
    }

    std::vector<std::unique_ptr<WorkerGroup>> groups;
    std::vector<std::thread> threads;
    groups.reserve(group_count);
    threads.reserve(cpu_core_count);

    for (int i = 0; i < group_count; ++i)
    {
        groups.push_back(std::make_unique<WorkerGroup>(i, &db, &scheduler));
    }
    for (int i = 0; i < cpu_core_count; ++i)
    {
        const int gid = GetGroupId(i);
        threads.emplace_back(&WorkerGroup::Run, groups[gid].get(), i);
    }

    std::vector<std::unique_ptr<Query>> queries;
    queries.reserve(prepared_queries.size());
    std::vector<Query *> query_ptrs;
    query_ptrs.reserve(prepared_queries.size());
    for (const auto &prepared : prepared_queries)
    {
        auto query = std::make_unique<Query>();
        query->PrepareMemoryEventSession(prepared.line_no, memory_events_log_enable != 0);
        {
            ScopedMemoryEventSession memory_event_scope(query->memory_event_session());
            query->Reset(prepared.query_vec, prepared.filter_expr, prepared.top_k, prepared.line_no);
        }
        query_ptrs.push_back(query.get());
        queries.push_back(std::move(query));
    }

    std::cout << std::fixed << std::setprecision(5);
    std::cout << "[ParallelTest] Query objects prepared. count=" << query_ptrs.size() << "\n";
    std::cout << "[ParallelTest] Batch submit begins. target_qps=10000\n";

    const auto batch_start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < query_ptrs.size(); ++i)
    {
        const auto scheduled_submit_time =
            batch_start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                              std::chrono::duration<double>(static_cast<double>(i) / target_qps));
        std::this_thread::sleep_until(scheduled_submit_time);
        scheduler.Push(query_ptrs[i]);
    }

    size_t completed_count = 0;
    auto next_progress_report = batch_start + std::chrono::seconds(1);
    while (completed_count < query_ptrs.size())
    {
        scheduler.PopResult();  // 阻塞等待，无 CPU 轮询
        ++completed_count;

        const auto now = std::chrono::steady_clock::now();
        if (now >= next_progress_report)
        {
            PrintParallelProgress(completed_count, query_ptrs.size());
            do
            {
                next_progress_report += std::chrono::seconds(1);
            } while (now >= next_progress_report);
        }
    }
    const auto batch_end = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> total_latency_ms = batch_end - batch_start;
    const double total_latency_seconds = total_latency_ms.count() / 1000.0;
    const double overall_qps = (total_latency_seconds > 0.0)
                                   ? (static_cast<double>(query_ptrs.size()) / total_latency_seconds)
                                   : 0.0;
    const double avg_query_latency_ms = query_ptrs.empty()
                                            ? 0.0
                                            : (total_latency_ms.count() / static_cast<double>(query_ptrs.size()));

    std::cout << "\n========================================================\n";
    std::cout << "                Parallel Batch Test Result              \n";
    std::cout << "========================================================\n";
    std::cout << "Query Count                    : " << query_ptrs.size() << "\n";
    std::cout << "Completed Query Count          : " << completed_count << "\n";
    std::cout << "Target QPS                     : " << target_qps << "\n";
    std::cout << "Total Batch Latency            : " << total_latency_ms.count() << " ms\n";
    std::cout << "Overall QPS                    : " << overall_qps << "\n";
    std::cout << "Average Query Latency(Batch/N) : " << avg_query_latency_ms << " ms\n";
    std::cout << "========================================================\n";

    std::cout << "[System] All queries completed. Shutting down worker threads...\n";
    for (auto &group : groups)
    {
        group->Stop();
    }
    for (auto &thread : threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }

    for (const auto &query : queries)
    {
        query->AppendMemoryEventLogs(memory_log_collector);
    }

    std::vector<double> total_end_to_end_ms;
    std::vector<double> bucket_level_ivf_ms;
    std::vector<double> candidate_bucket_merge_ms;
    std::vector<double> npu_async_launch_ms;
    std::vector<double> inbucket_attr_filter_overlapped_ms;
    std::vector<double> wait_npu_flag_ms;
    std::vector<double> result_collection_ms;
    std::vector<double> final_merge_ms;
    std::vector<size_t> process_round_count_level_1;
    std::vector<size_t> process_round_count_level_2;
    if (!CollectParallelLatencyMetrics(queries,
                                       total_end_to_end_ms,
                                       bucket_level_ivf_ms,
                                       candidate_bucket_merge_ms,
                                       npu_async_launch_ms,
                                       inbucket_attr_filter_overlapped_ms,
                                       wait_npu_flag_ms,
                                       result_collection_ms,
                                       final_merge_ms,
                                       process_round_count_level_1,
                                       process_round_count_level_2))
    {
        return -1;
    }

    std::vector<double> recall_rate_percent;
    if (!CollectParallelRecallMetrics(dataset,
                                      prepared_queries,
                                      queries,
                                      ground_truth_cache.get(),
                                      recall_rate_percent))
    {
        return -1;
    }

    if (!FlushParallelOutputs(query_result_root_dir,
                              total_end_to_end_ms,
                              recall_rate_percent,
                              bucket_level_ivf_ms,
                              candidate_bucket_merge_ms,
                              npu_async_launch_ms,
                              inbucket_attr_filter_overlapped_ms,
                              wait_npu_flag_ms,
                              result_collection_ms,
                              final_merge_ms,
                              process_round_count_level_1,
                              process_round_count_level_2,
                              overall_qps))
    {
        std::cerr << "[Fatal] Failed to write parallel outputs.\n";
        return -1;
    }
    if (!memory_log_collector.Flush(memory_log_file))
    {
        std::cerr << "[Fatal] Failed to write " << memory_log_file << ".\n";
        return -1;
    }

    std::cout << "[System] All threads stopped. Exiting safely.\n";
    return 0;
}
