#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
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
        process_round_count_level_1.resize(expected_queries);
        process_round_count_level_2.resize(expected_queries);
        recall_rate_percent.resize(expected_queries);
        bucket_level_ivf_ms.resize(expected_queries);
        candidate_bucket_merge_ms.resize(expected_queries);
        npu_async_launch_ms.resize(expected_queries);
        inbucket_attr_filter_overlapped_ms.resize(expected_queries);
        wait_npu_flag_ms.resize(expected_queries);
        result_collection_ms.resize(expected_queries);
        final_merge_ms.resize(expected_queries);
    }

    void Record(size_t query_index,
                const Query &query,
                double construct_ms,
                double total_ms)
    {
        query_construction_ms[query_index] = construct_ms;
        total_end_to_end_ms[query_index] = total_ms;
        ground_truth_count[query_index] = query.ground_truth_results.size();
        engine_result_count[query_index] = query.result.topk_results.size();
        process_round_count_level_1[query_index] =
            static_cast<size_t>(query.process_round_count_level_1);
        process_round_count_level_2[query_index] =
            static_cast<size_t>(query.process_round_count_level_2);
        recall_rate_percent[query_index] = static_cast<double>(query.recall_rate) * 100.0;
        bucket_level_ivf_ms[query_index] = query.timing_metrics.bucket_level_ivf_ms;
        candidate_bucket_merge_ms[query_index] = query.timing_metrics.candidate_bucket_merge_ms;
        npu_async_launch_ms[query_index] = query.timing_metrics.npu_async_launch_ms;
        inbucket_attr_filter_overlapped_ms[query_index] = query.timing_metrics.inbucket_attr_filter_overlapped_ms;
        wait_npu_flag_ms[query_index] = query.timing_metrics.wait_npu_flag_ms;
        result_collection_ms[query_index] = query.timing_metrics.result_collection_ms;
        final_merge_ms[query_index] = query.timing_metrics.final_merge_ms;
    }

    bool Flush(const fs::path &result_root) const
    {
        const fs::path recall_dir = result_root / "recall";
        const fs::path latency_dir = result_root / "latency";
        const fs::path log_dir = result_root / "log";

        return RunSupport::EnsureOutputDirectory(recall_dir) &&
               RunSupport::EnsureOutputDirectory(latency_dir) &&
               RunSupport::EnsureOutputDirectory(log_dir) &&
               RunSupport::WriteMetricFile(recall_dir, "recall_rate_percent.txt", recall_rate_percent) &&
               RunSupport::WriteMetricFile(recall_dir, "ground_truth_count.txt", ground_truth_count) &&
               RunSupport::WriteMetricFile(recall_dir, "engine_result_count.txt", engine_result_count) &&
               RunSupport::WriteMetricFile(latency_dir, "total_end_to_end_ms.txt", total_end_to_end_ms) &&
               RunSupport::WriteMetricFile(latency_dir, "query_construction_ms.txt", query_construction_ms) &&
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
                                          process_round_count_level_2);
    }

    std::vector<double> query_construction_ms;
    std::vector<double> total_end_to_end_ms;
    std::vector<size_t> ground_truth_count;
    std::vector<size_t> engine_result_count;
    std::vector<size_t> process_round_count_level_1;
    std::vector<size_t> process_round_count_level_2;
    std::vector<double> recall_rate_percent;
    std::vector<double> bucket_level_ivf_ms;
    std::vector<double> candidate_bucket_merge_ms;
    std::vector<double> npu_async_launch_ms;
    std::vector<double> inbucket_attr_filter_overlapped_ms;
    std::vector<double> wait_npu_flag_ms;
    std::vector<double> result_collection_ms;
    std::vector<double> final_merge_ms;
};
} // namespace

int main() {
    constexpr const char *kDatasetFile = "../../dataset_HW.bin";
    constexpr int kDefaultTopK = 100;
    constexpr int QueryNum = 10000;
    const std::vector<std::string> kQueryPaths = {
        "datasets/hw_queries.fvecs",
        "../datasets/hw_queries.fvecs",
        "../../datasets/hw_queries.fvecs",
        "../../../datasets/hw_queries.fvecs"
    };

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
    std::cout << "[Loader] Prepared query count: " << queries.size() << "\n";

    // Step 1: Ensure queries vector has exactly QueryNum items
    if (queries.size() > QueryNum) {
        queries.resize(QueryNum);
        std::cout << "[Loader] Truncated queries: " << queries.size() << "\n";
    } else if (queries.size() < QueryNum) {
        const size_t original_count = queries.size();
        queries.reserve(QueryNum);
        for (size_t i = original_count; i < QueryNum; ++i) {
            queries.push_back(queries[i % original_count]);
        }
        std::cout << "[Loader] Expanded queries: " << original_count << " -> " << queries.size() << "\n";
    }

    // Step 2: Load filter expressions: read 10 from file, replicate to QueryNum
    {
        std::vector<std::string> filter_exprs_10;
        const fs::path filter_expr_path = config_path.has_parent_path()
                                              ? (config_path.parent_path() / "filter_expr_example.txt")
                                              : fs::path("filter_expr_example.txt");
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
            std::cerr << "[Warn] filter_expr_example.txt not found or empty at " << filter_expr_path
                      << ". Using empty filters for all queries.\n";
        } else if (filter_exprs_10.size() < 10) {
            std::cerr << "[Warn] filter_expr_example.txt has fewer than 10 lines. Using available expressions.\n";
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

        for (size_t i = 0; i < queries.size(); ++i) {
            queries[i].filter_expr = filter_exprs_N[i];
            // queries[i].filter_expr = "";
        }
        std::cout << "[Loader] Assigned filter expressions to all " << queries.size() << " queries\n";
    }

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
        std::cerr << "[Fatal] Failed to build BucketDocTable from clustering output: " << e.what() << "\n";
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
    QueryPool query_pool(static_cast<size_t>(query_pool_capacity));

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

        std::vector<DataReader::GroundTruthResultItem> cached_ground_truth;
        if (ground_truth_cache &&
            ground_truth_cache->Lookup(prepared.query_vec,
                                       prepared.filter_expr,
                                       prepared.top_k,
                                       cached_ground_truth)) {
            completed_q->ground_truth_results = RunSupport::ConvertCacheItemsToQueryResults(cached_ground_truth);
        } else {
            completed_q->brute_force_search(dataset);

            if (ground_truth_cache &&
                !ground_truth_cache->Store(prepared.query_vec,
                                           prepared.filter_expr,
                                           prepared.top_k,
                                           RunSupport::ConvertQueryResultsToCacheItems(
                                               completed_q->ground_truth_results))) {
                std::cerr << "[Warn] Failed to append ground-truth cache entry for query line "
                          << prepared.line_no << ".\n";
            }
        }

        completed_q->calculate_recall();

        const double total_ms = construct_ms.count() + loop_ms.count();
        deferred_metric_files->Record(idx,
                                      *completed_q,
                                      construct_ms.count(),
                                      total_ms);
        completed_q->AppendMemoryEventLogs(memory_log_collector);

        query_pool.Release(completed_q);
        if ((idx + 1) % 100 == 0 || idx + 1 == queries.size()) {
            std::cout << (idx + 1) << "/" << queries.size() << std::endl;
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
    return 0;
}
