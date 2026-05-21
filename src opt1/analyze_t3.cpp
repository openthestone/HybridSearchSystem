#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "Clustering/run_clustering.h"
#include "DataBaseCPU/DataBaseCPU.h"
#include "Query/Query.h"
#include "Schedule/Scheduler.h"
#include "Schedule/WorkerGroup.h"
#include "utils/DataReader.h"
#include "utils/RunSupport.h"

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

class WorkerRuntime
{
public:
    WorkerRuntime(DataBaseCPU *db, Scheduler *scheduler)
    {
        groups_.reserve(static_cast<size_t>(group_count));
        threads_.reserve(static_cast<size_t>(cpu_core_count));

        for (int i = 0; i < group_count; ++i)
        {
            groups_.push_back(std::make_unique<WorkerGroup>(i, db, scheduler));
        }
        for (int i = 0; i < cpu_core_count; ++i)
        {
            const int gid = GetGroupId(i);
            threads_.emplace_back(&WorkerGroup::Run, groups_[static_cast<size_t>(gid)].get(), i);
        }
    }

    ~WorkerRuntime()
    {
        Stop();
    }

    void Stop()
    {
        if (stopped_)
        {
            return;
        }
        stopped_ = true;

        for (auto &group : groups_)
        {
            group->Stop();
        }
        for (auto &thread : threads_)
        {
            if (thread.joinable())
            {
                thread.join();
            }
        }
    }

private:
    bool stopped_ = false;
    std::vector<std::unique_ptr<WorkerGroup>> groups_;
    std::vector<std::thread> threads_;
};

void PrintParallelProgress(size_t completed_count, size_t total_count)
{
    std::cout << "[Progress] " << completed_count << "/" << total_count << std::endl;
}

bool LoadGroundTruthCache(std::unique_ptr<DataReader::GroundTruthCache> &ground_truth_cache)
{
    if (ground_truth_cache_file.empty())
    {
        std::cout << "[Loader] Ground-truth cache disabled by config.\n";
        return true;
    }

    std::cout << "[Loader] Loading ground-truth cache...\n";
    ground_truth_cache = std::make_unique<DataReader::GroundTruthCache>(ground_truth_cache_file);
    if (!ground_truth_cache->Load())
    {
        std::cerr << "[Warn] Failed to load ground-truth cache from "
                  << ground_truth_cache->path()
                  << ". Continue without cache.\n";
        ground_truth_cache.reset();
        return true;
    }

    std::cout << "[Loader] Ground-truth cache loaded from "
              << ground_truth_cache->path()
              << ". entries=" << ground_truth_cache->EntryCount() << "\n";
    return true;
}

bool PrecomputeGroundTruth(const InputDataset &dataset,
                           const std::vector<DataReader::PreparedQuery> &prepared_queries,
                           std::unique_ptr<DataReader::GroundTruthCache> &ground_truth_cache,
                           std::vector<std::vector<QueryResult::Item>> &ground_truth_by_query)
{
    ground_truth_by_query.clear();
    ground_truth_by_query.resize(prepared_queries.size());

    std::cout << "[AnalyzeT3] Preparing ground truth for recall calculation...\n";
    for (size_t idx = 0; idx < prepared_queries.size(); ++idx)
    {
        const auto &prepared = prepared_queries[idx];
        std::vector<DataReader::GroundTruthResultItem> cached_ground_truth;
        if (ground_truth_cache &&
            ground_truth_cache->Lookup(prepared.query_vec,
                                       prepared.filter_expr,
                                       prepared.top_k,
                                       cached_ground_truth))
        {
            ground_truth_by_query[idx] =
                RunSupport::ConvertCacheItemsToQueryResults(cached_ground_truth);
        }
        else
        {
            Query query;
            query.PrepareMemoryEventSession(prepared.line_no, false);
            query.Reset(prepared.query_vec, prepared.filter_expr, prepared.top_k, prepared.line_no);
            query.brute_force_search(dataset);
            ground_truth_by_query[idx] = std::move(query.ground_truth_results);

            if (ground_truth_cache &&
                !ground_truth_cache->Store(prepared.query_vec,
                                           prepared.filter_expr,
                                           prepared.top_k,
                                           RunSupport::ConvertQueryResultsToCacheItems(
                                               ground_truth_by_query[idx])))
            {
                std::cerr << "[Warn] Failed to append ground-truth cache entry for query line "
                          << prepared.line_no << ".\n";
            }
        }

        std::cout << "[AnalyzeT3][GT] " << (idx + 1) << "/"
                  << prepared_queries.size() << "\n";
    }

    return true;
}

bool RunParallelRound(const std::vector<DataReader::PreparedQuery> &prepared_queries,
                      const std::vector<std::vector<QueryResult::Item>> &ground_truth_by_query,
                      Scheduler &scheduler,
                      double test_k_expand_param,
                      double &average_recall)
{
    k_expand_param = test_k_expand_param;
    if (!RunSupport::ValidatePreparedQueriesAgainstPrealloc(prepared_queries))
    {
        return false;
    }

    std::vector<std::unique_ptr<Query>> queries;
    queries.reserve(prepared_queries.size());
    std::vector<Query *> query_ptrs;
    query_ptrs.reserve(prepared_queries.size());

    for (const auto &prepared : prepared_queries)
    {
        auto query = std::make_unique<Query>();
        query->PrepareMemoryEventSession(prepared.line_no, false);
        query->Reset(prepared.query_vec, prepared.filter_expr, prepared.top_k, prepared.line_no);
        query_ptrs.push_back(query.get());
        queries.push_back(std::move(query));
    }

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "[AnalyzeT3] k_expand_param=" << test_k_expand_param
              << ", query_count=" << query_ptrs.size() << "\n";
    std::cout << "[AnalyzeT3] Batch submit begins.\n";

    const auto batch_start = std::chrono::steady_clock::now();
    for (Query *query : query_ptrs)
    {
        scheduler.Push(query);
    }

    size_t completed_count = 0;
    auto next_progress_report = batch_start + std::chrono::seconds(1);
    while (completed_count < query_ptrs.size())
    {
        if (scheduler.PopResult() != nullptr)
        {
            ++completed_count;
        }

        const auto now = std::chrono::steady_clock::now();
        if (now >= next_progress_report)
        {
            PrintParallelProgress(completed_count, query_ptrs.size());
            do
            {
                next_progress_report += std::chrono::seconds(1);
            } while (now >= next_progress_report);
        }

        if (completed_count < query_ptrs.size())
        {
            std::this_thread::yield();
        }
    }
    const auto batch_end = std::chrono::steady_clock::now();

    double recall_sum = 0.0;
    for (size_t idx = 0; idx < queries.size(); ++idx)
    {
        queries[idx]->ground_truth_results = ground_truth_by_query[idx];
        queries[idx]->calculate_recall();
        recall_sum += static_cast<double>(queries[idx]->recall_rate);
    }
    average_recall = queries.empty() ? 0.0 : recall_sum / static_cast<double>(queries.size());

    std::chrono::duration<double, std::milli> total_latency_ms = batch_end - batch_start;
    const double total_latency_seconds = total_latency_ms.count() / 1000.0;
    const double overall_qps = (total_latency_seconds > 0.0)
                                   ? (static_cast<double>(query_ptrs.size()) / total_latency_seconds)
                                   : 0.0;
    const double avg_query_latency_ms = query_ptrs.empty()
                                            ? 0.0
                                            : (total_latency_ms.count() / static_cast<double>(query_ptrs.size()));

    std::cout << "\n========================================================\n";
    std::cout << "                 Analyze T3 Round Result                \n";
    std::cout << "========================================================\n";
    std::cout << "k_expand_param                : " << test_k_expand_param << "\n";
    std::cout << "Query Count                   : " << query_ptrs.size() << "\n";
    std::cout << "Completed Query Count         : " << completed_count << "\n";
    std::cout << "Total Batch Latency           : " << total_latency_ms.count() << " ms\n";
    std::cout << "Overall QPS                   : " << overall_qps << "\n";
    std::cout << "Average Query Latency(Batch/N): " << avg_query_latency_ms << " ms\n";
    std::cout << "Average Recall                : " << average_recall << "\n";
    std::cout << "========================================================\n";

    return true;
}
} // namespace

int main()
{
    constexpr int kDefaultTopK = 100;
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
    std::cout << "[AnalyzeT3] Loading config from " << config_path << "...\n";
    if (!LoadParams(config_path.string(), ResourceConfigProfile::Parallel))
    {
        std::cerr << "[Error] Failed to load config: " << config_path << "\n";
        return -1;
    }
    if (!LoadKExpandParamTestSet(config_path.string()))
    {
        return -1;
    }

    DataReader::DatasetBuffers dataset_buffers;
    DataReader::LoadFailureReason dataset_failure = DataReader::LoadFailureReason::None;
    if (!DataReader::LoadDatasetCache(kDatasetFile, dataset_buffers, &dataset_failure))
    {
        if (dataset_failure != DataReader::LoadFailureReason::VectorDimMismatch)
        {
            std::cerr << "[Fatal] Failed to load dataset cache: " << kDatasetFile << "\n";
        }
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
    FilterExpCompiler::WarmUpThreadLocalBuffers();

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

    std::unique_ptr<DataReader::GroundTruthCache> ground_truth_cache;
    if (!LoadGroundTruthCache(ground_truth_cache))
    {
        return -1;
    }

    std::vector<std::vector<QueryResult::Item>> ground_truth_by_query;
    if (!PrecomputeGroundTruth(dataset,
                               prepared_queries,
                               ground_truth_cache,
                               ground_truth_by_query))
    {
        return -1;
    }

    std::cout << "[System] Initializing DataBaseCPU...\n";
    DataBaseCPU db(dataset, std::move(bucket_layout));

    const size_t scheduler_capacity = prepared_queries.size() + 10000;
    std::cout << "[System] Initializing Scheduler. capacity=" << scheduler_capacity << "\n";
    Scheduler scheduler(scheduler_capacity);

    std::cout << "[System] Launching Worker Threads...\n";
    WorkerRuntime workers(&db, &scheduler);

    for (double test_value : k_expand_param_test_set)
    {
        double average_recall = 0.0;
        if (!RunParallelRound(prepared_queries,
                              ground_truth_by_query,
                              scheduler,
                              test_value,
                              average_recall))
        {
            return -1;
        }
    }

    workers.Stop();
    std::cout << "[System] All threads stopped. Exiting safely.\n";
    return 0;
}
