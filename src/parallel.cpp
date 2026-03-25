#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "utils/RunSupport.h"
#include "Clustering/run_clustering.h"
#include "DataBaseCPU/DataBaseCPU.h"
#include "utils/DataReader.h"
#include "Query/Query.h"
#include "Schedule/Scheduler.h"
#include "Schedule/WorkerGroup.h"

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
                                   std::vector<double> &total_end_to_end_ms)
{
    total_end_to_end_ms.resize(queries.size());
    for (size_t i = 0; i < queries.size(); ++i)
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
    }

    return true;
}

bool FlushParallelOutputs(const fs::path &result_root,
                          const std::vector<double> &total_end_to_end_ms,
                          double overall_qps)
{
    const fs::path latency_dir = result_root / "latency";
    const fs::path log_dir = result_root / "log";
    const std::vector<double> qps_values{overall_qps};
    return RunSupport::EnsureOutputDirectory(latency_dir) &&
           RunSupport::EnsureOutputDirectory(log_dir) &&
           RunSupport::WriteMetricFile(latency_dir, "total_end_to_end_ms.txt", total_end_to_end_ms) &&
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
    constexpr const char *kDatasetFile = "../../dataset.bin";
    constexpr int kDefaultTopK = 100;
    const std::vector<std::string> kQueryPaths = {
        "datasets/DEEP/deep1B_queries.fvecs",
        "../datasets/DEEP/deep1B_queries.fvecs",
        "../../datasets/DEEP/deep1B_queries.fvecs",
        "../../../datasets/DEEP/deep1B_queries.fvecs"
    };

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
              << ", Buckets=" << total_bucket_num << "\n";

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

    std::vector<uint64_t> bucket_doc_offsets;
    std::vector<uint32_t> bucket_doc_ids;
    int sync_result = SyncClusteringContextFromMain(dataset_buffers.vectors,
                                                    dataset_buffers.bitmaps,
                                                    bucket_doc_offsets,
                                                    bucket_doc_ids,
                                                    dataset_buffers.centroids,
                                                    total_bucket_num,
                                                    total_tag_num,
                                                    vector_dim,
                                                    total_doc_num,
                                                    max_doc_per_bucket);
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
    if (clustered_bucket_count != static_cast<size_t>(total_bucket_num))
    {
        std::cout << "[Clustering] bucket count updated by clustering: "
                  << total_bucket_num << " -> " << clustered_bucket_count << "\n";
        total_bucket_num = static_cast<int>(clustered_bucket_count);
    }

    if (dataset_buffers.centroids.size() !=
        static_cast<size_t>(total_bucket_num) * static_cast<size_t>(vector_dim))
    {
        std::cerr << "[Fatal] Invalid clustering output: centroids size mismatch.\n";
        return -1;
    }
    if (!FinalizePreallocationParams())
    {
        return -1;
    }

    InputDataset dataset = DataReader::BuildInputDataset(dataset_buffers);
    BucketDocTable bucket_doc_table;
    try
    {
        bucket_doc_table = RunSupport::BuildBucketDocTableFromOffsets(bucket_doc_offsets, bucket_doc_ids);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Fatal] Failed to build BucketDocTable from clustering output: "
                  << e.what() << "\n";
        return -1;
    }

    std::cout << "[Loader] BucketDocTable built. StoredDocRefs="
              << RunSupport::CountStoredDocRefs(bucket_doc_table)
              << ", BucketCount=" << bucket_doc_table.size() << "\n";

    std::cout << "[System] Initializing DataBaseCPU...\n";
    DataBaseCPU db(dataset, bucket_doc_table);

    const size_t scheduler_capacity = prepared_queries.size() + 10000;
    std::cout << "[System] Initializing Scheduler. capacity=" << scheduler_capacity << "\n";
    Scheduler scheduler(scheduler_capacity);

    std::cout << "[System] Launching Worker Threads...\n";
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
    std::cout << "[ParallelTest] Batch submit begins.\n";

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
    if (!CollectParallelLatencyMetrics(queries, total_end_to_end_ms))
    {
        return -1;
    }

    if (!FlushParallelOutputs(query_result_root_dir, total_end_to_end_ms, overall_qps))
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
