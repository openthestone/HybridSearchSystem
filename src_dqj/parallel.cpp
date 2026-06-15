#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
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

constexpr double kParallelTargetQps = 2000.0;
constexpr std::chrono::microseconds kInjectorTick(1000);

bool ParseConfigStringValue(const std::string &config_file,
                            const std::string &target_key,
                            std::string &value_out)
{
    std::ifstream file(config_file);
    if (!file.is_open())
    {
        return false;
    }

    std::string line;
    while (std::getline(file, line))
    {
        StripConfigComment(line);
        line = ParamTrim(line);
        if (line.empty())
        {
            continue;
        }

        const size_t equal_pos = line.find('=');
        if (equal_pos == std::string::npos)
        {
            continue;
        }

        const std::string key = ParamTrim(line.substr(0, equal_pos));
        if (key != target_key)
        {
            continue;
        }

        std::string parsed_value;
        if (!ParseConfigString(line.substr(equal_pos + 1), parsed_value))
        {
            return false;
        }
        value_out = parsed_value;
        return true;
    }

    return false;
}

void RestoreSrcStyleRelativePathsFromConfig(const std::string &config_file)
{
    auto restore_path = [&](const std::string &key, std::string &target) {
        std::string raw_value;
        if (ParseConfigStringValue(config_file, key, raw_value))
        {
            target = raw_value;
        }
    };

    restore_path("dataset_cache_file", dataset_cache_file);
    restore_path("tag_map_cache_file", tag_map_cache_file);
    restore_path("query_file", query_file);
    restore_path("query_result_root", query_result_root);
    restore_path("ground_truth_cache_file", ground_truth_cache_file);
    restore_path("bucket_ivf_index_file", bucket_ivf_index_file);
    restore_path("bucket_index_file", bucket_index_file);
    restore_path("raw_vector_file", raw_vector_file);
    restore_path("raw_attr_dir", raw_attr_dir);
    restore_path("raw_query_file", raw_query_file);
}

struct SharedProgressStats
{
    std::atomic<size_t> submitted_count{0};
    std::atomic<size_t> max_task_queue_size{0};
    std::atomic<unsigned long long> task_queue_size_sum{0};
    std::atomic<size_t> task_queue_sample_count{0};
};

struct ParallelRunStats
{
    double target_qps = 0.0;
    size_t query_count = 0;
    double measure_submit_duration_ms = 0.0;
    double measure_submit_qps = 0.0;
    double total_complete_duration_ms = 0.0;
    double complete_qps = 0.0;
    double injector_tick_us = 0.0;
    double max_submit_lag_ms = 0.0;
    size_t max_task_queue_size = 0;
    double avg_task_queue_size = 0.0;
};

struct ParallelRecallSummary
{
    double average_recall = 0.0;
    size_t cache_hit_count = 0;
    size_t brute_force_count = 0;
};

void UpdateAtomicMax(std::atomic<size_t> &target,
                     size_t candidate)
{
    size_t current = target.load(std::memory_order_relaxed);
    while (candidate > current &&
           !target.compare_exchange_weak(current,
                                         candidate,
                                         std::memory_order_relaxed,
                                         std::memory_order_relaxed))
    {
    }
}

void SampleTaskQueueSize(SharedProgressStats &stats,
                         size_t queue_size)
{
    UpdateAtomicMax(stats.max_task_queue_size, queue_size);
    stats.task_queue_size_sum.fetch_add(static_cast<unsigned long long>(queue_size),
                                        std::memory_order_relaxed);
    stats.task_queue_sample_count.fetch_add(1, std::memory_order_relaxed);
}

ParallelRunStats RunRateLimitedInjector(const std::vector<Query *> &query_ptrs,
                                        Scheduler &scheduler,
                                        SharedProgressStats &progress_stats,
                                        std::chrono::steady_clock::time_point measure_start)
{
    ParallelRunStats run_stats;
    run_stats.target_qps = kParallelTargetQps;
    run_stats.query_count = query_ptrs.size();
    run_stats.injector_tick_us = static_cast<double>(kInjectorTick.count());

    size_t submitted_count = 0;
    double max_submit_lag_ms = 0.0;

    while (submitted_count < query_ptrs.size())
    {
        const auto now = std::chrono::steady_clock::now();
        const double elapsed_seconds =
            std::chrono::duration<double>(now - measure_start).count();
        const size_t expected_submitted = std::min(
            query_ptrs.size(),
            static_cast<size_t>(elapsed_seconds * kParallelTargetQps));

        if (expected_submitted <= submitted_count)
        {
            std::this_thread::sleep_for(kInjectorTick);
            continue;
        }

        while (submitted_count < expected_submitted)
        {
            scheduler.Push(query_ptrs[submitted_count]);
            const auto enqueued_time = std::chrono::steady_clock::now();
            const auto scheduled_time =
                measure_start +
                std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(
                        static_cast<double>(submitted_count + 1) / kParallelTargetQps));
            if (enqueued_time > scheduled_time)
            {
                const double lag_ms =
                    std::chrono::duration<double, std::milli>(enqueued_time - scheduled_time)
                        .count();
                max_submit_lag_ms = std::max(max_submit_lag_ms, lag_ms);
            }

            ++submitted_count;
            progress_stats.submitted_count.store(submitted_count, std::memory_order_relaxed);
            SampleTaskQueueSize(progress_stats, scheduler.GetTaskQueueSize());
        }
    }

    const auto submit_end = std::chrono::steady_clock::now();
    run_stats.measure_submit_duration_ms =
        std::chrono::duration<double, std::milli>(submit_end - measure_start).count();
    const double submit_seconds = run_stats.measure_submit_duration_ms / 1000.0;
    run_stats.measure_submit_qps =
        (submit_seconds > 0.0) ? (static_cast<double>(submitted_count) / submit_seconds) : 0.0;
    run_stats.max_submit_lag_ms = max_submit_lag_ms;
    return run_stats;
}


bool CollectParallelLatencyMetrics(const std::vector<std::unique_ptr<Query>> &queries,
                                   std::vector<double> &total_end_to_end_ms,
                                   std::vector<double> &bucket_level_ivf_ms,
                                   std::vector<double> &bucket_level_ivf_filter_eval_ms,
                                   std::vector<double> &bucket_level_ivf_centroid_score_ms,
                                   std::vector<double> &bucket_level_ivf_result_pack_ms,
                                   std::vector<double> &candidate_bucket_merge_ms,
                                   std::vector<double> &npu_async_launch_ms,
                                   std::vector<double> &npu_submit_ms,
                                   std::vector<double> &inbucket_attr_filter_overlapped_ms,
                                   std::vector<double> &wait_npu_flag_ms,
                                   std::vector<double> &result_collection_ms,
                                   std::vector<double> &final_merge_ms,
                                   std::vector<double> &npu_mask_filter_launch_ms,
                                   std::vector<double> &npu_mask_filter_h2d_ms,
                                   std::vector<double> &npu_mask_filter_kernel_exec_ms,
                                   std::vector<double> &npu_mask_filter_d2h_ms,
                                   std::vector<size_t> &process_round_count,
                                   std::vector<size_t> &process_round_count_level_2,
                                   std::vector<size_t> &searched_l2_bucket_count,
                                   std::vector<size_t> &searched_l2_nonzero_count,
                                   std::vector<size_t> &searched_l2_allzero_count,
                                   std::vector<std::vector<double>> &bucket_selectivity_per_query)
{
    const size_t n = queries.size();
    total_end_to_end_ms.resize(n);
    bucket_level_ivf_ms.resize(n);
    bucket_level_ivf_filter_eval_ms.resize(n);
    bucket_level_ivf_centroid_score_ms.resize(n);
    bucket_level_ivf_result_pack_ms.resize(n);
    candidate_bucket_merge_ms.resize(n);
    npu_async_launch_ms.resize(n);
    npu_submit_ms.resize(n);
    inbucket_attr_filter_overlapped_ms.resize(n);
    wait_npu_flag_ms.resize(n);
    result_collection_ms.resize(n);
    final_merge_ms.resize(n);
    npu_mask_filter_launch_ms.resize(n);
    npu_mask_filter_h2d_ms.resize(n);
    npu_mask_filter_kernel_exec_ms.resize(n);
    npu_mask_filter_d2h_ms.resize(n);
    process_round_count.resize(n);
    process_round_count_level_2.resize(n);
    searched_l2_bucket_count.resize(n);
    searched_l2_nonzero_count.resize(n);
    searched_l2_allzero_count.resize(n);
    bucket_selectivity_per_query.resize(n);

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
        bucket_level_ivf_filter_eval_ms[i] = query.timing_metrics.bucket_level_ivf_filter_eval_ms;
        bucket_level_ivf_centroid_score_ms[i] = query.timing_metrics.bucket_level_ivf_centroid_score_ms;
        bucket_level_ivf_result_pack_ms[i] = query.timing_metrics.bucket_level_ivf_result_pack_ms;
        candidate_bucket_merge_ms[i] = query.timing_metrics.candidate_bucket_merge_ms;
        npu_async_launch_ms[i] = query.timing_metrics.npu_async_launch_ms;
        npu_submit_ms[i] = query.timing_metrics.npu_submit_ms;
        inbucket_attr_filter_overlapped_ms[i] = query.timing_metrics.inbucket_attr_filter_overlapped_ms;
        wait_npu_flag_ms[i] = query.timing_metrics.wait_npu_flag_ms;
        result_collection_ms[i] = query.timing_metrics.result_collection_ms;
        final_merge_ms[i] = query.timing_metrics.final_merge_ms;
        npu_mask_filter_launch_ms[i] = query.timing_metrics.npu_mask_filter_launch_ms;
        npu_mask_filter_h2d_ms[i] = query.timing_metrics.npu_mask_filter_h2d_ms;
        npu_mask_filter_kernel_exec_ms[i] = query.timing_metrics.npu_mask_filter_kernel_exec_ms;
        npu_mask_filter_d2h_ms[i] = query.timing_metrics.npu_mask_filter_d2h_ms;
        process_round_count[i] = static_cast<size_t>(query.process_round_count_l2);
        process_round_count_level_2[i] = static_cast<size_t>(query.process_round_count_level_2);
        searched_l2_bucket_count[i] = static_cast<size_t>(query.searched_bucket_count_level_2);
        searched_l2_nonzero_count[i] = static_cast<size_t>(query.searched_l2_count_nonzero.load());
        searched_l2_allzero_count[i] = static_cast<size_t>(query.searched_l2_count_allzero.load());
        {
            size_t sel_count = query.bucket_selectivity_idx_.load(std::memory_order_relaxed);
            bucket_selectivity_per_query[i].assign(
                query.bucket_selectivity_.begin(),
                query.bucket_selectivity_.begin() + std::min(sel_count, query.bucket_selectivity_.size()));
        }
    }

    return true;
}

bool CollectParallelRecallMetrics(const InputDataset &dataset,
                                  const std::vector<DataReader::PreparedQuery> &prepared_queries,
                                  std::vector<std::unique_ptr<Query>> &queries,
                                  DataReader::GroundTruthCache *ground_truth_cache,
                                  std::vector<double> &recall_rate_percent,
                                  ParallelRecallSummary &summary)
{
    if (queries.size() != prepared_queries.size())
    {
        std::cerr << "[Parallel] Query count mismatch when collecting recall metrics. prepared="
                  << prepared_queries.size() << ", runtime=" << queries.size() << "\n";
        return false;
    }

    const size_t n = queries.size();
    recall_rate_percent.resize(n);

    // Phase 1: Check cache
    std::vector<bool> needs_brute_force(n, false);
    size_t cache_hit_count = 0;

#pragma omp parallel for schedule(dynamic, 8) reduction(+:cache_hit_count)
    for (size_t i = 0; i < n; ++i)
    {
        std::vector<DataReader::GroundTruthResultItem> cached_ground_truth;
        if (ground_truth_cache &&
            ground_truth_cache->Lookup(prepared_queries[i].query_vec,
                                       prepared_queries[i].filter_expr,
                                       prepared_queries[i].top_k,
                                       cached_ground_truth))
        {
            queries[i]->ground_truth_results = RunSupport::ConvertCacheItemsToQueryResults(cached_ground_truth);
            ++cache_hit_count;
        }
        else
        {
            needs_brute_force[i] = true;
        }
    }

    // Phase 2: Group brute-force queries by filter expression, compute passing docs once per group
    struct FilterGroup {
        std::vector<uint32_t> passing_docs;
        std::vector<size_t> query_indices;
    };
    std::vector<FilterGroup> groups;
    std::unordered_map<std::string, size_t> expr_to_group;

    for (size_t i = 0; i < n; ++i)
    {
        if (!needs_brute_force[i]) continue;
        const auto &expr = prepared_queries[i].filter_expr;
        auto [it, inserted] = expr_to_group.emplace(expr, groups.size());
        if (inserted)
        {
            groups.emplace_back();
            Query &ref = *queries[i];
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
        groups[it->second].query_indices.push_back(i);
    }

    // Phase 3: Process brute-force queries with shared passing docs
    size_t brute_force_count = 0;
    if (!groups.empty())
    {
        struct Task { size_t query_idx; size_t group_idx; };
        std::vector<Task> tasks;
        for (size_t g = 0; g < groups.size(); ++g)
        {
            brute_force_count += groups[g].query_indices.size();
            for (size_t qi : groups[g].query_indices)
            {
                tasks.push_back({qi, g});
            }
        }

#pragma omp parallel for schedule(dynamic, 8)
        for (size_t t = 0; t < tasks.size(); ++t)
        {
            queries[tasks[t].query_idx]->brute_force_search_filtered(
                dataset, groups[tasks[t].group_idx].passing_docs);
        }
    }

    // Phase 4: Calculate recall for all queries
    double recall_sum = 0.0;
#pragma omp parallel for schedule(static) reduction(+:recall_sum)
    for (size_t i = 0; i < n; ++i)
    {
        queries[i]->calculate_recall();
        recall_rate_percent[i] = static_cast<double>(queries[i]->recall_rate) * 100.0;
        recall_sum += static_cast<double>(queries[i]->recall_rate);
    }

    // Phase 5: Deferred cache stores
    if (ground_truth_cache)
    {
        for (size_t i = 0; i < n; ++i)
        {
            if (!needs_brute_force[i]) continue;
            const auto &prepared = prepared_queries[i];
            Query &query = *queries[i];
            if (!ground_truth_cache->Store(prepared.query_vec,
                                           prepared.filter_expr,
                                           prepared.top_k,
                                           RunSupport::ConvertQueryResultsToCacheItems(query.ground_truth_results)))
            {
                std::cerr << "[Warn] Failed to append ground-truth cache entry for query line "
                          << prepared.line_no << ".\n";
            }
        }
    }

    summary.cache_hit_count = cache_hit_count;
    summary.brute_force_count = brute_force_count;
    summary.average_recall =
        n == 0 ? 0.0 : (recall_sum / static_cast<double>(n));
    return true;
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

bool FlushParallelOutputs(const fs::path &result_root,
                          const std::vector<double> &total_end_to_end_ms,
                          const std::vector<double> &recall_rate_percent,
                          const std::vector<double> &bucket_level_ivf_ms,
                          const std::vector<double> &bucket_level_ivf_filter_eval_ms,
                          const std::vector<double> &bucket_level_ivf_centroid_score_ms,
                          const std::vector<double> &bucket_level_ivf_result_pack_ms,
                          const std::vector<double> &candidate_bucket_merge_ms,
                          const std::vector<double> &npu_async_launch_ms,
                          const std::vector<double> &npu_submit_ms,
                          const std::vector<double> &inbucket_attr_filter_overlapped_ms,
                          const std::vector<double> &wait_npu_flag_ms,
                          const std::vector<double> &result_collection_ms,
                          const std::vector<double> &final_merge_ms,
                          const std::vector<double> &npu_mask_filter_launch_ms,
                          const std::vector<double> &npu_mask_filter_h2d_ms,
                          const std::vector<double> &npu_mask_filter_kernel_exec_ms,
                          const std::vector<double> &npu_mask_filter_d2h_ms,
                          const std::vector<size_t> &process_round_count,
                          const std::vector<size_t> &process_round_count_level_2,
                          const std::vector<size_t> &searched_l2_bucket_count,
                          const std::vector<size_t> &searched_l2_nonzero_count,
                          const std::vector<size_t> &searched_l2_allzero_count,
                          const std::vector<std::vector<double>> &bucket_selectivity_per_query,
                          double overall_qps)
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

    // Launch independent file writes in parallel
    auto f_recall = std::async(std::launch::async, [&] {
        return RunSupport::WriteMetricFile(recall_dir, "recall_rate_percent.txt", recall_rate_percent);
    });
    auto f_latency = std::async(std::launch::async, [&] {
        return RunSupport::WriteMetricFile(latency_dir, "total_end_to_end_ms.txt", total_end_to_end_ms) &&
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
    const std::vector<double> qps_values{overall_qps};
    bool qps_ok = RunSupport::WriteMetricFile(log_dir, "QPS.txt", qps_values);

    return f_recall.get() && f_latency.get() && f_log.get() && qps_ok;
}

void PrintParallelProgress(size_t submitted_count,
                           size_t completed_count,
                           size_t total_count)
{
    std::cout << "[Progress] submitted=" << submitted_count << "/" << total_count
              << ", completed=" << completed_count << "/" << total_count << std::endl;
}
} // namespace

int main()
{
    const fs::path config_path = RunSupport::ResolveConfigPath();
    std::cout << "[System] Loading config from " << config_path << "...\n";
    if (!LoadParams(config_path.string(), ResourceConfigProfile::Parallel))
    {
        std::cerr << "[Error] Failed to load config: " << config_path << "\n";
        return -1;
    }
    RestoreSrcStyleRelativePathsFromConfig(config_path.string());

    const fs::path query_result_root_dir =
        query_result_root.empty() ? fs::path(".") : fs::path(query_result_root);
    const fs::path memory_log_file = query_result_root_dir / "log" / "memory_logs.txt";

    std::cout << "[Loader] Loading dataset cache...\n";
    DataReader::DatasetBuffers dataset_buffers;
    std::unordered_map<std::string, int> tag_map;
    if (!DataReader::EnsureDatasetAndTagMapCaches(dataset_buffers, tag_map))
    {
        std::cerr << "[Fatal] Failed to prepare dataset/tag-map caches.\n";
        return -1;
    }
    std::cout << "[Loader] Dataset loaded. Docs=" << total_doc_num
              << ", Dim=" << vector_dim
              << ", Tags=" << total_tag_num
              << ", Buckets(Level1)=" << total_bucket_num_level_1 << "\n";
    std::cout << "[Loader] tag_map loaded. size=" << tag_map.size() << "\n";
    std::cout << "[Loader] Loading queries...\n";
    if (!DataReader::EnsurePreparedQueryFile())
    {
        std::cerr << "[Fatal] Failed to prepare query file: " << query_file << "\n";
        return -1;
    }

    std::vector<DataReader::PreparedQuery> prepared_queries;
    std::vector<std::string> query_warnings;
    DataReader::QueryLoadDiagnostics query_load_diagnostics;
    DataReader::LoadFailureReason query_failure = DataReader::LoadFailureReason::None;
    if (!DataReader::LoadPreparedQueries(query_file,
                                         tag_map,
                                         vector_dim,
                                         prepared_queries,
                                         query_warnings,
                                         &query_load_diagnostics,
                                         &query_failure))
    {
        if (query_failure != DataReader::LoadFailureReason::VectorDimMismatch)
        {
            std::cerr << "[Fatal] Failed to read query file: " << query_file << "\n";
        }
        return -1;
    }
    if (!RunSupport::WriteQueryLoadDiagnosticsFiles(query_result_root_dir, query_load_diagnostics))
    {
        std::cerr << "[Fatal] Failed to write query load diagnostics under "
                  << (query_result_root_dir / "log") << "\n";
        return -1;
    }
    if (!RunSupport::WritePreparedQueryFilterFile(query_result_root_dir, prepared_queries))
    {
        std::cerr << "[Fatal] Failed to write query boolean filters under "
                  << (query_result_root_dir / "log") << "\n";
        return -1;
    }

    for (const auto &warning : query_warnings)
    {
        std::cout << "[Warn] " << warning << "\n";
    }

    if (prepared_queries.empty())
    {
        std::cout << "[Info] No valid query found in " << query_file << ". Exit with code 0.\n";
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
    dataset.bucket_centroids = dataset_buffers.centroids.data();

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
    { // NPU scope: db, groups, scheduler must destruct before aclFinalize
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

    SharedProgressStats progress_stats;
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "[ParallelTest] Query objects prepared. count=" << query_ptrs.size() << "\n";

    // Warmup: push independent dummy queries through every WorkerGroup to pre-load
    // NPU kernels, warm HBM caches, and initialize thread-local buffers.
    // Must use separate Query objects to avoid contaminating result state.
    {
        const size_t warmup_count = std::max(static_cast<size_t>(group_count) * 32, static_cast<size_t>(64));
        std::vector<std::unique_ptr<Query>> warmup_queries;
        warmup_queries.reserve(warmup_count);
        const auto &proto = prepared_queries[0];
        for (size_t i = 0; i < warmup_count; ++i)
        {
            auto wq = std::make_unique<Query>();
            wq->Reset(proto.query_vec, proto.filter_expr, proto.top_k, 0);
            scheduler.Push(wq.get());
            warmup_queries.push_back(std::move(wq));
        }
        std::cout << "[Warmup] Submitting " << warmup_count << " warmup queries...\n";
        size_t warmup_done = 0;
        while (warmup_done < warmup_count)
        {
            if (scheduler.PopResult() != nullptr)
            {
                ++warmup_done;
            }
            else
            {
                std::this_thread::yield();
            }
        }
        std::cout << "[Warmup] " << warmup_count << " warmup queries completed.\n";
        warmup_queries.clear();
    }

    std::cout << "[ParallelTest] Sequential rate-limited submit begins. target_qps="
              << kParallelTargetQps
              << ", injector_tick_us=" << kInjectorTick.count() << "\n";

    const auto measure_start = std::chrono::steady_clock::now();
    ParallelRunStats run_stats;
    std::thread injector([&]() {
        run_stats = RunRateLimitedInjector(query_ptrs, scheduler, progress_stats, measure_start);
    });

    size_t completed_count = 0;
    auto next_progress_report = measure_start + std::chrono::seconds(1);
    while (completed_count < query_ptrs.size())
    {
        if (scheduler.PopResult() != nullptr)
        {
            ++completed_count;
            SampleTaskQueueSize(progress_stats, scheduler.GetTaskQueueSize());
        }

        const auto now = std::chrono::steady_clock::now();
        if (now >= next_progress_report)
        {
            PrintParallelProgress(progress_stats.submitted_count.load(std::memory_order_relaxed),
                                  completed_count,
                                  query_ptrs.size());
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
    if (injector.joinable())
    {
        injector.join();
    }
    const auto batch_end = std::chrono::steady_clock::now();

    std::chrono::duration<double, std::milli> total_latency_ms = batch_end - measure_start;
    run_stats.total_complete_duration_ms = total_latency_ms.count();
    const double total_latency_seconds = run_stats.total_complete_duration_ms / 1000.0;
    run_stats.complete_qps = (total_latency_seconds > 0.0)
                                 ? (static_cast<double>(query_ptrs.size()) / total_latency_seconds)
                                 : 0.0;
    const double avg_query_latency_ms =
        query_ptrs.empty()
            ? 0.0
            : (run_stats.total_complete_duration_ms / static_cast<double>(query_ptrs.size()));
    const unsigned long long queue_size_sum =
        progress_stats.task_queue_size_sum.load(std::memory_order_relaxed);
    const size_t queue_sample_count =
        progress_stats.task_queue_sample_count.load(std::memory_order_relaxed);
    run_stats.max_task_queue_size =
        progress_stats.max_task_queue_size.load(std::memory_order_relaxed);
    run_stats.avg_task_queue_size =
        (queue_sample_count > 0)
            ? (static_cast<double>(queue_size_sum) / static_cast<double>(queue_sample_count))
            : 0.0;

    std::cout << "\n========================================================\n";
    std::cout << "             Parallel Rate-Limited Test Result          \n";
    std::cout << "========================================================\n";
    std::cout << "Query Count                    : " << query_ptrs.size() << "\n";
    std::cout << "Submitted Query Count          : "
              << progress_stats.submitted_count.load(std::memory_order_relaxed) << "\n";
    std::cout << "Completed Query Count          : " << completed_count << "\n";
    std::cout << "Measure Submit Duration        : " << run_stats.measure_submit_duration_ms
              << " ms\n";
    std::cout << "Measure Submit QPS             : " << run_stats.measure_submit_qps << "\n";
    std::cout << "Total Complete Duration        : " << run_stats.total_complete_duration_ms
              << " ms\n";
    std::cout << "Complete QPS                   : " << run_stats.complete_qps << "\n";
    std::cout << "Average Query Latency(Batch/N) : " << avg_query_latency_ms << " ms\n";
    std::cout << "Max Submit Lag                 : " << run_stats.max_submit_lag_ms << " ms\n";
    std::cout << "Max Task Queue Size            : " << run_stats.max_task_queue_size << "\n";
    std::cout << "Avg Task Queue Size            : " << run_stats.avg_task_queue_size << "\n";
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
    std::vector<size_t> process_round_count;
    std::vector<size_t> process_round_count_level_2;
    std::vector<size_t> searched_l2_bucket_count;
    std::vector<size_t> searched_l2_nonzero_count;
    std::vector<size_t> searched_l2_allzero_count;
    std::vector<std::vector<double>> bucket_selectivity_per_query;
    if (!CollectParallelLatencyMetrics(queries,
                                       total_end_to_end_ms,
                                       bucket_level_ivf_ms,
                                       bucket_level_ivf_filter_eval_ms,
                                       bucket_level_ivf_centroid_score_ms,
                                       bucket_level_ivf_result_pack_ms,
                                       candidate_bucket_merge_ms,
                                       npu_async_launch_ms,
                                       npu_submit_ms,
                                       inbucket_attr_filter_overlapped_ms,
                                       wait_npu_flag_ms,
                                       result_collection_ms,
                                       final_merge_ms,
                                       npu_mask_filter_launch_ms,
                                       npu_mask_filter_h2d_ms,
                                       npu_mask_filter_kernel_exec_ms,
                                       npu_mask_filter_d2h_ms,
                                       process_round_count,
                                       process_round_count_level_2,
                                       searched_l2_bucket_count,
                                       searched_l2_nonzero_count,
                                       searched_l2_allzero_count,
                                       bucket_selectivity_per_query))
    {
        return -1;
    }

    std::vector<double> recall_rate_percent;
    ParallelRecallSummary recall_summary;
    if (!CollectParallelRecallMetrics(dataset,
                                      prepared_queries,
                                      queries,
                                      ground_truth_cache.get(),
                                      recall_rate_percent,
                                      recall_summary))
    {
        return -1;
    }

    std::cout << "\n========================================================\n";
    std::cout << "                 Parallel Recall Result                 \n";
    std::cout << "========================================================\n";
    std::cout << "Average Recall                : " << recall_summary.average_recall << "\n";
    std::cout << "Average Recall Percent        : "
              << (recall_summary.average_recall * 100.0) << " %\n";
    std::cout << "Ground-Truth Cache Hit Count  : " << recall_summary.cache_hit_count << "\n";
    std::cout << "Ground-Truth BruteForce Count : " << recall_summary.brute_force_count << "\n";
    std::cout << "========================================================\n";

    if (!FlushParallelOutputs(query_result_root_dir,
                              total_end_to_end_ms,
                              recall_rate_percent,
                              bucket_level_ivf_ms,
                              bucket_level_ivf_filter_eval_ms,
                              bucket_level_ivf_centroid_score_ms,
                              bucket_level_ivf_result_pack_ms,
                              candidate_bucket_merge_ms,
                              npu_async_launch_ms,
                              npu_submit_ms,
                              inbucket_attr_filter_overlapped_ms,
                              wait_npu_flag_ms,
                              result_collection_ms,
                              final_merge_ms,
                              npu_mask_filter_launch_ms,
                              npu_mask_filter_h2d_ms,
                              npu_mask_filter_kernel_exec_ms,
                              npu_mask_filter_d2h_ms,
                              process_round_count,
                              process_round_count_level_2,
                              searched_l2_bucket_count,
                              searched_l2_nonzero_count,
                              searched_l2_allzero_count,
                              bucket_selectivity_per_query,
                              run_stats.complete_qps))
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
    } // end NPU scope: groups, scheduler, db destruct here (releases pinned memory)

    std::cout << "[System] NPU scope ended. Calling aclFinalize...\n";
    npuAPI::Finalize();
    return 0;
}
