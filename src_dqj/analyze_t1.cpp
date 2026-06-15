#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Clustering/run_clustering.h"
#include "DataBaseCPU/DataBaseCPU.h"
#include "Query/Query.h"
#include "Schedule/Scheduler.h"
#include "Schedule/WorkerGroup.h"
#include "analyze/SelectionRateAnalyzer.h"
#include "utils/DataReader.h"
#include "utils/RunSupport.h"

namespace fs = RunSupport::fs;

namespace
{
// ---------------------------------------------------------------------------
//  Config path helpers (dqj: config-driven data loading)
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
//  Query object pool
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
//  Analysis data structures
// ---------------------------------------------------------------------------
struct RunStats
{
    size_t query_count = 0;
    double total_complete_duration_ms = 0.0;
    double complete_qps = 0.0;
};

enum class StageKind : size_t
{
    BucketLevelIvf = 0,
    CandidateBucketMerge = 1,
    InbucketAttrFilter = 2,
    WaitNpuFlag = 3,
    ResultCollection = 4,
    FinalMerge = 5
};

constexpr size_t kTrackedStageCount = 6;

struct AnalyzeQueryRecord
{
    size_t query_index = 0;
    size_t query_id = 0;
    double total_end_to_end_ms = 0.0;
    double bucket_level_ivf_ms = 0.0;
    double candidate_bucket_merge_ms = 0.0;
    double inbucket_attr_filter_overlapped_ms = 0.0;
    double wait_npu_flag_ms = 0.0;
    double result_collection_ms = 0.0;
    double final_merge_ms = 0.0;
    std::vector<uint32_t> searched_level_2_bucket_ids;
};

struct NumericSummary
{
    size_t count = 0;
    double mean = 0.0;
    double median = 0.0;
    double max = 0.0;
};

enum class QueryLatencyGroup : size_t
{
    BelowP50 = 0,
    P50ToP80 = 1,
    P80ToP90 = 2,
    P90ToP99 = 3,
    P99 = 4
};

constexpr size_t kQueryLatencyGroupCount = 5;

// ---------------------------------------------------------------------------
//  Utility functions
// ---------------------------------------------------------------------------
const char *StageName(StageKind stage)
{
    switch (stage)
    {
    case StageKind::BucketLevelIvf:
        return "bucket_level_ivf_ms";
    case StageKind::CandidateBucketMerge:
        return "candidate_bucket_merge_ms";
    case StageKind::InbucketAttrFilter:
        return "inbucket_attr_filter_overlapped_ms";
    case StageKind::WaitNpuFlag:
        return "wait_npu_flag_ms";
    case StageKind::ResultCollection:
        return "result_collection_ms";
    case StageKind::FinalMerge:
        return "final_merge_ms";
    }
    return "unknown_stage";
}

double GetStageValue(const AnalyzeQueryRecord &record, StageKind stage)
{
    switch (stage)
    {
    case StageKind::BucketLevelIvf:
        return record.bucket_level_ivf_ms;
    case StageKind::CandidateBucketMerge:
        return record.candidate_bucket_merge_ms;
    case StageKind::InbucketAttrFilter:
        return record.inbucket_attr_filter_overlapped_ms;
    case StageKind::WaitNpuFlag:
        return record.wait_npu_flag_ms;
    case StageKind::ResultCollection:
        return record.result_collection_ms;
    case StageKind::FinalMerge:
        return record.final_merge_ms;
    }
    return 0.0;
}

bool IsCpuStage(StageKind stage)
{
    return stage != StageKind::WaitNpuFlag;
}

std::string ComponentName(StageKind stage)
{
    return IsCpuStage(stage) ? "CPU" : "NPU";
}

NumericSummary ComputeNumericSummary(const std::vector<double> &values)
{
    NumericSummary summary;
    summary.count = values.size();
    if (values.empty())
    {
        return summary;
    }

    double sum = 0.0;
    for (double value : values)
    {
        sum += value;
        summary.max = std::max(summary.max, value);
    }
    summary.mean = sum / static_cast<double>(values.size());

    std::vector<double> sorted = values;
    std::sort(sorted.begin(), sorted.end());
    const size_t mid = sorted.size() / 2;
    if ((sorted.size() % 2U) == 0U)
    {
        summary.median = (sorted[mid - 1] + sorted[mid]) * 0.5;
    }
    else
    {
        summary.median = sorted[mid];
    }

    return summary;
}

double SafePercent(double numerator, double denominator)
{
    if (denominator <= 0.0)
    {
        return 0.0;
    }
    return numerator * 100.0 / denominator;
}

double SafeRatio(double numerator, double denominator)
{
    if (denominator == 0.0)
    {
        if (numerator == 0.0)
        {
            return 0.0;
        }
        return std::numeric_limits<double>::infinity();
    }
    return numerator / denominator;
}

StageKind DominantStage(const AnalyzeQueryRecord &record)
{
    StageKind best_stage = StageKind::BucketLevelIvf;
    double best_value = GetStageValue(record, best_stage);
    for (size_t stage_idx = 1; stage_idx < kTrackedStageCount; ++stage_idx)
    {
        const StageKind stage = static_cast<StageKind>(stage_idx);
        const double value = GetStageValue(record, stage);
        if (value > best_value)
        {
            best_value = value;
            best_stage = stage;
        }
    }
    return best_stage;
}

std::string VerdictFromDominantCounts(size_t cpu_dominant_count,
                                      size_t npu_dominant_count)
{
    const size_t total = cpu_dominant_count + npu_dominant_count;
    if (total == 0)
    {
        return "NoData";
    }

    const double cpu_share = static_cast<double>(cpu_dominant_count) / static_cast<double>(total);
    const double npu_share = static_cast<double>(npu_dominant_count) / static_cast<double>(total);
    if (cpu_share >= 0.7)
    {
        return "CPU-dominant";
    }
    if (npu_share >= 0.7)
    {
        return "NPU-dominant";
    }
    return "Mixed";
}

const char *QueryLatencyGroupName(QueryLatencyGroup group)
{
    switch (group)
    {
    case QueryLatencyGroup::BelowP50:
        return "p50_below";
    case QueryLatencyGroup::P50ToP80:
        return "p50_p80";
    case QueryLatencyGroup::P80ToP90:
        return "p80_p90";
    case QueryLatencyGroup::P90ToP99:
        return "p90_p99";
    case QueryLatencyGroup::P99:
        return "p99";
    }
    return "unknown";
}

size_t PercentileCount(size_t total_count, double ratio)
{
    if (total_count == 0)
    {
        return 0;
    }
    return std::min(total_count,
                    std::max<size_t>(1,
                                     static_cast<size_t>(
                                         std::ceil(static_cast<double>(total_count) * ratio))));
}

QueryLatencyGroup ClassifyLatencyGroupByDescendingRank(size_t descending_rank,
                                                       size_t total_count)
{
    const size_t p99_count = PercentileCount(total_count, 0.01);
    const size_t top_10_count = std::max(p99_count, PercentileCount(total_count, 0.10));
    const size_t top_20_count = std::max(top_10_count, PercentileCount(total_count, 0.20));
    const size_t top_50_count = std::max(top_20_count, PercentileCount(total_count, 0.50));

    if (descending_rank < p99_count)
    {
        return QueryLatencyGroup::P99;
    }
    if (descending_rank < top_10_count)
    {
        return QueryLatencyGroup::P90ToP99;
    }
    if (descending_rank < top_20_count)
    {
        return QueryLatencyGroup::P80ToP90;
    }
    if (descending_rank < top_50_count)
    {
        return QueryLatencyGroup::P50ToP80;
    }
    return QueryLatencyGroup::BelowP50;
}

// ---------------------------------------------------------------------------
//  Build analysis record from completed Query
// ---------------------------------------------------------------------------
bool BuildAnalyzeRecord(size_t query_index,
                        const Query &query,
                        AnalyzeQueryRecord &record)
{
    if (query.start_time_for_parallel_record == std::chrono::steady_clock::time_point{})
    {
        std::cerr << "[AnalyzeT1] Missing start_time_for_parallel_record for query line "
                  << query.query_id << "\n";
        return false;
    }
    if (query.end_time_for_parallel_record == std::chrono::steady_clock::time_point{})
    {
        std::cerr << "[AnalyzeT1] Missing end_time_for_parallel_record for query line "
                  << query.query_id << "\n";
        return false;
    }
    if (query.end_time_for_parallel_record < query.start_time_for_parallel_record)
    {
        std::cerr << "[AnalyzeT1] Invalid timing order for query line "
                  << query.query_id << "\n";
        return false;
    }

    record = AnalyzeQueryRecord{};
    record.query_index = query_index;
    record.query_id = query.query_id;
    record.total_end_to_end_ms =
        std::chrono::duration<double, std::milli>(query.end_time_for_parallel_record -
                                                  query.start_time_for_parallel_record)
            .count();
    record.bucket_level_ivf_ms = query.timing_metrics.bucket_level_ivf_ms;
    record.candidate_bucket_merge_ms = query.timing_metrics.candidate_bucket_merge_ms;
    record.inbucket_attr_filter_overlapped_ms =
        query.timing_metrics.inbucket_attr_filter_overlapped_ms;
    record.wait_npu_flag_ms = query.timing_metrics.wait_npu_flag_ms;
    record.result_collection_ms = query.timing_metrics.result_collection_ms;
    record.final_merge_ms = query.timing_metrics.final_merge_ms;
    record.searched_level_2_bucket_ids = query.searched_level_2_bucket_ids;
    return true;
}

// ---------------------------------------------------------------------------
//  Per-bucket selection rate computation
// ---------------------------------------------------------------------------
std::vector<double> BuildQuerySelectionRates(const InputDataset &dataset,
                                             const DataBaseCPU &db,
                                             const DataReader::PreparedQuery &prepared_query,
                                             const AnalyzeQueryRecord &record,
                                             bool parallelize_across_buckets)
{
    Query query;
    query.Reset(prepared_query.query_vec,
                prepared_query.filter_expr,
                prepared_query.top_k,
                prepared_query.line_no);

    const uint32_t bitmap_stride = static_cast<uint32_t>((total_tag_num + 63) / 64);
    const size_t bucket_count = record.searched_level_2_bucket_ids.size();
    std::vector<double> rates(bucket_count, 0.0);
    std::atomic<size_t> processed_bucket_count{0};
    const size_t progress_step = std::max<size_t>(1, bucket_count / 20);

    std::cout << "[AnalyzeT1] Query " << record.query_id
              << " selection-rate analysis starts. level2_bucket_count=" << bucket_count;
#ifdef _OPENMP
    std::cout << ", omp_max_threads=" << omp_get_max_threads();
#endif
    std::cout << "\n";

#if defined(_OPENMP)
#pragma omp parallel for if(parallelize_across_buckets) schedule(dynamic, 1)
#endif
    for (int bucket_idx = 0; bucket_idx < static_cast<int>(bucket_count); ++bucket_idx)
    {
        const uint32_t bucket_id = record.searched_level_2_bucket_ids[static_cast<size_t>(bucket_idx)];
        const Bucket &bucket = db.get_bucket(bucket_id);
        const uint32_t total_docs = bucket.get_doc_num();
        uint32_t matched_docs = 0;

        if (total_docs != 0)
        {
            for (uint32_t doc_id : bucket.get_global_ids())
            {
                if (query.evaluate_single_doc_filter(doc_id, dataset.tag_bitmaps, bitmap_stride))
                {
                    ++matched_docs;
                }
            }
        }

        const double selection_rate =
            (total_docs == 0)
                ? 0.0
                : (static_cast<double>(matched_docs) / static_cast<double>(total_docs));
        rates[static_cast<size_t>(bucket_idx)] = selection_rate;

        const size_t finished = processed_bucket_count.fetch_add(1, std::memory_order_relaxed) + 1;
        if (finished == bucket_count || (finished % progress_step) == 0)
        {
#if defined(_OPENMP)
#pragma omp critical(analyze_t1_selection_rate_progress_log)
#endif
            {
                std::cout << "[AnalyzeT1] Query " << record.query_id
                          << " bucket progress " << finished
                          << "/" << bucket_count << "\n";
            }
        }
    }

    std::cout << "[AnalyzeT1] Query " << record.query_id
              << " selection-rate analysis finished.\n";
    return rates;
}

// ---------------------------------------------------------------------------
//  Output writers
// ---------------------------------------------------------------------------
fs::path GetAnalyzeOutputDir(const fs::path &query_result_root_dir)
{
    return query_result_root_dir / "analyze";
}

bool WriteP99QueryIdFile(const fs::path &output_file,
                         const std::vector<AnalyzeQueryRecord> &records,
                         const std::vector<size_t> &p99_query_id_sorted_indices)
{
    std::ofstream out(output_file, std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
        std::cerr << "[Writer] Failed to open p99 query id file: " << output_file << "\n";
        return false;
    }

    for (size_t query_index : p99_query_id_sorted_indices)
    {
        out << records[query_index].query_id << '\n';
    }

    return out.good();
}

std::vector<size_t> BuildP99QueryIdSortedIndices(const std::vector<AnalyzeQueryRecord> &records,
                                                 const std::vector<size_t> &p99_sorted_indices)
{
    std::vector<size_t> query_id_sorted_indices = p99_sorted_indices;
    std::sort(query_id_sorted_indices.begin(),
              query_id_sorted_indices.end(),
              [&](size_t lhs, size_t rhs) {
                  if (records[lhs].query_id != records[rhs].query_id)
                  {
                      return records[lhs].query_id < records[rhs].query_id;
                  }
                  return records[lhs].query_index < records[rhs].query_index;
              });
    return query_id_sorted_indices;
}

bool WriteGroupedStatsFile(
    const fs::path &output_file,
    const std::array<std::vector<double>, kQueryLatencyGroupCount> &grouped_values,
    const std::array<size_t, kQueryLatencyGroupCount> &query_count_per_group)
{
    std::ofstream out(output_file, std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
        std::cerr << "[Writer] Failed to open grouped stats file: " << output_file << "\n";
        return false;
    }

    out << std::fixed << std::setprecision(4);
    out << std::left  << std::setw(12) << "group"
        << std::right << std::setw(12) << "query_count"
        << std::right << std::setw(14) << "sample_count"
        << std::right << std::setw(12) << "mean"
        << std::right << std::setw(12) << "median"
        << std::right << std::setw(12) << "max" << '\n';
    for (size_t group_idx = 0; group_idx < kQueryLatencyGroupCount; ++group_idx)
    {
        const QueryLatencyGroup group = static_cast<QueryLatencyGroup>(group_idx);
        const NumericSummary summary = ComputeNumericSummary(grouped_values[group_idx]);
        out << std::left  << std::setw(12) << QueryLatencyGroupName(group)
            << std::right << std::setw(12) << query_count_per_group[group_idx]
            << std::right << std::setw(14) << grouped_values[group_idx].size()
            << std::right << std::setw(12) << summary.mean
            << std::right << std::setw(12) << summary.median
            << std::right << std::setw(12) << summary.max << '\n';
    }

    return out.good();
}

bool WriteSummaryFile(const fs::path &output_file,
                      const std::vector<AnalyzeQueryRecord> &records,
                      const std::vector<size_t> &p99_sorted_indices,
                      const std::vector<bool> &is_p99,
                      const RunStats &run_stats)
{
    std::ofstream out(output_file, std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
        std::cerr << "[Writer] Failed to open analyze_t1 summary file: " << output_file << "\n";
        return false;
    }

    std::array<std::vector<double>, kTrackedStageCount> p99_stage_values;
    std::array<std::vector<double>, kTrackedStageCount> non_p99_stage_values;
    std::array<size_t, kTrackedStageCount> dominant_stage_counts{};
    std::vector<double> p99_total_latency;
    std::vector<double> p99_wait_share_pct;
    size_t cpu_dominant_count = 0;
    size_t npu_dominant_count = 0;

    for (size_t i = 0; i < records.size(); ++i)
    {
        const AnalyzeQueryRecord &record = records[i];
        if (is_p99[i])
        {
            p99_total_latency.push_back(record.total_end_to_end_ms);
            p99_wait_share_pct.push_back(
                SafePercent(record.wait_npu_flag_ms, record.total_end_to_end_ms));
            for (size_t stage_idx = 0; stage_idx < kTrackedStageCount; ++stage_idx)
            {
                p99_stage_values[stage_idx].push_back(
                    GetStageValue(record, static_cast<StageKind>(stage_idx)));
            }

            const StageKind dominant_stage = DominantStage(record);
            dominant_stage_counts[static_cast<size_t>(dominant_stage)] += 1;
            if (dominant_stage == StageKind::WaitNpuFlag)
            {
                ++npu_dominant_count;
            }
            else
            {
                ++cpu_dominant_count;
            }
        }
        else
        {
            for (size_t stage_idx = 0; stage_idx < kTrackedStageCount; ++stage_idx)
            {
                non_p99_stage_values[stage_idx].push_back(
                    GetStageValue(record, static_cast<StageKind>(stage_idx)));
            }
        }
    }

    const NumericSummary p99_latency_summary = ComputeNumericSummary(p99_total_latency);
    const NumericSummary p99_wait_share_summary = ComputeNumericSummary(p99_wait_share_pct);
    const double p99_threshold_ms = p99_sorted_indices.empty()
                                        ? 0.0
                                        : records[p99_sorted_indices.back()].total_end_to_end_ms;

    out << std::fixed << std::setprecision(5);
    out << "========================================================\n";
    out << "                 Analyze T1 P99 Summary                 \n";
    out << "========================================================\n";
    out << "Total Query Count              : " << records.size() << "\n";
    out << "P99 Query Count                : " << p99_sorted_indices.size() << "\n";
    out << "P99 Threshold(total_end_to_end): " << p99_threshold_ms << " ms\n";
    out << "P99 Latency Mean               : " << p99_latency_summary.mean << " ms\n";
    out << "P99 Latency Median             : " << p99_latency_summary.median << " ms\n";
    out << "P99 Latency Max                : " << p99_latency_summary.max << " ms\n";
    out << "Total Complete Duration        : " << run_stats.total_complete_duration_ms << " ms\n";
    out << "Complete QPS                   : " << run_stats.complete_qps << "\n";
    out << "--------------------------------------------------------\n";
    out << "Stage Comparison\n";
    out << std::left  << std::setw(36) << "stage"
        << std::left  << std::setw(10) << "comp"
        << std::right << std::setw(14) << "p99_mean"
        << std::right << std::setw(14) << "p99_median"
        << std::right << std::setw(14) << "p99_max"
        << std::right << std::setw(14) << "other_mean"
        << std::right << std::setw(14) << "other_median"
        << std::right << std::setw(14) << "other_max"
        << std::right << std::setw(14) << "med_ratio"
        << std::right << std::setw(14) << "mean_ratio"
        << std::right << std::setw(14) << "dominant_cnt"
        << '\n';

    for (size_t stage_idx = 0; stage_idx < kTrackedStageCount; ++stage_idx)
    {
        const StageKind stage = static_cast<StageKind>(stage_idx);
        const NumericSummary p99_stage_summary = ComputeNumericSummary(p99_stage_values[stage_idx]);
        const NumericSummary non_p99_stage_summary =
            ComputeNumericSummary(non_p99_stage_values[stage_idx]);
        out << std::left  << std::setw(36) << StageName(stage)
            << std::left  << std::setw(10) << ComponentName(stage)
            << std::right << std::setw(14) << p99_stage_summary.mean
            << std::right << std::setw(14) << p99_stage_summary.median
            << std::right << std::setw(14) << p99_stage_summary.max
            << std::right << std::setw(14) << non_p99_stage_summary.mean
            << std::right << std::setw(14) << non_p99_stage_summary.median
            << std::right << std::setw(14) << non_p99_stage_summary.max
            << std::right << std::setw(14) << SafeRatio(p99_stage_summary.median, non_p99_stage_summary.median)
            << std::right << std::setw(14) << SafeRatio(p99_stage_summary.mean, non_p99_stage_summary.mean)
            << std::right << std::setw(14) << dominant_stage_counts[stage_idx]
            << '\n';
    }

    out << "--------------------------------------------------------\n";
    out << "CPU/NPU Dominance Among P99 (by max tracked stage)\n";
    out << "CPU-Dominant Query Count       : " << cpu_dominant_count << "\n";
    out << "NPU-Dominant Query Count       : " << npu_dominant_count << "\n";
    out << "Wait NPU Share Mean            : " << p99_wait_share_summary.mean << " %\n";
    out << "Wait NPU Share Median          : " << p99_wait_share_summary.median << " %\n";
    out << "Wait NPU Share Max             : " << p99_wait_share_summary.max << " %\n";
    out << "Count-based Verdict            : "
        << VerdictFromDominantCounts(cpu_dominant_count, npu_dominant_count) << "\n";
    out << "========================================================\n";

    return out.good();
}

bool RemoveObsoleteAnalyzeFiles(const fs::path &analyze_output_dir)
{
    const std::array<const char *, 5> obsolete_files = {
        "analyze_t1_p99_queries.txt",
        "analyze_t1_summary.txt",
        "analyze_t1_p99_query_ids.txt",
        "analyze_t1_p99_level_2_bucket_selection_rates.txt",
        "analyze_t1_p99_level_2_bucket_selection_rate_stats.txt"};

    for (const char *file_name : obsolete_files)
    {
        const fs::path file_path = analyze_output_dir / file_name;
        std::error_code ec;
        fs::remove(file_path, ec);
        if (ec)
        {
            std::cerr << "[Writer] Failed to remove obsolete analyze file: "
                      << file_path << ", error=" << ec.message() << "\n";
            return false;
        }
    }

    return true;
}
} // namespace

// ---------------------------------------------------------------------------
//  main
// ---------------------------------------------------------------------------
int main()
{
    // --- Config ---
    const fs::path config_path = RunSupport::ResolveConfigPath();
    std::cout << "[AnalyzeT1] Loading config from " << config_path << "...\n";
    if (!LoadParams(config_path.string(), ResourceConfigProfile::Parallel))
    {
        std::cerr << "[Error] Failed to load config: " << config_path << "\n";
        return -1;
    }
    RestoreSrcStyleRelativePathsFromConfig(config_path.string());

#ifdef _OPENMP
    if (cpu_core_count > 0)
    {
        omp_set_num_threads(cpu_core_count);
    }
#endif

    const fs::path query_result_root_dir =
        query_result_root.empty() ? fs::path(".") : fs::path(query_result_root);
    const fs::path analyze_output_dir = GetAnalyzeOutputDir(query_result_root_dir);
    const fs::path memory_log_file = query_result_root_dir / "log" / "memory_logs.txt";

    // --- Dataset ---
    std::cout << "[Loader] Loading dataset cache...\n";
    DataReader::DatasetBuffers dataset_buffers;
    std::unordered_map<std::string, int> tag_map;
    DataReader::LoadFailureReason dataset_failure = DataReader::LoadFailureReason::None;
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

    // --- Queries (dqj: config-driven with tag_map) ---
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
    std::cout << "[Analyze] analyze_t1=" << ::analyze_t1
              << ", batch0=" << ::valid_bucket_num_base_level_1
              << ", incremental=" << ::valid_bucket_num_incremental_level_1 << "\n";
    FilterExpCompiler::WarmUpThreadLocalBuffers();

    // --- Clustering ---
    std::cout << "[Clustering] Syncing clustering context...\n";
    std::vector<uint64_t> bucket_doc_offsets;
    std::vector<uint32_t> bucket_doc_ids;
    const int sync_result = SyncClusteringContextFromMain(dataset_buffers.vectors,
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

    // --- DataBaseCPU ---
    std::cout << "[System] Initializing DataBaseCPU...\n";
    DataBaseCPU db(dataset, std::move(bucket_layout));

    // --- Scheduler + WorkerGroups ---
    std::cout << "[System] Launching Worker Threads...\n";
    Scheduler scheduler;
    std::vector<std::unique_ptr<WorkerGroup>> groups;
    std::vector<std::thread> threads;
    QueryPool query_pool(static_cast<size_t>(query_pool_capacity));
    for (int i = 0; i < group_count; ++i)
    {
        groups.push_back(std::make_unique<WorkerGroup>(i, &db, &scheduler));
    }
    for (int i = 0; i < cpu_core_count; ++i)
    {
        const int gid = GetGroupId(i);
        threads.emplace_back(&WorkerGroup::Run, groups[gid].get(), i);
    }

    // --- Serial query execution loop ---
    std::cout << std::fixed << std::setprecision(5);
    std::cout << "[AnalyzeT1] Serial query loop begins. count="
              << prepared_queries.size() << "\n";

    RunStats run_stats;
    run_stats.query_count = prepared_queries.size();
    std::vector<AnalyzeQueryRecord> records(prepared_queries.size());
    MemoryEventLogCollector memory_log_collector(memory_events_log_enable != 0);
    if (memory_events_log_enable != 0)
    {
        memory_log_collector.Reserve(prepared_queries.size() * 8);
    }

    const auto batch_start = std::chrono::steady_clock::now();
    for (size_t idx = 0; idx < prepared_queries.size(); ++idx)
    {
        const auto &prepared = prepared_queries[idx];
        Query *query = query_pool.Acquire();
        if (query == nullptr)
        {
            std::cerr << "[Fatal] QueryPool exhausted. capacity=" << query_pool_capacity << "\n";
            return -1;
        }
        query->PrepareMemoryEventSession(prepared.line_no, memory_events_log_enable != 0);
        {
            ScopedMemoryEventSession memory_event_scope(query->memory_event_session());
            query->Reset(prepared.query_vec, prepared.filter_expr, prepared.top_k, prepared.line_no);
        }

        scheduler.Push(query);
        Query *completed_query = nullptr;
        while (completed_query == nullptr)
        {
            completed_query = scheduler.PopResult();
            if (completed_query == nullptr)
            {
                std::this_thread::yield();
            }
        }

        if (!BuildAnalyzeRecord(idx, *completed_query, records[idx]))
        {
            return -1;
        }
        completed_query->AppendMemoryEventLogs(memory_log_collector);
        query_pool.Release(completed_query);

        std::cout << "[Progress] completed=" << (idx + 1)
                  << "/" << prepared_queries.size() << std::endl;
    }
    const auto batch_end = std::chrono::steady_clock::now();

    run_stats.total_complete_duration_ms =
        std::chrono::duration<double, std::milli>(batch_end - batch_start).count();
    const double total_latency_seconds = run_stats.total_complete_duration_ms / 1000.0;
    run_stats.complete_qps = (total_latency_seconds > 0.0)
                                 ? (static_cast<double>(prepared_queries.size()) / total_latency_seconds)
                                 : 0.0;
    const double avg_query_latency_ms =
        prepared_queries.empty()
            ? 0.0
            : (run_stats.total_complete_duration_ms / static_cast<double>(prepared_queries.size()));

    std::cout << "\n========================================================\n";
    std::cout << "                  Serial Analyze T1 Result              \n";
    std::cout << "========================================================\n";
    std::cout << "Query Count                    : " << prepared_queries.size() << "\n";
    std::cout << "Total Complete Duration        : " << run_stats.total_complete_duration_ms
              << " ms\n";
    std::cout << "Complete QPS                   : " << run_stats.complete_qps << "\n";
    std::cout << "Average Query Latency(Batch/N) : " << avg_query_latency_ms << " ms\n";
    std::cout << "========================================================\n";

    // --- Shutdown worker threads ---
    std::cout << "[System] All queries completed. Shutting down worker threads...\n";
    for (auto &group : groups)
    {
        group->Stop();
    }
    std::cout << "[AnalyzeT1] Stop signal sent to all worker groups. Waiting for worker threads to exit...\n";
    for (auto &thread : threads)
    {
        if (thread.joinable())
        {
            thread.join();
        }
    }
    std::cout << "[AnalyzeT1] All worker threads joined.\n";

    // --- Classify queries into latency groups ---
    std::vector<size_t> sorted_indices(records.size());
    for (size_t i = 0; i < sorted_indices.size(); ++i)
    {
        sorted_indices[i] = i;
    }
    std::sort(sorted_indices.begin(),
              sorted_indices.end(),
              [&](size_t lhs, size_t rhs) {
                  return records[lhs].total_end_to_end_ms > records[rhs].total_end_to_end_ms;
              });

    std::vector<size_t> p99_sorted_indices;
    std::vector<bool> is_p99(records.size(), false);
    std::vector<QueryLatencyGroup> query_groups(records.size(), QueryLatencyGroup::BelowP50);
    std::array<size_t, kQueryLatencyGroupCount> query_count_per_group{};
    std::array<std::vector<double>, kQueryLatencyGroupCount> searched_bucket_count_values;
    std::array<size_t, kQueryLatencyGroupCount> selection_rate_sample_count_per_group{};

    p99_sorted_indices.reserve(PercentileCount(records.size(), 0.01));
    for (size_t rank = 0; rank < sorted_indices.size(); ++rank)
    {
        const size_t query_index = sorted_indices[rank];
        const QueryLatencyGroup group =
            ClassifyLatencyGroupByDescendingRank(rank, sorted_indices.size());
        query_groups[query_index] = group;
        ++query_count_per_group[static_cast<size_t>(group)];

        const double searched_bucket_count =
            static_cast<double>(records[query_index].searched_level_2_bucket_ids.size());
        searched_bucket_count_values[static_cast<size_t>(group)].push_back(searched_bucket_count);
        selection_rate_sample_count_per_group[static_cast<size_t>(group)] +=
            records[query_index].searched_level_2_bucket_ids.size();

        if (group == QueryLatencyGroup::P99)
        {
            p99_sorted_indices.push_back(query_index);
            is_p99[query_index] = true;
        }
    }
    const std::vector<size_t> p99_query_id_sorted_indices =
        BuildP99QueryIdSortedIndices(records, p99_sorted_indices);

    // --- Per-bucket selection rate analysis ---
    std::array<std::vector<double>, kQueryLatencyGroupCount> selection_rate_values_by_group;
    for (size_t group_idx = 0; group_idx < kQueryLatencyGroupCount; ++group_idx)
    {
        selection_rate_values_by_group[group_idx].reserve(
            selection_rate_sample_count_per_group[group_idx]);
    }

    std::cout << "[AnalyzeT1] Query groups ready. P99 count="
              << p99_query_id_sorted_indices.size()
              << ". Starting level-2 bucket selection-rate analysis for all queries...\n";

    std::vector<std::vector<double>> query_selection_rates_by_query(records.size());
    std::atomic<size_t> completed_selection_rate_queries{0};

#if defined(_OPENMP)
#pragma omp parallel
    {
        FilterExpCompiler::WarmUpThreadLocalBuffers();
#pragma omp for schedule(dynamic, 1)
        for (int i = 0; i < static_cast<int>(records.size()); ++i)
        {
            const size_t query_index = static_cast<size_t>(i);
            const AnalyzeQueryRecord &record = records[query_index];
            const QueryLatencyGroup group = query_groups[query_index];
#pragma omp critical(analyze_t1_selection_rate_query_log)
            {
                std::cout << "[AnalyzeT1] Selection-rate start "
                          << (query_index + 1)
                          << "/" << records.size()
                          << ", query_id=" << record.query_id
                          << ", group=" << QueryLatencyGroupName(group)
                          << ", level2_bucket_count="
                          << record.searched_level_2_bucket_ids.size()
                          << "\n";
            }
            query_selection_rates_by_query[query_index] =
                BuildQuerySelectionRates(dataset,
                                         db,
                                         prepared_queries[record.query_index],
                                         record,
                                         false);

            const size_t finished =
                completed_selection_rate_queries.fetch_add(1, std::memory_order_relaxed) + 1;
#pragma omp critical(analyze_t1_selection_rate_query_log)
            {
                std::cout << "[AnalyzeT1] Selection-rate finished "
                          << finished
                          << "/" << records.size()
                          << ", query_id=" << record.query_id
                          << "\n";
            }
        }
    }
#else
    for (size_t query_index = 0; query_index < records.size(); ++query_index)
    {
        const AnalyzeQueryRecord &record = records[query_index];
        const QueryLatencyGroup group = query_groups[query_index];
        std::cout << "[AnalyzeT1] Selection-rate start "
                  << (query_index + 1)
                  << "/" << records.size()
                  << ", query_id=" << record.query_id
                  << ", group=" << QueryLatencyGroupName(group)
                  << ", level2_bucket_count=" << record.searched_level_2_bucket_ids.size()
                  << "\n";
        query_selection_rates_by_query[query_index] =
            BuildQuerySelectionRates(dataset,
                                     db,
                                     prepared_queries[record.query_index],
                                     record,
                                     true);
        const size_t finished =
            completed_selection_rate_queries.fetch_add(1, std::memory_order_relaxed) + 1;
        std::cout << "[AnalyzeT1] Selection-rate finished "
                  << finished
                  << "/" << records.size()
                  << ", query_id=" << record.query_id
                  << "\n";
    }
#endif

    for (size_t query_index = 0; query_index < records.size(); ++query_index)
    {
        const QueryLatencyGroup group = query_groups[query_index];
        std::vector<double> &group_values =
            selection_rate_values_by_group[static_cast<size_t>(group)];
        const std::vector<double> &query_selection_rates =
            query_selection_rates_by_query[query_index];
        group_values.insert(group_values.end(),
                            query_selection_rates.begin(),
                            query_selection_rates.end());
    }
    std::cout << "[AnalyzeT1] Level-2 bucket selection-rate analysis finished.\n";

    // --- Write output files ---
    std::cout << "[AnalyzeT1] Writing analyze result files into " << analyze_output_dir << " ...\n";
    if (!RunSupport::EnsureOutputDirectory(analyze_output_dir))
    {
        std::cerr << "[Fatal] Failed to create analyze output directory: "
                  << analyze_output_dir << "\n";
        return -1;
    }
    std::cout << "[AnalyzeT1] Analyze output directory is ready.\n";
    if (!RemoveObsoleteAnalyzeFiles(analyze_output_dir))
    {
        return -1;
    }
    if (!WriteSummaryFile(analyze_output_dir / "summary.txt",
                          records,
                          p99_sorted_indices,
                          is_p99,
                          run_stats))
    {
        return -1;
    }
    std::cout << "[AnalyzeT1] Wrote " << (analyze_output_dir / "summary.txt") << "\n";
    if (!WriteP99QueryIdFile(analyze_output_dir / "p99_query_ids.txt",
                             records,
                             p99_query_id_sorted_indices))
    {
        return -1;
    }
    std::cout << "[AnalyzeT1] Wrote " << (analyze_output_dir / "p99_query_ids.txt") << "\n";
    if (!WriteGroupedStatsFile(
            analyze_output_dir / "level_2_bucket_selection_rate_stats.txt",
            selection_rate_values_by_group,
            query_count_per_group))
    {
        return -1;
    }
    std::cout << "[AnalyzeT1] Wrote "
              << (analyze_output_dir / "level_2_bucket_selection_rate_stats.txt") << "\n";
    if (!WriteGroupedStatsFile(
            analyze_output_dir / "level_2_bucket_count_stats.txt",
            searched_bucket_count_values,
            query_count_per_group))
    {
        return -1;
    }
    std::cout << "[AnalyzeT1] Wrote "
              << (analyze_output_dir / "level_2_bucket_count_stats.txt") << "\n";
    if (memory_events_log_enable != 0)
    {
        if (!memory_log_collector.Flush(memory_log_file))
        {
            std::cerr << "[Fatal] Failed to write " << memory_log_file << ".\n";
            return -1;
        }
        std::cout << "[AnalyzeT1] Memory logs flushed to " << memory_log_file << "\n";
    }

    std::cout << "[AnalyzeT1] Summary written to "
              << (analyze_output_dir / "summary.txt") << "\n";
    std::cout << "[AnalyzeT1] P99 query ids written to "
              << (analyze_output_dir / "p99_query_ids.txt") << "\n";
    std::cout << "[AnalyzeT1] Grouped level-2 bucket selection-rate stats written to "
              << (analyze_output_dir / "level_2_bucket_selection_rate_stats.txt")
              << "\n";
    std::cout << "[AnalyzeT1] Grouped level-2 bucket count stats written to "
              << (analyze_output_dir / "level_2_bucket_count_stats.txt")
              << "\n";
    std::cout << "[System] All threads stopped. Exiting safely.\n";
    return 0;
}
