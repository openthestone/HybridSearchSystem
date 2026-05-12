#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
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

#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = RunSupport::fs;

namespace
{

// =========================================================
// Thresholds
// =========================================================
constexpr double kSlowQueryFilterMsThreshold = 5.0;
constexpr uint32_t kLargeBucketDocThreshold = 50000;
constexpr uint32_t kManyBucketsThreshold = 200;
constexpr int kManyPredicatesThreshold = 10;
constexpr int kLargeOrWidthThreshold = 8;
constexpr double kLowSelectivityRatio = 0.3;
constexpr double kHighSurvivorRatio = 0.5;
constexpr double kManyBitmapMbThreshold = 5.0;
constexpr double kWastedScanBitmapMbThreshold = 2.0;
constexpr double kWastedScanSurvivorRatio = 0.01;
constexpr double kLowFilterPowerBitmapMb = 2.0;
constexpr double kLowFilterPowerSurvivorRatio = 0.3;
constexpr double kDenseBitmapSelectivity = 0.5;
constexpr double kResultCollectionPctThreshold = 0.2;
constexpr double kNpuWaitPctThreshold = 0.2;
constexpr double kEstimatedMemBandwidthBytesPerMs = 40.0 * 1024.0 * 1024.0; // 40 GB/s
constexpr double kUnaccountedWaitPctThreshold = 0.15;
constexpr double kLowSelectivityPredThreshold = 0.5;

// =========================================================
// Reason flags
// =========================================================
enum ReasonFlag : uint32_t
{
    REASON_LARGE_BUCKET = 1u << 0,
    REASON_MANY_BUCKETS = 1u << 1,
    REASON_MANY_PREDICATES = 1u << 2,
    REASON_LARGE_OR = 1u << 3,
    REASON_HAS_NOT = 1u << 4,
    REASON_LOW_SELECTIVITY = 1u << 5,
    REASON_HIGH_SURVIVOR_RATIO = 1u << 6,
    REASON_MANY_BITMAP_BYTES = 1u << 7,
    REASON_WASTED_BITMAP_SCAN = 1u << 8,
    REASON_FILTER_DOMINANT = 1u << 9,
    REASON_LOW_FILTER_POWER = 1u << 10,
    REASON_DENSE_BITMAP = 1u << 11,
    REASON_RESULT_COLLECTION_HEAVY = 1u << 12,
    REASON_NPU_WAIT = 1u << 13,
    REASON_UNACCOUNTED_WAIT = 1u << 14,
};

constexpr int kNumReasonFlags = 15;

struct ExprStats
{
    int num_predicates = 0;
    int num_and = 0;
    int num_or = 0;
    int num_not = 0;
    int max_or_width = 0;
    int expr_depth = 0;
    int total_or_terms = 0;
};

struct BitmapVolumeStats
{
    uint64_t words_read = 0;
    uint64_t words_written = 0;
    uint64_t and_ops = 0;
    uint64_t or_ops = 0;
    uint64_t not_ops = 0;
    uint64_t num_popcount_ops = 0;
};

struct BucketDistStats
{
    uint32_t num_buckets = 0;
    uint64_t total_candidates = 0;
    uint64_t total_survivors = 0;
    uint32_t largest_bucket = 0;
    uint32_t smallest_bucket = UINT32_MAX;
    double mean_bucket = 0.0;
    double survivor_ratio = 0.0;
    uint32_t num_empty_buckets = 0;
    uint32_t num_dense_buckets = 0;
};

struct SelectivityStats
{
    double min_pred_selectivity = 0.0;
    double max_pred_selectivity = 0.0;
    double mean_pred_selectivity = 0.0;
    double estimated_final_selectivity = 0.0;
    double not_result_density = 0.0;
};

struct TimingRow
{
    size_t query_index = 0;
    size_t query_id = 0;
    double total_end_to_end_ms = 0.0;
    double construct_ms = 0.0;
    int searched_bucket_count_level_1 = 0;
    double bucket_level_ivf_ms = 0.0;
    double candidate_bucket_merge_ms = 0.0;
    double npu_async_launch_ms = 0.0;
    double inbucket_attr_filter_ms = 0.0;
    double wait_npu_flag_ms = 0.0;
    double result_collection_ms = 0.0;
    double final_merge_ms = 0.0;
    std::string filter_expr;
};

struct AnalyzeRow
{
    size_t query_index = 0;
    size_t query_id = 0;
    bool is_p99 = false;
    int group_id = 0;
    int leader_id = 0;

    double total_end_to_end_ms = 0.0;
    double inbucket_attr_filter_ms = 0.0;
    double filter_pct = 0.0;
    double bucket_level_ivf_ms = 0.0;
    double candidate_bucket_merge_ms = 0.0;
    double npu_async_launch_ms = 0.0;
    double result_collection_ms = 0.0;
    double final_merge_ms = 0.0;
    double wait_npu_flag_ms = 0.0;
    double unaccounted_wait_ms = 0.0;

    ExprStats expr;
    BitmapVolumeStats bitmap;
    double bitmap_read_mb = 0.0;
    double bitmap_write_mb = 0.0;
    uint64_t bitmap_words_scanned = 0;
    BucketDistStats bucket;
    SelectivityStats selectivity;
    double estimated_cost_ms = 0.0;
    double bucket_imbalance_ratio = 0.0;
    int num_low_sel_pred = 0;
    std::string top5_bucket_sizes_str;
    std::string attr_expr_normalized;

    uint32_t reason_flags = 0;
    std::string reason_string;
    std::string filter_expr;
};

// =========================================================
// QueryPool (same as serial.cpp)
// =========================================================
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

// =========================================================
// Expression analysis
// =========================================================
ExprStats AnalyzeExpression(const std::vector<RPNItem> &rpn)
{
    ExprStats stats;
    if (rpn.empty())
    {
        return stats;
    }

    struct StackEntry
    {
        int depth;
        int or_terms; // number of leaf predicates under OR chain
    };
    std::vector<StackEntry> stack;
    stack.reserve(rpn.size());

    for (const auto &item : rpn)
    {
        if (!item.is_op)
        {
            ++stats.num_predicates;
            if (item.flags & 1)
            {
                ++stats.num_not;
            }
            stack.push_back({1, 0});
        }
        else
        {
            if (item.value == FilterOp8::OP_AND)
            {
                ++stats.num_and;
            }
            else if (item.value == FilterOp8::OP_OR)
            {
                ++stats.num_or;
            }

            if (stack.size() < 2)
            {
                continue;
            }
            StackEntry right = stack.back();
            stack.pop_back();
            StackEntry left = stack.back();
            stack.pop_back();

            int merged_or_terms = left.or_terms + right.or_terms;
            if (item.value == FilterOp8::OP_OR)
            {
                if (left.or_terms == 0)
                {
                    ++merged_or_terms;
                }
                if (right.or_terms == 0)
                {
                    ++merged_or_terms;
                }
                if (merged_or_terms > stats.max_or_width)
                {
                    stats.max_or_width = merged_or_terms;
                }
            }

            stack.push_back({std::max(left.depth, right.depth) + 1, merged_or_terms});
        }
    }

    stats.total_or_terms = (!stack.empty()) ? stack[0].or_terms : 0;

    if (!stack.empty())
    {
        stats.expr_depth = stack[0].depth;
    }

    return stats;
}

// =========================================================
// Normalized expression string from RPN
// =========================================================
std::string GenerateNormalizedExpr(const std::vector<RPNItem> &rpn)
{
    if (rpn.empty())
    {
        return "PASS_ALL";
    }

    struct StackStr
    {
        std::string str;
        int precedence; // 0=operand, 1=OR, 2=AND
    };

    std::vector<StackStr> stack;
    stack.reserve(rpn.size());

    for (const auto &item : rpn)
    {
        if (!item.is_op)
        {
            std::string s;
            if (item.flags & 1)
            {
                s = "!T" + std::to_string(item.value);
            }
            else
            {
                s = "T" + std::to_string(item.value);
            }
            stack.push_back({std::move(s), 0});
        }
        else
        {
            if (stack.size() < 2)
            {
                continue;
            }
            StackStr right = std::move(stack.back());
            stack.pop_back();
            StackStr left = std::move(stack.back());
            stack.pop_back();

            const char *op_str = (item.value == FilterOp8::OP_AND) ? "&" : "|";
            int new_prec = (item.value == FilterOp8::OP_AND) ? 2 : 1;

            std::string combined;
            auto wrap = [&](const StackStr &child) -> std::string
            {
                if (child.precedence > 0 && child.precedence < new_prec)
                {
                    return "(" + child.str + ")";
                }
                return child.str;
            };

            combined += wrap(left);
            combined += op_str;
            combined += wrap(right);

            stack.push_back({std::move(combined), new_prec});
        }
    }

    return stack.empty() ? "EMPTY" : stack[0].str;
}

// =========================================================
// Bitmap volume (pure counting, same block decomposition as search_bucket)
// =========================================================
BitmapVolumeStats ComputeBitmapVolume(const std::vector<RPNItem> &rpn, const Bucket &bucket)
{
    BitmapVolumeStats stats;
    if (rpn.empty())
    {
        return stats;
    }

    uint32_t stride = bucket.get_stride();
    if (stride == 0)
    {
        return stats;
    }

    uint32_t cache_line_bytes = cpu_cache_line_size > 0 ? static_cast<uint32_t>(cpu_cache_line_size) : 64;
    uint32_t block_u64 = std::max<uint32_t>(1, (cache_line_bytes / sizeof(uint64_t)) * 4);

    for (uint32_t block_start = 0; block_start < stride; block_start += block_u64)
    {
        uint32_t block_len = std::min(block_u64, stride - block_start);
        for (const auto &item : rpn)
        {
            if (!item.is_op)
            {
                if (item.flags & 1)
                {
                    // Inverted operand (NOT): read + write for inversion.
                    stats.words_read += block_len;
                    stats.words_written += block_len;
                    stats.not_ops++;
                }
                // Non-inverted operand: no read at this point —
                // the consuming operator accounts for reading it.
            }
            else
            {
                stats.words_read += 2 * block_len;
                stats.words_written += block_len;
                if (item.value == FilterOp8::OP_AND)
                {
                    stats.and_ops++;
                }
                else
                {
                    stats.or_ops++;
                }
            }
        }
    }

    return stats;
}

// =========================================================
// Bucket selection (IVF score + sort + expand to L2)
// =========================================================
struct BucketScore
{
    uint32_t bucket_id = 0;
    float score = 0.0f;
};

float ComputeIP(const float *lhs, const float *rhs)
{
    float dot = 0.0f;
#if defined(__aarch64__) || defined(__arm__)
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    int d = 0;
    for (; d + 3 < vector_dim; d += 4)
    {
        float32x4_t lv = vld1q_f32(lhs + d);
        float32x4_t rv = vld1q_f32(rhs + d);
        sum_vec = vmlaq_f32(sum_vec, lv, rv);
    }
    dot = vaddvq_f32(sum_vec);
    for (; d < vector_dim; ++d)
    {
        dot += lhs[d] * rhs[d];
    }
#else
    for (int d = 0; d < vector_dim; ++d)
    {
        dot += lhs[d] * rhs[d];
    }
#endif
    return dot;
}

std::vector<uint32_t> SelectCandidateBuckets(DataBaseCPU &db,
                                              Query &query,
                                              int target_l1_count)
{
    const float *centroids = db.get_centroids();
    const float *q_vec = query.query_vector.data();

    // Step 1: Determine valid L1 buckets via IVF filtering.
    std::vector<BucketScore> scores;

    if (query.filter_exp.BucketLevelIVF_RPN.empty())
    {
        // No IVF filter: all L1 buckets are valid candidates.
        scores.reserve(static_cast<size_t>(total_bucket_num_level_1));
        for (int b = 0; b < total_bucket_num_level_1; ++b)
        {
            const float *c = centroids + static_cast<size_t>(b) * static_cast<size_t>(vector_dim);
            float s = ComputeIP(q_vec, c);
            scores.push_back({static_cast<uint32_t>(b), s});
        }
    }
    else
    {
        // Evaluate IVF filter across all cores in group 0.
        // search_ivf returns LOCAL masks: bit i in core c's mask corresponds to
        // global L1 bucket (c * buckets_per_core + i). We must translate each
        // local set bit to a global bucket ID before scoring.
        // This matches WorkerGroup::ExecuteIVF (WorkerGroup.h:1044-1066).
        const BucketLevelIVF &ivf = db.get_bucket_level_ivf();
        const uint32_t bpc = ivf.get_buckets_per_core();
        std::vector<uint64_t, AlignedAllocator<uint64_t>> scratch_ivf;

        // Collect global L1 bucket IDs that pass the IVF filter.
        std::vector<uint32_t> valid_l1_ids;
        for (int core = 0; core < cores_per_group; ++core)
        {
            std::vector<uint64_t> core_mask;
            query.search_ivf(ivf, core_mask, scratch_ivf, core);
            const uint32_t base = static_cast<uint32_t>(core) * bpc;

            for (uint32_t wi = 0; wi < static_cast<uint32_t>(core_mask.size()); ++wi)
            {
                uint64_t word = core_mask[wi];
                if (word == 0)
                {
                    continue;
                }
                uint32_t word_base = wi * 64;
                while (word)
                {
                    int bit = __builtin_ctzll(word);
                    word &= word - 1;
                    uint32_t local_bid = word_base + static_cast<uint32_t>(bit);
                    if (local_bid >= bpc)
                    {
                        continue;
                    }
                    uint32_t global_bid = base + local_bid;
                    if (global_bid < static_cast<uint32_t>(total_bucket_num_level_1))
                    {
                        valid_l1_ids.push_back(global_bid);
                    }
                }
            }
        }

        // Score only valid L1 buckets.
        scores.reserve(valid_l1_ids.size());
        for (uint32_t global_bid : valid_l1_ids)
        {
            const float *c = centroids + static_cast<size_t>(global_bid) * static_cast<size_t>(vector_dim);
            float s = ComputeIP(q_vec, c);
            scores.push_back({global_bid, s});
        }
    }

    // Step 2: Sort by score descending.
    std::sort(scores.begin(), scores.end(), [](const BucketScore &a, const BucketScore &b)
              {
        if (a.score != b.score) return a.score > b.score;
        return a.bucket_id < b.bucket_id; });

    // Step 3: Take top N L1 buckets (matching real pipeline's searched count).
    int l1_count;
    if (target_l1_count > 0)
    {
        l1_count = std::min(static_cast<int>(scores.size()), target_l1_count);
    }
    else
    {
        l1_count = std::min(static_cast<int>(scores.size()),
                            valid_bucket_num_base_level_1);
    }

    // Step 4: Expand to L2 buckets.
    std::vector<uint32_t> l2_bucket_ids;
    for (int i = 0; i < l1_count; ++i)
    {
        uint32_t l1_id = scores[static_cast<size_t>(i)].bucket_id;
        uint32_t begin = db.get_level_2_bucket_begin(l1_id);
        uint32_t end = db.get_level_2_bucket_end(l1_id);
        for (uint32_t bid = begin; bid < end; ++bid)
        {
            l2_bucket_ids.push_back(bid);
        }
    }

    return l2_bucket_ids;
}

// =========================================================
// Survivors via popcount
// =========================================================
uint64_t PopcountMask(const std::vector<uint64_t> &mask, uint32_t doc_num)
{
    uint64_t count = 0;
    size_t full_words = static_cast<size_t>(doc_num / 64);
    uint32_t remaining_bits = doc_num % 64;

    for (size_t i = 0; i < full_words && i < mask.size(); ++i)
    {
        count += static_cast<uint64_t>(__builtin_popcountll(mask[i]));
    }

    if (remaining_bits > 0 && full_words < mask.size())
    {
        uint64_t last = mask[full_words];
        uint64_t valid_bits = (1ULL << remaining_bits) - 1;
        count += static_cast<uint64_t>(__builtin_popcountll(last & valid_bits));
    }

    return count;
}

// =========================================================
// Reason classification
// =========================================================
uint32_t ClassifyReason(const AnalyzeRow &row)
{
    uint32_t flags = 0;

    if (row.bucket.largest_bucket > kLargeBucketDocThreshold)
    {
        flags |= REASON_LARGE_BUCKET;
    }
    if (row.bucket.num_buckets > kManyBucketsThreshold)
    {
        flags |= REASON_MANY_BUCKETS;
    }
    if (row.expr.num_predicates > kManyPredicatesThreshold)
    {
        flags |= REASON_MANY_PREDICATES;
    }
    if (row.expr.max_or_width > kLargeOrWidthThreshold)
    {
        flags |= REASON_LARGE_OR;
    }
    if (row.expr.num_not > 0)
    {
        flags |= REASON_HAS_NOT;
    }
    if (row.bucket.survivor_ratio > kLowSelectivityRatio)
    {
        flags |= REASON_LOW_SELECTIVITY;
    }
    if (row.bucket.survivor_ratio > kHighSurvivorRatio)
    {
        flags |= REASON_HIGH_SURVIVOR_RATIO;
    }
    if (row.bitmap_read_mb > kManyBitmapMbThreshold)
    {
        flags |= REASON_MANY_BITMAP_BYTES;
    }
    if (row.bitmap_read_mb > kWastedScanBitmapMbThreshold &&
        row.bucket.survivor_ratio < kWastedScanSurvivorRatio)
    {
        flags |= REASON_WASTED_BITMAP_SCAN;
    }
    if (row.total_end_to_end_ms > 0.0 &&
        row.inbucket_attr_filter_ms / row.total_end_to_end_ms > 0.5)
    {
        flags |= REASON_FILTER_DOMINANT;
    }
    if (row.bitmap_read_mb > kLowFilterPowerBitmapMb &&
        row.bucket.survivor_ratio > kLowFilterPowerSurvivorRatio)
    {
        flags |= REASON_LOW_FILTER_POWER;
    }
    if (row.selectivity.max_pred_selectivity > kDenseBitmapSelectivity)
    {
        flags |= REASON_DENSE_BITMAP;
    }
    if (row.total_end_to_end_ms > 0.0 &&
        row.result_collection_ms / row.total_end_to_end_ms > kResultCollectionPctThreshold)
    {
        flags |= REASON_RESULT_COLLECTION_HEAVY;
    }
    if (row.total_end_to_end_ms > 0.0 &&
        row.wait_npu_flag_ms / row.total_end_to_end_ms > kNpuWaitPctThreshold)
    {
        flags |= REASON_NPU_WAIT;
    }
    if (row.total_end_to_end_ms > 0.0 &&
        row.unaccounted_wait_ms / row.total_end_to_end_ms > kUnaccountedWaitPctThreshold)
    {
        flags |= REASON_UNACCOUNTED_WAIT;
    }

    return flags;
}

std::string ReasonFlagsToString(uint32_t flags)
{
    if (flags == 0)
    {
        return "NONE";
    }

    struct FlagDef
    {
        uint32_t bit;
        const char *name;
    };
    static const FlagDef defs[] = {
        {REASON_LARGE_BUCKET, "LARGE_BUCKET"},
        {REASON_MANY_BUCKETS, "MANY_BUCKETS"},
        {REASON_MANY_PREDICATES, "MANY_PREDICATES"},
        {REASON_LARGE_OR, "LARGE_OR"},
        {REASON_HAS_NOT, "HAS_NOT"},
        {REASON_LOW_SELECTIVITY, "LOW_SELECTIVITY"},
        {REASON_HIGH_SURVIVOR_RATIO, "HIGH_SURVIVOR_RATIO"},
        {REASON_MANY_BITMAP_BYTES, "MANY_BITMAP_BYTES"},
        {REASON_WASTED_BITMAP_SCAN, "WASTED_BITMAP_SCAN"},
        {REASON_FILTER_DOMINANT, "FILTER_DOMINANT"},
        {REASON_LOW_FILTER_POWER, "LOW_FILTER_POWER"},
        {REASON_DENSE_BITMAP, "DENSE_BITMAP"},
        {REASON_RESULT_COLLECTION_HEAVY, "RESULT_COLLECTION_HEAVY"},
        {REASON_NPU_WAIT, "NPU_WAIT"},
        {REASON_UNACCOUNTED_WAIT, "UNACCOUNTED_WAIT"},
    };

    std::string result;
    for (const auto &d : defs)
    {
        if (flags & d.bit)
        {
            if (!result.empty())
            {
                result += "|";
            }
            result += d.name;
        }
    }
    return result;
}

// =========================================================
// CSV helpers
// =========================================================
std::string SanitizeCSV(const std::string &s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s)
    {
        if (c == ',')
        {
            out += ';';
        }
        else if (c == '\n' || c == '\r')
        {
            out += ' ';
        }
        else if (c == '"')
        {
            out += '\'';
        }
        else
        {
            out += c;
        }
    }
    return out;
}

bool WriteCSVHeader(std::ofstream &out)
{
    out << "query_idx,query_id,is_p99,group_id,leader_id,"
        << "total_ms,filter_ms,filter_pct,"
        << "ivf_ms,merge_ms,npu_launch_ms,collect_ms,final_merge_ms,unaccounted_ms,"
        << "num_predicates,num_and,num_or,num_not,max_or_width,expr_depth,total_or_terms,num_low_sel_pred,"
        << "bitmap_read_mb,bitmap_write_mb,bitmap_words_scanned,and_ops,or_ops,not_ops,popcount_ops,"
        << "num_buckets,candidates,survivors,survivor_ratio,"
        << "largest_bucket,smallest_bucket,mean_bucket,num_empty_buckets,num_dense_buckets,"
        << "bucket_imbalance_ratio,top5_bucket_sizes,"
        << "min_pred_sel,max_pred_sel,mean_pred_sel,est_final_sel,not_result_density,"
        << "estimated_cost_ms,reason,attr_expr_normalized,filter_expr\n";
    return out.good();
}

bool WriteCSVRow(std::ofstream &out, const AnalyzeRow &row)
{
    out << row.query_index << ','
        << row.query_id << ','
        << (row.is_p99 ? 1 : 0) << ','
        << row.group_id << ','
        << row.leader_id << ','
        << std::fixed << std::setprecision(3)
        << row.total_end_to_end_ms << ','
        << row.inbucket_attr_filter_ms << ','
        << row.filter_pct << ','
        << row.bucket_level_ivf_ms << ','
        << row.candidate_bucket_merge_ms << ','
        << row.npu_async_launch_ms << ','
        << row.result_collection_ms << ','
        << row.final_merge_ms << ','
        << row.unaccounted_wait_ms << ','
        << row.expr.num_predicates << ','
        << row.expr.num_and << ','
        << row.expr.num_or << ','
        << row.expr.num_not << ','
        << row.expr.max_or_width << ','
        << row.expr.expr_depth << ','
        << row.expr.total_or_terms << ','
        << row.num_low_sel_pred << ','
        << std::fixed << std::setprecision(4)
        << row.bitmap_read_mb << ','
        << row.bitmap_write_mb << ','
        << row.bitmap_words_scanned << ','
        << row.bitmap.and_ops << ','
        << row.bitmap.or_ops << ','
        << row.bitmap.not_ops << ','
        << row.bitmap.num_popcount_ops << ','
        << row.bucket.num_buckets << ','
        << row.bucket.total_candidates << ','
        << row.bucket.total_survivors << ','
        << std::fixed << std::setprecision(6)
        << row.bucket.survivor_ratio << ','
        << row.bucket.largest_bucket << ','
        << row.bucket.smallest_bucket << ','
        << std::fixed << std::setprecision(1)
        << row.bucket.mean_bucket << ','
        << row.bucket.num_empty_buckets << ','
        << row.bucket.num_dense_buckets << ','
        << std::fixed << std::setprecision(2)
        << row.bucket_imbalance_ratio << ','
        << row.top5_bucket_sizes_str << ','
        << std::fixed << std::setprecision(6)
        << row.selectivity.min_pred_selectivity << ','
        << row.selectivity.max_pred_selectivity << ','
        << row.selectivity.mean_pred_selectivity << ','
        << row.selectivity.estimated_final_selectivity << ','
        << row.selectivity.not_result_density << ','
        << std::fixed << std::setprecision(3)
        << row.estimated_cost_ms << ','
        << row.reason_string << ','
        << SanitizeCSV(row.attr_expr_normalized) << ','
        << SanitizeCSV(row.filter_expr) << '\n';
    return out.good();
}

bool WriteSummary(const fs::path &output_file,
                  const std::vector<AnalyzeRow> &rows,
                  size_t slow_count)
{
    std::ofstream out(output_file, std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
        return false;
    }

    const size_t total = rows.size();

    // Aggregate stats
    double sum_total_ms = 0.0;
    double sum_filter_ms = 0.0;
    uint64_t sum_candidates = 0;
    uint64_t sum_survivors = 0;
    double sum_bitmap_read_mb = 0.0;
    double sum_estimated_cost_ms = 0.0;
    int max_predicates = 0;
    int max_or_width = 0;
    uint32_t largest_bucket = 0;
    size_t p99_count = 0;

    // Reason flag frequency
    uint32_t reason_counts[kNumReasonFlags] = {};
    const char *reason_names[] = {
        "LARGE_BUCKET", "MANY_BUCKETS", "MANY_PREDICATES", "LARGE_OR",
        "HAS_NOT", "LOW_SELECTIVITY", "HIGH_SURVIVOR_RATIO",
        "MANY_BITMAP_BYTES", "WASTED_BITMAP_SCAN", "FILTER_DOMINANT",
        "LOW_FILTER_POWER", "DENSE_BITMAP", "RESULT_COLLECTION_HEAVY",
        "NPU_WAIT", "UNACCOUNTED_WAIT"};
    const uint32_t reason_bits[] = {
        REASON_LARGE_BUCKET, REASON_MANY_BUCKETS, REASON_MANY_PREDICATES,
        REASON_LARGE_OR, REASON_HAS_NOT, REASON_LOW_SELECTIVITY,
        REASON_HIGH_SURVIVOR_RATIO, REASON_MANY_BITMAP_BYTES,
        REASON_WASTED_BITMAP_SCAN, REASON_FILTER_DOMINANT,
        REASON_LOW_FILTER_POWER, REASON_DENSE_BITMAP,
        REASON_RESULT_COLLECTION_HEAVY, REASON_NPU_WAIT,
        REASON_UNACCOUNTED_WAIT};

    for (const auto &row : rows)
    {
        sum_total_ms += row.total_end_to_end_ms;
        sum_filter_ms += row.inbucket_attr_filter_ms;
        sum_candidates += row.bucket.total_candidates;
        sum_survivors += row.bucket.total_survivors;
        sum_bitmap_read_mb += row.bitmap_read_mb;
        sum_estimated_cost_ms += row.estimated_cost_ms;
        if (row.expr.num_predicates > max_predicates)
        {
            max_predicates = row.expr.num_predicates;
        }
        if (row.expr.max_or_width > max_or_width)
        {
            max_or_width = row.expr.max_or_width;
        }
        if (row.bucket.largest_bucket > largest_bucket)
        {
            largest_bucket = row.bucket.largest_bucket;
        }
        if (row.is_p99)
        {
            ++p99_count;
        }

        for (int i = 0; i < kNumReasonFlags; ++i)
        {
            if (row.reason_flags & reason_bits[i])
            {
                ++reason_counts[i];
            }
        }
    }

    out << "=== analyze_t8 Summary ===\n\n";
    out << std::fixed << std::setprecision(3);
    out << "Total queries          : " << total << "\n";
    out << "P99 queries            : " << p99_count << "\n";
    out << "Slow queries (>" << kSlowQueryFilterMsThreshold << "ms) : " << slow_count << "\n\n";
    out << "Avg total_ms           : " << (total > 0 ? sum_total_ms / total : 0.0) << "\n";
    out << "Avg filter_ms          : " << (total > 0 ? sum_filter_ms / total : 0.0) << "\n";
    out << "Avg bitmap_read_mb     : " << (total > 0 ? sum_bitmap_read_mb / total : 0.0) << "\n";
    out << "Avg estimated_cost_ms  : " << (total > 0 ? sum_estimated_cost_ms / total : 0.0) << "\n";
    out << "Avg candidates/query   : " << (total > 0 ? static_cast<double>(sum_candidates) / total : 0.0) << "\n";
    out << "Avg survivors/query    : " << (total > 0 ? static_cast<double>(sum_survivors) / total : 0.0) << "\n";
    out << "Overall survivor ratio : " << (sum_candidates > 0 ? static_cast<double>(sum_survivors) / sum_candidates : 0.0) << "\n";
    out << "Max predicates         : " << max_predicates << "\n";
    out << "Max or_width           : " << max_or_width << "\n";
    out << "Largest bucket (docs)  : " << largest_bucket << "\n";

    out << "\n--- Reason Flag Frequency (across all queries) ---\n";
    for (int i = 0; i < kNumReasonFlags; ++i)
    {
        out << "  " << std::left << std::setw(28) << reason_names[i]
            << " : " << reason_counts[i];
        if (total > 0)
        {
            out << " (" << std::fixed << std::setprecision(1)
                << (100.0 * reason_counts[i] / total) << "%)";
        }
        out << "\n";
    }

    return out.good();
}

// =========================================================
// Load queries from fvecs (same as serial/parallel)
// =========================================================
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

// =========================================================
// Main
// =========================================================
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

    const fs::path config_path = RunSupport::ResolveConfigPath();
    std::cout << "[T8] Loading config from " << config_path << "...\n";
    if (!LoadParams(config_path.string(), ResourceConfigProfile::Serial))
    {
        std::cerr << "[Error] Failed to load config: " << config_path << "\n";
        return -1;
    }
    const fs::path query_result_root_dir =
        query_result_root.empty() ? fs::path(".") : fs::path(query_result_root);

    // =============================
    // Phase A: Pipeline execution (collect real timing)
    // =============================
    std::cout << "[T8] Phase A: Running serial pipeline for timing data...\n";

    DataReader::DatasetBuffers dataset_buffers;
    if (!DataReader::LoadDatasetCache(kDatasetFile, dataset_buffers))
    {
        std::cerr << "[Fatal] Failed to load dataset cache: " << kDatasetFile << "\n";
        return -1;
    }
    std::cout << "[T8] Dataset loaded. Docs=" << total_doc_num
              << ", Dim=" << vector_dim
              << ", Tags=" << total_tag_num
              << ", Buckets(Level1)=" << total_bucket_num_level_1 << "\n";

    std::vector<DataReader::PreparedQuery> queries;
    std::string loaded_query_path;
    if (!LoadPreparedQueriesFromFvec(kQueryPaths,
                                     vector_dim,
                                     kDefaultTopK,
                                     queries,
                                     loaded_query_path))
    {
        std::cerr << "[Fatal] Failed to read queries from deep1B_queries.fvecs\n";
        return -1;
    }
    std::cout << "[T8] Queries loaded from " << loaded_query_path << "\n";

    if (queries.empty())
    {
        std::cout << "[T8] No valid queries. Exiting.\n";
        return 0;
    }

    // Ensure queries vector has exactly QueryNum items
    if (queries.size() > static_cast<size_t>(QueryNum))
    {
        queries.resize(QueryNum);
        std::cout << "[T8] Truncated queries: " << queries.size() << "\n";
    }
    else if (queries.size() < static_cast<size_t>(QueryNum))
    {
        const size_t original_count = queries.size();
        queries.reserve(QueryNum);
        for (size_t i = original_count; i < static_cast<size_t>(QueryNum); ++i)
        {
            queries.push_back(queries[i % original_count]);
        }
        std::cout << "[T8] Expanded queries: " << original_count << " -> " << queries.size() << "\n";
    }

    // Load filter expressions from filter_expr.txt
    {
        std::vector<std::string> filter_exprs_pool;
        const fs::path filter_expr_path = config_path.has_parent_path()
                                              ? (config_path.parent_path() / "filter_expr_600.txt")
                                              : fs::path("filter_expr_600.txt");
        {
            std::ifstream fexpr_file(filter_expr_path);
            std::string line;
            while (std::getline(fexpr_file, line))
            {
                if (!line.empty())
                {
                    filter_exprs_pool.push_back(line);
                }
            }
        }
        if (filter_exprs_pool.empty())
        {
            std::cerr << "[Warn] filter_expr.txt not found or empty at " << filter_expr_path
                      << ". Using empty filters for all queries.\n";
        }
        else
        {
            std::mt19937 rng(42);
            std::uniform_int_distribution<size_t> dist(0, filter_exprs_pool.size() - 1);
            for (size_t i = 0; i < queries.size(); ++i)
            {
                queries[i].filter_expr = filter_exprs_pool[dist(rng)];
            }
        }
        std::cout << "[T8] Assigned filter expressions to all " << queries.size() << " queries\n";
    }

    std::cout << "[T8] Prepared query count: " << queries.size() << "\n";
    if (!RunSupport::ValidatePreparedQueriesAgainstPrealloc(queries))
    {
        return -1;
    }
    FilterExpCompiler::WarmUpThreadLocalBuffers();

    std::cout << "[T8] Syncing clustering context...\n";
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
        std::cerr << "[Fatal] Failed to sync clustering context.\n";
        return -1;
    }

    InputDataset dataset = DataReader::BuildInputDataset(dataset_buffers);
    BucketDocTable level_1_bdt;
    try
    {
        level_1_bdt = RunSupport::BuildBucketDocTableFromOffsets(bucket_doc_offsets, bucket_doc_ids);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Fatal] " << e.what() << "\n";
        return -1;
    }
    TwoLevelBucketLayout bucket_layout;
    try
    {
        bucket_layout = RunSupport::BuildTwoLevelBucketLayout(std::move(level_1_bdt),
                                                              max_doc_per_bucket_level_2);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Fatal] " << e.what() << "\n";
        return -1;
    }
    total_bucket_num_level_2 = static_cast<int>(bucket_layout.Level2BucketCount());
    if (!FinalizePreallocationParams())
    {
        return -1;
    }

    std::cout << "[T8] Building DataBaseCPU...\n";
    DataBaseCPU db(dataset, std::move(bucket_layout));

    std::cout << "[T8] Starting serial pipeline...\n";
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

    // Collect timing for each query.
    std::vector<TimingRow> timing_rows;
    timing_rows.reserve(queries.size());

    for (size_t idx = 0; idx < queries.size(); ++idx)
    {
        const auto &prepared = queries[idx];

        auto start_construct = std::chrono::high_resolution_clock::now();
        Query *q = query_pool.Acquire();
        if (q == nullptr)
        {
            std::cerr << "[Fatal] QueryPool exhausted.\n";
            return -1;
        }
        q->Reset(prepared.query_vec, prepared.filter_expr, prepared.top_k, prepared.line_no);
        auto end_construct = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> construct_ms = end_construct - start_construct;

        auto start_loop = std::chrono::high_resolution_clock::now();
        scheduler.Push(q);
        Query *completed_q = scheduler.PopResult();
        auto end_loop = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> loop_ms = end_loop - start_loop;

        TimingRow tr;
        tr.query_index = idx;
        tr.query_id = prepared.line_no;
        tr.total_end_to_end_ms = construct_ms.count() + loop_ms.count();
        tr.construct_ms = construct_ms.count();
        tr.searched_bucket_count_level_1 = completed_q->searched_bucket_count_level_1;
        tr.bucket_level_ivf_ms = completed_q->timing_metrics.bucket_level_ivf_ms;
        tr.candidate_bucket_merge_ms = completed_q->timing_metrics.candidate_bucket_merge_ms;
        tr.npu_async_launch_ms = completed_q->timing_metrics.npu_async_launch_ms;
        tr.inbucket_attr_filter_ms = completed_q->timing_metrics.inbucket_attr_filter_overlapped_ms;
        tr.wait_npu_flag_ms = completed_q->timing_metrics.wait_npu_flag_ms;
        tr.result_collection_ms = completed_q->timing_metrics.result_collection_ms;
        tr.final_merge_ms = completed_q->timing_metrics.final_merge_ms;
        tr.filter_expr = prepared.filter_expr;
        timing_rows.push_back(std::move(tr));

        query_pool.Release(completed_q);
        if ((idx + 1) % 100 == 0 || idx + 1 == queries.size())
        {
            std::cout << "[T8] Pipeline: " << (idx + 1) << "/" << queries.size() << "\n";
        }
    }

    for (auto &g : groups)
    {
        g->Stop();
    }
    for (auto &t : threads)
    {
        if (t.joinable())
        {
            t.join();
        }
    }

    // =============================
    // Phase B: Offline analysis
    // =============================
    std::cout << "\n[T8] Phase B: Offline diagnostic analysis...\n";

    // Compute is_p99 threshold: top 1% by total_ms.
    std::vector<size_t> sorted_by_total(queries.size());
    for (size_t i = 0; i < queries.size(); ++i)
    {
        sorted_by_total[i] = i;
    }
    std::sort(sorted_by_total.begin(), sorted_by_total.end(),
              [&](size_t a, size_t b)
              {
                  return timing_rows[a].total_end_to_end_ms > timing_rows[b].total_end_to_end_ms;
              });
    size_t p99_cutoff = std::max<size_t>(1, queries.size() / 100);
    std::vector<bool> is_p99_flag(queries.size(), false);
    for (size_t i = 0; i < p99_cutoff && i < sorted_by_total.size(); ++i)
    {
        is_p99_flag[sorted_by_total[i]] = true;
    }

    // Create reusable Query objects for offline analysis.
    Query analysis_query;
    std::vector<uint64_t> temp_mask;
    std::vector<uint64_t, AlignedAllocator<uint64_t>> scratch;

    // Per-predicate detail rows for separate CSV.
    struct PredicateDetailRow
    {
        size_t query_index;
        size_t query_id;
        int pred_idx;
        uint32_t tag_id;
        bool is_inverted;
        uint64_t matched_count;
        uint64_t total_candidates;
        double selectivity;
    };
    std::vector<PredicateDetailRow> pred_detail_rows;

    std::vector<AnalyzeRow> analyze_rows;
    analyze_rows.reserve(queries.size());

    for (size_t idx = 0; idx < queries.size(); ++idx)
    {
        const auto &prepared = queries[idx];
        const TimingRow &tr = timing_rows[idx];

        AnalyzeRow row;
        row.query_index = tr.query_index;
        row.query_id = tr.query_id;
        row.is_p99 = is_p99_flag[idx];
        row.group_id = 0;  // serial mode: single group
        row.leader_id = 0; // serial mode: rank 0
        row.total_end_to_end_ms = tr.total_end_to_end_ms;
        row.inbucket_attr_filter_ms = tr.inbucket_attr_filter_ms;
        row.filter_pct = (tr.total_end_to_end_ms > 0.0)
                             ? (tr.inbucket_attr_filter_ms / tr.total_end_to_end_ms * 100.0)
                             : 0.0;
        row.bucket_level_ivf_ms = tr.bucket_level_ivf_ms;
        row.candidate_bucket_merge_ms = tr.candidate_bucket_merge_ms;
        row.npu_async_launch_ms = tr.npu_async_launch_ms;
        row.result_collection_ms = tr.result_collection_ms;
        row.final_merge_ms = tr.final_merge_ms;
        row.wait_npu_flag_ms = tr.wait_npu_flag_ms;
        row.filter_expr = tr.filter_expr;

        // Unaccounted wait: pipeline time - critical path stages.
        // Pipeline time = total - construct_ms (construction is outside the pipeline).
        {
            double pipeline_ms = tr.total_end_to_end_ms - tr.construct_ms;
            double overlap_stage = std::max(tr.inbucket_attr_filter_ms, tr.npu_async_launch_ms);
            double accounted = tr.bucket_level_ivf_ms + tr.candidate_bucket_merge_ms +
                               overlap_stage + tr.wait_npu_flag_ms +
                               tr.result_collection_ms + tr.final_merge_ms;
            row.unaccounted_wait_ms = pipeline_ms - accounted;
        }

        // Build analysis query to get the compiled RPN.
        analysis_query.Reset(prepared.query_vec, prepared.filter_expr, prepared.top_k, prepared.line_no);
        const auto &rpn = analysis_query.filter_exp.Bucket_RPN;

        // Expression analysis.
        row.expr = AnalyzeExpression(rpn);
        row.attr_expr_normalized = GenerateNormalizedExpr(rpn);

        // Bucket selection (uses real pipeline's searched_bucket_count_level_1).
        std::vector<uint32_t> l2_ids = SelectCandidateBuckets(db, analysis_query,
                                                               tr.searched_bucket_count_level_1);

        // Collect tag operand IDs for per-predicate selectivity.
        struct TagOperand
        {
            uint32_t tag_id;
            bool is_inverted;
        };
        std::vector<TagOperand> tag_operands;
        for (const auto &item : rpn)
        {
            if (!item.is_op)
            {
                tag_operands.push_back({static_cast<uint32_t>(item.value),
                                        (item.flags & 1) != 0});
            }
        }

        // Per-bucket analysis: bitmap volume + survivors + per-tag cardinality.
        BitmapVolumeStats total_bitmap;
        BucketDistStats bds;
        bds.num_buckets = static_cast<uint32_t>(l2_ids.size());

        // Per-tag cardinality accumulators.
        std::vector<uint64_t> per_tag_matched(tag_operands.size(), 0);

        // Collect bucket sizes for distribution analysis.
        std::vector<uint32_t> bucket_sizes;
        bucket_sizes.reserve(l2_ids.size());

        for (uint32_t bid : l2_ids)
        {
            const Bucket &bucket = db.get_bucket(bid);
            uint32_t doc_num = bucket.get_doc_num();

            bds.total_candidates += doc_num;
            bucket_sizes.push_back(doc_num);

            if (doc_num > bds.largest_bucket)
            {
                bds.largest_bucket = doc_num;
            }
            if (doc_num < bds.smallest_bucket)
            {
                bds.smallest_bucket = doc_num;
            }
            if (doc_num == 0)
            {
                ++bds.num_empty_buckets;
            }

            // Bitmap volume.
            BitmapVolumeStats bvs = ComputeBitmapVolume(rpn, bucket);
            total_bitmap.words_read += bvs.words_read;
            total_bitmap.words_written += bvs.words_written;
            total_bitmap.and_ops += bvs.and_ops;
            total_bitmap.or_ops += bvs.or_ops;
            total_bitmap.not_ops += bvs.not_ops;

            // Per-tag bitmap popcount for selectivity.
            for (size_t t = 0; t < tag_operands.size(); ++t)
            {
                const uint64_t *bits = bucket.get_tag_bits(tag_operands[t].tag_id);
                uint64_t matched = 0;
                for (uint32_t w = 0; w < bucket.get_stride(); ++w)
                {
                    matched += static_cast<uint64_t>(__builtin_popcountll(bits[w]));
                }
                per_tag_matched[t] += matched;
            }

            // Survivors via actual search_bucket.
            analysis_query.search_bucket(bucket, temp_mask, scratch);
            uint64_t survivors = PopcountMask(temp_mask, doc_num);
            bds.total_survivors += survivors;
            total_bitmap.num_popcount_ops++;
        }

        if (bds.num_buckets > 0)
        {
            bds.mean_bucket = static_cast<double>(bds.total_candidates) / bds.num_buckets;
        }
        // Count dense buckets (size > 2x mean).
        if (bds.mean_bucket > 0.0)
        {
            for (uint32_t sz : bucket_sizes)
            {
                if (static_cast<double>(sz) > 2.0 * bds.mean_bucket)
                {
                    ++bds.num_dense_buckets;
                }
            }
        }
        if (bds.smallest_bucket == UINT32_MAX)
        {
            bds.smallest_bucket = 0;
        }
        bds.survivor_ratio = (bds.total_candidates > 0)
                                 ? static_cast<double>(bds.total_survivors) / bds.total_candidates
                                 : 0.0;

        // Top-5 bucket sizes.
        {
            std::sort(bucket_sizes.begin(), bucket_sizes.end(), std::greater<uint32_t>());
            std::ostringstream oss;
            int cnt = 0;
            for (uint32_t sz : bucket_sizes)
            {
                if (cnt > 0)
                {
                    oss << '|';
                }
                oss << sz;
                if (++cnt >= 5)
                {
                    break;
                }
            }
            row.top5_bucket_sizes_str = oss.str();
        }

        // Bucket imbalance ratio.
        row.bucket_imbalance_ratio = (bds.mean_bucket > 0.0)
                                         ? static_cast<double>(bds.largest_bucket) / bds.mean_bucket
                                         : 0.0;

        // Per-predicate selectivity + low-selectivity count + detail rows.
        row.num_low_sel_pred = 0;
        if (!tag_operands.empty() && bds.total_candidates > 0)
        {
            double sum_sel = 0.0;
            double min_sel = 1.0;
            double max_sel = 0.0;
            double not_matched_sum = 0.0;

            for (size_t t = 0; t < tag_operands.size(); ++t)
            {
                double sel = static_cast<double>(per_tag_matched[t]) / bds.total_candidates;
                if (tag_operands[t].is_inverted)
                {
                    sel = 1.0 - sel;
                    not_matched_sum += per_tag_matched[t];
                }
                sum_sel += sel;
                if (sel < min_sel)
                {
                    min_sel = sel;
                }
                if (sel > max_sel)
                {
                    max_sel = sel;
                }
                if (sel > kLowSelectivityPredThreshold)
                {
                    ++row.num_low_sel_pred;
                }

                // Per-predicate detail row.
                pred_detail_rows.push_back({idx, row.query_id, static_cast<int>(t),
                                            tag_operands[t].tag_id, tag_operands[t].is_inverted,
                                            per_tag_matched[t], bds.total_candidates, sel});
            }

            row.selectivity.min_pred_selectivity = min_sel;
            row.selectivity.max_pred_selectivity = max_sel;
            row.selectivity.mean_pred_selectivity = sum_sel / tag_operands.size();
            row.selectivity.estimated_final_selectivity = bds.survivor_ratio;
            row.selectivity.not_result_density = (bds.total_candidates > 0)
                                                     ? not_matched_sum / bds.total_candidates
                                                     : 0.0;
        }

        row.bitmap = total_bitmap;
        row.bitmap_read_mb = static_cast<double>(total_bitmap.words_read) * sizeof(uint64_t) / (1024.0 * 1024.0);
        row.bitmap_write_mb = static_cast<double>(total_bitmap.words_written) * sizeof(uint64_t) / (1024.0 * 1024.0);
        row.bitmap_words_scanned = total_bitmap.words_read + total_bitmap.words_written;
        row.bucket = bds;

        // Estimated cost model.
        {
            double read_cost_ms = row.bitmap_read_mb * 1024.0 * 1024.0 / kEstimatedMemBandwidthBytesPerMs;
            double write_cost_ms = row.bitmap_write_mb * 1024.0 * 1024.0 / kEstimatedMemBandwidthBytesPerMs;
            double ops_cost_ms = static_cast<double>(total_bitmap.and_ops + total_bitmap.or_ops + total_bitmap.not_ops) * 0.5 / 1e6;
            double survivor_cost_ms = static_cast<double>(bds.total_survivors) * 2.0 / 1e6;
            row.estimated_cost_ms = read_cost_ms + write_cost_ms + ops_cost_ms + survivor_cost_ms;
        }

        // Classification.
        row.reason_flags = ClassifyReason(row);
        row.reason_string = ReasonFlagsToString(row.reason_flags);

        analyze_rows.push_back(std::move(row));

        if ((idx + 1) % 100 == 0 || idx + 1 == queries.size())
        {
            std::cout << "[T8] Analysis: " << (idx + 1) << "/" << queries.size() << "\n";
        }
    }

    // =============================
    // Output
    // =============================
    const fs::path log_dir = query_result_root_dir / "log";
    if (!RunSupport::EnsureOutputDirectory(log_dir))
    {
        std::cerr << "[Fatal] Cannot create log directory: " << log_dir << "\n";
        return -1;
    }

    // Full CSV
    {
        const fs::path csv_path = log_dir / "analyze_t8.csv";
        std::ofstream csv_out(csv_path, std::ios::out | std::ios::trunc);
        if (!csv_out.is_open())
        {
            std::cerr << "[Fatal] Cannot open " << csv_path << "\n";
            return -1;
        }
        WriteCSVHeader(csv_out);
        for (const auto &row : analyze_rows)
        {
            WriteCSVRow(csv_out, row);
        }
        std::cout << "[T8] Full CSV written to " << csv_path << "\n";
    }

    // Slow query CSV
    size_t slow_count = 0;
    {
        const fs::path slow_path = log_dir / "analyze_t8_slow.csv";
        std::ofstream slow_out(slow_path, std::ios::out | std::ios::trunc);
        if (!slow_out.is_open())
        {
            std::cerr << "[Fatal] Cannot open " << slow_path << "\n";
            return -1;
        }
        WriteCSVHeader(slow_out);
        for (const auto &row : analyze_rows)
        {
            if (row.inbucket_attr_filter_ms > kSlowQueryFilterMsThreshold)
            {
                WriteCSVRow(slow_out, row);
                ++slow_count;
            }
        }
        std::cout << "[T8] Slow query CSV (" << slow_count << " rows) written to " << slow_path << "\n";
    }

    // Per-predicate detail CSV
    {
        const fs::path pred_path = log_dir / "analyze_t8_predicates.csv";
        std::ofstream pred_out(pred_path, std::ios::out | std::ios::trunc);
        if (!pred_out.is_open())
        {
            std::cerr << "[Fatal] Cannot open " << pred_path << "\n";
            return -1;
        }
        pred_out << "query_idx,query_id,pred_idx,tag_id,is_inverted,"
                 << "matched_count,total_candidates,selectivity\n";
        for (const auto &p : pred_detail_rows)
        {
            pred_out << p.query_index << ','
                     << p.query_id << ','
                     << p.pred_idx << ','
                     << p.tag_id << ','
                     << (p.is_inverted ? 1 : 0) << ','
                     << p.matched_count << ','
                     << p.total_candidates << ','
                     << std::fixed << std::setprecision(6)
                     << p.selectivity << '\n';
        }
        std::cout << "[T8] Per-predicate CSV (" << pred_detail_rows.size()
                  << " rows) written to " << pred_path << "\n";
    }

    // Summary
    {
        const fs::path summary_path = log_dir / "analyze_t8_summary.txt";
        if (!WriteSummary(summary_path, analyze_rows, slow_count))
        {
            std::cerr << "[Fatal] Cannot write summary to " << summary_path << "\n";
            return -1;
        }
        std::cout << "[T8] Summary written to " << summary_path << "\n";
    }

    std::cout << "\n[T8] Done. " << queries.size() << " queries analyzed, "
              << slow_count << " slow queries identified.\n";
    return 0;
}
