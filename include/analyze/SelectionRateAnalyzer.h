#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include "Query/Query.h"
#include "DataBaseCPU/AlignedAllocator.h"
#include "DataBaseCPU/BucketLevelIVF.h"
#include "utils/RunSupport.h"

namespace Analyze
{
namespace fs = std::filesystem;

struct QuerySelectionRateRow
{
    std::vector<double> batch_rates;
    double global_rate = 0.0;
};

class SelectionRateAnalyzer
{
public:
    SelectionRateAnalyzer(const InputDataset &dataset,
                          const BucketDocTable &bucket_doc_table)
        : dataset_(dataset),
          bucket_doc_table_(bucket_doc_table),
          bucket_layout_hash_(ComputeBucketDocTableHash(bucket_doc_table)),
          bitmap_stride_(static_cast<uint32_t>((total_tag_num + 63) / 64))
    {
        InitializeBucketLevelIvf();
    }

    QuerySelectionRateRow AnalyzeQuery(const DataReader::PreparedQuery &prepared) const
    {
        Query query;
        query.Reset(prepared.query_vec, prepared.filter_expr, prepared.top_k, prepared.line_no);

        QuerySelectionRateRow row;
        row.batch_rates.assign(static_cast<size_t>(std::max(::analyze_t1, 0)),
                               std::numeric_limits<double>::quiet_NaN());

        const std::vector<uint32_t> sorted_bucket_ids = FilterAndSortBuckets(query);
        const BatchAssignment assignment = AssignAnalyzedBuckets(sorted_bucket_ids);
        const Counts counts = CountBatchMatches(query, assignment.assigned_buckets);

        for (int batch_idx = 0; batch_idx < ::analyze_t1; ++batch_idx)
        {
            if (assignment.bucket_count_per_batch[static_cast<size_t>(batch_idx)] == 0)
            {
                continue;
            }

            const uint64_t total_docs = counts.total_docs_per_batch[static_cast<size_t>(batch_idx)];
            if (total_docs == 0)
            {
                row.batch_rates[static_cast<size_t>(batch_idx)] = 0.0;
                continue;
            }

            const uint64_t matched_docs = counts.matched_docs_per_batch[static_cast<size_t>(batch_idx)];
            row.batch_rates[static_cast<size_t>(batch_idx)] =
                static_cast<double>(matched_docs) / static_cast<double>(total_docs);
        }

        const uint64_t global_matched_docs = CountGlobalMatches(query);
        row.global_rate = (::total_doc_num <= 0)
                              ? 0.0
                              : static_cast<double>(global_matched_docs) /
                                    static_cast<double>(::total_doc_num);
        return row;
    }

    static bool WriteRows(const fs::path &output_file,
                          const std::vector<QuerySelectionRateRow> &rows)
    {
        if (!RunSupport::EnsureOutputDirectory(output_file.parent_path()))
        {
            return false;
        }

        std::ofstream out(output_file, std::ios::out | std::ios::trunc);
        if (!out.is_open())
        {
            std::cerr << "[Writer] Failed to open analysis file: " << output_file << "\n";
            return false;
        }

        out << std::fixed << std::setprecision(8);
        for (const auto &row : rows)
        {
            for (size_t i = 0; i < row.batch_rates.size(); ++i)
            {
                if (i != 0)
                {
                    out << '\t';
                }
                WriteValue(out, row.batch_rates[i]);
            }
            if (!row.batch_rates.empty())
            {
                out << '\t';
            }
            WriteValue(out, row.global_rate);
            out << '\n';
        }

        if (!out.good())
        {
            std::cerr << "[Writer] Failed while writing analysis file: " << output_file << "\n";
            return false;
        }
        return true;
    }

private:
    struct BucketScore
    {
        uint32_t bucket_id = 0;
        float score = 0.0f;
    };

    struct AssignedBucket
    {
        uint32_t bucket_id = 0;
        int batch_idx = 0;
    };

    struct BatchAssignment
    {
        std::vector<AssignedBucket> assigned_buckets;
        std::vector<size_t> bucket_count_per_batch;
    };

    struct Counts
    {
        std::vector<uint64_t> matched_docs_per_batch;
        std::vector<uint64_t> total_docs_per_batch;
    };

    static void WriteValue(std::ofstream &out, double value)
    {
        if (std::isnan(value))
        {
            out << "nan";
            return;
        }
        out << value;
    }

    static float ComputeInnerProduct(const float *lhs, const float *rhs)
    {
        float dot_product = 0.0f;
#if defined(__aarch64__) || defined(__arm__)
        float32x4_t sum_vec = vdupq_n_f32(0.0f);
        int d = 0;
        for (; d + 3 < vector_dim; d += 4)
        {
            const float32x4_t lhs_vec = vld1q_f32(lhs + d);
            const float32x4_t rhs_vec = vld1q_f32(rhs + d);
            sum_vec = vmlaq_f32(sum_vec, lhs_vec, rhs_vec);
        }
        dot_product = vaddvq_f32(sum_vec);
        for (; d < vector_dim; ++d)
        {
            dot_product += lhs[d] * rhs[d];
        }
#else
        for (int d = 0; d < vector_dim; ++d)
        {
            dot_product += lhs[d] * rhs[d];
        }
#endif
        return dot_product;
    }

    void InitializeBucketLevelIvf()
    {
        if (TryLoadBucketLevelIvfCache())
        {
            std::cout << "[Analyze] Bucket-level IVF loaded from cache." << std::endl;
            return;
        }

        std::cout << "[Analyze] Rebuilding bucket-level IVF for analysis..." << std::endl;
        BuildBucketLevelIvfFromDataset();
    }

    bool TryLoadBucketLevelIvfCache()
    {
        if (bucket_ivf_index_file.empty())
        {
            return false;
        }

        std::ifstream in(bucket_ivf_index_file, std::ios::binary);
        if (!in.is_open())
        {
            return false;
        }

        DataReader::BucketIvfIndexFileHeaderDisk header{};
        if (!DataReader::ReadBinaryExact(in, &header, sizeof(header)))
        {
            return false;
        }

        const char expected_magic[8] = "BIVFIDX";
        if (std::memcmp(header.magic, expected_magic, sizeof(header.magic)) != 0 ||
            header.version != 2 ||
            header.total_tag_num != static_cast<uint32_t>(total_tag_num) ||
            header.total_bucket_num != static_cast<uint32_t>(total_bucket_num_level_1) ||
            header.cores_per_group != static_cast<uint32_t>(cores_per_group) ||
            header.cpu_cache_line_size != static_cast<uint32_t>(cpu_cache_line_size) ||
            header.buckets_per_core != bucket_level_ivf_.get_buckets_per_core() ||
            header.aligned_stride != bucket_level_ivf_.get_aligned_stride() ||
            header.layout_hash != bucket_layout_hash_)
        {
            return false;
        }

        return bucket_level_ivf_.Deserialize(in);
    }

    void BuildBucketLevelIvfFromDataset()
    {
        std::vector<uint64_t> agg_or_mask(bitmap_stride_, 0ULL);
        std::vector<uint64_t> agg_and_mask(bitmap_stride_, ~0ULL);

        for (size_t bucket_idx = 0; bucket_idx < bucket_doc_table_.size(); ++bucket_idx)
        {
            const std::vector<uint32_t> &doc_ids = bucket_doc_table_[bucket_idx];
            if (doc_ids.empty())
            {
                continue;
            }

            std::fill(agg_or_mask.begin(), agg_or_mask.end(), 0ULL);
            std::fill(agg_and_mask.begin(), agg_and_mask.end(), ~0ULL);

            for (uint32_t doc_id : doc_ids)
            {
                const uint64_t *doc_bitmap =
                    dataset_.tag_bitmaps + static_cast<size_t>(doc_id) * static_cast<size_t>(bitmap_stride_);

                uint32_t word_idx = 0;
#if defined(__aarch64__) || defined(__arm__)
                for (; word_idx + 1 < bitmap_stride_; word_idx += 2)
                {
                    const uint64x2_t doc_vec = vld1q_u64(doc_bitmap + word_idx);
                    uint64x2_t or_vec = vld1q_u64(agg_or_mask.data() + word_idx);
                    uint64x2_t and_vec = vld1q_u64(agg_and_mask.data() + word_idx);

                    or_vec = vorrq_u64(or_vec, doc_vec);
                    and_vec = vandq_u64(and_vec, doc_vec);

                    vst1q_u64(agg_or_mask.data() + word_idx, or_vec);
                    vst1q_u64(agg_and_mask.data() + word_idx, and_vec);
                }
#endif
                for (; word_idx < bitmap_stride_; ++word_idx)
                {
                    agg_or_mask[word_idx] |= doc_bitmap[word_idx];
                    agg_and_mask[word_idx] &= doc_bitmap[word_idx];
                }
            }

            bucket_level_ivf_.RegisterBucket(static_cast<uint32_t>(bucket_idx),
                                             agg_or_mask,
                                             agg_and_mask);
        }
    }

    std::vector<uint32_t> FilterAndSortBuckets(Query &query) const
    {
        std::vector<std::vector<BucketScore>> per_rank_candidates(static_cast<size_t>(cores_per_group));
        const float *centroids = dataset_.bucket_centroids;
        const float *q_vec = query.query_vector.data();
        const uint32_t buckets_per_core = bucket_level_ivf_.get_buckets_per_core();

#pragma omp parallel for schedule(static)
        for (int rank = 0; rank < cores_per_group; ++rank)
        {
            std::vector<uint64_t> local_mask;
            std::vector<uint64_t, AlignedAllocator<uint64_t>> scratch_buffer;
            query.search_ivf(bucket_level_ivf_, local_mask, scratch_buffer, rank);

            std::vector<BucketScore> &candidates = per_rank_candidates[static_cast<size_t>(rank)];
            candidates.reserve(buckets_per_core);
            const uint32_t base_global_bucket_idx = static_cast<uint32_t>(rank) * buckets_per_core;

            for (uint32_t word_idx = 0; word_idx < local_mask.size(); ++word_idx)
            {
                const uint64_t word = local_mask[word_idx];
                if (word == 0)
                {
                    continue;
                }

                const uint32_t word_base_offset = word_idx * 64;
                for (int bit = 0; bit < 64; ++bit)
                {
                    if (((word >> bit) & 1ULL) == 0)
                    {
                        continue;
                    }

                    const uint32_t local_bucket_idx = word_base_offset + static_cast<uint32_t>(bit);
                    if (local_bucket_idx >= buckets_per_core)
                    {
                        continue;
                    }

                    const uint32_t bucket_id = base_global_bucket_idx + local_bucket_idx;
                    if (bucket_id >= static_cast<uint32_t>(total_bucket_num_level_1))
                    {
                        continue;
                    }

                    const float *centroid =
                        centroids + static_cast<size_t>(bucket_id) * static_cast<size_t>(vector_dim);
                    candidates.push_back({bucket_id, ComputeInnerProduct(q_vec, centroid)});
                }
            }
        }

        std::vector<BucketScore> scores;
        size_t total_candidates = 0;
        for (const auto &rank_candidates : per_rank_candidates)
        {
            total_candidates += rank_candidates.size();
        }
        scores.reserve(total_candidates);
        for (const auto &rank_candidates : per_rank_candidates)
        {
            scores.insert(scores.end(), rank_candidates.begin(), rank_candidates.end());
        }

        std::sort(scores.begin(), scores.end(), [](const BucketScore &lhs, const BucketScore &rhs) {
            if (lhs.score != rhs.score)
            {
                return lhs.score > rhs.score;
            }
            return lhs.bucket_id < rhs.bucket_id;
        });

        std::vector<uint32_t> sorted_bucket_ids(scores.size(), 0);
        for (size_t i = 0; i < scores.size(); ++i)
        {
            sorted_bucket_ids[i] = scores[i].bucket_id;
        }
        return sorted_bucket_ids;
    }

    BatchAssignment AssignAnalyzedBuckets(const std::vector<uint32_t> &sorted_bucket_ids) const
    {
        BatchAssignment assignment;
        assignment.bucket_count_per_batch.assign(static_cast<size_t>(std::max(::analyze_t1, 0)), 0);

        if (::analyze_t1 <= 0)
        {
            return assignment;
        }

        size_t cursor = 0;
        for (int batch_idx = 0; batch_idx < ::analyze_t1 && cursor < sorted_bucket_ids.size(); ++batch_idx)
        {
            // 先做桶级过滤，再对过滤后的桶按顺序分批。
            const size_t batch_size = static_cast<size_t>((batch_idx == 0)
                                                              ? ::valid_bucket_num_base_level_2
                                                              : ::valid_bucket_num_incremental_level_2);
            const size_t batch_end = std::min(cursor + batch_size, sorted_bucket_ids.size());
            assignment.bucket_count_per_batch[static_cast<size_t>(batch_idx)] = batch_end - cursor;
            for (; cursor < batch_end; ++cursor)
            {
                assignment.assigned_buckets.push_back(
                    AssignedBucket{sorted_bucket_ids[cursor], batch_idx});
            }
        }
        return assignment;
    }

    Counts CountBatchMatches(const Query &query,
                             const std::vector<AssignedBucket> &assigned_buckets) const
    {
        Counts counts;
        counts.matched_docs_per_batch.assign(static_cast<size_t>(std::max(::analyze_t1, 0)), 0);
        counts.total_docs_per_batch.assign(static_cast<size_t>(std::max(::analyze_t1, 0)), 0);

        if (assigned_buckets.empty() || ::analyze_t1 <= 0)
        {
            return counts;
        }

#pragma omp parallel
        {
            std::vector<uint64_t> local_matched(static_cast<size_t>(::analyze_t1), 0);
            std::vector<uint64_t> local_total(static_cast<size_t>(::analyze_t1), 0);

#pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < static_cast<int>(assigned_buckets.size()); ++i)
            {
                const AssignedBucket &assigned = assigned_buckets[static_cast<size_t>(i)];
                const std::vector<uint32_t> &doc_ids = bucket_doc_table_[assigned.bucket_id];
                local_total[static_cast<size_t>(assigned.batch_idx)] += doc_ids.size();

                for (uint32_t doc_id : doc_ids)
                {
                    if (query.evaluate_single_doc_filter(doc_id, dataset_.tag_bitmaps, bitmap_stride_))
                    {
                        ++local_matched[static_cast<size_t>(assigned.batch_idx)];
                    }
                }
            }

#pragma omp critical
            {
                for (int batch_idx = 0; batch_idx < ::analyze_t1; ++batch_idx)
                {
                    counts.matched_docs_per_batch[static_cast<size_t>(batch_idx)] +=
                        local_matched[static_cast<size_t>(batch_idx)];
                    counts.total_docs_per_batch[static_cast<size_t>(batch_idx)] +=
                        local_total[static_cast<size_t>(batch_idx)];
                }
            }
        }

        return counts;
    }

    uint64_t CountGlobalMatches(const Query &query) const
    {
        unsigned long long global_matched_docs = 0;

#pragma omp parallel for reduction(+ : global_matched_docs) schedule(static)
        for (int doc_id = 0; doc_id < ::total_doc_num; ++doc_id)
        {
            if (query.evaluate_single_doc_filter(static_cast<uint32_t>(doc_id),
                                                 dataset_.tag_bitmaps,
                                                 bitmap_stride_))
            {
                ++global_matched_docs;
            }
        }

        return static_cast<uint64_t>(global_matched_docs);
    }

    const InputDataset &dataset_;
    const BucketDocTable &bucket_doc_table_;
    BucketLevelIVF bucket_level_ivf_;
    uint64_t bucket_layout_hash_ = 0;
    uint32_t bitmap_stride_ = 0;
};
} // namespace Analyze
