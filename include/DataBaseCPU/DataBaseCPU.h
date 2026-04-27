#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "utils/DataReader.h"
#include "Bucket.h"
#include "BucketLevelIVF.h"
#include "InputDataset.h"
// 引入 NPU 接口，用于将向量数据上传至 Device 侧 HBM
#include "NPU/npuAPI.h"

class DataBaseCPU
{
public:
    DataBaseCPU(const InputDataset &dataset, TwoLevelBucketLayout bucket_layout)
    {
        std::cout << "[DB] Starting Index Construction..." << std::endl;

        if (bucket_layout.Level1BucketCount() != static_cast<size_t>(total_bucket_num_level_1))
        {
            throw std::invalid_argument("[DB] level-1 bucket count does not match total_bucket_num_level_1.");
        }
        if (bucket_layout.Level2BucketCount() != static_cast<size_t>(total_bucket_num_level_2))
        {
            throw std::invalid_argument("[DB] level-2 bucket count does not match total_bucket_num_level_2.");
        }
        if (bucket_layout.level_1_to_level_2_offsets.size() != bucket_layout.Level1BucketCount() + 1)
        {
            throw std::invalid_argument("[DB] invalid level_1_to_level_2_offsets size.");
        }
        if (bucket_layout.level_1_to_level_2_offsets.back() != bucket_layout.Level2BucketCount())
        {
            throw std::invalid_argument("[DB] invalid level_1_to_level_2_offsets tail.");
        }

        ValidateQueryScoreSlotCapacities(bucket_layout.level_1_bucket_doc_table);

        BucketDocTable normalized_level_2_bucket_doc_table =
            NormalizeLevel2BucketDocTable(std::move(bucket_layout.level_2_bucket_doc_table));
        level_1_to_level_2_offsets_ = std::move(bucket_layout.level_1_to_level_2_offsets);
        level_1_bucket_layout_hash_ = ComputeBucketDocTableHash(bucket_layout.level_1_bucket_doc_table);
        level_2_bucket_layout_hash_ = ComputeBucketDocTableHash(normalized_level_2_bucket_doc_table);

        // ==========================================
        // 1. 存储一级桶中心向量
        // ==========================================
        bucket_centroids_.resize(static_cast<size_t>(total_bucket_num_level_1) * static_cast<size_t>(vector_dim));
        std::memcpy(bucket_centroids_.data(),
                    dataset.bucket_centroids,
                    static_cast<size_t>(total_bucket_num_level_1) * static_cast<size_t>(vector_dim) * sizeof(float));

        // ==========================================
        // 2. CPU 侧索引：优先从文件加载，否则从 dataset 重建
        // ==========================================
        if (!TryLoadCpuIndexes(normalized_level_2_bucket_doc_table))
        {
            std::cout << "[DB] CPU index cache unavailable. Rebuilding from dataset..." << std::endl;
            BuildCpuIndexesFromDataset(dataset,
                                       bucket_layout.level_1_bucket_doc_table,
                                       normalized_level_2_bucket_doc_table);
            if (!SaveCpuIndexes())
            {
                std::cout << "[DB] Warning: Failed to persist CPU indexes. "
                          << "Next run will rebuild them again." << std::endl;
            }
        }

        npuAPI::Init(dataset, normalized_level_2_bucket_doc_table);
        std::cout << "[DB] Index Construction Complete." << std::endl;
    }

    // 析构时清理 NPU 资源
    ~DataBaseCPU()
    {
        npuAPI::Finalize();
    }

    /**
     * @brief 获取指定二级桶的引用
     */
    const Bucket &get_bucket(uint32_t level_2_bucket_id) const
    {
        return buckets_[level_2_bucket_id];
    }

    /**
     * @brief 获取一级桶级倒排索引
     */
    const BucketLevelIVF &get_bucket_level_ivf() const
    {
        return bucket_level_ivf_;
    }

    /**
     * @brief 获取所有一级桶中心向量的数组首地址
     */
    const float *get_centroids() const
    {
        return bucket_centroids_.data();
    }

    uint32_t get_level_2_bucket_begin(uint32_t level_1_bucket_id) const
    {
        return level_1_to_level_2_offsets_[level_1_bucket_id];
    }

    uint32_t get_level_2_bucket_end(uint32_t level_1_bucket_id) const
    {
        return level_1_to_level_2_offsets_[level_1_bucket_id + 1];
    }

    // Compute total number of L2 buckets for a contiguous range of L1 bucket IDs.
    // Enables pre-allocation of L2 enumeration vector in caller.
    uint32_t get_total_level_2_bucket_count(uint32_t level_1_bucket_id_first, uint32_t level_1_count) const
    {
        const uint32_t begin_offset = level_1_to_level_2_offsets_[level_1_bucket_id_first];
        const uint32_t end_offset = level_1_to_level_2_offsets_[level_1_bucket_id_first + level_1_count];
        return end_offset - begin_offset;
    }

private:
    static constexpr uint32_t kBucketIvfIndexFileVersion = 2;
    static constexpr uint32_t kBucketIndexFileVersion = 5;
    static constexpr size_t kQueryScoreCols = 16;

    static void ValidateQueryScoreSlotCapacities(const BucketDocTable &level_1_bucket_doc_table)
    {
        std::vector<size_t> level_1_doc_counts;
        level_1_doc_counts.reserve(level_1_bucket_doc_table.size());
        for (const auto &bucket_docs : level_1_bucket_doc_table)
        {
            level_1_doc_counts.push_back(bucket_docs.size());
        }
        std::sort(level_1_doc_counts.begin(), level_1_doc_counts.end(), std::greater<size_t>());

        auto sum_top_doc_counts = [&](size_t top_n) -> size_t
        {
            const size_t limit = std::min(top_n, level_1_doc_counts.size());
            size_t total_docs = 0;
            for (size_t i = 0; i < limit; ++i)
            {
                total_docs += level_1_doc_counts[i];
            }
            return total_docs;
        };

        const size_t first_round_required_floats =
            sum_top_doc_counts(static_cast<size_t>(std::max(valid_bucket_num_base_level_1, 0))) * kQueryScoreCols;
        const size_t incremental_required_floats =
            sum_top_doc_counts(static_cast<size_t>(std::max(valid_bucket_num_incremental_level_1, 0))) * kQueryScoreCols;

        const size_t first_round_slot_capacity_floats =
            static_cast<size_t>(valid_bucket_num_base_level_1) *
            static_cast<size_t>(max_process_bucket_num_level_2) *
            static_cast<size_t>(max_doc_per_bucket_level_2) *
            kQueryScoreCols;
        const size_t incremental_slot_capacity_floats =
            static_cast<size_t>(valid_bucket_num_incremental_level_1) *
            static_cast<size_t>(max_process_bucket_num_level_2) *
            static_cast<size_t>(max_doc_per_bucket_level_2) *
            kQueryScoreCols;

        if (first_round_required_floats > first_round_slot_capacity_floats)
        {
            throw std::invalid_argument("[DB] first-round score slot capacity is insufficient: required_floats=" +
                                        std::to_string(first_round_required_floats) +
                                        ", capacity_floats=" +
                                        std::to_string(first_round_slot_capacity_floats));
        }
        if (incremental_required_floats > incremental_slot_capacity_floats)
        {
            throw std::invalid_argument("[DB] incremental score slot capacity is insufficient: required_floats=" +
                                        std::to_string(incremental_required_floats) +
                                        ", capacity_floats=" +
                                        std::to_string(incremental_slot_capacity_floats));
        }
    }

    static DataReader::BucketIvfIndexFileHeaderDisk MakeBucketIvfIndexFileHeader(const BucketLevelIVF &ivf,
                                                                                 uint64_t layout_hash)
    {
        DataReader::BucketIvfIndexFileHeaderDisk header{};
        const char magic[8] = "BIVFIDX";
        std::memcpy(header.magic, magic, sizeof(header.magic));
        header.version = kBucketIvfIndexFileVersion;
        header.total_tag_num = static_cast<uint32_t>(total_tag_num);
        header.total_bucket_num = static_cast<uint32_t>(total_bucket_num_level_1);
        header.cores_per_group = static_cast<uint32_t>(cores_per_group);
        header.cpu_cache_line_size = static_cast<uint32_t>(cpu_cache_line_size);
        header.buckets_per_core = ivf.get_buckets_per_core();
        header.aligned_stride = ivf.get_aligned_stride();
        header.layout_hash = layout_hash;
        return header;
    }

    static DataReader::BucketIndexFileHeaderDisk MakeBucketIndexFileHeader(uint64_t layout_hash)
    {
        DataReader::BucketIndexFileHeaderDisk header{};
        const char magic[8] = "BKTIDX1";
        std::memcpy(header.magic, magic, sizeof(header.magic));
        header.version = kBucketIndexFileVersion;
        header.total_doc_num = static_cast<uint32_t>(total_doc_num);
        header.total_tag_num = static_cast<uint32_t>(total_tag_num);
        header.total_bucket_num = static_cast<uint32_t>(total_bucket_num_level_2);
        header.vector_dim = static_cast<uint32_t>(vector_dim);
        header.max_doc_per_bucket = static_cast<uint32_t>(max_doc_per_bucket_level_2);
        header.layout_hash = layout_hash;
        return header;
    }

    static bool IsValidBucketIvfIndexFileHeader(const DataReader::BucketIvfIndexFileHeaderDisk &header,
                                                const BucketLevelIVF &ivf,
                                                uint64_t expected_layout_hash)
    {
        const char expected_magic[8] = "BIVFIDX";
        return std::memcmp(header.magic, expected_magic, sizeof(header.magic)) == 0 &&
               header.version == kBucketIvfIndexFileVersion &&
               header.total_tag_num == static_cast<uint32_t>(total_tag_num) &&
               header.total_bucket_num == static_cast<uint32_t>(total_bucket_num_level_1) &&
               header.cores_per_group == static_cast<uint32_t>(cores_per_group) &&
               header.cpu_cache_line_size == static_cast<uint32_t>(cpu_cache_line_size) &&
               header.buckets_per_core == ivf.get_buckets_per_core() &&
               header.aligned_stride == ivf.get_aligned_stride() &&
               header.layout_hash == expected_layout_hash;
    }

    static bool IsValidBucketIndexFileHeader(const DataReader::BucketIndexFileHeaderDisk &header,
                                             uint64_t expected_layout_hash)
    {
        const char expected_magic[8] = "BKTIDX1";
        return std::memcmp(header.magic, expected_magic, sizeof(header.magic)) == 0 &&
               header.version == kBucketIndexFileVersion &&
               header.total_doc_num == static_cast<uint32_t>(total_doc_num) &&
               header.total_tag_num == static_cast<uint32_t>(total_tag_num) &&
               header.total_bucket_num == static_cast<uint32_t>(total_bucket_num_level_2) &&
               header.vector_dim == static_cast<uint32_t>(vector_dim) &&
               header.max_doc_per_bucket == static_cast<uint32_t>(max_doc_per_bucket_level_2) &&
               header.layout_hash == expected_layout_hash;
    }

    static void ComputeAggMasksForDocs(const InputDataset &dataset,
                                       const std::vector<uint32_t> &doc_ids,
                                       uint32_t input_bitmap_stride,
                                       std::vector<uint64_t> &out_or_mask,
                                       std::vector<uint64_t> &out_and_mask)
    {
        std::fill(out_or_mask.begin(), out_or_mask.end(), 0ULL);
        std::fill(out_and_mask.begin(), out_and_mask.end(), ~0ULL);

        for (uint32_t doc_id : doc_ids)
        {
            const uint64_t *doc_bitmap =
                dataset.tag_bitmaps + static_cast<size_t>(doc_id) * static_cast<size_t>(input_bitmap_stride);

            uint32_t word_idx = 0;
#if defined(__aarch64__) || defined(__arm__)
            for (; word_idx + 1 < input_bitmap_stride; word_idx += 2)
            {
                const uint64x2_t doc_vec = vld1q_u64(doc_bitmap + word_idx);
                uint64x2_t or_vec = vld1q_u64(out_or_mask.data() + word_idx);
                uint64x2_t and_vec = vld1q_u64(out_and_mask.data() + word_idx);

                or_vec = vorrq_u64(or_vec, doc_vec);
                and_vec = vandq_u64(and_vec, doc_vec);

                vst1q_u64(out_or_mask.data() + word_idx, or_vec);
                vst1q_u64(out_and_mask.data() + word_idx, and_vec);
            }
#endif
            for (; word_idx < input_bitmap_stride; ++word_idx)
            {
                out_or_mask[word_idx] |= doc_bitmap[word_idx];
                out_and_mask[word_idx] &= doc_bitmap[word_idx];
            }
        }
    }

    void BuildCpuIndexesFromDataset(const InputDataset &dataset,
                                    const BucketDocTable &level_1_bucket_doc_table,
                                    const BucketDocTable &normalized_level_2_bucket_doc_table)
    {
        buckets_.clear();
        buckets_.resize(static_cast<size_t>(total_bucket_num_level_2));

        const uint32_t input_bitmap_stride = (total_tag_num + 63) / 64;

#pragma omp parallel num_threads(GetNonQueryOpenMpThreadCount())
        {
            std::vector<uint64_t> agg_or_mask(input_bitmap_stride);
            std::vector<uint64_t> agg_and_mask(input_bitmap_stride);

#pragma omp for schedule(dynamic, 16)
            for (int bid = 0; bid < total_bucket_num_level_2; ++bid)
            {
                Bucket current_bucket(normalized_level_2_bucket_doc_table[static_cast<size_t>(bid)]);

                if (current_bucket.get_doc_num() == 0)
                {
                    buckets_[static_cast<size_t>(bid)] = std::move(current_bucket);
                    continue;
                }

                current_bucket.ImportDataAndComputeStats(dataset,
                                                        input_bitmap_stride,
                                                        agg_or_mask,
                                                        agg_and_mask);
                current_bucket.FinalizeTagLayout();
                buckets_[static_cast<size_t>(bid)] = std::move(current_bucket);
            }
        }

        std::vector<uint64_t> agg_or_mask(input_bitmap_stride);
        std::vector<uint64_t> agg_and_mask(input_bitmap_stride);

        for (uint32_t bid = 0; bid < total_bucket_num_level_1; ++bid)
        {
            const std::vector<uint32_t> &doc_ids = level_1_bucket_doc_table[bid];
            if (doc_ids.empty())
            {
                continue;
            }

            ComputeAggMasksForDocs(dataset, doc_ids, input_bitmap_stride, agg_or_mask, agg_and_mask);
            bucket_level_ivf_.RegisterBucket(bid, agg_or_mask, agg_and_mask);
        }
    }

    bool TryLoadCpuIndexes(const BucketDocTable &normalized_level_2_bucket_doc_table)
    {
        if (bucket_ivf_index_file.empty() || bucket_index_file.empty())
        {
            std::cout << "[DB] CPU index cache disabled by config." << std::endl;
            return false;
        }

        const std::filesystem::path bucket_ivf_path(bucket_ivf_index_file);
        const std::filesystem::path bucket_index_path(bucket_index_file);

        std::error_code ivf_ec;
        std::error_code bucket_ec;
        const bool bucket_ivf_exists = std::filesystem::exists(bucket_ivf_path, ivf_ec);
        const bool bucket_index_exists = std::filesystem::exists(bucket_index_path, bucket_ec);
        if (ivf_ec || bucket_ec || !bucket_ivf_exists || !bucket_index_exists)
        {
            if (!bucket_index_exists)
            {
                std::cout << "[DB] Warning: bucket index file not found: "
                          << bucket_index_path << std::endl;
            }
            if (!bucket_ivf_exists)
            {
                std::cout << "[DB] Warning: bucket IVF index file not found: "
                          << bucket_ivf_path << std::endl;
            }
            return false;
        }

        std::vector<Bucket> loaded_buckets;
        if (!LoadBucketIndexFile(bucket_index_path,
                                 normalized_level_2_bucket_doc_table,
                                 level_2_bucket_layout_hash_,
                                 loaded_buckets))
        {
            std::cout << "[DB] Warning: bucket index file parameter mismatch: "
                      << bucket_index_path << std::endl;
            return false;
        }

        BucketLevelIVF loaded_bucket_level_ivf;
        if (!LoadBucketIvfIndexFile(bucket_ivf_path,
                                    level_1_bucket_layout_hash_,
                                    loaded_bucket_level_ivf))
        {
            std::cout << "[DB] Warning: bucket IVF index file parameter mismatch: "
                      << bucket_ivf_path << std::endl;
            return false;
        }

        buckets_ = std::move(loaded_buckets);
        bucket_level_ivf_ = std::move(loaded_bucket_level_ivf);
        return true;
    }

    bool SaveCpuIndexes() const
    {
        if (bucket_ivf_index_file.empty() || bucket_index_file.empty())
        {
            return true;
        }

        return SaveBucketIvfIndexFile(std::filesystem::path(bucket_ivf_index_file)) &&
               SaveBucketIndexFile(std::filesystem::path(bucket_index_file));
    }

    static bool LoadBucketIvfIndexFile(const std::filesystem::path &path,
                                       uint64_t expected_layout_hash,
                                       BucketLevelIVF &bucket_level_ivf)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            return false;
        }

        DataReader::BucketIvfIndexFileHeaderDisk header{};
        if (!DataReader::ReadBinaryExact(in, &header, sizeof(header)) ||
            !IsValidBucketIvfIndexFileHeader(header, bucket_level_ivf, expected_layout_hash)) {
            return false;
        }

        return bucket_level_ivf.Deserialize(in);
    }

    static bool LoadBucketIndexFile(const std::filesystem::path &path,
                                    const BucketDocTable &normalized_level_2_bucket_doc_table,
                                    uint64_t expected_layout_hash,
                                    std::vector<Bucket> &buckets_out)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            return false;
        }

        DataReader::BucketIndexFileHeaderDisk header{};
        uint64_t bucket_count = 0;
        if (!DataReader::ReadBinaryExact(in, &header, sizeof(header)) ||
            !IsValidBucketIndexFileHeader(header, expected_layout_hash) ||
            !DataReader::ReadBinaryExact(in, &bucket_count, sizeof(bucket_count))) {
            return false;
        }

        if (bucket_count != normalized_level_2_bucket_doc_table.size()) {
            return false;
        }

        std::vector<DataReader::BucketIndexEntryDisk> entries(static_cast<size_t>(bucket_count));
        if (!DataReader::ReadBinaryExact(in,
                                         entries.data(),
                                         entries.size() * sizeof(DataReader::BucketIndexEntryDisk))) {
            return false;
        }

        std::vector<Bucket> loaded_buckets(static_cast<size_t>(bucket_count));
        std::vector<uint8_t> load_ok(static_cast<size_t>(bucket_count), 0);
        int stream_open_failed = 0;
#pragma omp parallel num_threads(GetNonQueryOpenMpThreadCount())
        {
            std::ifstream bucket_stream(path, std::ios::binary);
            if (!bucket_stream.is_open()) {
                #pragma omp atomic write
                stream_open_failed = 1;
            }
#pragma omp for schedule(static, 256)
            for (int bid = 0; bid < static_cast<int>(bucket_count); ++bid)
            {
                int local_stream_open_failed = 0;
                #pragma omp atomic read
                local_stream_open_failed = stream_open_failed;
                if (local_stream_open_failed) {
                    continue;
                }
                const auto &entry = entries[static_cast<size_t>(bid)];
                bucket_stream.clear();
                bucket_stream.seekg(static_cast<std::streamoff>(entry.offset), std::ios::beg);
                if (!bucket_stream.good()) {
                    continue;
                }

                if (!loaded_buckets[static_cast<size_t>(bid)].Deserialize(bucket_stream)) {
                    continue;
                }
                load_ok[static_cast<size_t>(bid)] = 1;
            }
        }

        if (stream_open_failed != 0)
        {
            return false;
        }

        for (size_t bid = 0; bid < loaded_buckets.size(); ++bid)
        {
            if (!load_ok[bid] ||
                loaded_buckets[bid].get_doc_num() != normalized_level_2_bucket_doc_table[bid].size())
            {
                return false;
            }
        }

        buckets_out = std::move(loaded_buckets);
        return true;
    }

    bool SaveBucketIvfIndexFile(const std::filesystem::path &path) const
    {
        if (!DataReader::EnsureBinaryParentDirectory(path)) {
            return false;
        }

        const std::filesystem::path temp_path = DataReader::BuildBinaryTempPath(path);
        std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            return false;
        }

        const DataReader::BucketIvfIndexFileHeaderDisk header =
            MakeBucketIvfIndexFileHeader(bucket_level_ivf_, level_1_bucket_layout_hash_);
        const bool ok = DataReader::WriteBinaryExact(out, &header, sizeof(header)) &&
                        bucket_level_ivf_.Serialize(out);
        out.flush();
        const bool final_ok = ok && out.good();
        out.close();
        if (!final_ok) {
            std::error_code ec;
            std::filesystem::remove(temp_path, ec);
            return false;
        }

        return DataReader::ReplaceBinaryFileAtomically(temp_path, path);
    }

    bool SaveBucketIndexFile(const std::filesystem::path &path) const
    {
        if (!DataReader::EnsureBinaryParentDirectory(path)) {
            return false;
        }

        const std::filesystem::path temp_path = DataReader::BuildBinaryTempPath(path);
        std::ofstream out(temp_path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            return false;
        }

        const DataReader::BucketIndexFileHeaderDisk header =
            MakeBucketIndexFileHeader(level_2_bucket_layout_hash_);
        const uint64_t bucket_count = static_cast<uint64_t>(buckets_.size());
        std::vector<DataReader::BucketIndexEntryDisk> entries(static_cast<size_t>(bucket_count));
        const uint64_t entry_table_bytes = static_cast<uint64_t>(entries.size() * sizeof(DataReader::BucketIndexEntryDisk));
        uint64_t current_offset = static_cast<uint64_t>(sizeof(header)) + sizeof(bucket_count) + entry_table_bytes;

        // Pre-calculate each bucket's serialized size and offset
        for (size_t bid = 0; bid < buckets_.size(); ++bid)
        {
            std::ostringstream bucket_stream(std::ios::binary);
            if (!buckets_[bid].SerializeToStream(bucket_stream))
            {
                return false;
            }
            const std::string payload = bucket_stream.str();
            entries[bid].offset = current_offset;
            entries[bid].size = static_cast<uint64_t>(payload.size());
            current_offset += entries[bid].size;
        }

        // Write header + bucket_count + entry_table
        bool ok = DataReader::WriteBinaryExact(out, &header, sizeof(header)) &&
                  DataReader::WriteBinaryExact(out, &bucket_count, sizeof(bucket_count)) &&
                  DataReader::WriteBinaryExact(out,
                                               entries.data(),
                                               entries.size() * sizeof(DataReader::BucketIndexEntryDisk));

        // Write each bucket's serialized data at its offset
        for (const Bucket &bucket : buckets_)
        {
            ok = ok && bucket.SerializeToStream(out);
        }
        out.flush();
        const bool final_ok = ok && out.good();
        out.close();
        if (!final_ok) {
            std::error_code ec;
            std::filesystem::remove(temp_path, ec);
            return false;
        }

        return DataReader::ReplaceBinaryFileAtomically(temp_path, path);
    }

    static BucketDocTable NormalizeLevel2BucketDocTable(BucketDocTable bucket_doc_table)
    {
        if (bucket_doc_table.size() != static_cast<size_t>(total_bucket_num_level_2))
        {
            throw std::invalid_argument("[DB] level-2 bucket_doc_table size does not match total_bucket_num_level_2.");
        }

        BucketDocTable normalized(std::move(bucket_doc_table));
        size_t total_stored_doc_refs = 0;

        for (size_t bid = 0; bid < normalized.size(); ++bid)
        {
            auto &doc_ids = normalized[bid];
            std::sort(doc_ids.begin(), doc_ids.end());
            doc_ids.erase(std::unique(doc_ids.begin(), doc_ids.end()), doc_ids.end());

            if (doc_ids.size() > static_cast<size_t>(max_doc_per_bucket_level_2))
            {
                throw std::runtime_error("[DB] level-2 bucket_doc_table contains a bucket that exceeds max_doc_per_bucket_level_2.");
            }

            for (uint32_t doc_id : doc_ids)
            {
                if (doc_id >= static_cast<uint32_t>(total_doc_num))
                {
                    throw std::out_of_range("[DB] bucket_doc_table contains an invalid doc_id.");
                }
            }

            total_stored_doc_refs += doc_ids.size();
        }

        std::cout << "[DB] BucketDocTable normalized. StoredDocRefs="
                  << total_stored_doc_refs
                  << ", MaxAllowedRefs="
                  << static_cast<size_t>(total_bucket_num_level_2) *
                         static_cast<size_t>(max_doc_per_bucket_level_2)
                  << std::endl;
        return normalized;
    }

    std::vector<Bucket> buckets_;
    BucketLevelIVF bucket_level_ivf_;
    std::vector<uint32_t> level_1_to_level_2_offsets_;

    // 扁平化存储的桶中心向量
    std::vector<float, AlignedAllocator<float>> bucket_centroids_;
    uint64_t level_1_bucket_layout_hash_ = 0;
    uint64_t level_2_bucket_layout_hash_ = 0;
};
