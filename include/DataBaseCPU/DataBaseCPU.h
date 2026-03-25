#pragma once

#include <vector>
#include <memory>
#include <iostream>
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "utils/DataReader.h"
#include "Bucket.h"
#include "BucketLevelIVF.h"
#include "InputDataset.h"
// 引入 NPU 接口，用于将向量数据上传至 Device 侧 HBM
#include "NPU/npuAPI.h"

class DataBaseCPU
{
public:
    /**
     * @brief 构造函数：初始化并构建整个索引系统
     * @param dataset 包含所有元数据的输入集
     * @param bucket_doc_table 每个桶包含的全局 doc_id 列表
     */
    DataBaseCPU(const InputDataset &dataset, const BucketDocTable &bucket_doc_table)
    {
        std::cout << "[DB] Starting Index Construction..." << std::endl;

        BucketDocTable normalized_bucket_doc_table = NormalizeBucketDocTable(bucket_doc_table);

        // 将 dataset.vectors 转换成half并上传到 8 张 NPU 的 HBM 中
        npuAPI::Init(dataset, normalized_bucket_doc_table);

        // ==========================================
        // 1. 存储桶中心向量
        // ==========================================
        bucket_centroids_.resize(static_cast<size_t>(total_bucket_num) * static_cast<size_t>(vector_dim));
        std::memcpy(bucket_centroids_.data(),
                    dataset.bucket_centroids,
                    static_cast<size_t>(total_bucket_num) * static_cast<size_t>(vector_dim) * sizeof(float));

        // ==========================================
        // 2. CPU 侧索引优先从文件加载，失败时回退到重建
        // ==========================================
        if (TryLoadCpuIndexes(normalized_bucket_doc_table))
        {
            std::cout << "[DB] CPU indexes loaded from cache files." << std::endl;
            std::cout << "[DB] Index Construction Complete." << std::endl;
            return;
        }

        std::cout << "[DB] CPU index cache unavailable. Rebuilding from dataset..." << std::endl;
        BuildCpuIndexesFromDataset(dataset, normalized_bucket_doc_table);
        if (!SaveCpuIndexes())
        {
            std::cout << "[DB] Warning: Failed to persist CPU indexes. "
                      << "Next run will rebuild them again." << std::endl;
        }

        std::cout << "[DB] Index Construction Complete." << std::endl;
    }

    // 析构时清理 NPU 资源
    ~DataBaseCPU()
    {
        npuAPI::Finalize();
    }

    /**
     * @brief 获取指定桶的引用
     */
    const Bucket &get_bucket(uint32_t bucket_id) const
    {
        return buckets_[bucket_id];
    }

    /**
     * @brief 获取桶级倒排索引
     */
    const BucketLevelIVF &get_bucket_level_ivf() const
    {
        return bucket_level_ivf_;
    }

    /**
     * @brief 获取所有桶中心向量的数组首地址
     */
    const float *get_centroids() const
    {
        return bucket_centroids_.data();
    }

private:
    static constexpr uint32_t kBucketIvfIndexFileVersion = 1;
    static constexpr uint32_t kBucketIndexFileVersion = 1;

    static DataReader::BucketIvfIndexFileHeaderDisk MakeBucketIvfIndexFileHeader(const BucketLevelIVF &ivf)
    {
        DataReader::BucketIvfIndexFileHeaderDisk header{};
        const char magic[8] = "BIVFIDX";
        std::memcpy(header.magic, magic, sizeof(header.magic));
        header.version = kBucketIvfIndexFileVersion;
        header.total_tag_num = static_cast<uint32_t>(total_tag_num);
        header.total_bucket_num = static_cast<uint32_t>(total_bucket_num);
        header.cores_per_group = static_cast<uint32_t>(cores_per_group);
        header.cpu_cache_line_size = static_cast<uint32_t>(cpu_cache_line_size);
        header.buckets_per_core = ivf.get_buckets_per_core();
        header.aligned_stride = ivf.get_aligned_stride();
        return header;
    }

    static DataReader::BucketIndexFileHeaderDisk MakeBucketIndexFileHeader()
    {
        DataReader::BucketIndexFileHeaderDisk header{};
        const char magic[8] = "BKTIDX1";
        std::memcpy(header.magic, magic, sizeof(header.magic));
        header.version = kBucketIndexFileVersion;
        header.total_doc_num = static_cast<uint32_t>(total_doc_num);
        header.total_tag_num = static_cast<uint32_t>(total_tag_num);
        header.total_bucket_num = static_cast<uint32_t>(total_bucket_num);
        header.vector_dim = static_cast<uint32_t>(vector_dim);
        header.max_doc_per_bucket = static_cast<uint32_t>(max_doc_per_bucket);
        return header;
    }

    static bool IsValidBucketIvfIndexFileHeader(const DataReader::BucketIvfIndexFileHeaderDisk &header,
                                                const BucketLevelIVF &ivf)
    {
        const char expected_magic[8] = "BIVFIDX";
        return std::memcmp(header.magic, expected_magic, sizeof(header.magic)) == 0 &&
               header.version == kBucketIvfIndexFileVersion &&
               header.total_tag_num == static_cast<uint32_t>(total_tag_num) &&
               header.total_bucket_num == static_cast<uint32_t>(total_bucket_num) &&
               header.cores_per_group == static_cast<uint32_t>(cores_per_group) &&
               header.cpu_cache_line_size == static_cast<uint32_t>(cpu_cache_line_size) &&
               header.buckets_per_core == ivf.get_buckets_per_core() &&
               header.aligned_stride == ivf.get_aligned_stride();
    }

    static bool IsValidBucketIndexFileHeader(const DataReader::BucketIndexFileHeaderDisk &header)
    {
        const char expected_magic[8] = "BKTIDX1";
        return std::memcmp(header.magic, expected_magic, sizeof(header.magic)) == 0 &&
               header.version == kBucketIndexFileVersion &&
               header.total_doc_num == static_cast<uint32_t>(total_doc_num) &&
               header.total_tag_num == static_cast<uint32_t>(total_tag_num) &&
               header.total_bucket_num == static_cast<uint32_t>(total_bucket_num) &&
               header.vector_dim == static_cast<uint32_t>(vector_dim) &&
               header.max_doc_per_bucket == static_cast<uint32_t>(max_doc_per_bucket);
    }

    void BuildCpuIndexesFromDataset(const InputDataset &dataset,
                                    const BucketDocTable &normalized_bucket_doc_table)
    {
        buckets_.clear();
        buckets_.reserve(static_cast<size_t>(total_bucket_num));

        uint32_t input_bitmap_stride = (total_tag_num + 63) / 64;
        std::vector<uint64_t> agg_or_mask(input_bitmap_stride);
        std::vector<uint64_t> agg_and_mask(input_bitmap_stride);
        /*
            agg意为聚合
            agg_or_mask：
                操作：对桶内所有文档的bitmap进行 或 运算
                结果含义：如果某一位是 1，表示桶内至少有一个文档拥有该 Tag
                用于构建 table_and_or（桶级倒排索引的其中一张表）
            agg_and_mask：
                操作：对桶内所有文档的bitmap进行 与 运算
                结果含义：如果某一位是 1，表示桶内所有文档都拥有该 Tag
                用于推导 table_not（桶级倒排索引的另一张表）

            构建BucketLevelIVF的过程如下：
                遍历所有桶：
                    每个桶都会遍历自己桶内的所有doc并聚合得出长度为total_tag_num个bit（向uint64_t取整）的两个agg_mask，然后将其注册到BucketLevelIVF中
        */

        for (uint32_t bid = 0; bid < total_bucket_num; ++bid)
        {
            // 1. 初始化 Bucket
            buckets_.emplace_back(normalized_bucket_doc_table[bid]);
            Bucket &current_bucket = buckets_.back();
            /*
              在 buckets_ 容器的末尾构造一个新的 Bucket 对象。
              normalized_bucket_doc_table[bid] 存放了属于当前桶（bid）的所有 doc 的全局 ID，
              且已经过排序、去重和容量校验。
            */

            // 如果桶为空，直接跳过后续处理
            if (current_bucket.get_doc_num() == 0)
            {
                std::cout << "Warning: Bucket " << bid << " is empty. Skipping.\n";
                continue;
            }

            // 2. current_bucket基于自己桶内所有doc的全局ID和dataset，完成自我构建，并聚合得到agg_or_mask和agg_and_mask
            // 注意：虽然向量数据主要在 NPU 使用，但 CPU 这里仍需访问 dataset.vectors
            // 来计算模长 (Norm) 并存储在 Bucket 对象中，这是 L2 距离分解计算所必需的。
            current_bucket.ImportDataAndComputeStats(dataset, input_bitmap_stride, agg_or_mask, agg_and_mask);

            // 3. current_bucket将统计信息注册到 BucketLevelIVF 中
            bucket_level_ivf_.RegisterBucket(bid, agg_or_mask, agg_and_mask);
        }
    }

    bool TryLoadCpuIndexes(const BucketDocTable &normalized_bucket_doc_table)
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
                                 normalized_bucket_doc_table,
                                 loaded_buckets))
        {
            std::cout << "[DB] Warning: bucket index file parameter mismatch: "
                      << bucket_index_path << std::endl;
            return false;
        }

        BucketLevelIVF loaded_bucket_level_ivf;
        if (!LoadBucketIvfIndexFile(bucket_ivf_path, loaded_bucket_level_ivf))
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
                                       BucketLevelIVF &bucket_level_ivf)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            return false;
        }

        DataReader::BucketIvfIndexFileHeaderDisk header{};
        if (!DataReader::ReadBinaryExact(in, &header, sizeof(header)) ||
            !IsValidBucketIvfIndexFileHeader(header, bucket_level_ivf)) {
            return false;
        }

        return bucket_level_ivf.Deserialize(in);
    }

    static bool LoadBucketIndexFile(const std::filesystem::path &path,
                                    const BucketDocTable &normalized_bucket_doc_table,
                                    std::vector<Bucket> &buckets_out)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in.is_open()) {
            return false;
        }

        DataReader::BucketIndexFileHeaderDisk header{};
        uint64_t bucket_count = 0;
        if (!DataReader::ReadBinaryExact(in, &header, sizeof(header)) ||
            !IsValidBucketIndexFileHeader(header) ||
            !DataReader::ReadBinaryExact(in, &bucket_count, sizeof(bucket_count))) {
            return false;
        }

        if (bucket_count != normalized_bucket_doc_table.size()) {
            return false;
        }

        std::vector<Bucket> loaded_buckets;
        loaded_buckets.reserve(static_cast<size_t>(bucket_count));
        for (size_t bid = 0; bid < static_cast<size_t>(bucket_count); ++bid)
        {
            Bucket bucket;
            if (!bucket.Deserialize(in) ||
                !bucket.MatchesGlobalIds(normalized_bucket_doc_table[bid])) {
                return false;
            }
            loaded_buckets.push_back(std::move(bucket));
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

        const DataReader::BucketIvfIndexFileHeaderDisk header = MakeBucketIvfIndexFileHeader(bucket_level_ivf_);
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

        const DataReader::BucketIndexFileHeaderDisk header = MakeBucketIndexFileHeader();
        const uint64_t bucket_count = static_cast<uint64_t>(buckets_.size());
        bool ok = DataReader::WriteBinaryExact(out, &header, sizeof(header)) &&
                  DataReader::WriteBinaryExact(out, &bucket_count, sizeof(bucket_count));
        for (const Bucket &bucket : buckets_)
        {
            ok = ok && bucket.Serialize(out);
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

    static BucketDocTable NormalizeBucketDocTable(const BucketDocTable &bucket_doc_table)
    {
        if (bucket_doc_table.size() != static_cast<size_t>(total_bucket_num))
        {
            throw std::invalid_argument("[DB] bucket_doc_table size does not match total_bucket_num.");
        }

        BucketDocTable normalized(bucket_doc_table);
        size_t total_stored_doc_refs = 0;

        for (size_t bid = 0; bid < normalized.size(); ++bid)
        {
            auto &doc_ids = normalized[bid];
            std::sort(doc_ids.begin(), doc_ids.end());
            doc_ids.erase(std::unique(doc_ids.begin(), doc_ids.end()), doc_ids.end());

            if (doc_ids.size() > static_cast<size_t>(max_doc_per_bucket))
            {
                throw std::runtime_error("[DB] bucket_doc_table contains a bucket that exceeds max_doc_per_bucket.");
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
                  << static_cast<size_t>(total_bucket_num) * static_cast<size_t>(max_doc_per_bucket)
                  << std::endl;
        return normalized;
    }

    std::vector<Bucket> buckets_;
    BucketLevelIVF bucket_level_ivf_;

    // 扁平化存储的桶中心向量
    std::vector<float, AlignedAllocator<float>> bucket_centroids_;
};
