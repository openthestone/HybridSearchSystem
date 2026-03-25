#pragma once

#include <vector>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>

#include "utils/DataReader.h"
#include "AlignedAllocator.h"

class BucketLevelIVF
{
public:
    // 桶级属性倒排
    // 改造为 物理分片 (Sharding per Core) + Cache Line 对齐
    // 第一维：Core ID (0 ~ cores_per_group-1)
    // 第二维：Tag ID (扁平化存储， stride 为 aligned_stride_)
    BucketLevelIVF()
    {
        // 1. 计算每个 Core 负责的桶数量 (向上取整)
        buckets_per_core_ = (total_bucket_num + cores_per_group - 1) / cores_per_group;

        // 2. 计算每个 Core 需要的 bits 和 uint64 数量
        uint32_t bits_per_core = buckets_per_core_;
        uint32_t u64_per_core = (bits_per_core + 63) / 64;

        // 3. 计算 Cache Line 对齐步长 (Padding)
        // cpu_cache_line_size 通常为 128 (Kunpeng 920) 或 64
        // 我们需要 stride 是 cache_line_size 的倍数
        uint32_t cache_line_u64 = cpu_cache_line_size / 8;

        // aligned_stride_ 是每个 Tag 在每个 Core 分片中占用的 uint64 数量
        aligned_stride_ = (u64_per_core + cache_line_u64 - 1) / cache_line_u64 * cache_line_u64;

        size_t shard_size_u64 = (size_t)total_tag_num * aligned_stride_;

        // 4. 初始化分片
        shards_and_or_.resize(cores_per_group);
        shards_not_.resize(cores_per_group);

        for (int i = 0; i < cores_per_group; ++i)
        {
            // 使用 resize 初始化为 0
            shards_and_or_[i].resize(shard_size_u64, 0);
            shards_not_[i].resize(shard_size_u64, 0);
        }

        std::cout << "[BucketLevelIVF] Optimized Layout: "
                  << "BucketsPerCore=" << buckets_per_core_
                  << ", RawU64=" << u64_per_core
                  << ", AlignedStride=" << aligned_stride_
                  << " (" << aligned_stride_ * 8 << " Bytes)" << std::endl;
    }

    // ==========================================
    // 注册接口：根据agg_or_mask和agg_and_mask更新索引
    // ==========================================
    void RegisterBucket(uint32_t bucket_id,
                        const std::vector<uint64_t> &agg_or_mask,
                        const std::vector<uint64_t> &agg_and_mask)
    {
        // 1. 计算该桶属于哪个 Core 的分片
        int core_idx = bucket_id / buckets_per_core_;
        int local_bucket_idx = bucket_id % buckets_per_core_;

        if (core_idx >= cores_per_group)
            return; // 越界保护

        // 2. 遍历所有 Tag，更新对应 Core 的分片
        // 这里的循环是 Tag-Major 的，但在 Register 阶段性能不是瓶颈
        for (uint32_t tag_idx = 0; tag_idx < total_tag_num; ++tag_idx)
        {
            uint32_t word_idx = tag_idx / 64;
            uint32_t bit_idx = tag_idx % 64;

            // 提取该 Tag 在该 Bucket 的统计位
            bool has_tag = (agg_or_mask[word_idx] >> bit_idx) & 1ULL;
            bool all_have_tag = (agg_and_mask[word_idx] >> bit_idx) & 1ULL;

            // 计算在 Shard 中的位置
            // Row_Start = tag_idx * aligned_stride_
            // Word_Offset = local_bucket_idx / 64
            // Bit_Offset = local_bucket_idx % 64
            size_t row_start = (size_t)tag_idx * aligned_stride_;
            size_t word_offset = local_bucket_idx / 64;
            size_t bit_offset = local_bucket_idx % 64;

            // 写入 table_and_or_ (语义: 该桶是否有 doc 包含该 tag)
            if (has_tag)
            {
                shards_and_or_[core_idx][row_start + word_offset] |= (1ULL << bit_offset);
            }

            // 写入 table_not_ (语义: 该桶是否有 doc *不* 包含该 tag)
            if (!all_have_tag)
            {
                shards_not_[core_idx][row_start + word_offset] |= (1ULL << bit_offset);
            }
        }
    }

    // ==========================================
    // 查询接口 (返回行指针，供高效位运算使用)
    // ==========================================

    // 获取指定 Core 分片中，指定 Tag 的行起始指针
    const uint64_t *get_row_and_or(int core_idx, uint32_t tag_id) const
    {
        if (tag_id >= total_tag_num || core_idx >= cores_per_group)
            return nullptr;
        return &shards_and_or_[core_idx][(size_t)tag_id * aligned_stride_];
    }

    const uint64_t *get_row_not(int core_idx, uint32_t tag_id) const
    {
        if (tag_id >= total_tag_num || core_idx >= cores_per_group)
            return nullptr;
        return &shards_not_[core_idx][(size_t)tag_id * aligned_stride_];
    }

    // 返回对齐后的步长 (每个 Core 处理的 uint64 数量，包含 padding)
    uint32_t get_aligned_stride() const
    {
        return aligned_stride_;
    }

    // 返回每个 Core 负责的逻辑桶数量 (用于反算全局 ID)
    uint32_t get_buckets_per_core() const
    {
        return buckets_per_core_;
    }

    bool Serialize(std::ofstream &out) const
    {
        const uint32_t shard_count = static_cast<uint32_t>(shards_and_or_.size());
        const uint64_t shard_size_u64 = shard_count == 0 ? 0 : static_cast<uint64_t>(shards_and_or_[0].size());

        if (!DataReader::WriteBinaryExact(out, &buckets_per_core_, sizeof(buckets_per_core_)) ||
            !DataReader::WriteBinaryExact(out, &aligned_stride_, sizeof(aligned_stride_)) ||
            !DataReader::WriteBinaryExact(out, &shard_count, sizeof(shard_count)) ||
            !DataReader::WriteBinaryExact(out, &shard_size_u64, sizeof(shard_size_u64))) {
            return false;
        }

        for (const auto &shard : shards_and_or_) {
            if (shard.size() != static_cast<size_t>(shard_size_u64) ||
                !DataReader::WriteBinaryExact(out,
                                              shard.data(),
                                              static_cast<size_t>(shard_size_u64) * sizeof(uint64_t))) {
                return false;
            }
        }
        for (const auto &shard : shards_not_) {
            if (shard.size() != static_cast<size_t>(shard_size_u64) ||
                !DataReader::WriteBinaryExact(out,
                                              shard.data(),
                                              static_cast<size_t>(shard_size_u64) * sizeof(uint64_t))) {
                return false;
            }
        }

        return true;
    }

    bool Deserialize(std::ifstream &in)
    {
        uint32_t file_buckets_per_core = 0;
        uint32_t file_aligned_stride = 0;
        uint32_t file_shard_count = 0;
        uint64_t file_shard_size_u64 = 0;

        if (!DataReader::ReadBinaryExact(in, &file_buckets_per_core, sizeof(file_buckets_per_core)) ||
            !DataReader::ReadBinaryExact(in, &file_aligned_stride, sizeof(file_aligned_stride)) ||
            !DataReader::ReadBinaryExact(in, &file_shard_count, sizeof(file_shard_count)) ||
            !DataReader::ReadBinaryExact(in, &file_shard_size_u64, sizeof(file_shard_size_u64))) {
            return false;
        }

        if (file_buckets_per_core != buckets_per_core_ ||
            file_aligned_stride != aligned_stride_ ||
            file_shard_count != static_cast<uint32_t>(shards_and_or_.size()) ||
            file_shard_count != static_cast<uint32_t>(shards_not_.size())) {
            return false;
        }

        for (size_t core_idx = 0; core_idx < shards_and_or_.size(); ++core_idx) {
            if (shards_and_or_[core_idx].size() != static_cast<size_t>(file_shard_size_u64) ||
                !DataReader::ReadBinaryExact(in,
                                             shards_and_or_[core_idx].data(),
                                             static_cast<size_t>(file_shard_size_u64) * sizeof(uint64_t))) {
                return false;
            }
        }
        for (size_t core_idx = 0; core_idx < shards_not_.size(); ++core_idx) {
            if (shards_not_[core_idx].size() != static_cast<size_t>(file_shard_size_u64) ||
                !DataReader::ReadBinaryExact(in,
                                             shards_not_[core_idx].data(),
                                             static_cast<size_t>(file_shard_size_u64) * sizeof(uint64_t))) {
                return false;
            }
        }

        return true;
    }

private:
    uint32_t buckets_per_core_;
    uint32_t aligned_stride_; // Cache Line Aligned Stride

    // 物理分片存储
    // shards_[core_idx] 是一个扁平化的 vector
    std::vector<std::vector<uint64_t, AlignedAllocator<uint64_t>>> shards_and_or_;
    std::vector<std::vector<uint64_t, AlignedAllocator<uint64_t>>> shards_not_;
};
