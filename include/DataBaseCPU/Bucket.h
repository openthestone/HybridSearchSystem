#pragma once

#include <vector>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>

// 引入 NEON 头文件，用于ARM平台加速
#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#endif

#include "utils/DataReader.h"
#include "AlignedAllocator.h"
#include "InputDataset.h"

class Bucket
{
public:
    Bucket() : doc_num_(0), stride_(0), current_build_idx_(0) {}

    /**
     * @brief 构造函数
     * @param ids 传入该桶包含的所有 doc 的全局 ID
     */
    Bucket(std::vector<uint32_t> ids) : global_ids_(std::move(ids))
    {
        // 在 DataBaseCPU 构建流程中，传入的 ids 天然有序

        doc_num_ = (uint32_t)global_ids_.size();
        current_build_idx_ = 0; // 初始化构建游标

        // 计算步长
        if (doc_num_ > 0)
        {
            stride_ = (doc_num_ + 63) / 64;
        }
        else
        {
            stride_ = 0;
        }

        // 初始化 Offset 数组 (全部指向哨兵行)
        // 修改：防止溢出，底层存储已改为 uint32_t
        tag_offsets_.assign(total_tag_num, 0);

        // 初始化 Bitmap 数据区
        // 第 0 行作为哨兵行（全0），未分配的 tag 指向此处
        if (stride_ > 0)
        {
            bitmap_data_.resize(stride_, 0);
        }

        // 新增：预留 Norm 存储空间
        if (doc_num_ > 0)
        {
            precomputed_norms_.reserve(doc_num_);
        }
    }

    // ==========================================
    // 构建接口：导入dataset + 自我构建 + 计算agg_or_mask和agg_and_mask
    // ==========================================
    void ImportDataAndComputeStats(const InputDataset &dataset,
                                   uint32_t input_bitmap_stride,
                                   std::vector<uint64_t> &out_or_mask,
                                   std::vector<uint64_t> &out_and_mask)
    {
        // 1. 重置输出掩码
        // OR 初始化为 0，AND 初始化为全 1 (全1表示"目前为止所有doc都有该tag")
        std::fill(out_or_mask.begin(), out_or_mask.end(), 0ULL);
        std::fill(out_and_mask.begin(), out_and_mask.end(), ~0ULL);

        if (doc_num_ == 0)
            return;

        // 2. 遍历桶内所有文档
        for (uint32_t global_id : global_ids_)
        {
            // --- A. 处理属性 ---
            // 定位到 InputDataset 中该 doc 的 tags 起始位置
            const uint64_t *doc_tags_ptr = dataset.tag_bitmaps + (size_t)global_id * input_bitmap_stride;

            // 将 Tags 写入 Bucket (内部自动转置存储)
            append_doc_tags(doc_tags_ptr, input_bitmap_stride);

            // 计算局部agg_mask
            uint32_t k = 0;

            // 使用 NEON SIMD 加速属性聚合
#if defined(__aarch64__) || defined(__arm__)
            // 每次处理 128 位 (2 个 uint64)
            for (; k + 1 < input_bitmap_stride; k += 2)
            {
                // 加载当前 doc 的 tags
                uint64x2_t v_doc = vld1q_u64(doc_tags_ptr + k);

                // 加载当前的累积掩码
                uint64x2_t v_out_or = vld1q_u64(out_or_mask.data() + k);
                uint64x2_t v_out_and = vld1q_u64(out_and_mask.data() + k);

                // 执行位运算: OR 和 AND
                v_out_or = vorrq_u64(v_out_or, v_doc);
                v_out_and = vandq_u64(v_out_and, v_doc);

                // 写回结果
                vst1q_u64(out_or_mask.data() + k, v_out_or);
                vst1q_u64(out_and_mask.data() + k, v_out_and);
            }
#endif
            // 处理剩余部分 (Scalar Fallback)
            for (; k < input_bitmap_stride; ++k)
            {
                uint64_t block = doc_tags_ptr[k];
                out_or_mask[k] |= block;
                out_and_mask[k] &= block;
            }

            // --- B. 新增：处理向量 Norm ---
            // 依据方案：ScoreL2 = 0.5 * ||v||^2 - <v, q>
            // CPU 必须预存 0.5 * ||v||^2
            const float *vec_ptr = dataset.vectors + (size_t)global_id * vector_dim;
            float norm_sq = 0.0f;
            int d = 0;

            // 使用 NEON SIMD 加速向量模长计算
#if defined(__aarch64__) || defined(__arm__)
            // 假设 vector_dim 是 4 的倍数 (例如 32)
            float32x4_t v_sum = vdupq_n_f32(0.0f);
            for (; d + 3 < vector_dim; d += 4)
            {
                float32x4_t v = vld1q_f32(vec_ptr + d);
                // Fused Multiply-Add: v_sum += v * v
                v_sum = vmlaq_f32(v_sum, v, v);
            }
            // 归约求和
            norm_sq = vaddvq_f32(v_sum);
#endif
            // 处理剩余部分
            for (; d < vector_dim; ++d)
            {
                norm_sq += vec_ptr[d] * vec_ptr[d];
            }
            precomputed_norms_.push_back(norm_sq * 0.5f);
        }
    }

    // ==========================================
    // 写属性倒排表的接口
    // 有append_doc_tags()和set_bit()两个接口，主要用前者
    // ==========================================

    /**
     * @brief 离线构建时使用的接口：追加一个 Doc 的所有 Tags
     */
    void append_doc_tags(const uint64_t *input_bitmap_ptr, uint32_t bitmap_len)
    {
        if (current_build_idx_ >= doc_num_)
            return; // 越界保护

        // 遍历输入 bitmap 的每一个 uint64
        for (uint32_t i = 0; i < bitmap_len; ++i)
        {
            uint64_t word = input_bitmap_ptr[i];
            if (word == 0)
                continue; // 快速跳过全0块

            // 遍历该 uint64 中的每一个 bit 1
            for (int bit = 0; bit < 64; ++bit)
            {
                if ((word >> bit) & 1ULL)
                {
                    uint32_t tag_id = i * 64 + bit;
                    if (tag_id < total_tag_num)
                    {
                        set_bit_internal(tag_id, current_build_idx_);
                    }
                }
            }
        }
        // 移动游标指向下一个 doc
        current_build_idx_++;
    }

    /**
     * @brief 通过doc的全局 ID 设置属性位 (用于离线构建完成后的补充修改)
     */
    void set_bit(uint32_t tag_id, uint32_t global_doc_id)
    {
        if (tag_id >= total_tag_num || doc_num_ == 0)
            return;

        auto it = std::lower_bound(global_ids_.begin(), global_ids_.end(), global_doc_id);
        if (it == global_ids_.end() || *it != global_doc_id)
        {
            std::cout << "Error: Attempt to set tag for doc_id not in this bucket: " << global_doc_id << std::endl;
            return;
        }

        uint32_t local_doc_idx = (uint32_t)std::distance(global_ids_.begin(), it);
        set_bit_internal(tag_id, local_doc_idx);
    }

    // ==========================================
    // 查询接口
    // ==========================================

    const uint64_t *get_tag_bits(uint32_t tag_id) const
    {
        if (tag_id >= total_tag_num)
            return &bitmap_data_[0];
        uint32_t offset = tag_offsets_[tag_id];

        // 如果 offset 为 0，说明该 tag 指向哨兵行（本桶无此 tag），返回全 0 行
        if (offset == 0)
            return &bitmap_data_[0];

        return &bitmap_data_[offset];
    }

    // 获取预计算的 Norm 数组
    const float *get_norms() const { return precomputed_norms_.data(); }

    uint32_t get_doc_num() const { return doc_num_; }
    uint32_t get_stride() const { return stride_; }
    const std::vector<uint32_t> &get_global_ids() const { return global_ids_; }

    bool Serialize(std::ofstream &out) const
    {
        return DataReader::WriteBinaryExact(out, &doc_num_, sizeof(doc_num_)) &&
               DataReader::WriteBinaryExact(out, &stride_, sizeof(stride_)) &&
               DataReader::WriteBinaryVector(out, global_ids_) &&
               DataReader::WriteBinaryVector(out, tag_offsets_) &&
               DataReader::WriteBinaryVector(out, bitmap_data_) &&
               DataReader::WriteBinaryVector(out, precomputed_norms_);
    }

    bool Deserialize(std::ifstream &in)
    {
        Bucket loaded;

        if (!DataReader::ReadBinaryExact(in, &loaded.doc_num_, sizeof(loaded.doc_num_)) ||
            !DataReader::ReadBinaryExact(in, &loaded.stride_, sizeof(loaded.stride_)) ||
            !DataReader::ReadBinaryVector(in, loaded.global_ids_) ||
            !DataReader::ReadBinaryVector(in, loaded.tag_offsets_) ||
            !DataReader::ReadBinaryVector(in, loaded.bitmap_data_) ||
            !DataReader::ReadBinaryVector(in, loaded.precomputed_norms_)) {
            return false;
        }

        const uint32_t expected_stride = (loaded.doc_num_ == 0) ? 0 : (loaded.doc_num_ + 63) / 64;
        if (loaded.stride_ != expected_stride) {
            return false;
        }
        if (loaded.global_ids_.size() != static_cast<size_t>(loaded.doc_num_)) {
            return false;
        }
        if (loaded.tag_offsets_.size() != static_cast<size_t>(total_tag_num)) {
            return false;
        }
        if (loaded.precomputed_norms_.size() != static_cast<size_t>(loaded.doc_num_)) {
            return false;
        }
        if (loaded.doc_num_ > 0 && loaded.bitmap_data_.size() < static_cast<size_t>(loaded.stride_)) {
            return false;
        }

        loaded.current_build_idx_ = loaded.doc_num_;
        *this = std::move(loaded);
        return true;
    }

    bool MatchesGlobalIds(const std::vector<uint32_t> &expected_global_ids) const
    {
        return global_ids_ == expected_global_ids;
    }

private:
    uint32_t doc_num_;
    uint32_t stride_;
    uint32_t current_build_idx_;
    std::vector<uint32_t> global_ids_;

    // 从 uint16_t 升级为 uint32_t，消除单桶 tag 种类过多时的溢出风险
    std::vector<uint32_t> tag_offsets_;

    // 属性 bitmap 数据
    std::vector<uint64_t, AlignedAllocator<uint64_t>> bitmap_data_;

    // 存储 0.5 * ||v||^2，用于 L2 距离修正
    std::vector<float, AlignedAllocator<float>> precomputed_norms_;

    void set_bit_internal(uint32_t tag_id, uint32_t local_doc_idx)
    {
        if (tag_offsets_[tag_id] == 0)
        {
            allocate_row_for_tag(tag_id);
        }
        uint32_t row_start_idx = tag_offsets_[tag_id];
        uint32_t word_offset = local_doc_idx / 64;
        uint32_t bit_idx = local_doc_idx % 64;
        bitmap_data_[row_start_idx + word_offset] |= (1ULL << bit_idx);
    }

    void allocate_row_for_tag(uint32_t tag_id)
    {
        size_t new_start_idx = bitmap_data_.size();
        if (new_start_idx > std::numeric_limits<uint32_t>::max())
        {
            throw std::overflow_error("Bucket bitmap too large, offset overflow!");
        }
        tag_offsets_[tag_id] = (uint32_t)new_start_idx;
        bitmap_data_.resize(new_start_idx + stride_, 0);
    }
};
