#pragma once

#include <cstdint>
#include <vector>

using BucketDocTable = std::vector<std::vector<uint32_t>>;

struct TwoLevelBucketLayout {
    BucketDocTable level_1_bucket_doc_table;
    BucketDocTable level_2_bucket_doc_table;

    // level_1_to_level_2_offsets[i] ~ level_1_to_level_2_offsets[i + 1]
    // 对应第 i 个一级桶名下的连续二级桶 ID 范围。
    std::vector<uint32_t> level_1_to_level_2_offsets;

    size_t Level1BucketCount() const {
        return level_1_bucket_doc_table.size();
    }
    size_t Level2BucketCount() const {
        return level_2_bucket_doc_table.size();
    }
};

struct InputDataset {
    // uint32_t total_doc_num 已定义

    // 1. 向量数据
    // 布局: [total_doc_num * vector_dim]
    const float* vectors;

    // 2. 属性 Tag 位图
    // 每个 doc 需要 ceil(total_tag_num/64) = 2 个 uint64，这里的2就是 bitmap_stride
    // 大小 = total_doc_num * bitmap_stride
    // tag_bitmaps[i * bitmap_stride] 开始的 bits 表示第 i 个 doc 的 tags
    const uint64_t* tag_bitmaps;

    // 3. 一级桶中心向量
    // 布局: [total_bucket_num_level_1 * vector_dim]
    const float* bucket_centroids;

    // 4. L2→L0 mapping (optional, for L0 NPU score caching)
    // Layout: [total_bucket_num_level_2], maps each L2 bucket to its L0 mesocluster
    const uint32_t* l1_to_l0_map = nullptr;
};
