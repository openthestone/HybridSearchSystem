#pragma once

#include <cstdint>
#include <vector>

using BucketDocTable = std::vector<std::vector<uint32_t>>;

struct InputDataset
{
    // uint32_t total_doc_num 已定义

    // 1. 向量数据
    // 布局: [total_doc_num * vector_dim]
    const float *vectors;

    // 2. 默认分桶结果
    // doc_bucket_ids[i] 表示第 i 个 doc 在“单桶归属”缓存中的默认桶号。
    // 多桶归属场景下，实际建库入口改为显式传入 BucketDocTable。
    const uint32_t *doc_bucket_ids;

    // 3. 属性 Tag 位图
    // 每个 doc 需要 ceil(total_tag_num/64) = 2 个 uint64，这里的2就是 bitmap_stride
    // 大小 = total_doc_num * bitmap_stride
    // tag_bitmaps[i * bitmap_stride] 开始的 bits 表示第 i 个 doc 的 tags
    const uint64_t *tag_bitmaps;

    // 4. 桶中心向量
    // 布局: [total_bucket_num * vector_dim]
    const float *bucket_centroids;
};
