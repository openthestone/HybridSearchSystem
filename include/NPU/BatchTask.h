#pragma once
#include <cstdint>

// 定义一个桶的计算任务
struct BatchTaskData
{
    uint32_t offset_A; // 指定该数据桶（Bucket）的向量矩阵在 NPU HBM 中的起始位置
    uint32_t offset_C; // 指定计算结果（内积得分）应写入结果缓冲区（Result Workspace，在HBM上）的起始位置
    uint32_t m;        // 该桶内的doc数量
};