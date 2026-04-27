#pragma once
#include <cstdint>

// 定义一个桶的计算任务
struct BatchTaskData
{
    uint64_t offset_A = 0; // 指定该数据桶（Bucket）的向量矩阵在 NPU HBM 中的起始位置
    uint64_t offset_C = 0; // 指定计算结果（内积得分）应写入结果缓冲区（Result Workspace，在HBM上）的起始位置
    uint32_t m = 0;        // 该桶内的doc数量
    uint32_t reserved = 0; // 显式补齐，保证 Host/Device 结构布局一致
};
