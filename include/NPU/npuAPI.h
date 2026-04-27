#pragma once

#include <vector>
#include <cstdint>
#include <memory>
#include "utils/DataReader.h"
#include "DataBaseCPU/InputDataset.h"

// 引入 Ascend 运行时头文件
#include <acl/acl.h>

namespace npuAPI
{
    // Stream 句柄包装
    using Stream = aclrtStream;
    constexpr int kGroupFlagSlotCount = 3;

    /**
     * @brief 索引初始化
     * 1. Float32 -> Float16 (Half)
     * 2. 上传至 8 张 NPU 的 HBM (保持 ND 格式)
     * 3. 预分配每个 Group 的 Device Workspace (Query, Doc, Result)
     * 4. 初始化 Flag 同步机制资源
     */
    void Init(const InputDataset &dataset, const BucketDocTable &bucket_doc_table);

    /**
     * @brief 获取当前 CPU Group 绑定的 NPU Stream
     */
    Stream GetGroupStream(int group_id);

    // --- Pinned Memory 管理接口 ---
    
    /**
     * @brief 分配 Host 侧的 Pinned Memory (页锁定内存)
     * 用于 Flag 和 Result Buffer，支持设备直接 DMA 访问
     */
    void AllocateHostPinned(void** ptr, size_t size);

    /**
     * @brief 释放 Host Pinned Memory
     */
    void FreeHostPinned(void* ptr);

    // --- Flag 同步接口 ---

    /**
     * @brief 获取 Group 对应槽位的 Host Flag 指针
     * Worker 线程通过轮询此地址判断 NPU 任务是否完成
     */
    volatile uint32_t* GetGroupFlag(int group_id, int slot_id);

    /**
     * @brief 将槽位 Flag 置为 0 (idle/ready)，用于新查询初始化或回收
     */
    void ClearGroupFlag(int group_id, int slot_id);

    /**
     * @brief 将槽位 Flag 置为 1 (busy)，表示逻辑预取任务开始
     */
    void ResetGroupFlag(int group_id, int slot_id);

    /**
     * @brief 在 Stream 尾部记录完成事件，替代原来的 D2H Flag 拷贝
     */
    double EnqueueGroupCompletion(Stream stream, int group_id, int slot_id);

    /**
     * @brief 查询事件是否完成
     */
    bool IsGroupCompletionReady(int group_id, int slot_id);

    /**
     * @brief 每个 query 调用一次：FP32→FP16 转换 + H2D 拷贝 query 向量
     * 后续 LaunchBatchKernel 不再重复拷贝 query
     * @param mmad_query_padded_fp16 若提供已预处理的 FP16 buffer，则直接使用；否则在函数内转换
     */
    void UploadQuery(const float *query_vector,
                     const uint16_t *mmad_query_padded_fp16,
                     int group_id);

    /**
     * @brief 启动异步批量计算任务链
     * 前置条件：已调用 UploadQuery 上传 query 向量
     * * 流程 (全异步):
     * 1. H2D: 拷贝任务参数 BatchTaskData
     * 2. Kernel: 启动计算核函数
     * 3. D2H: 将计算结果 DMA 到 host_output_buffer（单次拷贝，紧凑布局）
     * * @param host_output_buffer 必须是 Pinned Memory，用于接收结果
     */
    void LaunchBatchKernel(Stream stream,
                           const std::vector<uint32_t> &bucket_ids,
                           float *host_output_buffer,
                           int group_id);

    void DebugVerifyBatchResults(const std::vector<uint32_t> &bucket_ids,
                                 const float *query_vector,
                                 const float *host_output_buffer,
                                 int group_id);

    void SynchronizeStream(Stream stream);
    void Finalize();
}
