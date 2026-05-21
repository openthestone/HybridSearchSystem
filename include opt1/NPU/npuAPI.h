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

    // GatherL0Scores 内部各阶段耗时
    struct GatherTimingMs
    {
        double host_offset_build_ms = 0; // 构建 src/dst/count 偏移数组
        double d2d_submit_ms = 0;        // 提交 D2D memcpy (HBM→HBM)
        double d2h_submit_ms = 0;        // 提交 D2H memcpy
        double sync_wait_ms = 0;         // aclrtSynchronizeStream 墙钟时间
    };

    // L2 桶在 HBM L0 缓冲区中的分数位置
    struct L2ScoreLocation
    {
        size_t offset_bytes = 0; // 在 dev_l0_cache 中的字节偏移
        uint32_t doc_count = 0;  // 该 L2 桶的文档数
    };

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
     * @brief 重置 L0 cache 写入偏移（每条查询开始时调用）
     */
    void ResetL0CacheOffset(int group_id);

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

    /**
     * @brief L0 轮次专用：算分到 dev_l0_cache，记录偏移映射，不执行 D2H
     * @param l2_score_map 输出参数，记录每个 L2 桶在 HBM 中的偏移
     */
    void LaunchL0BatchKernel(Stream stream,
                              const std::vector<uint32_t> &bucket_ids,
                              int group_id,
                              std::vector<L2ScoreLocation> &l2_score_map);

    /**
     * @brief Gather kernel: 从 dev_l0_cache 紧凑提取指定 L2 桶分数到 d_result_ws，再 D2H
     * @param l2_ids 需要的 L2 桶 ID 列表（按 first_round 的 L2 顺序）
     * @param l2_score_map L2→HBM 偏移映射
     * @param host_output_buffer Host pinned memory 接收紧凑结果
     * @return 紧凑结果的总字节数
     */
    size_t GatherL0Scores(Stream stream,
                           int group_id,
                           const std::vector<uint32_t> &l2_ids,
                           const std::vector<L2ScoreLocation> &l2_score_map,
                           float *host_output_buffer,
                           GatherTimingMs *timing = nullptr);

    void DebugVerifyBatchResults(const std::vector<uint32_t> &bucket_ids,
                                 const float *query_vector,
                                 const float *host_output_buffer,
                                 int group_id);

    void Finalize();
}
