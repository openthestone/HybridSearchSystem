#include "NPU/npuAPI.h"
#include "DataBaseCPU/InputDataset.h"
#include <atomic>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <vector>
#include "acl/acl.h"
#include "utils/DataReader.h"
#include "NPU/BatchTask.h" //避免acl头文件污染

// 引用内核启动头文件
#include "aclrtlaunch_kernel_vector_mmad.h"

namespace npuAPI
{
    uint16_t Float32ToFp16(float value)
    {
        uint32_t f32;
        std::memcpy(&f32, &value, sizeof(float));
        uint16_t f16 = 0;
        uint32_t sign = (f32 >> 16) & 0x8000;
        int16_t exponent = ((f32 >> 23) & 0xFF) - 127;
        uint32_t mantissa = f32 & 0x007FFFFF;
        if (exponent > 15)
        {
            f16 = sign | 0x7C00;
        }
        else if (exponent <= -15)
        {
            f16 = sign;
        }
        else
        {
            exponent += 15;
            mantissa >>= 13;
            f16 = sign | (exponent << 10) | mantissa;
        }
        return f16;
    }

    struct DeviceContext
    {
        //定义一张NPU设备的信息
        int device_id = -1;   //0~7
        void *dev_base_addr = nullptr; //该NPU的 HBM 上存储全体向量数据的基地址
        std::vector<aclrtStream> streams;  //每个NPU上有3个group，每个group有一个 Stream
    };

    struct GroupContext
    {
        /*
            定义一个group的device侧的workspace
            指针指向的是HBM上的地址
            被group循环利用
        */
        void *d_query_ws = nullptr;   //存放查询向量
        void *d_result_ws = nullptr;  //存放计算结果
        void *d_task_ws = nullptr;    // 存放CPU发来的BatchTaskData 数组（比如有200个element）

        // --- Flag 同步资源 ---
        uint32_t *h_flags[kGroupFlagSlotCount] = {}; // Host 侧 Flag 槽位 (Pinned Memory)
        aclrtEvent completion_events[kGroupFlagSlotCount] = {}; // Stream 尾部完成事件
        void *d_flag_zero = nullptr; // Device 侧存放 0 值的内存
    };

    struct BucketMeta
    {
        /*
            在 CPU 侧记录每个桶在 NPU HBM 中的物理位置
            在Init 索引构建时：根据 BucketMeta 将向量数据上传至 DeviceContext 的 dev_base_addr内存块
            在LaunchBatchKernel时，cpu发来一批桶id，然后依据 BucketMeta 构建 BatchTaskData 数组
        */
        size_t byte_offset; //桶起始位置相对于 dev_base_addr 的偏移量
        uint32_t doc_num;   //桶内doc数量
        size_t doc_id_offset = 0; // 调试校验时用于回溯桶内 doc 的全局 ID 顺序
    };

    // 注意：这里不能在静态初始化阶段使用配置全局变量，
    // 否则会在 main()->LoadParams() 之前使用默认值。
    static std::vector<DeviceContext> g_devices;
    static std::vector<GroupContext> g_group_ctxs;
    static std::vector<BucketMeta> g_bucket_metas;
    static std::vector<uint32_t> g_bucket_doc_ids;
    static const float *g_host_vectors = nullptr;
    static bool g_acl_initialized = false;
    static bool g_initialized = false;
    static std::mutex g_debug_log_mutex;
    static std::atomic<uint64_t> g_debug_batch_counter{0};

    namespace
    {
        constexpr uint32_t kKernelResultCols = 16;
        constexpr uint32_t kKernelMinRowTile = 16;
        constexpr uint32_t kKernelReferenceVectorDim = 64;
        constexpr uint32_t kKernelReferenceRowTile = 256;
        constexpr uint32_t kKernelATileElementBudget =
            kKernelReferenceVectorDim * kKernelReferenceRowTile;

        struct DebugMismatch
        {
            uint32_t bucket_id = 0;
            uint32_t local_doc_idx = 0;
            uint32_t global_doc_id = 0;
            float cpu_score = 0.0f;
            float npu_score = 0.0f;
            float abs_err = 0.0f;
            float allowed_err = 0.0f;
        };

        inline float ComputeCpuDotProduct(const float *lhs, const float *rhs, int dim)
        {
            float sum = 0.0f;
            for (int d = 0; d < dim; ++d)
            {
                sum += lhs[d] * rhs[d];
            }
            return sum;
        }

        inline uint32_t RoundDownToMultiple(uint32_t value, uint32_t multiple)
        {
            return (multiple == 0) ? value : (value / multiple) * multiple;
        }

        inline uint32_t GetKernelRowTile()
        {
            if (vector_dim <= 0)
            {
                return kKernelMinRowTile;
            }

            uint32_t row_tile =
                kKernelATileElementBudget / static_cast<uint32_t>(vector_dim);
            row_tile = std::max<uint32_t>(kKernelMinRowTile, row_tile);
            row_tile = std::max<uint32_t>(kKernelMinRowTile,
                                          RoundDownToMultiple(row_tile, kKernelMinRowTile));
            return std::min<uint32_t>(row_tile, static_cast<uint32_t>(max_doc_per_bucket_level_2));
        }

        inline size_t GetMaxKernelTasksPerBatch()
        {
            const uint32_t row_tile = GetKernelRowTile();
            const size_t max_tiles_per_bucket =
                (static_cast<size_t>(max_doc_per_bucket_level_2) + row_tile - 1) / row_tile;
            return static_cast<size_t>(max_process_bucket_num_level_2) * max_tiles_per_bucket;
        }

        void LogAclCleanupError(const char *op, aclError ret, int device_id, int group_id = -1)
        {
            std::cerr << "[NPU] " << op << " failed";
            if (device_id >= 0)
            {
                std::cerr << ", device=" << device_id;
            }
            if (group_id >= 0)
            {
                std::cerr << ", group=" << group_id;
            }
            std::cerr << ", code=" << ret << std::endl;
        }

        void ResetRuntimeState()
        {
            g_devices.clear();
            g_group_ctxs.clear();
            g_bucket_metas.clear();
            g_bucket_doc_ids.clear();
            g_host_vectors = nullptr;
            g_initialized = false;
            g_debug_batch_counter.store(0, std::memory_order_relaxed);
        }

        void FreeDeviceBuffer(void *&ptr, const char *label, int device_id, int group_id = -1)
        {
            if (!ptr)
            {
                return;
            }
            const auto ret = aclrtFree(ptr);
            if (ret != ACL_SUCCESS)
            {
                LogAclCleanupError(label, ret, device_id, group_id);
            }
            ptr = nullptr;
        }

        void FreeHostFlag(uint32_t *&ptr, int device_id, int group_id)
        {
            if (!ptr)
            {
                return;
            }
            const auto ret = aclrtFreeHost(ptr);
            if (ret != ACL_SUCCESS)
            {
                LogAclCleanupError("aclrtFreeHost(h_flag)", ret, device_id, group_id);
            }
            ptr = nullptr;
        }
    } // namespace

    // --- Pinned Memory Helper ---
    void AllocateHostPinned(void** ptr, size_t size)
    {
        // aclrtMallocHost 分配的是 Pinned Memory，物理地址连续，支持 Device 直接 DMA 访问
        auto ret = aclrtMallocHost(ptr, size);
        if (ret != ACL_SUCCESS) {
            std::cerr << "[NPU] aclrtMallocHost failed, code: " << ret << std::endl;
        }
    }

    void FreeHostPinned(void* ptr)
    {
        if (!ptr)
        {
            return;
        }
        auto ret = aclrtFreeHost(ptr);
        if (ret != ACL_SUCCESS)
        {
            std::cerr << "[NPU] aclrtFreeHost failed, code: " << ret << std::endl;
        }
    }

    void Init(const InputDataset &dataset, const BucketDocTable &bucket_doc_table)
    {
        if (g_initialized)
            return;

        g_devices.resize(npu_device_count);
        g_group_ctxs.resize(group_count);
        g_bucket_metas.resize(total_bucket_num_level_2);

        auto ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS)
        {
            std::cout << "aclInit fail: " << ret << std::endl;
            return;
        }
        g_acl_initialized = true;
        std::cout << "[NPU] Init: Loading Data..." << std::endl;
        g_host_vectors = dataset.vectors;
        g_bucket_doc_ids.clear();

        size_t total_stored_docs = 0;
        for (const auto &bucket_docs : bucket_doc_table)
        {
            total_stored_docs += bucket_docs.size();
        }

        if (npu_debug_verify != 0)
        {
            g_bucket_doc_ids.reserve(total_stored_docs);
            std::cout << "[NPU][Debug] CPU verification is enabled. abs_tol="
                      << npu_debug_verify_abs_tol
                      << ", rel_tol=" << npu_debug_verify_rel_tol
                      << ", max_report=" << npu_debug_verify_max_report
                      << std::endl;
        }

        size_t total_elements = total_stored_docs * static_cast<size_t>(vector_dim);
        for (int i = 0; i < total_bucket_num_level_2; ++i)
            g_bucket_metas[i].doc_num = static_cast<uint32_t>(bucket_doc_table[i].size());

        std::vector<uint16_t> host_buffer(total_elements);
        size_t cursor = 0;
        size_t current_byte_offset = 0;

        for (int bid = 0; bid < total_bucket_num_level_2; ++bid)
        {
            const auto &bucket_docs = bucket_doc_table[bid];
            g_bucket_metas[bid].byte_offset = current_byte_offset;
            if (npu_debug_verify != 0)
            {
                g_bucket_metas[bid].doc_id_offset = g_bucket_doc_ids.size();
            }
            for (uint32_t doc_id : bucket_docs)
            {
                if (npu_debug_verify != 0)
                {
                    g_bucket_doc_ids.push_back(doc_id);
                }
                const float *src_vec = dataset.vectors + (size_t)doc_id * vector_dim;
                for (int d = 0; d < vector_dim; ++d)
                {
                    host_buffer[cursor++] = Float32ToFp16(src_vec[d]);
                }
            }
            current_byte_offset += g_bucket_metas[bid].doc_num * vector_dim * sizeof(uint16_t);
        }
        //  (f32toFp16数据转换结束)

        size_t total_bytes = total_elements * sizeof(uint16_t);
        std::cout << "[NPU] StoredDocRefs=" << total_stored_docs
                  << ", HBMVectorBytesPerDevice=" << total_bytes
                  << std::endl;
        uint32_t max_docs_in_batch =
            static_cast<uint32_t>(max_process_bucket_num_level_2 * max_doc_per_bucket_level_2);

        size_t ws_query_size =
            static_cast<size_t>(vector_dim) * kKernelResultCols * sizeof(uint16_t); // 每个query向量都要padding到[k,16]

        // 结果按紧凑 [doc_num, 1] 布局写回（Fixpipe nSize=1 仅输出 column 0）。
        size_t ws_result_size = max_docs_in_batch * sizeof(float);

        // Task Workspace: 一个 bucket 可能被拆成多个 row-tile task。
        size_t ws_task_size = GetMaxKernelTasksPerBatch() * sizeof(BatchTaskData);

        for (int dev_id = npu_device_id_start; dev_id < npu_device_id_start + npu_device_count; ++dev_id)
        {
            int local_idx = dev_id - npu_device_id_start;
            g_devices[local_idx].device_id = dev_id;
            aclrtSetDevice(dev_id);

            const int dev_group_count = GroupCountForDevice(local_idx);
            for (int j = 0; j < dev_group_count; ++j)
            {
                int group_id = GroupIdForDeviceStream(local_idx, j);
                aclrtStream s = nullptr;
                aclrtCreateStream(&s);
                g_devices[local_idx].streams.push_back(s);

                aclrtMalloc(&g_group_ctxs[group_id].d_query_ws, ws_query_size, ACL_MEM_MALLOC_HUGE_FIRST);
                aclrtMalloc(&g_group_ctxs[group_id].d_result_ws, ws_result_size, ACL_MEM_MALLOC_HUGE_FIRST);
                aclrtMalloc(&g_group_ctxs[group_id].d_task_ws, ws_task_size, ACL_MEM_MALLOC_HUGE_FIRST);

                // --- Flag 资源初始化 ---
                for (int slot_id = 0; slot_id < kGroupFlagSlotCount; ++slot_id)
                {
                    AllocateHostPinned((void**)&g_group_ctxs[group_id].h_flags[slot_id], sizeof(uint32_t));
                    *g_group_ctxs[group_id].h_flags[slot_id] = 0U; // 初始为 idle
                    aclrtEvent event;
                    aclrtCreateEvent(&event);
                    g_group_ctxs[group_id].completion_events[slot_id] = event;
                }

                aclrtMalloc(&g_group_ctxs[group_id].d_flag_zero, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
                const uint32_t zero_val = 0U;
                aclrtMemcpy(g_group_ctxs[group_id].d_flag_zero, sizeof(uint32_t),
                            &zero_val, sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE);
            }

            aclrtMalloc(&g_devices[local_idx].dev_base_addr, total_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            aclrtMemcpy(g_devices[local_idx].dev_base_addr, total_bytes, host_buffer.data(), total_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        }
        g_initialized = true;
    }

    Stream GetGroupStream(int group_id)
    {
        int dev_idx = g_group_to_device[group_id];
        int stream_idx = g_group_to_stream[group_id];
        return g_devices[dev_idx].streams[stream_idx];
    }

    void UploadQuery(const float *query_vector,
                     const uint16_t *mmad_query_padded_fp16,
                     int group_id)
    {
        Stream stream = GetGroupStream(group_id);
        void *d_query = g_group_ctxs[group_id].d_query_ws;

        const size_t query_bytes = static_cast<size_t>(vector_dim) * kKernelResultCols * sizeof(uint16_t);
        if (mmad_query_padded_fp16 != nullptr)
        {
            aclrtMemcpyAsync(d_query, query_bytes, mmad_query_padded_fp16,
                             query_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
            return;
        }

        static thread_local std::vector<uint16_t> h_query_padded;
        h_query_padded.assign(static_cast<size_t>(vector_dim) * kKernelResultCols, 0);
        for (int i = 0; i < vector_dim; ++i)
        {
            h_query_padded[static_cast<size_t>(i) * kKernelResultCols] = Float32ToFp16(query_vector[i]);
        }
        aclrtMemcpyAsync(d_query, query_bytes, h_query_padded.data(),
                         query_bytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
    }

    volatile uint32_t* GetGroupFlag(int group_id, int slot_id) {
        return g_group_ctxs[group_id].h_flags[slot_id];
    }

    void ClearGroupFlag(int group_id, int slot_id) {
        *g_group_ctxs[group_id].h_flags[slot_id] = 0U;
    }

    void ResetGroupFlag(int group_id, int slot_id) {
        *g_group_ctxs[group_id].h_flags[slot_id] = 1U;
    }

    double EnqueueGroupCompletion(Stream stream, int group_id, int slot_id)
    {
        auto start = std::chrono::high_resolution_clock::now();
        aclrtEvent event = g_group_ctxs[group_id].completion_events[slot_id];
        aclError ret = (event == nullptr)
                           ? ACL_SUCCESS
                           : aclrtRecordEvent(event, stream);
        auto end = std::chrono::high_resolution_clock::now();
        if (ret != ACL_SUCCESS)
        {
            std::cerr << "[NPU] completion event submit failed for group=" << group_id
                      << " slot=" << slot_id << " code=" << ret << std::endl;
            return -1.0;
        }
        return std::chrono::duration<double, std::milli>(end - start).count();
    }

    bool IsGroupCompletionReady(int group_id, int slot_id)
    {
        aclrtEvent event = g_group_ctxs[group_id].completion_events[slot_id];
        if (event == nullptr)
        {
            return *g_group_ctxs[group_id].h_flags[slot_id] == 0U;
        }

        aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
        const aclError ret = aclrtQueryEventStatus(event, &status);
        if (ret != ACL_SUCCESS)
        {
            return *g_group_ctxs[group_id].h_flags[slot_id] == 0U;
        }

        if (status == ACL_EVENT_RECORDED_STATUS_COMPLETE)
        {
            *g_group_ctxs[group_id].h_flags[slot_id] = 0U;
            return true;
        }
        return false;
    }

    void SynchronizeStream(Stream s) { aclrtSynchronizeStream(s); }   //优化后这个函数不再被使用
    void Finalize()
    {
        if (!g_acl_initialized)
        {
            ResetRuntimeState();
            return;
        }

        for (int local_idx = 0; local_idx < static_cast<int>(g_devices.size()); ++local_idx)
        {
            auto &device = g_devices[local_idx];
            const int device_id = device.device_id;
            const bool valid_device = (device_id >= 0);
            bool device_ready = false;

            if (valid_device)
            {
                const auto ret = aclrtSetDevice(device_id);
                if (ret != ACL_SUCCESS)
                {
                    LogAclCleanupError("aclrtSetDevice", ret, device_id);
                }
                else
                {
                    device_ready = true;
                }
            }

            const int dev_group_count = GroupCountForDevice(local_idx);
            for (int j = 0; j < dev_group_count; ++j)
            {
                const int group_id = GroupIdForDeviceStream(local_idx, j);
                if (group_id >= static_cast<int>(g_group_ctxs.size()))
                {
                    break;
                }

                if (device_ready && j < static_cast<int>(device.streams.size()) && device.streams[j] != nullptr)
                {
                    const auto ret = aclrtSynchronizeStream(device.streams[j]);
                    if (ret != ACL_SUCCESS)
                    {
                        LogAclCleanupError("aclrtSynchronizeStream", ret, device_id, group_id);
                    }
                }
            }

            for (int j = 0; j < dev_group_count; ++j)
            {
                const int group_id = GroupIdForDeviceStream(local_idx, j);
                if (group_id >= static_cast<int>(g_group_ctxs.size()))
                {
                    break;
                }

                auto &group = g_group_ctxs[group_id];
                if (device_ready)
                {
                    FreeDeviceBuffer(group.d_query_ws, "aclrtFree(d_query_ws)", device_id, group_id);
                    FreeDeviceBuffer(group.d_result_ws, "aclrtFree(d_result_ws)", device_id, group_id);
                    FreeDeviceBuffer(group.d_task_ws, "aclrtFree(d_task_ws)", device_id, group_id);
                    FreeDeviceBuffer(group.d_flag_zero, "aclrtFree(d_flag_zero)", device_id, group_id);
                }
                for (int slot_id = 0; slot_id < kGroupFlagSlotCount; ++slot_id)
                {
                    FreeHostFlag(group.h_flags[slot_id], device_id, group_id);
                    if (group.completion_events[slot_id] != nullptr)
                    {
                        aclrtEvent event = group.completion_events[slot_id];
                        aclrtDestroyEvent(event);
                        group.completion_events[slot_id] = nullptr;
                    }
                }
            }

            if (device_ready)
            {
                FreeDeviceBuffer(device.dev_base_addr, "aclrtFree(dev_base_addr)", device_id);

                for (size_t stream_idx = 0; stream_idx < device.streams.size(); ++stream_idx)
                {
                    if (device.streams[stream_idx] == nullptr)
                    {
                        continue;
                    }

                    const auto ret = aclrtDestroyStream(device.streams[stream_idx]);
                    if (ret != ACL_SUCCESS)
                    {
                        LogAclCleanupError("aclrtDestroyStream",
                                           ret,
                                           device_id,
                                           GroupIdForDeviceStream(local_idx, static_cast<int>(stream_idx)));
                    }
                    device.streams[stream_idx] = nullptr;
                }

                const auto ret = aclrtResetDevice(device_id);
                if (ret != ACL_SUCCESS)
                {
                    LogAclCleanupError("aclrtResetDevice", ret, device_id);
                }
            }

            device.streams.clear();
            device.dev_base_addr = nullptr;
        }

        const auto ret = aclFinalize();
        if (ret != ACL_SUCCESS)
        {
            LogAclCleanupError("aclFinalize", ret, -1);
        }

        g_acl_initialized = false;
        ResetRuntimeState();
    }

    // -----------------------------------------------------------
    // 优化版 LaunchBatchKernel (Async Chain, no query upload)
    // 前置条件：已调用 UploadQuery 上传 query 向量
    // -----------------------------------------------------------
    void LaunchBatchKernel(Stream stream,
                           const std::vector<uint32_t> &bucket_ids,
                           float *host_output_buffer,
                           int group_id)
    {
        int local_idx = g_group_to_device[group_id];

        void *d_query = g_group_ctxs[group_id].d_query_ws;
        void *d_task = g_group_ctxs[group_id].d_task_ws;
        void *dev_base_doc = g_devices[local_idx].dev_base_addr;
        void *dev_base_result = g_group_ctxs[group_id].d_result_ws;

        // 1. Prepare Task Data (compact offset_C layout)
        static thread_local std::vector<BatchTaskData> h_tasks;
        h_tasks.clear();
        h_tasks.reserve(GetMaxKernelTasksPerBatch());
        const uint32_t kernel_row_tile = GetKernelRowTile();

        size_t current_res_offset = 0;

        for (uint32_t bid : bucket_ids)
        {
            auto &meta = g_bucket_metas[bid];
            if (meta.doc_num == 0)
            {
                continue;
            }

            for (uint32_t row_begin = 0; row_begin < meta.doc_num; row_begin += kernel_row_tile)
            {
                const uint32_t tile_rows = std::min<uint32_t>(kernel_row_tile, meta.doc_num - row_begin);
                BatchTaskData task;
                task.offset_A = static_cast<uint64_t>(meta.byte_offset) +
                                static_cast<size_t>(row_begin) * static_cast<size_t>(vector_dim) * sizeof(uint16_t);
                // Compact offset_C: kernel outputs only column 0 (stride-1)
                task.offset_C = static_cast<uint64_t>(current_res_offset) +
                                static_cast<size_t>(row_begin) * sizeof(float);
                task.m = tile_rows;
                h_tasks.push_back(task);
            }
            current_res_offset +=
                static_cast<size_t>(meta.doc_num) * sizeof(float);
        }

        if (h_tasks.empty())
            return;

        const size_t max_tasks = GetMaxKernelTasksPerBatch();
        if (h_tasks.size() > max_tasks)
        {
            std::cerr << "[NPU] Too many kernel tasks in one batch: "
                      << h_tasks.size() << " > " << max_tasks << std::endl;
            return;
        }

        // 2. Copy Task Data to Device
        size_t task_data_size = h_tasks.size() * sizeof(BatchTaskData);
        aclrtMemcpyAsync(d_task, task_data_size, h_tasks.data(), task_data_size, ACL_MEMCPY_HOST_TO_DEVICE, stream);

        // 3. Single Launch
        ACLRT_LAUNCH_KERNEL(kernel_vector_mmad)(
            (uint32_t)h_tasks.size(), stream,
            d_query,
            d_task,
            dev_base_doc,
            dev_base_result,
            static_cast<uint32_t>(vector_dim));

        // 4. Single D2H copy (compact layout: device and host use same offsets)
        if (current_res_offset > 0)
        {
            aclrtMemcpyAsync(host_output_buffer, current_res_offset,
                             dev_base_result, current_res_offset,
                             ACL_MEMCPY_DEVICE_TO_HOST, stream);
        }
    }

    void DebugVerifyBatchResults(const std::vector<uint32_t> &bucket_ids,
                                 const float *query_vector,
                                 const float *host_output_buffer,
                                 int group_id)
    {
        if (npu_debug_verify == 0 || !query_vector || !host_output_buffer || !g_host_vectors)
        {
            return;
        }

        const int report_limit = npu_debug_verify_max_report;
        const int dim = vector_dim;
        size_t checked_docs = 0;
        size_t mismatch_count = 0;
        float max_abs_err = 0.0f;
        float max_rel_err = 0.0f;
        DebugMismatch worst_case;
        std::vector<DebugMismatch> mismatch_reports;
        if (report_limit > 0)
        {
            mismatch_reports.reserve(report_limit);
        }

        for (size_t bucket_idx = 0; bucket_idx < bucket_ids.size(); ++bucket_idx)
        {
            const uint32_t bucket_id = bucket_ids[bucket_idx];
            if (bucket_id >= g_bucket_metas.size())
            {
                continue;
            }

            const BucketMeta &meta = g_bucket_metas[bucket_id];
            const uint32_t *doc_ids = g_bucket_doc_ids.data() + meta.doc_id_offset;
            const float *npu_scores = host_output_buffer;

            for (uint32_t local_doc_idx = 0; local_doc_idx < meta.doc_num; ++local_doc_idx)
            {
                const uint32_t global_doc_id = doc_ids[local_doc_idx];
                const float *doc_vec = g_host_vectors + static_cast<size_t>(global_doc_id) * dim;
                const float cpu_score = ComputeCpuDotProduct(doc_vec, query_vector, dim);
                const float npu_score = npu_scores[local_doc_idx];
                const bool invalid = !std::isfinite(cpu_score) || !std::isfinite(npu_score);
                const float abs_err = invalid ? std::numeric_limits<float>::infinity()
                                              : std::fabs(cpu_score - npu_score);
                const float ref_scale = invalid ? 0.0f
                                                : std::max(std::fabs(cpu_score), std::fabs(npu_score));
                const float allowed_err = npu_debug_verify_abs_tol + npu_debug_verify_rel_tol * ref_scale;
                const float rel_err = invalid ? std::numeric_limits<float>::infinity()
                                              : abs_err / std::max(1e-6f, std::fabs(cpu_score));

                ++checked_docs;
                if (checked_docs == 1 || abs_err > max_abs_err)
                {
                    max_abs_err = abs_err;
                    max_rel_err = rel_err;
                    worst_case = {bucket_id, local_doc_idx, global_doc_id, cpu_score, npu_score, abs_err, allowed_err};
                }
                else if (rel_err > max_rel_err)
                {
                    max_rel_err = rel_err;
                }

                if (invalid || abs_err > allowed_err)
                {
                    ++mismatch_count;
                    if (report_limit > 0 && static_cast<int>(mismatch_reports.size()) < report_limit)
                    {
                        mismatch_reports.push_back({bucket_id, local_doc_idx, global_doc_id,
                                                    cpu_score, npu_score, abs_err, allowed_err});
                    }
                }
            }
            host_output_buffer += meta.doc_num;
        }

        const uint64_t batch_id = g_debug_batch_counter.fetch_add(1, std::memory_order_relaxed) + 1;
        std::lock_guard<std::mutex> lock(g_debug_log_mutex);
        std::cout << std::fixed
                  << "[NPU][Debug] group=" << group_id
                  << " batch=" << batch_id
                  << " buckets=" << bucket_ids.size()
                  << " docs=" << checked_docs
                  << " mismatches=" << mismatch_count
                  << " max_abs_err=" << max_abs_err
                  << " max_rel_err=" << max_rel_err
                  << std::endl;

        if (checked_docs > 0)
        {
            std::cout << "[NPU][Debug] worst_case"
                      << " bucket=" << worst_case.bucket_id
                      << " local_doc=" << worst_case.local_doc_idx
                      << " global_doc=" << worst_case.global_doc_id
                      << " cpu=" << worst_case.cpu_score
                      << " npu=" << worst_case.npu_score
                      << " abs_err=" << worst_case.abs_err
                      << " allowed=" << worst_case.allowed_err
                      << std::endl;
        }

        for (const auto &item : mismatch_reports)
        {
            std::cout << "[NPU][Debug][Mismatch]"
                      << " bucket=" << item.bucket_id
                      << " local_doc=" << item.local_doc_idx
                      << " global_doc=" << item.global_doc_id
                      << " cpu=" << item.cpu_score
                      << " npu=" << item.npu_score
                      << " abs_err=" << item.abs_err
                      << " allowed=" << item.allowed_err
                      << std::endl;
        }
    }
}
