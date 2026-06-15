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
#include "acl/acl_op.h"
#include "acl/acl_op_compiler.h"
#include "utils/DataReader.h"
#include "NPU/BatchTask.h" //避免acl头文件污染

// 引用内核启动头文件
#include "aclrtlaunch_kernel_vector_mmad.h"

// MaskFilterTask must match the definition in kernel_mask_filter_op.h
// but is redeclared here to avoid including kernel_operator.h in host code.
struct MaskFilterTask {
    uint64_t scores_offset;
    uint64_t mask_offset;
    uint64_t output_offset;
    uint64_t index_offset;
    uint32_t doc_num;
    uint32_t mask_words;
};

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
        void *d_l0_cache = nullptr;   // L0 分数 HBM 缓冲区（跨轮次持久化，按需选择性 D2H）
        size_t l0_cache_write_offset = 0; // 当前写入偏移（跨 LaunchL0BatchKernel 调用累加）

        // Mask filter kernel buffers
        void *d_mf_mask = nullptr;    // H2D mask bitmap destination
        void *d_mf_output = nullptr;  // compact scores output (float per match)
        void *d_mf_index = nullptr;   // compact local-doc-idx output (uint32 per match)
        void *d_mf_count = nullptr;   // per-block match count (uint32 per block)
        void *d_mf_task = nullptr;    // MaskFilterTask array

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
    static size_t g_total_stored_docs = 0;
    static int g_total_bucket_num_level_2 = 0;

    // Precomputed per-bucket tile data (offset_A and m fixed at Init; offset_C patched at launch)
    struct PrecomputedBucket {
        size_t task_offset;     // start index in g_precomputed_tasks
        uint32_t tile_count;    // number of tiles for this bucket
    };
    static std::vector<BatchTaskData> g_precomputed_tasks;
    static std::vector<PrecomputedBucket> g_precomputed_buckets;

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
            return static_cast<size_t>(g_total_bucket_num_level_2) * max_tiles_per_bucket;
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
        g_total_bucket_num_level_2 = total_bucket_num_level_2;

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
        g_total_stored_docs = total_stored_docs;

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

        // Precompute BatchTaskData templates for all buckets (offset_A and m are fixed)
        {
            const uint32_t row_tile = GetKernelRowTile();
            g_precomputed_buckets.resize(total_bucket_num_level_2);
            size_t total_tiles = 0;
            for (int bid = 0; bid < total_bucket_num_level_2; ++bid)
            {
                g_precomputed_buckets[bid].task_offset = total_tiles;
                uint32_t doc_num = g_bucket_metas[bid].doc_num;
                uint32_t tiles = (doc_num + row_tile - 1) / row_tile;
                g_precomputed_buckets[bid].tile_count = tiles;
                total_tiles += tiles;
            }
            g_precomputed_tasks.resize(total_tiles);
            size_t pc = 0;
            for (int bid = 0; bid < total_bucket_num_level_2; ++bid)
            {
                auto &meta = g_bucket_metas[bid];
                for (uint32_t row_begin = 0; row_begin < meta.doc_num; row_begin += row_tile)
                {
                    uint32_t tile_rows = std::min(row_tile, meta.doc_num - row_begin);
                    g_precomputed_tasks[pc].offset_A =
                        meta.byte_offset + static_cast<size_t>(row_begin) * static_cast<size_t>(vector_dim) * sizeof(uint16_t);
                    g_precomputed_tasks[pc].offset_C = 0;
                    g_precomputed_tasks[pc].m = tile_rows;
                    g_precomputed_tasks[pc].reserved = 0;
                    ++pc;
                }
            }
            std::cout << "[NPU] PrecomputedTasks=" << total_tiles << std::endl;
        }

        size_t total_bytes = total_elements * sizeof(uint16_t);
        std::cout << "[NPU] StoredDocRefs=" << total_stored_docs
                  << ", HBMVectorBytesPerDevice=" << total_bytes
                  << std::endl;

        size_t ws_query_size =
            static_cast<size_t>(vector_dim) * kKernelResultCols * sizeof(uint16_t); // 每个query向量都要padding到[k,16]

        // 结果按紧凑 [doc_num, 1] 布局写回（Fixpipe nSize=1 仅输出 column 0）。
        // 使用 total_stored_docs 支持单次 merged launch 处理整轮所有文档
        size_t ws_result_size = total_stored_docs * sizeof(float);
        // L0 cache needs extra padding: each bucket's area is rounded up to 32B
        size_t ws_l0_cache_size = ws_result_size + static_cast<size_t>(g_total_bucket_num_level_2) * 32;

        // Task Workspace: 一个 bucket 可能被拆成多个 row-tile task。
        size_t max_total_tasks = (total_stored_docs + GetKernelRowTile() - 1) / GetKernelRowTile();
        size_t ws_task_size = max_total_tasks * sizeof(BatchTaskData);

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
                aclrtMalloc(&g_group_ctxs[group_id].d_l0_cache, ws_l0_cache_size, ACL_MEM_MALLOC_HUGE_FIRST);

                // Mask filter kernel buffers
                {
                    size_t mf_max_mask_bytes = ((total_stored_docs + 63) / 64) * sizeof(uint64_t);
                    size_t mf_max_output_bytes = total_stored_docs * sizeof(float);
                    size_t mf_max_index_bytes = total_stored_docs * sizeof(uint32_t);
                    size_t mf_max_count_bytes = static_cast<size_t>(g_total_bucket_num_level_2) * sizeof(uint32_t);
                    size_t mf_max_task_bytes = sizeof(uint64_t) + static_cast<size_t>(g_total_bucket_num_level_2) * sizeof(MaskFilterTask);

                    aclrtMalloc(&g_group_ctxs[group_id].d_mf_mask, mf_max_mask_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
                    aclrtMalloc(&g_group_ctxs[group_id].d_mf_output, mf_max_output_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
                    aclrtMalloc(&g_group_ctxs[group_id].d_mf_index, mf_max_index_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
                    aclrtMalloc(&g_group_ctxs[group_id].d_mf_count, mf_max_count_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
                    aclrtMalloc(&g_group_ctxs[group_id].d_mf_task, mf_max_task_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
                }

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

        InitMaskFilterOp();
        g_initialized = true;
    }

    Stream GetGroupStream(int group_id)
    {
        int dev_idx = g_group_to_device[group_id];
        int stream_idx = g_group_to_stream[group_id];
        return g_devices[dev_idx].streams[stream_idx];
    }

    void ResetL0CacheOffset(int group_id)
    {
        g_group_ctxs[group_id].l0_cache_write_offset = 0;
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
                    FreeDeviceBuffer(group.d_l0_cache, "aclrtFree(d_l0_cache)", device_id, group_id);
                    FreeDeviceBuffer(group.d_flag_zero, "aclrtFree(d_flag_zero)", device_id, group_id);
                    FreeDeviceBuffer(group.d_mf_mask, "aclrtFree(d_mf_mask)", device_id, group_id);
                    FreeDeviceBuffer(group.d_mf_output, "aclrtFree(d_mf_output)", device_id, group_id);
                    FreeDeviceBuffer(group.d_mf_index, "aclrtFree(d_mf_index)", device_id, group_id);
                    FreeDeviceBuffer(group.d_mf_count, "aclrtFree(d_mf_count)", device_id, group_id);
                    FreeDeviceBuffer(group.d_mf_task, "aclrtFree(d_mf_task)", device_id, group_id);
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
                           int group_id,
                           const std::vector<size_t> *bucket_result_offsets_bytes)
    {
        int local_idx = g_group_to_device[group_id];

        void *d_query = g_group_ctxs[group_id].d_query_ws;
        void *d_task = g_group_ctxs[group_id].d_task_ws;
        void *dev_base_doc = g_devices[local_idx].dev_base_addr;
        void *dev_base_result = g_group_ctxs[group_id].d_result_ws;

        // 1. Prepare Task Data using precomputed templates
        static thread_local std::vector<BatchTaskData> h_tasks;
        static thread_local std::vector<size_t> bucket_res_offsets;
        bucket_res_offsets.clear();
        bucket_res_offsets.reserve(bucket_ids.size());

        const uint32_t kernel_row_tile = GetKernelRowTile();
        size_t total_tiles = 0;
        size_t current_res_offset = 0;

        // Pass 1: compute per-bucket result offsets and total tile count
        for (size_t i = 0; i < bucket_ids.size(); ++i)
        {
            uint32_t bid = bucket_ids[i];
            size_t off = bucket_result_offsets_bytes
                             ? (*bucket_result_offsets_bytes)[i]
                             : current_res_offset;
            bucket_res_offsets.push_back(off);
            total_tiles += g_precomputed_buckets[bid].tile_count;
            if (!bucket_result_offsets_bytes)
                current_res_offset += static_cast<size_t>(g_bucket_metas[bid].doc_num) * sizeof(float);
        }

        if (total_tiles == 0)
            return;

        h_tasks.resize(total_tiles);

        // Pass 2: memcpy precomputed tasks + patch offset_C
        size_t dst_base = 0;
        for (size_t i = 0; i < bucket_ids.size(); ++i)
        {
            uint32_t bid = bucket_ids[i];
            const auto &pb = g_precomputed_buckets[bid];
            if (pb.tile_count == 0)
                continue;
            size_t src_base = pb.task_offset;
            size_t res_base = bucket_res_offsets[i];
            memcpy(&h_tasks[dst_base], &g_precomputed_tasks[src_base],
                   static_cast<size_t>(pb.tile_count) * sizeof(BatchTaskData));
            for (uint32_t t = 0; t < pb.tile_count; ++t)
            {
                h_tasks[dst_base + t].offset_C =
                    static_cast<uint64_t>(res_base) +
                    static_cast<size_t>(t) * kernel_row_tile * sizeof(float);
            }
            dst_base += pb.tile_count;
        }

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
        if (host_output_buffer && !bucket_result_offsets_bytes && current_res_offset > 0)
        {
            aclrtMemcpyAsync(host_output_buffer, current_res_offset,
                             dev_base_result, current_res_offset,
                             ACL_MEMCPY_DEVICE_TO_HOST, stream);
        }
        else if (host_output_buffer && bucket_result_offsets_bytes)
        {
            // With custom offsets, compute the total span for D2H
            size_t total_span = 0;
            for (size_t i = 0; i < bucket_ids.size(); ++i)
                total_span = std::max(total_span,
                    (*bucket_result_offsets_bytes)[i] +
                    static_cast<size_t>(g_bucket_metas[bucket_ids[i]].doc_num) * sizeof(float));
            if (total_span > 0)
                aclrtMemcpyAsync(host_output_buffer, total_span,
                                 dev_base_result, total_span,
                                 ACL_MEMCPY_DEVICE_TO_HOST, stream);
        }
    }

    void LaunchL0BatchKernel(Stream stream,
                              const std::vector<uint32_t> &bucket_ids,
                              int group_id,
                              std::vector<L2ScoreLocation> &l2_score_map)
    {
        int local_idx = g_group_to_device[group_id];

        void *d_query = g_group_ctxs[group_id].d_query_ws;
        void *d_task = g_group_ctxs[group_id].d_task_ws;
        void *dev_base_doc = g_devices[local_idx].dev_base_addr;
        void *dev_l0_cache = g_group_ctxs[group_id].d_l0_cache;

        // 1. Prepare Task Data using precomputed templates (compact offset_C into dev_l0_cache)
        static thread_local std::vector<BatchTaskData> h_tasks;
        static thread_local std::vector<size_t> l0_bucket_res_offsets;
        l0_bucket_res_offsets.clear();
        l0_bucket_res_offsets.reserve(bucket_ids.size());

        const uint32_t kernel_row_tile = GetKernelRowTile();

        l2_score_map.resize(g_total_bucket_num_level_2);
        size_t total_tiles = 0;
        size_t current_res_offset = g_group_ctxs[group_id].l0_cache_write_offset;

        // Pass 1: compute per-bucket result offsets + l2_score_map + total tile count
        for (uint32_t bid : bucket_ids)
        {
            l0_bucket_res_offsets.push_back(current_res_offset);
            uint32_t doc_num = g_bucket_metas[bid].doc_num;
            l2_score_map[bid].offset_bytes = current_res_offset;
            l2_score_map[bid].doc_count = doc_num;
            total_tiles += g_precomputed_buckets[bid].tile_count;
            current_res_offset += static_cast<size_t>(doc_num) * sizeof(float);
            // Pad to 32B boundary so next bucket's offset is aligned for gather DataCopy
            current_res_offset = (current_res_offset + 31) & ~(size_t)31;
        }

        if (total_tiles == 0)
            return;

        h_tasks.resize(total_tiles);

        // Pass 2: memcpy precomputed tasks + patch offset_C
        size_t dst_base = 0;
        for (size_t i = 0; i < bucket_ids.size(); ++i)
        {
            uint32_t bid = bucket_ids[i];
            const auto &pb = g_precomputed_buckets[bid];
            if (pb.tile_count == 0)
                continue;
            size_t res_base = l0_bucket_res_offsets[i];
            memcpy(&h_tasks[dst_base], &g_precomputed_tasks[pb.task_offset],
                   static_cast<size_t>(pb.tile_count) * sizeof(BatchTaskData));
            for (uint32_t t = 0; t < pb.tile_count; ++t)
            {
                h_tasks[dst_base + t].offset_C =
                    static_cast<uint64_t>(res_base) +
                    static_cast<size_t>(t) * kernel_row_tile * sizeof(float);
            }
            dst_base += pb.tile_count;
        }

        const size_t max_tasks = GetMaxKernelTasksPerBatch();
        if (h_tasks.size() > max_tasks)
        {
            std::cerr << "[NPU] Too many kernel tasks in L0 batch: "
                      << h_tasks.size() << " > " << max_tasks << std::endl;
            return;
        }

        // 2. Copy Task Data to Device
        size_t task_data_size = h_tasks.size() * sizeof(BatchTaskData);
        aclrtMemcpyAsync(d_task, task_data_size, h_tasks.data(), task_data_size, ACL_MEMCPY_HOST_TO_DEVICE, stream);

        // 3. Launch kernel — writes to dev_l0_cache instead of d_result_ws
        ACLRT_LAUNCH_KERNEL(kernel_vector_mmad)(
            (uint32_t)h_tasks.size(), stream,
            d_query,
            d_task,
            dev_base_doc,
            dev_l0_cache,
            static_cast<uint32_t>(vector_dim));

        // Scores stay in dev_l0_cache for subsequent GatherL0Scores call
        // Persist the write offset for the next LaunchL0BatchKernel call in this query
        g_group_ctxs[group_id].l0_cache_write_offset = current_res_offset;
    }

    size_t GatherL0Scores(Stream stream,
                           int group_id,
                           const std::vector<uint32_t> &l2_ids,
                           const std::vector<L2ScoreLocation> &l2_score_map,
                           float *host_output_buffer,
                           GatherTimingMs *timing,
                           const std::vector<size_t> *dst_offsets_override)
    {
        void *dev_l0_cache = g_group_ctxs[group_id].d_l0_cache;
        void *dev_result = g_group_ctxs[group_id].d_result_ws;

        using Clock = std::chrono::high_resolution_clock;

        // 1. Build compact offset map for the selected L2 buckets
        auto t_offset_start = Clock::now();
        static thread_local std::vector<size_t> src_offsets;
        src_offsets.clear();
        src_offsets.reserve(l2_ids.size());

        static thread_local std::vector<size_t> dst_offsets;
        dst_offsets.clear();
        dst_offsets.reserve(l2_ids.size());

        static thread_local std::vector<uint32_t> counts;
        counts.clear();
        counts.reserve(l2_ids.size());

        size_t compact_dst_floats = 0;

        for (size_t idx = 0; idx < l2_ids.size(); ++idx)
        {
            uint32_t l2_id = l2_ids[idx];
            const auto &loc = l2_score_map[l2_id];
            if (loc.doc_count == 0)
                continue;

            src_offsets.push_back(loc.offset_bytes);
            if (dst_offsets_override)
                dst_offsets.push_back((*dst_offsets_override)[idx]);
            else
                dst_offsets.push_back(compact_dst_floats * sizeof(float));
            counts.push_back(loc.doc_count);
            compact_dst_floats += loc.doc_count;
        }

        if (counts.empty())
        {
            if (timing) *timing = GatherTimingMs{};
            return 0;
        }

        size_t compact_bytes = compact_dst_floats * sizeof(float);
        auto t_offset_end = Clock::now();

        // 2. D2D gather: scatter-read from dev_l0_cache into dev_result
        auto t_d2d_start = Clock::now();
        for (size_t i = 0; i < counts.size(); ++i)
        {
            size_t copy_bytes = counts[i] * sizeof(float);
            aclrtMemcpyAsync(
                reinterpret_cast<char *>(dev_result) + dst_offsets[i],
                copy_bytes,
                reinterpret_cast<char *>(dev_l0_cache) + src_offsets[i],
                copy_bytes,
                ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        }
        auto t_d2d_end = Clock::now();

        // 3. D2H: skip when host_output_buffer is nullptr (mask filter reads from device)
        auto t_d2h_start = Clock::now();
        if (host_output_buffer)
        {
            size_t d2h_span = dst_offsets_override
                ? dst_offsets.back() + counts.back() * sizeof(float)
                : compact_bytes;
            aclrtMemcpyAsync(host_output_buffer, d2h_span,
                             dev_result, d2h_span,
                             ACL_MEMCPY_DEVICE_TO_HOST, stream);
        }
        auto t_d2h_end = Clock::now();

        // 4. Synchronize (only when D2H was done — mask filter path skips sync here)
        auto t_sync_start = Clock::now();
        if (host_output_buffer)
            aclrtSynchronizeStream(stream);
        auto t_sync_end = Clock::now();

        if (timing)
        {
            timing->host_offset_build_ms =
                std::chrono::duration<double, std::milli>(t_offset_end - t_offset_start).count();
            timing->d2d_submit_ms =
                std::chrono::duration<double, std::milli>(t_d2d_end - t_d2d_start).count();
            timing->d2h_submit_ms =
                std::chrono::duration<double, std::milli>(t_d2h_end - t_d2h_start).count();
            timing->sync_wait_ms =
                std::chrono::duration<double, std::milli>(t_sync_end - t_sync_start).count();
        }

        return compact_bytes;
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

    void *GetL0CacheBuffer(int group_id)
    {
        if (group_id < 0 || group_id >= static_cast<int>(g_group_ctxs.size()))
            return nullptr;
        return g_group_ctxs[group_id].d_l0_cache;
    }

    // ----------------------------------------------------------------
    // Mask Filter AI CPU Op: one-time compile.
    // Call after aclInit + aclrtSetDevice.
    // ----------------------------------------------------------------

    static bool s_mask_filter_compiled = false;

    void InitMaskFilterOp()
    {
        if (s_mask_filter_compiled) return;

        int64_t dims1024[1] = {1024};
        int64_t dims16[1] = {16};
        int64_t dims1[1] = {1};

        aclTensorDesc *input_desc[6];
        input_desc[0] = aclCreateTensorDesc(ACL_FLOAT, 1, dims1024, ACL_FORMAT_ND);
        input_desc[1] = aclCreateTensorDesc(ACL_UINT64, 1, dims16, ACL_FORMAT_ND);
        input_desc[2] = aclCreateTensorDesc(ACL_FLOAT, 1, dims1024, ACL_FORMAT_ND);
        input_desc[3] = aclCreateTensorDesc(ACL_INT32, 1, dims1024, ACL_FORMAT_ND);
        input_desc[4] = aclCreateTensorDesc(ACL_INT32, 1, dims1, ACL_FORMAT_ND);
        input_desc[5] = aclCreateTensorDesc(ACL_UINT64, 1, dims1024, ACL_FORMAT_ND);

        aclTensorDesc *output_desc[1];
        output_desc[0] = aclCreateTensorDesc(ACL_FLOAT, 1, dims1, ACL_FORMAT_ND);

        aclopAttr *attr = aclopCreateAttr();

        aclError ret = aclopCompile("MaskFilter", 6, input_desc, 1, output_desc,
                                    attr, ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL);

        if (ret == ACL_SUCCESS) {
            s_mask_filter_compiled = true;
            printf("[MaskFilter] AI CPU op compiled successfully.\n");
        } else {
            fprintf(stderr, "[MaskFilter] FATAL: aclopCompile failed (%d). AI CPU is required.\n", ret);
            exit(1);
        }

        for (int i = 0; i < 6; i++) aclDestroyTensorDesc(input_desc[i]);
        aclDestroyTensorDesc(output_desc[0]);
        aclopDestroyAttr(attr);
    }

    // ----------------------------------------------------------------
    // Mask Filter: AI CPU only (exit on failure).
    // ----------------------------------------------------------------

    void LaunchMaskFilter(Stream stream,
                           int group_id,
                           const std::vector<uint32_t> &bucket_ids,
                           const std::vector<size_t> &score_offsets_bytes,
                           const uint64_t *host_mask_storage,
                           const std::vector<BucketMaskInfo> &mask_infos,
                           size_t mask_stride_u64,
                           float *host_score_output,
                           uint32_t *host_index_output,
                           uint32_t *host_count_output,
                           MaskFilterTimingMs *timing)
    {
        if (bucket_ids.empty())
            return;

        auto &ctx = g_group_ctxs[group_id];
        const uint32_t N = static_cast<uint32_t>(bucket_ids.size());

        auto t0 = std::chrono::high_resolution_clock::now();
        aclrtSynchronizeStream(stream);
        auto t1 = std::chrono::high_resolution_clock::now();

        struct MFTask {
            uint64_t scores_offset;
            uint64_t mask_offset;
            uint64_t output_offset;
            uint64_t index_offset;
            uint32_t doc_num;
            uint32_t mask_words;
        };

        size_t task_bytes = sizeof(uint64_t) + N * sizeof(MFTask);
        static thread_local std::vector<uint8_t> task_buf;
        task_buf.resize(task_bytes, 0);

        uint64_t *hdr = reinterpret_cast<uint64_t *>(task_buf.data());
        hdr[0] = N;
        MFTask *tasks = reinterpret_cast<MFTask *>(task_buf.data() + sizeof(uint64_t));

        size_t total_output_floats = 0;
        size_t total_mask_u64 = 0;
        for (uint32_t i = 0; i < N; ++i) {
            int dn = g_bucket_metas[bucket_ids[i]].doc_num;
            total_output_floats += dn;
            total_mask_u64 += mask_infos[i].mask_words;
        }

        void *d_mf_scores = ctx.d_result_ws;

        size_t mask_total_bytes = total_mask_u64 * sizeof(uint64_t);
        size_t output_bytes = total_output_floats * sizeof(float);
        size_t index_bytes = total_output_floats * sizeof(uint32_t);
        size_t count_bytes = N * sizeof(uint32_t);

        static thread_local void *td_mf_mask = nullptr;
        static thread_local size_t td_mf_mask_cap = 0;
        static thread_local void *td_mf_output = nullptr;
        static thread_local size_t td_mf_output_cap = 0;
        static thread_local void *td_mf_index = nullptr;
        static thread_local size_t td_mf_index_cap = 0;
        static thread_local void *td_mf_count = nullptr;
        static thread_local size_t td_mf_count_cap = 0;
        static thread_local void *td_mf_task = nullptr;
        static thread_local size_t td_mf_task_cap = 0;

        if (mask_total_bytes > td_mf_mask_cap) {
            if (td_mf_mask) aclrtFree(td_mf_mask);
            aclrtMalloc(&td_mf_mask, mask_total_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            td_mf_mask_cap = mask_total_bytes;
        }
        if (output_bytes > td_mf_output_cap) {
            if (td_mf_output) aclrtFree(td_mf_output);
            aclrtMalloc(&td_mf_output, output_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            td_mf_output_cap = output_bytes;
        }
        if (index_bytes > td_mf_index_cap) {
            if (td_mf_index) aclrtFree(td_mf_index);
            aclrtMalloc(&td_mf_index, index_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            td_mf_index_cap = index_bytes;
        }
        if (count_bytes > td_mf_count_cap) {
            if (td_mf_count) aclrtFree(td_mf_count);
            aclrtMalloc(&td_mf_count, count_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            td_mf_count_cap = count_bytes;
        }
        if (task_bytes > td_mf_task_cap) {
            if (td_mf_task) aclrtFree(td_mf_task);
            aclrtMalloc(&td_mf_task, task_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
            td_mf_task_cap = task_bytes;
        }

        static thread_local std::vector<uint64_t> host_mask_concat;
        host_mask_concat.resize(total_mask_u64);
        size_t mask_offset_acc = 0;
        size_t output_offset_acc = 0;

        for (uint32_t i = 0; i < N; ++i) {
            int doc_num = g_bucket_metas[bucket_ids[i]].doc_num;
            size_t mw = mask_infos[i].mask_words;

            memcpy(host_mask_concat.data() + mask_offset_acc,
                   host_mask_storage + mask_infos[i].mask_offset_u64,
                   mw * sizeof(uint64_t));

            tasks[i].scores_offset = score_offsets_bytes[i];
            tasks[i].mask_offset = mask_offset_acc * sizeof(uint64_t);
            tasks[i].output_offset = output_offset_acc * sizeof(float);
            tasks[i].index_offset = output_offset_acc * sizeof(uint32_t);
            tasks[i].doc_num = static_cast<uint32_t>(doc_num);
            tasks[i].mask_words = static_cast<uint32_t>(mw);

            mask_offset_acc += mw;
            output_offset_acc += doc_num;
        }

        aclrtMemcpy(td_mf_mask, mask_total_bytes, host_mask_concat.data(),
                     mask_total_bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        aclrtMemcpy(td_mf_task, task_bytes, task_buf.data(),
                     task_bytes, ACL_MEMCPY_HOST_TO_DEVICE);

        auto t2 = std::chrono::high_resolution_clock::now();

        int64_t dims_scores[1] = {static_cast<int64_t>(total_output_floats)};
        int64_t dims_mask[1] = {static_cast<int64_t>(total_mask_u64)};
        int64_t dims_count[1] = {static_cast<int64_t>(N)};
        int64_t dims_task[1] = {static_cast<int64_t>(task_bytes / sizeof(uint64_t))};
        int64_t dims1[1] = {1};

        aclTensorDesc *input_desc[6];
        input_desc[0] = aclCreateTensorDesc(ACL_FLOAT, 1, dims_scores, ACL_FORMAT_ND);
        input_desc[1] = aclCreateTensorDesc(ACL_UINT64, 1, dims_mask, ACL_FORMAT_ND);
        input_desc[2] = aclCreateTensorDesc(ACL_FLOAT, 1, dims_scores, ACL_FORMAT_ND);
        input_desc[3] = aclCreateTensorDesc(ACL_INT32, 1, dims_scores, ACL_FORMAT_ND);
        input_desc[4] = aclCreateTensorDesc(ACL_INT32, 1, dims_count, ACL_FORMAT_ND);
        input_desc[5] = aclCreateTensorDesc(ACL_UINT64, 1, dims_task, ACL_FORMAT_ND);

        aclTensorDesc *output_desc[1];
        output_desc[0] = aclCreateTensorDesc(ACL_FLOAT, 1, dims1, ACL_FORMAT_ND);

        aclDataBuffer *input_buf[6];
        input_buf[0] = aclCreateDataBuffer(d_mf_scores, dims_scores[0] * sizeof(float));
        input_buf[1] = aclCreateDataBuffer(td_mf_mask, mask_total_bytes);
        input_buf[2] = aclCreateDataBuffer(td_mf_output, output_bytes);
        input_buf[3] = aclCreateDataBuffer(td_mf_index, index_bytes);
        input_buf[4] = aclCreateDataBuffer(td_mf_count, count_bytes);
        input_buf[5] = aclCreateDataBuffer(td_mf_task, task_bytes);

        aclDataBuffer *output_buf[1];
        output_buf[0] = aclCreateDataBuffer(td_mf_output, sizeof(float));

        aclopAttr *attr = aclopCreateAttr();

        aclError ret = aclopExecuteV2("MaskFilter", 6, input_desc, input_buf,
                                      1, output_desc, output_buf, attr, stream);

        if (ret != ACL_SUCCESS) {
            for (int i = 0; i < 6; i++) { aclDestroyDataBuffer(input_buf[i]); aclDestroyTensorDesc(input_desc[i]); }
            aclDestroyDataBuffer(output_buf[0]); aclDestroyTensorDesc(output_desc[0]);
            aclopDestroyAttr(attr);
            fprintf(stderr, "[MaskFilter] FATAL: aclopExecuteV2 failed (%d). AI CPU is required.\n", ret);
            exit(1);
        }

        aclrtSynchronizeStream(stream);

        auto t3 = std::chrono::high_resolution_clock::now();

        static thread_local std::vector<float> h_out_scores;
        static thread_local std::vector<uint32_t> h_out_indices;
        static thread_local std::vector<uint32_t> h_out_counts;
        h_out_scores.resize(total_output_floats);
        h_out_indices.resize(total_output_floats);
        h_out_counts.resize(N);

        aclrtMemcpy(h_out_scores.data(), output_bytes, td_mf_output, output_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(h_out_indices.data(), index_bytes, td_mf_index, index_bytes, ACL_MEMCPY_DEVICE_TO_HOST);
        aclrtMemcpy(h_out_counts.data(), count_bytes, td_mf_count, count_bytes, ACL_MEMCPY_DEVICE_TO_HOST);

        auto t4 = std::chrono::high_resolution_clock::now();

        float *out_s = host_score_output;
        uint32_t *out_i = host_index_output;
        for (uint32_t i = 0; i < N; ++i) {
            int doc_num = g_bucket_metas[bucket_ids[i]].doc_num;
            uint32_t cnt = h_out_counts[i];
            size_t out_offset = 0;
            for (uint32_t j = 0; j < i; ++j)
                out_offset += g_bucket_metas[bucket_ids[j]].doc_num;

            memcpy(out_s, h_out_scores.data() + out_offset, cnt * sizeof(float));
            memcpy(out_i, h_out_indices.data() + out_offset, cnt * sizeof(uint32_t));
            host_count_output[i] = cnt;
            out_s += doc_num;
            out_i += doc_num;
        }

        for (int i = 0; i < 6; i++) { aclDestroyDataBuffer(input_buf[i]); aclDestroyTensorDesc(input_desc[i]); }
        aclDestroyDataBuffer(output_buf[0]); aclDestroyTensorDesc(output_desc[0]);
        aclopDestroyAttr(attr);

        if (timing) {
            // NOTE: h2d_ms measures stream sync wait (waiting for prior NPU scoring),
            // not actual H2D copy time (which uses async copies above).
            timing->h2d_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
            // kernel_exec_ms covers: H2D async completion + AI CPU kernel launch + sync
            timing->kernel_exec_ms += std::chrono::duration<double, std::milli>(t3 - t2).count();
            // d2h_ms covers: D2H memcpy (may include residual AI CPU completion wait)
            timing->d2h_ms += std::chrono::duration<double, std::milli>(t4 - t3).count();
        }
    }
}
