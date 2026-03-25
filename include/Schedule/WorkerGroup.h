#pragma once
#include <thread>
#include <atomic>
#include <vector>
#include <cmath>
#include <algorithm>
#include <queue>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstring>

// 引入 NEON 头文件
#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#endif

#include "utils/DataReader.h"
#include "utils/MemoryEventLogger.h"
#include "Query/Query.h"
#include "DataBaseCPU/DataBaseCPU.h"
#include "Schedule/Scheduler.h"
#include "NPU/npuAPI.h"
#include "Schedule/ThreadUtils.h"

template <typename Vec>
inline void WarmReserveVectorStorage(Vec &vec, size_t target_capacity)
{
    if (vec.capacity() < target_capacity)
    {
        vec.reserve(target_capacity);
    }
    if (target_capacity != 0 && vec.size() < target_capacity)
    {
        vec.resize(target_capacity);
    }
    vec.clear();
}

class WorkerGroup
{
public:
    WorkerGroup(int group_id, DataBaseCPU *db, Scheduler *sched)
        : group_id_(group_id), db_(db), sched_(sched)
    {
        npu_stream_ = npuAPI::GetGroupStream(group_id_);
        
        // --- 内存预分配优化 (Pinned Memory for Batch Scores) ---
        // 使用 Pinned Memory 替代 std::vector，支持 NPU 直接 DMA 写回
        size_t scores_size_bytes = valid_bucket_num_base * max_doc_per_bucket * 16 * sizeof(float);
        npuAPI::AllocateHostPinned((void**)&batch_scores_, scores_size_bytes);

        tls_scratch_pools_.resize(cores_per_group);
        tls_ivf_masks_.resize(cores_per_group);
        tls_batch_masks_.resize(cores_per_group);

        for (int i = 0; i < cores_per_group; ++i)
        {
            tls_scratch_pools_[i].resize(static_cast<size_t>(group_tls_scratch_pool_reserve_u64));
            tls_ivf_masks_[i].resize(static_cast<size_t>(group_tls_ivf_mask_reserve_u64));
            tls_batch_masks_[i].resize(static_cast<size_t>(group_tls_batch_mask_reserve_u64));
        }

        thread_bucket_results_.resize(cores_per_group);
        thread_doc_results_.resize(cores_per_group);
        for (int i = 0; i < cores_per_group; ++i)
        {
            WarmReserveVectorStorage(thread_bucket_results_[i],
                                     static_cast<size_t>(group_thread_bucket_results_reserve_items_per_rank));
            WarmReserveVectorStorage(thread_doc_results_[i].items,
                                     static_cast<size_t>(group_thread_doc_results_reserve_items_per_rank));
        }

        WarmReserveVectorStorage(sorted_buckets_,
                                 static_cast<size_t>(group_sorted_buckets_reserve_items));
        WarmReserveVectorStorage(batch_bucket_ids_,
                                 static_cast<size_t>(group_batch_bucket_ids_reserve_items));

    }

    // 析构函数：释放 Pinned Memory
    ~WorkerGroup()
    {
        if (batch_scores_) {
            npuAPI::FreeHostPinned(batch_scores_);
            batch_scores_ = nullptr;
        }
    }

    // 【新增】停止标志位
    void Stop()
    {
        running_ = false;
    }

    // 线程入口
    void Run(int core_id)
    {
        BindThreadToCore(core_id);

        int device_id = npu_device_id_start + (group_id_ / groups_per_device);
        auto ret = aclrtSetDevice(device_id);
        if (ret != ACL_SUCCESS)
        {
            std::cerr << "Worker Thread " << core_id
                      << " aclrtSetDevice failed, code: " << ret << std::endl;
            return;
        }
        bool is_leader = IsLeader(core_id);
        int rank = GetInGroupRank(core_id);
        uint64_t observed_stage_epoch = 0;
        WarmUpThreadLocalBuffers(is_leader);
        Query::WarmUpThreadLocalExecutionBuffers();

        // 【修改】循环检查 running_ 标志
        while (running_)
        {
            if (is_leader)
            {
                LeaderLoop();
            }
            else
            {
                FollowerLoop(rank, observed_stage_epoch);
            }
        }
    }

private:
    int group_id_;
    DataBaseCPU *db_;
    Scheduler *sched_;
    npuAPI::Stream npu_stream_;

    // 控制线程退出的原子标志，用于安全shut down系统
    std::atomic<bool> running_{true};

    Query *current_query_ = nullptr;

    enum State
    {
        IDLE = 0,
        IVF_FILTER = 1,
        BATCH_ATTR_FILTER_MASK = 2,
        BATCH_COLLECT_RESULTS = 3
    };
    // 组内 stage 分发信号，低位存 State，高位存单调递增的 epoch。
    // follower 只执行自己尚未消费过的 epoch，避免不同 query / probe round 复用同一 State 时发生 ABA。
    static constexpr uint64_t kStageStateBits = 8;
    static constexpr uint64_t kStageStateMask = (1ULL << kStageStateBits) - 1;
    static constexpr uint64_t PackStageSignal(State state, uint64_t epoch)
    {
        return (epoch << kStageStateBits) | static_cast<uint64_t>(state);
    }
    static constexpr State UnpackStageState(uint64_t signal)
    {
        return static_cast<State>(signal & kStageStateMask);
    }
    static constexpr uint64_t UnpackStageEpoch(uint64_t signal)
    {
        return signal >> kStageStateBits;
    }

    std::atomic<uint64_t> stage_signal_{PackStageSignal(IDLE, 0)};
    uint64_t next_stage_epoch_ = 0;
    std::atomic<int> worker_counter_{0};

    std::vector<uint32_t> sorted_buckets_;
    uint32_t current_batch_start_ = 0;
    uint32_t current_batch_size_ = 0;
    std::vector<uint32_t> batch_bucket_ids_;
    
    // 修改为 Pinned Memory 指针
    float* batch_scores_ = nullptr;

    struct CandidateBucket
    {
        uint32_t bucket_id;
        float score;
    };
    std::vector<std::vector<CandidateBucket>> thread_bucket_results_;

    struct ThreadDocResult
    {
        std::vector<QueryResult::Item> items;
    };
    std::vector<ThreadDocResult> thread_doc_results_;

    std::vector<std::vector<uint64_t, AlignedAllocator<uint64_t>>> tls_scratch_pools_;
    std::vector<std::vector<uint64_t>> tls_ivf_masks_;
    std::vector<std::vector<uint64_t>> tls_batch_masks_;

    static constexpr const char *kBatchBucketIdsLabel = "WorkerGroup.h:batch_bucket_ids_";
    static constexpr const char *kTlsIvfMasksLabel = "WorkerGroup.h:tls_ivf_masks_";
    static constexpr const char *kTlsScratchPoolsLabel = "WorkerGroup.h:tls_scratch_pools_";
    static constexpr const char *kExecuteIvfHeapLabel = "WorkerGroup.h:ExecuteIVF.local_pq";
    static constexpr const char *kThreadBucketResultsLabel = "WorkerGroup.h:thread_bucket_results_";
    static constexpr const char *kCandidateMergeBufferLabel = "WorkerGroup.h:CandidateBucketMerge.candidates";
    static constexpr const char *kSortedBucketsLabel = "WorkerGroup.h:sorted_buckets_";
    static constexpr const char *kTlsBatchMasksLabel = "WorkerGroup.h:tls_batch_masks_";
    static constexpr const char *kTempDocMaskLabel = "WorkerGroup.h:temp_doc_mask";
    static constexpr const char *kThreadDocResultsItemsLabel = "WorkerGroup.h:thread_doc_results_.items";
    static constexpr const char *kAllItemsLabel = "WorkerGroup.h:MergeBatchResults.all_items";

    static std::vector<CandidateBucket> &ThreadLocalCandidateHeapStorage()
    {
        static thread_local std::vector<CandidateBucket> storage;
        return storage;
    }

    static std::vector<CandidateBucket> &ThreadLocalCandidateMergeBuffer()
    {
        static thread_local std::vector<CandidateBucket> buffer;
        return buffer;
    }

    static std::vector<QueryResult::Item> &ThreadLocalAllItemsBuffer()
    {
        static thread_local std::vector<QueryResult::Item> buffer;
        return buffer;
    }

    static std::vector<uint64_t> &ThreadLocalTempDocMask()
    {
        static thread_local std::vector<uint64_t> buffer;
        return buffer;
    }

    static void WarmUpThreadLocalBuffers(bool is_leader)
    {
        WarmReserveVectorStorage(ThreadLocalCandidateHeapStorage(),
                                 static_cast<size_t>(group_local_pq_reserve_items_per_rank));
        WarmReserveVectorStorage(ThreadLocalTempDocMask(),
                                 static_cast<size_t>(group_temp_doc_mask_reserve_u64_per_rank));
        if (is_leader)
        {
            WarmReserveVectorStorage(ThreadLocalCandidateMergeBuffer(),
                                     static_cast<size_t>(group_candidate_merge_reserve_items));
            WarmReserveVectorStorage(ThreadLocalAllItemsBuffer(),
                                     static_cast<size_t>(group_all_items_reserve_items));
        }
    }

    // --- Leader Logic ---
    void PublishStage(State stage)
    {
        worker_counter_.store(0, std::memory_order_release);
        const uint64_t epoch = ++next_stage_epoch_;
        stage_signal_.store(PackStageSignal(stage, epoch), std::memory_order_release);
    }

    void LeaderLoop()
    {
        // 1. Fetch Query
        current_query_ = sched_->Pop();
        if (!current_query_)
        {
            // 如果队列空且收到停止信号，则退出
            if (!running_)
                return;
            std::this_thread::yield();
            return;
        }

        ScopedMemoryEventSession memory_event_scope(current_query_->memory_event_session());
        MemoryEventSession *session = current_query_->memory_event_session();
        current_query_->MarkProcessingStart();

        auto start = std::chrono::high_resolution_clock::now();
        // 2. Stage 1
        PublishStage(IVF_FILTER);
        ExecuteIVF(0);
        WaitFollowers();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        current_query_->timing_metrics.bucket_level_ivf_ms = elapsed.count();

        start = std::chrono::high_resolution_clock::now();
        CandidateBucketMerge();
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        current_query_->timing_metrics.candidate_bucket_merge_ms = elapsed.count();

        // 3. Stage 2
        int offset = 0;
        bool satisfied = false;
        int probe_expand_rounds = 0;

        while (!satisfied && offset < (int)sorted_buckets_.size())
        {
            ++probe_expand_rounds;
            const int bucket_count_this_round =
                (probe_expand_rounds == 1) ? probe_initial_bucket_num : valid_bucket_num_incremental;
            int count = std::min(bucket_count_this_round, (int)sorted_buckets_.size() - offset);
            current_batch_start_ = offset;
            current_batch_size_ = count;

            const size_t batch_bucket_ids_old_capacity = session == nullptr ? 0 : batch_bucket_ids_.capacity();
            batch_bucket_ids_.resize(static_cast<size_t>(count));
            for (int i = 0; i < count; ++i)
                batch_bucket_ids_[static_cast<size_t>(i)] = sorted_buckets_[static_cast<size_t>(offset + i)];
            RecordCapacityGrowth<uint32_t>(session,
                                           kBatchBucketIdsLabel,
                                           batch_bucket_ids_old_capacity,
                                           batch_bucket_ids_.capacity());

            start = std::chrono::high_resolution_clock::now();
            
            // --- 全异步提交 ---
            // A. 重置 Flag 为忙
            npuAPI::ResetGroupFlag(group_id_);
            // B. 启动任务链 (包含计算、结果回写、Flag回写)
            npuAPI::LaunchBatchKernel(npu_stream_, batch_bucket_ids_, 
                                      current_query_->query_vector.data(), 
                                      batch_scores_, // Pinned Buffer
                                      group_id_);
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            current_query_->timing_metrics.npu_async_launch_ms += elapsed.count();

            // --- CPU 并行处理 (与 NPU 重叠) ---
            start = std::chrono::high_resolution_clock::now();
            PublishStage(BATCH_ATTR_FILTER_MASK);
            InBucketAttrFilter(0);
            WaitFollowers(); // 等待 Follower 完成 CPU 属性过滤
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            current_query_->timing_metrics.inbucket_attr_filter_overlapped_ms += elapsed.count();

            // --- 轮询等待 NPU 完成 ---
            start = std::chrono::high_resolution_clock::now();
            
            volatile bool* flag_ptr = npuAPI::GetGroupFlag(group_id_);
            while (*flag_ptr == true) {
#if defined(__aarch64__) || defined(__arm__)
                __asm__ volatile("yield");
#else
                _mm_pause();
#endif
            }
            // Flag 变 False，说明 batch_scores_ 数据已就绪

            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            current_query_->timing_metrics.wait_npu_flag_ms += elapsed.count();

            // Debug 路径：CPU 复算当前 batch 的内积，校验 NPU 返回结果是否在容差范围内。
            npuAPI::DebugVerifyBatchResults(batch_bucket_ids_,
                                           current_query_->query_vector.data(),
                                           batch_scores_,
                                           group_id_);

            // --- 结果收集 ---
            start = std::chrono::high_resolution_clock::now();
            PublishStage(BATCH_COLLECT_RESULTS);
            ExecuteResultCollection(0);
            WaitFollowers();
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            current_query_->timing_metrics.result_collection_ms += elapsed.count();

            start = std::chrono::high_resolution_clock::now();
            MergeBatchResults();
            satisfied = current_query_->result.check_satisfaction(current_query_->expanded_k);
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            current_query_->timing_metrics.final_merge_ms += elapsed.count();

            offset += count;
        }
        current_query_->timing_metrics.probe_expand_rounds = probe_expand_rounds;
        current_query_->process_round_count = probe_expand_rounds;
        current_query_->finalize_results();
        current_query_->MarkProcessingEnd();

        sched_->PushResult(current_query_);
        current_query_ = nullptr;
        current_batch_start_ = 0;
        current_batch_size_ = 0;
    }

    // --- Follower Logic ---
    void FollowerLoop(int rank, uint64_t &observed_stage_epoch)
    {
        const uint64_t signal = stage_signal_.load(std::memory_order_acquire);
        const uint64_t stage_epoch = UnpackStageEpoch(signal);
        if (stage_epoch == observed_stage_epoch)
        {
            if (!running_)
                return;
            std::this_thread::yield();
            return;
        }

        State s = UnpackStageState(signal);
        if (s == IDLE)
        {
            observed_stage_epoch = stage_epoch;
            if (!running_)
                return;
            std::this_thread::yield();
            return;
        }

        if (s == IVF_FILTER)
        {
            ScopedMemoryEventSession memory_event_scope(current_query_ == nullptr
                                                            ? nullptr
                                                            : current_query_->memory_event_session());
            ExecuteIVF(rank);
            observed_stage_epoch = stage_epoch;
            SignalDone();
        }
        else if (s == BATCH_ATTR_FILTER_MASK)
        {
            ScopedMemoryEventSession memory_event_scope(current_query_ == nullptr
                                                            ? nullptr
                                                            : current_query_->memory_event_session());
            InBucketAttrFilter(rank);
            observed_stage_epoch = stage_epoch;
            SignalDone();
        }
        else if (s == BATCH_COLLECT_RESULTS)
        {
            ScopedMemoryEventSession memory_event_scope(current_query_ == nullptr
                                                            ? nullptr
                                                            : current_query_->memory_event_session());
            ExecuteResultCollection(rank);
            observed_stage_epoch = stage_epoch;
            SignalDone();
        }
    }

    void SignalDone() { worker_counter_.fetch_add(1, std::memory_order_release); }
    void WaitFollowers()
    {
        while (worker_counter_.load(std::memory_order_acquire) < (cores_per_group - 1))
        {
#if defined(__aarch64__) || defined(__arm__)
            __asm__ volatile("yield");
#else
            _mm_pause();
#endif
        }
    }

    // --- Execution Logic (Keep Unchanged) ---
    void ExecuteIVF(int rank)
    {
        MemoryEventSession *session = current_query_ == nullptr ? nullptr : current_query_->memory_event_session();
        std::vector<uint64_t> &local_mask = tls_ivf_masks_[rank];
        const size_t local_mask_old_capacity = session == nullptr ? 0 : local_mask.capacity();
        const size_t scratch_old_capacity = session == nullptr ? 0 : tls_scratch_pools_[rank].capacity();
        local_mask.clear();
        current_query_->search_ivf(db_->get_bucket_level_ivf(), local_mask, tls_scratch_pools_[rank], rank);
        RecordCapacityGrowth<uint64_t>(session,
                                       kTlsIvfMasksLabel,
                                       local_mask_old_capacity,
                                       local_mask.capacity());
        RecordCapacityGrowth<uint64_t>(session,
                                       kTlsScratchPoolsLabel,
                                       scratch_old_capacity,
                                       tls_scratch_pools_[rank].capacity());
        const int local_candidate_limit = db_->get_bucket_level_ivf().get_buckets_per_core();

        auto cmp = [](const CandidateBucket &a, const CandidateBucket &b)
        { return a.score > b.score; };
        std::vector<CandidateBucket> &heap = ThreadLocalCandidateHeapStorage();
        const size_t heap_old_capacity = session == nullptr ? 0 : heap.capacity();
        heap.clear();

        const float *centroids = db_->get_centroids();
        const float *q_vec = current_query_->query_vector.data();
        uint32_t buckets_per_core = db_->get_bucket_level_ivf().get_buckets_per_core();
        uint32_t base_global_bucket_idx = rank * buckets_per_core;
        uint32_t mask_size = local_mask.size();

        for (uint32_t i = 0; i < mask_size; ++i)
        {
            uint64_t word = local_mask[i];
            if (word == 0)
                continue;
            uint32_t word_base_offset = i * 64;
            for (int bit = 0; bit < 64; ++bit)
            {
                if ((word >> bit) & 1ULL)
                {
                    uint32_t local_bid = word_base_offset + bit;
                    if (local_bid >= buckets_per_core)
                        continue;
                    uint32_t bucket_id = base_global_bucket_idx + local_bid;
                    if (bucket_id >= total_bucket_num)
                        continue;

                    const float *c_vec = centroids + bucket_id * vector_dim;
                    float dot_product = 0.0f;
#if defined(__aarch64__) || defined(__arm__)
                    float32x4_t sum_vec = vdupq_n_f32(0.0f);
                    for (int d = 0; d < vector_dim; d += 4)
                    {
                        float32x4_t va = vld1q_f32(q_vec + d);
                        float32x4_t vb = vld1q_f32(c_vec + d);
                        sum_vec = vmlaq_f32(sum_vec, va, vb);
                    }
                    dot_product = vaddvq_f32(sum_vec);
#else
                    for (int d = 0; d < vector_dim; ++d)
                        dot_product += q_vec[d] * c_vec[d];
#endif
                    if (heap.size() < static_cast<size_t>(local_candidate_limit))
                    {
                        heap.push_back({bucket_id, dot_product});
                        std::push_heap(heap.begin(), heap.end(), cmp);
                    }
                    else if (!heap.empty() && dot_product > heap.front().score)
                    {
                        std::pop_heap(heap.begin(), heap.end(), cmp);
                        heap.back() = {bucket_id, dot_product};
                        std::push_heap(heap.begin(), heap.end(), cmp);
                    }
                }
            }
        }
        auto &my_res = thread_bucket_results_[rank];
        const size_t my_res_old_capacity = session == nullptr ? 0 : my_res.capacity();
        my_res.clear();
        RecordCapacityGrowth<CandidateBucket>(session,
                                              kExecuteIvfHeapLabel,
                                              heap_old_capacity,
                                              heap.capacity());
        while (!heap.empty())
        {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            my_res.push_back(heap.back());
            heap.pop_back();
        }
        RecordCapacityGrowth<CandidateBucket>(session,
                                              kThreadBucketResultsLabel,
                                              my_res_old_capacity,
                                              my_res.capacity());
    }

    void CandidateBucketMerge()
    {
        MemoryEventSession *session = current_query_ == nullptr ? nullptr : current_query_->memory_event_session();
        std::vector<CandidateBucket> &candidates = ThreadLocalCandidateMergeBuffer();
        const size_t candidates_old_capacity = session == nullptr ? 0 : candidates.capacity();
        candidates.clear();
        for (const auto &bucket_list : thread_bucket_results_)
            candidates.insert(candidates.end(), bucket_list.begin(), bucket_list.end());

        std::sort(candidates.begin(), candidates.end(), [](const CandidateBucket &a, const CandidateBucket &b)
                  { return a.score > b.score; });
        RecordCapacityGrowth<CandidateBucket>(session,
                                              kCandidateMergeBufferLabel,
                                              candidates_old_capacity,
                                              candidates.capacity());

        const size_t sorted_buckets_old_capacity = session == nullptr ? 0 : sorted_buckets_.capacity();
        int count = static_cast<int>(candidates.size());
        sorted_buckets_.resize(static_cast<size_t>(count));
        for (int i = 0; i < count; ++i)
            sorted_buckets_[static_cast<size_t>(i)] = candidates[static_cast<size_t>(i)].bucket_id;
        RecordCapacityGrowth<uint32_t>(session,
                                       kSortedBucketsLabel,
                                       sorted_buckets_old_capacity,
                                       sorted_buckets_.capacity());
    }

    void InBucketAttrFilter(int rank)
    {
        MemoryEventSession *session = current_query_ == nullptr ? nullptr : current_query_->memory_event_session();
        int buckets_per_core = (current_batch_size_ + cores_per_group - 1) / cores_per_group;
        int start = rank * buckets_per_core;
        int end = std::min(start + buckets_per_core, (int)current_batch_size_);

        std::vector<uint64_t> &my_masks_buf = tls_batch_masks_[rank];
        const size_t my_masks_buf_old_capacity = session == nullptr ? 0 : my_masks_buf.capacity();
        my_masks_buf.clear();
        std::vector<uint64_t> &temp_doc_mask = ThreadLocalTempDocMask();
        const size_t temp_doc_mask_old_capacity = session == nullptr ? 0 : temp_doc_mask.capacity();
        const size_t scratch_old_capacity = session == nullptr ? 0 : tls_scratch_pools_[rank].capacity();
        temp_doc_mask.clear();

        for (int i = start; i < end; ++i)
        {
            uint32_t bid = batch_bucket_ids_[i];
            const Bucket &bucket = db_->get_bucket(bid);
            uint32_t bucket_stride = bucket.get_stride();
            current_query_->search_bucket(bucket, temp_doc_mask, tls_scratch_pools_[rank]);
            size_t write_offset = my_masks_buf.size();
            my_masks_buf.resize(write_offset + bucket_stride);
            std::memcpy(my_masks_buf.data() + write_offset,
                        temp_doc_mask.data(),
                        (size_t)bucket_stride * sizeof(uint64_t));
        }
        RecordCapacityGrowth<uint64_t>(session,
                                       kTlsBatchMasksLabel,
                                       my_masks_buf_old_capacity,
                                       my_masks_buf.capacity());
        RecordCapacityGrowth<uint64_t>(session,
                                       kTempDocMaskLabel,
                                       temp_doc_mask_old_capacity,
                                       temp_doc_mask.capacity());
        RecordCapacityGrowth<uint64_t>(session,
                                       kTlsScratchPoolsLabel,
                                       scratch_old_capacity,
                                       tls_scratch_pools_[rank].capacity());
    }

    void ExecuteResultCollection(int rank)
    {
        MemoryEventSession *session = current_query_ == nullptr ? nullptr : current_query_->memory_event_session();
        int buckets_per_core = (current_batch_size_ + cores_per_group - 1) / cores_per_group;
        int start = rank * buckets_per_core;
        int end = std::min(start + buckets_per_core, (int)current_batch_size_);

        auto &my_res = thread_doc_results_[rank];
        const size_t my_items_old_capacity = session == nullptr ? 0 : my_res.items.capacity();
        my_res.items.clear();
        const std::vector<uint64_t> &my_masks_buf = tls_batch_masks_[rank];
        size_t mask_read_offset = 0;

        for (int i = start; i < end; ++i)
        {
            uint32_t bid = batch_bucket_ids_[i];
            const Bucket &bucket = db_->get_bucket(bid);
            
            // 使用 Pinned Memory 的 float* 指针进行访问
            float *score_ptr = batch_scores_ + i * max_doc_per_bucket * 16;
            
            int doc_num = bucket.get_doc_num();
            const auto &gids = bucket.get_global_ids();
            uint32_t bucket_stride = bucket.get_stride();
            const uint64_t *current_mask = my_masks_buf.data() + mask_read_offset;
            mask_read_offset += bucket_stride;

            for (int d = 0; d < doc_num; ++d)
            {
                if (d >= max_doc_per_bucket)
                    break;
                bool keep = (current_mask[d / 64] >> (d % 64)) & 1ULL;
                if (keep)
                    my_res.items.push_back({gids[d], score_ptr[(size_t)d * 16]});
            }
        }

        // 多桶归属场景下，同一 doc 可能在当前 core 负责的多个桶里重复出现。
        // 先在 core 内去重，再截断到局部 Top-K 后交给 leader。
        QueryResult::DeduplicateAndKeepTopK(my_res.items, current_query_->expanded_k);
        RecordCapacityGrowth<QueryResult::Item>(session,
                                               kThreadDocResultsItemsLabel,
                                               my_items_old_capacity,
                                               my_res.items.capacity());
    }

    void MergeBatchResults()
    {
        MemoryEventSession *session = current_query_ == nullptr ? nullptr : current_query_->memory_event_session();
        std::vector<QueryResult::Item> &all_items = ThreadLocalAllItemsBuffer();
        const size_t all_items_old_capacity = session == nullptr ? 0 : all_items.capacity();
        all_items.clear();

        int k = current_query_->expanded_k;
        for (const auto &tr : thread_doc_results_)
            all_items.insert(all_items.end(), tr.items.begin(), tr.items.end());

        QueryResult::DeduplicateAndKeepTopK(all_items, k);
        current_query_->merge_batch_results(all_items);
        RecordCapacityGrowth<QueryResult::Item>(session,
                                               kAllItemsLabel,
                                               all_items_old_capacity,
                                               all_items.capacity());
    }
};
