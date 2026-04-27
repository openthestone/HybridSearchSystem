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
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <string>

#include <memory>

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

        prefetch_slot_capacity_floats_[kFirstRoundSlotId] =
            static_cast<size_t>(valid_bucket_num_base_level_1) *
            static_cast<size_t>(max_process_bucket_num_level_2) *
            static_cast<size_t>(max_doc_per_bucket_level_2) *
            kScoreCols;
        prefetch_slot_capacity_floats_[kIncrementalSlotAId] =
            static_cast<size_t>(valid_bucket_num_incremental_level_1) *
            static_cast<size_t>(max_process_bucket_num_level_2) *
            static_cast<size_t>(max_doc_per_bucket_level_2) *
            kScoreCols;
        prefetch_slot_capacity_floats_[kIncrementalSlotBId] =
            prefetch_slot_capacity_floats_[kIncrementalSlotAId];

        for (int slot_id = 0; slot_id < npuAPI::kGroupFlagSlotCount; ++slot_id)
        {
            const size_t slot_score_bytes = prefetch_slot_capacity_floats_[slot_id] * sizeof(float);
            prefetch_slots_[slot_id].flag = npuAPI::GetGroupFlag(group_id_, slot_id);
            npuAPI::AllocateHostPinned((void **)&prefetch_slots_[slot_id].score_buffer, slot_score_bytes);
            npuAPI::ClearGroupFlag(group_id_, slot_id);
        }

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

        WarmReserveVectorStorage(sorted_level_1_buckets_,
                                 static_cast<size_t>(group_sorted_buckets_reserve_items));
        for (auto &round_plan : round_plan_buffers_)
        {
            (void)round_plan;
            // mask_storage is allocated per-batch in PrepareRoundMasks
        }
    }

    // 析构函数：释放 Pinned Memory
    ~WorkerGroup()
    {
        for (int slot_id = 0; slot_id < npuAPI::kGroupFlagSlotCount; ++slot_id)
        {
            if (prefetch_slots_[slot_id].score_buffer)
            {
                npuAPI::FreeHostPinned(prefetch_slots_[slot_id].score_buffer);
                prefetch_slots_[slot_id].score_buffer = nullptr;
            }
        }
    }

    // 【新增】停止标志位
    void Stop()
    {
        running_ = false;
    }

    uint64_t GetProcessedCount() const
    {
        return queries_processed_.load(std::memory_order_relaxed);
    }

    // 线程入口
    void Run(int core_id)
    {
        BindThreadToCore(core_id);

        int device_id = npu_device_id_start + g_group_to_device[group_id_];
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
    std::atomic<uint64_t> queries_processed_{0};

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

    static constexpr size_t kScoreCols = 1;

    struct RoundBatch
    {
        std::vector<uint32_t> bucket_ids;
        size_t score_offset_floats = 0;
        size_t score_count_floats = 0;
        // Precomputed score offsets for each bucket in the batch (for dynamic work distribution)
        std::vector<size_t> score_offsets_per_bucket;
    };

    struct RoundPlan
    {
        bool valid = false;
        int level_1_count = 0;
        std::vector<RoundBatch> batches;
        // Shared mask storage: all workers write/read masks here
        // Mask for batch position i is at mask_storage[i * mask_stride ... (i+1) * mask_stride)
        std::vector<uint64_t> mask_storage;
        size_t mask_stride = 0;  // u64 per bucket slot
    };

    struct PrefetchSlot
    {
        volatile uint32_t *flag = nullptr;
        float *score_buffer = nullptr;
    };

    static constexpr int kFirstRoundSlotId = 0;
    static constexpr int kIncrementalSlotAId = 1;
    static constexpr int kIncrementalSlotBId = 2;

    std::vector<uint32_t> sorted_level_1_buckets_;
    uint32_t current_batch_size_ = 0;
    PrefetchSlot prefetch_slots_[npuAPI::kGroupFlagSlotCount];
    size_t prefetch_slot_capacity_floats_[npuAPI::kGroupFlagSlotCount] = {0, 0, 0};
    RoundPlan round_plan_buffers_[3];
    RoundPlan *active_round_ = nullptr;
    RoundBatch *active_batch_ = nullptr;
    float *active_batch_scores_ = nullptr;

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

    // Dynamic work distribution: atomic counter for balanced work across workers
    alignas(64) std::atomic<uint32_t> next_work_idx_{0};

    static constexpr const char *kTlsIvfMasksLabel = "WorkerGroup.h:tls_ivf_masks_";
    static constexpr const char *kTlsScratchPoolsLabel = "WorkerGroup.h:tls_scratch_pools_";
    static constexpr const char *kExecuteIvfHeapLabel = "WorkerGroup.h:ExecuteIVF.local_pq";
    static constexpr const char *kThreadBucketResultsLabel = "WorkerGroup.h:thread_bucket_results_";
    static constexpr const char *kCandidateMergeBufferLabel = "WorkerGroup.h:CandidateBucketMerge.candidates";
    static constexpr const char *kSortedBucketsLabel = "WorkerGroup.h:sorted_level_1_buckets_";
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
        next_work_idx_.store(0, std::memory_order_release);
        const uint64_t epoch = ++next_stage_epoch_;
        stage_signal_.store(PackStageSignal(stage, epoch), std::memory_order_release);
    }

    [[noreturn]] void AbortWithInvariantError(const std::string &msg) const
    {
        std::cerr << "[WorkerGroup] " << msg
                  << " group_id=" << group_id_
                  << " query_id=" << (current_query_ == nullptr ? 0 : current_query_->query_id)
                  << std::endl;
        std::abort();
    }

    void ClearActiveBatchContext()
    {
        active_round_ = nullptr;
        active_batch_ = nullptr;
        active_batch_scores_ = nullptr;
        current_batch_size_ = 0;
    }

    void ResetRoundPlan(RoundPlan &round)
    {
        round.valid = false;
        round.level_1_count = 0;
        round.batches.clear();
        round.mask_storage.clear();
        round.mask_stride = 0;
    }

    void InitializeQueryPrefetchState()
    {
        for (int slot_id = 0; slot_id < npuAPI::kGroupFlagSlotCount; ++slot_id)
        {
            npuAPI::ClearGroupFlag(group_id_, slot_id);
        }
        for (auto &round : round_plan_buffers_)
        {
            ResetRoundPlan(round);
        }
        ClearActiveBatchContext();
    }

    bool BuildRoundPlan(RoundPlan &round, int round_index, int &level_1_offset)
    {
        ResetRoundPlan(round);

        if (level_1_offset >= static_cast<int>(sorted_level_1_buckets_.size()))
        {
            return false;
        }

        const int requested_level_1_count =
            (round_index == 0) ? valid_bucket_num_base_level_1 : valid_bucket_num_incremental_level_1;
        const int level_1_count =
            std::min(requested_level_1_count, static_cast<int>(sorted_level_1_buckets_.size()) - level_1_offset);
        if (level_1_count <= 0)
        {
            return false;
        }

        size_t total_level_2_count = 0;
        for (int i = 0; i < level_1_count; ++i)
        {
            const uint32_t level_1_bucket_id =
                sorted_level_1_buckets_[static_cast<size_t>(level_1_offset + i)];
            total_level_2_count += static_cast<size_t>(db_->get_level_2_bucket_end(level_1_bucket_id) -
                                                       db_->get_level_2_bucket_begin(level_1_bucket_id));
        }

        const size_t estimated_batch_count =
            (total_level_2_count + static_cast<size_t>(max_process_bucket_num_level_2) - 1) /
            static_cast<size_t>(max_process_bucket_num_level_2);
        round.batches.reserve(estimated_batch_count);

        for (int i = 0; i < level_1_count; ++i)
        {
            const uint32_t level_1_bucket_id =
                sorted_level_1_buckets_[static_cast<size_t>(level_1_offset + i)];
            const uint32_t level_2_begin = db_->get_level_2_bucket_begin(level_1_bucket_id);
            const uint32_t level_2_end = db_->get_level_2_bucket_end(level_1_bucket_id);
            for (uint32_t level_2_bucket_id = level_2_begin; level_2_bucket_id < level_2_end; ++level_2_bucket_id)
            {
                if (round.batches.empty() ||
                    round.batches.back().bucket_ids.size() >= static_cast<size_t>(max_process_bucket_num_level_2))
                {
                    round.batches.emplace_back();
                    RoundBatch &new_batch = round.batches.back();
                    new_batch.bucket_ids.reserve(static_cast<size_t>(group_batch_bucket_ids_reserve_items));
                }

                RoundBatch &batch = round.batches.back();
                batch.bucket_ids.push_back(level_2_bucket_id);

                const Bucket &bucket = db_->get_bucket(level_2_bucket_id);
                batch.score_count_floats += static_cast<size_t>(bucket.get_doc_num()) * kScoreCols;
            }
        }

        level_1_offset += level_1_count;
        round.valid = !round.batches.empty();
        round.level_1_count = round.valid ? level_1_count : 0;
        if (!round.valid)
        {
            return false;
        }

        // Compute max stride across all buckets for shared mask buffer
        size_t max_stride = 0;
        for (const RoundBatch &batch : round.batches)
        {
            for (uint32_t bid : batch.bucket_ids)
            {
                max_stride = std::max(max_stride, static_cast<size_t>(db_->get_bucket(bid).get_stride()));
            }
        }
        round.mask_stride = max_stride;

        // Precompute score offsets and allocate shared mask buffer per batch
        size_t total_mask_words = 0;
        for (RoundBatch &batch : round.batches)
        {
            const size_t batch_size = batch.bucket_ids.size();
            batch.score_offsets_per_bucket.resize(batch_size);
            size_t score_off = 0;
            for (size_t i = 0; i < batch_size; ++i)
            {
                batch.score_offsets_per_bucket[i] = score_off;
                score_off += static_cast<size_t>(db_->get_bucket(batch.bucket_ids[i]).get_doc_num()) * kScoreCols;
            }
            total_mask_words += batch_size * max_stride;
        }

        // Allocate shared mask storage
        if (round.mask_storage.capacity() < total_mask_words)
        {
            round.mask_storage.reserve(total_mask_words);
        }

        return true;
    }

    bool MaxProbeL1BucketLimitEnabled() const
    {
        return max_probe_l1_bucket_num_enable == 1;
    }

    bool ReachesMaxProbeL1BucketLimit(int searched_level_1_count, const RoundPlan &round) const
    {
        return MaxProbeL1BucketLimitEnabled() &&
               searched_level_1_count + round.level_1_count >= max_probe_l1_bucket_num;
    }

    void AccountConsumedRound(const RoundPlan &round,
                              int &searched_level_1_count,
                              int &consumed_round_count,
                              int &process_round_count_level_2)
    {
        ++consumed_round_count;
        searched_level_1_count += round.level_1_count;
        process_round_count_level_2 += static_cast<int>(round.batches.size());
    }

    void AssignRoundToSlot(RoundPlan &round, int slot_id)
    {
        size_t used_floats = 0;
        for (RoundBatch &batch : round.batches)
        {
            batch.score_offset_floats = used_floats;
            used_floats += batch.score_count_floats;
            if (used_floats > prefetch_slot_capacity_floats_[slot_id])
            {
                throw std::runtime_error("[WorkerGroup] score slot capacity exceeded: slot_id=" +
                                         std::to_string(slot_id) +
                                         ", required_floats=" + std::to_string(used_floats) +
                                         ", capacity_floats=" +
                                         std::to_string(prefetch_slot_capacity_floats_[slot_id]));
            }
        }
    }

    void LaunchRoundToSlot(RoundPlan &round, int slot_id)
    {
        if (!round.valid)
        {
            return;
        }

        AssignRoundToSlot(round, slot_id);

        auto start = std::chrono::high_resolution_clock::now();
        npuAPI::ResetGroupFlag(group_id_, slot_id);
        for (const RoundBatch &batch : round.batches)
        {
            npuAPI::LaunchBatchKernel(npu_stream_,
                                      batch.bucket_ids,
                                      prefetch_slots_[slot_id].score_buffer + batch.score_offset_floats,
                                      group_id_);
        }
        npuAPI::EnqueueGroupCompletion(npu_stream_, group_id_, slot_id);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        current_query_->timing_metrics.npu_async_launch_ms += elapsed.count();
    }

    void WaitForSlot(int slot_id)
    {
        auto start = std::chrono::high_resolution_clock::now();
        // Pure spin on ACL event completion. This keeps the low-overhead wait strategy
        // from the compact-output path, but removes the extra D2H flag-ready copy from
        // the critical path.
        while (!npuAPI::IsGroupCompletionReady(group_id_, slot_id))
        {
#if defined(__aarch64__) || defined(__arm__)
            __asm__ volatile("yield");
#else
            _mm_pause();
#endif
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        current_query_->timing_metrics.wait_npu_flag_ms += elapsed.count();
    }

    void PrepareRoundMasks(RoundPlan &round)
    {
        auto start = std::chrono::high_resolution_clock::now();

        if (current_query_->filter_exp.Bucket_RPN.empty())
        {
            // No filter: all masks are 0xFF — skip per-batch barrier syncs entirely.
            // Allocate mask_storage for the largest batch and fill with all-ones.
            size_t max_needed = 0;
            for (const auto &batch : round.batches)
            {
                max_needed = std::max(max_needed,
                    static_cast<size_t>(batch.bucket_ids.size()) * round.mask_stride);
            }
            if (max_needed > 0)
            {
                if (round.mask_storage.size() < max_needed)
                    round.mask_storage.resize(max_needed);
                std::memset(round.mask_storage.data(), 0xFF, max_needed * sizeof(uint64_t));
            }
        }
        else
        {
            for (size_t bi = 0; bi < round.batches.size(); ++bi)
            {
                RoundBatch &batch = round.batches[bi];
                active_round_ = &round;
                active_batch_ = &batch;
                active_batch_scores_ = nullptr;
                current_batch_size_ = static_cast<uint32_t>(batch.bucket_ids.size());

                // Allocate shared mask buffer for this batch
                const size_t needed = static_cast<size_t>(current_batch_size_) * round.mask_stride;
                if (round.mask_storage.size() < needed)
                {
                    round.mask_storage.resize(needed, 0);
                }
                else
                {
                    std::memset(round.mask_storage.data(), 0, needed * sizeof(uint64_t));
                }

                PublishStage(BATCH_ATTR_FILTER_MASK);
                InBucketAttrFilter(0);
                WaitFollowers();
            }
        }
        ClearActiveBatchContext();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        current_query_->timing_metrics.inbucket_attr_filter_overlapped_ms += elapsed.count();
    }

    void CollectRoundResults(RoundPlan &round, int slot_id, bool &satisfied)
    {
        if (!round.valid)
        {
            return;
        }

        WaitForSlot(slot_id);

        for (RoundBatch &batch : round.batches)
        {
            if (satisfied)
            {
                break;
            }

            active_round_ = &round;
            active_batch_ = &batch;
            active_batch_scores_ = prefetch_slots_[slot_id].score_buffer + batch.score_offset_floats;
            current_batch_size_ = static_cast<uint32_t>(batch.bucket_ids.size());

            npuAPI::DebugVerifyBatchResults(batch.bucket_ids,
                                           current_query_->query_vector.data(),
                                           active_batch_scores_,
                                           group_id_);

            auto start = std::chrono::high_resolution_clock::now();
            PublishStage(BATCH_COLLECT_RESULTS);
            ExecuteResultCollection(0);
            WaitFollowers();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            current_query_->timing_metrics.result_collection_ms += elapsed.count();

            start = std::chrono::high_resolution_clock::now();
            MergeBatchResults();
            satisfied = current_query_->result.check_satisfaction(current_query_->expanded_k);
            end = std::chrono::high_resolution_clock::now();
            elapsed = end - start;
            current_query_->timing_metrics.final_merge_ms += elapsed.count();
        }
        ClearActiveBatchContext();
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
        current_query_->MarkProcessingStart();
        InitializeQueryPrefetchState();

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

        // 3. Stage 2 — Upload query to NPU once per query
        npuAPI::UploadQuery(current_query_->query_vector.data(),
                            current_query_->mmad_query_padded_fp16.empty()
                                ? nullptr
                                : current_query_->mmad_query_padded_fp16.data(),
                            group_id_);

        int level_1_offset = 0;
        int next_round_index = 0;
        bool satisfied = false;
        int searched_level_1_count = 0;
        int consumed_round_count = 0;
        int process_round_count_level_2 = 0;

        RoundPlan *first_round = &round_plan_buffers_[0];
        RoundPlan *current_round = &round_plan_buffers_[1];
        RoundPlan *future_round = &round_plan_buffers_[2];

        if (BuildRoundPlan(*first_round, next_round_index, level_1_offset))
        {
            ++next_round_index;
            LaunchRoundToSlot(*first_round, kFirstRoundSlotId);

            // Pre-submit next incremental round on NPU while CPU processes first round.
            // Skip when no filter: round 1 has ~78K docs >> expanded_k, always satisfies.
            // Pre-submitting wastes NPU bandwidth and delays the next query's stream.
            int pre_slot_id = kIncrementalSlotAId;
            bool has_pre_round = false;
            if (!current_query_->filter_exp.Bucket_RPN.empty() &&
                !ReachesMaxProbeL1BucketLimit(searched_level_1_count, *first_round))
            {
                has_pre_round = !satisfied && BuildRoundPlan(*current_round, next_round_index, level_1_offset);
                if (has_pre_round)
                {
                    ++next_round_index;
                    LaunchRoundToSlot(*current_round, pre_slot_id);
                }
            }

            PrepareRoundMasks(*first_round);
            CollectRoundResults(*first_round, kFirstRoundSlotId, satisfied);
            AccountConsumedRound(*first_round,
                                 searched_level_1_count,
                                 consumed_round_count,
                                 process_round_count_level_2);
            if (!satisfied && MaxProbeL1BucketLimitEnabled() &&
                searched_level_1_count >= max_probe_l1_bucket_num)
            {
                satisfied = true;
            }
            ResetRoundPlan(*first_round);

            // Process the pre-submitted round (NPU likely already done)
            if (has_pre_round && !satisfied)
            {
                PrepareRoundMasks(*current_round);
                CollectRoundResults(*current_round, pre_slot_id, satisfied);
                AccountConsumedRound(*current_round,
                                     searched_level_1_count,
                                     consumed_round_count,
                                     process_round_count_level_2);
                if (!satisfied && MaxProbeL1BucketLimitEnabled() &&
                    searched_level_1_count >= max_probe_l1_bucket_num)
                {
                    satisfied = true;
                }
                ResetRoundPlan(*current_round);
            }
        } else {
        }

        int current_slot_id = kIncrementalSlotAId;
        int future_slot_id = kIncrementalSlotBId;
        bool has_current_round = false;
        if (!satisfied && BuildRoundPlan(*current_round, next_round_index, level_1_offset))
        {
            ++next_round_index;
            LaunchRoundToSlot(*current_round, current_slot_id);
            has_current_round = true;
        }

        int round_iter = 0;
        while (!satisfied && has_current_round)
        {
            bool has_future_round = false;
            if (!ReachesMaxProbeL1BucketLimit(searched_level_1_count, *current_round) &&
                BuildRoundPlan(*future_round, next_round_index, level_1_offset))
            {
                ++next_round_index;
                LaunchRoundToSlot(*future_round, future_slot_id);
                has_future_round = true;
            }

            PrepareRoundMasks(*current_round);
            CollectRoundResults(*current_round, current_slot_id, satisfied);
            AccountConsumedRound(*current_round,
                                 searched_level_1_count,
                                 consumed_round_count,
                                 process_round_count_level_2);
            if (!satisfied && MaxProbeL1BucketLimitEnabled() &&
                searched_level_1_count >= max_probe_l1_bucket_num)
            {
                satisfied = true;
            }
            ResetRoundPlan(*current_round);
            ++round_iter;

            if (satisfied)
            {
                if (has_future_round)
                {
                    WaitForSlot(future_slot_id);
                    ResetRoundPlan(*future_round);
                }
                break;
            }

            if (!has_future_round)
            {
                break;
            }

            std::swap(current_round, future_round);
            std::swap(current_slot_id, future_slot_id);
        }

        current_query_->timing_metrics.probe_expand_rounds = consumed_round_count;
        current_query_->process_round_count_level_1 = consumed_round_count;
        current_query_->process_round_count_level_2 = process_round_count_level_2;
        current_query_->finalize_results();
        current_query_->MarkProcessingEnd();

        sched_->PushResult(current_query_);
        queries_processed_.fetch_add(1, std::memory_order_relaxed);
        current_query_ = nullptr;
        ClearActiveBatchContext();
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
#if defined(__aarch64__) || defined(__arm__)
            __asm__ volatile("yield");
#else
            std::this_thread::yield();
#endif
            return;
        }

        State s = UnpackStageState(signal);
        if (s == IDLE)
        {
            observed_stage_epoch = stage_epoch;
            if (!running_)
                return;
#if defined(__aarch64__) || defined(__arm__)
            __asm__ volatile("yield");
#else
            std::this_thread::yield();
#endif
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

    // --- Execution Logic ---
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
            // Use ctzll to skip zero bits: iterate only over set bits
            while (word)
            {
                int bit = __builtin_ctzll(word);
                word &= word - 1;  // clear lowest set bit

                uint32_t local_bid = word_base_offset + bit;
                if (local_bid >= buckets_per_core)
                    continue;
                uint32_t bucket_id = base_global_bucket_idx + local_bid;
                if (bucket_id >= static_cast<uint32_t>(total_bucket_num_level_1))
                    continue;

                const float *c_vec = centroids + bucket_id * vector_dim;
                // Software prefetch: bring c_vec into L1/L2 cache ahead of the computation
                // Use a look-ahead distance of 1 iteration to overlap memory fetch with compute
                __builtin_prefetch(c_vec, 0, 3);  // rw=0 (read), locality=3 (keep in all caches)
                float dot_product = 0.0f;
#if defined(__aarch64__) || defined(__arm__)
                // ---- NEON 256-bit logical concatenation ----
                // 将两个 float32x4_t (128-bit) 拼接为逻辑 256-bit
                // 每轮循环处理 8 元素，理论上吞吐量翻倍
                float32x4_t sum0 = vdupq_n_f32(0.0f);
                float32x4_t sum1 = vdupq_n_f32(0.0f);
                int d = 0;
                for (; d + 7 < vector_dim; d += 8)
                {
                    // ---- 加载 8 元素（两个 128-bit 寄存器）----
                    float32x4_t va0 = vld1q_f32(q_vec + d);
                    float32x4_t va1 = vld1q_f32(q_vec + d + 4);
                    float32x4_t vb0 = vld1q_f32(c_vec + d);
                    float32x4_t vb1 = vld1q_f32(c_vec + d + 4);
                    // ---- 两个 128-bit FMA 并行执行 ----
                    sum0 = vfmaq_f32(sum0, va0, vb0);
                    sum1 = vfmaq_f32(sum1, va1, vb1);
                }
                // ---- 横向求和：256-bit → 128-bit → scalar ----
                dot_product = vaddvq_f32(sum0) + vaddvq_f32(sum1);
                // ---- 尾部残余 (< 8 元素) ----
                for (; d < vector_dim; ++d)
                    dot_product += q_vec[d] * c_vec[d];
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

        auto cmp_desc = [](const CandidateBucket &a, const CandidateBucket &b)
                  { return a.score > b.score; };
        std::sort(candidates.begin(), candidates.end(), cmp_desc);
        const size_t count = candidates.size();
        RecordCapacityGrowth<CandidateBucket>(session,
                                              kCandidateMergeBufferLabel,
                                              candidates_old_capacity,
                                              candidates.capacity());

        const size_t sorted_buckets_old_capacity = session == nullptr ? 0 : sorted_level_1_buckets_.capacity();
        // Pre-size vector and fill via indexed assignment (Abseil hint: avoids N push_backs)
        sorted_level_1_buckets_.resize(count);
        for (size_t i = 0; i < count; ++i)
            sorted_level_1_buckets_[i] = candidates[i].bucket_id;
        RecordCapacityGrowth<uint32_t>(session,
                                       kSortedBucketsLabel,
                                       sorted_buckets_old_capacity,
                                       sorted_level_1_buckets_.capacity());
    }

    void InBucketAttrFilter(int rank)
    {
        if (active_round_ == nullptr || active_batch_ == nullptr)
        {
            return;
        }

        MemoryEventSession *session = current_query_ == nullptr ? nullptr : current_query_->memory_event_session();
        const int total = static_cast<int>(current_batch_size_);
        const std::vector<uint32_t> &batch_bucket_ids = active_batch_->bucket_ids;

        std::vector<uint64_t> &temp_doc_mask = ThreadLocalTempDocMask();
        const size_t temp_doc_mask_old_capacity = session == nullptr ? 0 : temp_doc_mask.capacity();
        const size_t scratch_old_capacity = session == nullptr ? 0 : tls_scratch_pools_[rank].capacity();
        temp_doc_mask.clear();

        // Dynamic work distribution: each worker atomically grabs the next bucket
        while (true)
        {
            int i = static_cast<int>(next_work_idx_.fetch_add(1, std::memory_order_relaxed));
            if (i >= total) break;

            uint32_t bid = batch_bucket_ids[static_cast<size_t>(i)];
            const Bucket &bucket = db_->get_bucket(bid);
            uint32_t bucket_stride = bucket.get_stride();
            current_query_->search_bucket(bucket, temp_doc_mask, tls_scratch_pools_[rank]);
            if (temp_doc_mask.size() < static_cast<size_t>(bucket_stride))
            {
                AbortWithInvariantError("search_bucket returned undersized mask buffer: bucket_id=" +
                                        std::to_string(bid) +
                                        ", mask_size=" + std::to_string(temp_doc_mask.size()) +
                                        ", bucket_stride=" + std::to_string(bucket_stride));
            }
            // Write mask to shared buffer at fixed position for this batch index
            size_t write_offset = static_cast<size_t>(i) * active_round_->mask_stride;
            std::memcpy(active_round_->mask_storage.data() + write_offset,
                        temp_doc_mask.data(),
                        static_cast<size_t>(bucket_stride) * sizeof(uint64_t));
        }
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
        if (active_round_ == nullptr || active_batch_ == nullptr || active_batch_scores_ == nullptr)
        {
            return;
        }

        MemoryEventSession *session = current_query_ == nullptr ? nullptr : current_query_->memory_event_session();
        const int total = static_cast<int>(current_batch_size_);
        const std::vector<uint32_t> &batch_bucket_ids = active_batch_->bucket_ids;
        const std::vector<size_t> &score_offsets = active_batch_->score_offsets_per_bucket;

        auto &my_res = thread_doc_results_[rank];
        const size_t my_items_old_capacity = session == nullptr ? 0 : my_res.items.capacity();
        my_res.items.clear();

        // Per-thread min-heap: keep only the top expanded_k items per thread.
        const int heap_k = current_query_->expanded_k;
        auto heap_worse = [](const QueryResult::Item &a, const QueryResult::Item &b) {
            return a.score > b.score;
        };
        bool heap_full = false;

        const bool no_filter = current_query_->filter_exp.Bucket_RPN.empty();

        // Dynamic work distribution: each worker atomically grabs the next bucket
        while (true)
        {
            int i = static_cast<int>(next_work_idx_.fetch_add(1, std::memory_order_relaxed));
            if (i >= total) break;

            uint32_t bid = batch_bucket_ids[static_cast<size_t>(i)];
            const Bucket &bucket = db_->get_bucket(bid);
            size_t score_read_offset = score_offsets[static_cast<size_t>(i)];
            float *score_ptr = active_batch_scores_ + score_read_offset;
            int doc_num = bucket.get_doc_num();
            const auto &gids = bucket.get_global_ids();

            if (no_filter)
            {
                // Fast path: all docs pass filter — skip mask scanning entirely.
                // Direct linear scan over compact scores (stride-1, cache-friendly).
                for (int id = 0; id < doc_num; ++id)
                {
                    float score = score_ptr[id];
                    if (!heap_full)
                    {
                        my_res.items.emplace_back(gids[id], score);
                        if (static_cast<int>(my_res.items.size()) == heap_k)
                        {
                            std::make_heap(my_res.items.begin(), my_res.items.end(), heap_worse);
                            heap_full = true;
                        }
                    }
                    else if (score > my_res.items.front().score)
                    {
                        std::pop_heap(my_res.items.begin(), my_res.items.end(), heap_worse);
                        my_res.items.back() = QueryResult::Item(gids[id], score);
                        std::push_heap(my_res.items.begin(), my_res.items.end(), heap_worse);
                    }
                }
            }
            else
            {
                // Filter path: scan mask bits to find passing docs.
                uint32_t bucket_stride = bucket.get_stride();
                if (score_read_offset + static_cast<size_t>(doc_num) * kScoreCols > active_batch_->score_count_floats)
                {
                    AbortWithInvariantError("score_read_offset exceeded stored batch score range: bucket_id=" +
                                            std::to_string(bid) +
                                            ", score_read_offset=" + std::to_string(score_read_offset) +
                                            ", doc_num=" + std::to_string(doc_num) +
                                            ", batch_score_count_floats=" +
                                            std::to_string(active_batch_->score_count_floats));
                }

                const uint64_t *pmask64 = active_round_->mask_storage.data() +
                                          static_cast<size_t>(i) * active_round_->mask_stride;

                for (uint32_t d = 0; d < bucket_stride; d++)
                {
                    uint64_t cur_mask = pmask64[d];
                    int base_id = static_cast<int>(d) << 6;
                    while (cur_mask)
                    {
                        int bit = __builtin_ctzll(cur_mask);
                        int id = base_id + bit;
                        if (id < doc_num)
                        {
                            float score = score_ptr[id];
                            if (!heap_full)
                            {
                                my_res.items.emplace_back(gids[id], score);
                                if (static_cast<int>(my_res.items.size()) == heap_k)
                                {
                                    std::make_heap(my_res.items.begin(), my_res.items.end(), heap_worse);
                                    heap_full = true;
                                }
                            }
                            else if (score > my_res.items.front().score)
                            {
                                std::pop_heap(my_res.items.begin(), my_res.items.end(), heap_worse);
                                my_res.items.back() = QueryResult::Item(gids[id], score);
                                std::push_heap(my_res.items.begin(), my_res.items.end(), heap_worse);
                            }
                        }
                        cur_mask &= cur_mask - 1;
                    }
                }
            }
        }

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

        // Collect all items from all cores, then sort once on Leader
        size_t total_items = 0;
        for (int t = 0; t < cores_per_group; ++t) {
            total_items += thread_doc_results_[t].items.size();
        }

        all_items.reserve(total_items);
        for (int t = 0; t < cores_per_group; ++t) {
            auto &thread_items = thread_doc_results_[t].items;
            all_items.insert(all_items.end(), thread_items.begin(), thread_items.end());
        }

        // Leader single KeepTopK sort
        QueryResult::KeepTopK(all_items, k);
        if (static_cast<int>(all_items.size()) > k) {
            all_items.resize(static_cast<size_t>(k));
        }

        current_query_->merge_batch_results(all_items);
        RecordCapacityGrowth<QueryResult::Item>(session,
                                               kAllItemsLabel,
                                               all_items_old_capacity,
                                               all_items.capacity());
    }
};
