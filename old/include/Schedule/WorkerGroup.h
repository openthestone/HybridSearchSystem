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
#include <unordered_set>

template <typename Vec>
inline void WarmReserveVectorStorage(Vec& vec, size_t target_capacity) {
    if (vec.capacity() < target_capacity) {
        vec.reserve(target_capacity);
    }
    if (target_capacity != 0 && vec.size() < target_capacity) {
        vec.resize(target_capacity);
    }
    vec.clear();
}

class WorkerGroup {
   public:
    WorkerGroup(int group_id, DataBaseCPU* db, Scheduler* sched)
        : group_id_(group_id), db_(db), sched_(sched), use_npu_(db != nullptr && db->UsesNpuVectorCompute()) {
        if (use_npu_) {
            npu_stream_ = npuAPI::GetGroupStream(group_id_);
        }

        prefetch_slot_capacity_floats_[kFirstRoundSlotId] =
            static_cast<size_t>(valid_bucket_num_base_level_2) * static_cast<size_t>(max_process_bucket_num_level_2) *
            static_cast<size_t>(max_doc_per_bucket_level_2) * kScoreCols;
        prefetch_slot_capacity_floats_[kIncrementalSlotAId] =
            static_cast<size_t>(valid_bucket_num_incremental_level_2) *
            static_cast<size_t>(max_process_bucket_num_level_2) * static_cast<size_t>(max_doc_per_bucket_level_2) *
            kScoreCols;
        prefetch_slot_capacity_floats_[kIncrementalSlotBId] = prefetch_slot_capacity_floats_[kIncrementalSlotAId];

        for (int slot_id = 0; slot_id < npuAPI::kGroupFlagSlotCount; ++slot_id) {
            const size_t slot_score_bytes = prefetch_slot_capacity_floats_[slot_id] * sizeof(float);
            if (use_npu_) {
                prefetch_slots_[slot_id].flag = npuAPI::GetGroupFlag(group_id_, slot_id);
                npuAPI::AllocateHostPinned((void**)&prefetch_slots_[slot_id].score_buffer, slot_score_bytes);
                npuAPI::ClearGroupFlag(group_id_, slot_id);
            } else {
                // Pure-CPU mode: score_buffer is consumed by ComputeBatchScoresOnCpu.
                // Allocate plain heap memory (no pinned host needed without NPU DMA).
                void* raw = std::malloc(slot_score_bytes);
                if (raw == nullptr && slot_score_bytes > 0) {
                    throw std::runtime_error("[WorkerGroup] failed to alloc CPU score_buffer");
                }
                prefetch_slots_[slot_id].score_buffer = static_cast<float*>(raw);
                prefetch_slots_[slot_id].owns_plain_buffer = true;
            }
        }

        tls_scratch_pools_.resize(cores_per_group);
        tls_ivf_masks_.resize(cores_per_group);
        tls_batch_masks_.resize(cores_per_group);

        for (int i = 0; i < cores_per_group; ++i) {
            tls_scratch_pools_[i].resize(static_cast<size_t>(group_tls_scratch_pool_reserve_u64));
            tls_ivf_masks_[i].resize(static_cast<size_t>(group_tls_ivf_mask_reserve_u64));
            tls_batch_masks_[i].resize(static_cast<size_t>(group_tls_batch_mask_reserve_u64));
        }

        thread_bucket_results_.resize(cores_per_group);
        thread_doc_results_.resize(cores_per_group);
        for (int i = 0; i < cores_per_group; ++i) {
            WarmReserveVectorStorage(thread_bucket_results_[i],
                                     static_cast<size_t>(group_thread_bucket_results_reserve_items_per_rank));
            WarmReserveVectorStorage(thread_doc_results_[i].items,
                                     static_cast<size_t>(group_thread_doc_results_reserve_items_per_rank));
        }

        // Initialize L0 score cache if DataBaseCPU has L0 mapping
        l0_enabled_ = (db_ != nullptr && db_->HasL0Mapping());
        l0_commit_threshold_ = l0_commit_threshold;
        use_mask_filter_kernel_ = use_npu_;
        if (l0_enabled_) {
            l0_cache_.Init(static_cast<size_t>(total_bucket_num_level_2), static_cast<size_t>(db_->get_l0_count()));
        }

        thread_ivf_step_timings_.resize(cores_per_group);

        WarmReserveVectorStorage(sorted_l2_buckets_, static_cast<size_t>(group_sorted_buckets_reserve_items));
        for (auto& round_plan : round_plan_buffers_) {
            (void)round_plan;
            // mask_storage is allocated per-batch in PrepareRoundMasks
        }
    }

    // 析构函数：释放 Pinned Memory
    ~WorkerGroup() {
        for (int slot_id = 0; slot_id < npuAPI::kGroupFlagSlotCount; ++slot_id) {
            if (prefetch_slots_[slot_id].score_buffer) {
                if (use_npu_) {
                    npuAPI::FreeHostPinned(prefetch_slots_[slot_id].score_buffer);
                } else if (prefetch_slots_[slot_id].owns_plain_buffer) {
                    std::free(prefetch_slots_[slot_id].score_buffer);
                }
                prefetch_slots_[slot_id].score_buffer = nullptr;
            }
        }
        if (pinned_scratch_) {
            npuAPI::FreeHostPinned(pinned_scratch_);
            pinned_scratch_ = nullptr;
        }
        if (mf_scores_buf_) {
            npuAPI::FreeHostPinned(mf_scores_buf_);
            mf_scores_buf_ = nullptr;
        }
        if (mf_indices_buf_) {
            npuAPI::FreeHostPinned(mf_indices_buf_);
            mf_indices_buf_ = nullptr;
        }
        if (mf_counts_buf_) {
            npuAPI::FreeHostPinned(mf_counts_buf_);
            mf_counts_buf_ = nullptr;
        }
    }

    // 【新增】停止标志位
    void Stop() {
        running_ = false;
    }

    // 线程入口
    void Run(int core_id) {
        BindThreadToCore(core_id);

        if (use_npu_) {
            int device_id = npu_device_id_start + g_group_to_device[group_id_];
            auto ret = aclrtSetDevice(device_id);
            if (ret != ACL_SUCCESS) {
                std::cerr << "Worker Thread " << core_id << " aclrtSetDevice failed, code: " << ret << std::endl;
                return;
            }
        }
        bool is_leader = IsLeader(core_id);
        int rank = GetInGroupRank(core_id);
        uint64_t observed_stage_epoch = 0;
        WarmUpThreadLocalBuffers(is_leader);

        // 【修改】循环检查 running_ 标志
        while (running_) {
            if (is_leader) {
                LeaderLoop();
            } else {
                FollowerLoop(rank, observed_stage_epoch);
            }
        }
    }

   private:
    int group_id_;
    DataBaseCPU* db_;
    Scheduler* sched_;
    npuAPI::Stream npu_stream_;

    // 控制线程退出的原子标志，用于安全shut down系统
    std::atomic<bool> running_{true};
    Query* current_query_ = nullptr;

    enum State { IDLE = 0, IVF_FILTER = 1, BATCH_ATTR_FILTER_MASK = 2, BATCH_COLLECT_RESULTS = 3 };
    // 组内 stage 分发信号，低位存 State，高位存单调递增的 epoch。
    // follower 只执行自己尚未消费过的 epoch，避免不同 query / probe round 复用同一 State 时发生 ABA。
    static constexpr uint64_t kStageStateBits = 8;
    static constexpr uint64_t kStageStateMask = (1ULL << kStageStateBits) - 1;
    static constexpr uint64_t PackStageSignal(State state, uint64_t epoch) {
        return (epoch << kStageStateBits) | static_cast<uint64_t>(state);
    }
    static constexpr State UnpackStageState(uint64_t signal) {
        return static_cast<State>(signal & kStageStateMask);
    }
    static constexpr uint64_t UnpackStageEpoch(uint64_t signal) {
        return signal >> kStageStateBits;
    }

    std::atomic<uint64_t> stage_signal_{PackStageSignal(IDLE, 0)};
    uint64_t next_stage_epoch_ = 0;
    std::atomic<int> worker_counter_{0};

    static constexpr size_t kScoreCols = 1;

    struct RoundBatch {
        std::vector<uint32_t> bucket_ids;
        size_t score_offset_floats = 0;
        size_t score_count_floats = 0;
        // Precomputed score offsets for each bucket in the batch (for dynamic work distribution)
        std::vector<size_t> score_offsets_per_bucket;
        // Offset into RoundPlan::mask_storage where this batch's masks start (u64 units)
        size_t mask_offset = 0;
        // L0 hybrid scoring: true = score from L0 cache (GatherL0Scores), false = original LaunchBatchKernel
        bool l0_cached = false;
    };

    struct RoundPlan {
        bool valid = false;
        int l2_count = 0;
        std::vector<RoundBatch> batches;
        // Shared mask storage: all workers write/read masks here
        // Mask for batch position i is at mask_storage[i * mask_stride ... (i+1) * mask_stride)
        std::vector<uint64_t> mask_storage;
        size_t mask_stride = 0;  // u64 per bucket slot
    };

    struct PrefetchSlot {
        volatile uint32_t* flag = nullptr;
        float* score_buffer = nullptr;
        bool owns_plain_buffer = false;  // true when score_buffer was std::malloc'd in CPU mode
    };

    static constexpr int kFirstRoundSlotId = 0;
    static constexpr int kIncrementalSlotAId = 1;
    static constexpr int kIncrementalSlotBId = 2;

    // ============================================================
    // L0 (mesocluster) score cache for reuse across expansion rounds
    // ============================================================
    static constexpr size_t kL2NotScored = ~size_t(0);

    // L0 HBM cache: tracks which L0/L2 buckets have been scored in NPU HBM
    struct L0HbmCache {
        // Per-L0 flag: has this L0 been fully scored (scores in HBM)?
        std::vector<bool> l0_scored;
        // Per-L2 HBM location (offset + doc_count) in dev_l0_cache
        std::vector<npuAPI::L2ScoreLocation> l2_locations;

        void Init(size_t l2_count, size_t l0_count) {
            l0_scored.assign(l0_count, false);
            l2_locations.assign(l2_count, {0, 0});
        }

        bool IsL0Scored(uint32_t l0_id) const {
            return l0_id < l0_scored.size() && l0_scored[l0_id];
        }
    };

    L0HbmCache l0_cache_;
    bool l0_enabled_ = false;           // true if DataBaseCPU has L0 mapping
    float l0_commit_threshold_ = 0.0f;  // ratio threshold for full L0 commit vs per-L2 scoring
    std::vector<float> rearrange_buf_;  // temp buffer for GatherL0Scores D2H output (synced internally)
    float* pinned_scratch_ = nullptr;   // pinned memory for LaunchBatchKernel DMA output (Path B)
    size_t pinned_scratch_floats_ = 0;

    // Mask filter kernel: compact results buffers (pinned memory)
    bool use_mask_filter_kernel_ = false;
    float* mf_scores_buf_ = nullptr;      // compact scores (D2H from mask filter)
    uint32_t* mf_indices_buf_ = nullptr;  // compact local doc indices
    uint32_t* mf_counts_buf_ = nullptr;   // per-bucket match counts
    size_t mf_buf_capacity_ = 0;          // max floats/uint32s in scores/indices buffer
    size_t mf_counts_capacity_ = 0;       // max uint32s in counts buffer

    std::vector<uint32_t> sorted_l2_buckets_;
    uint32_t current_batch_size_ = 0;
    PrefetchSlot prefetch_slots_[npuAPI::kGroupFlagSlotCount];
    size_t prefetch_slot_capacity_floats_[npuAPI::kGroupFlagSlotCount] = {0, 0, 0};
    RoundPlan round_plan_buffers_[3];
    RoundPlan* active_round_ = nullptr;
    RoundBatch* active_batch_ = nullptr;
    float* active_batch_scores_ = nullptr;

    struct CandidateBucket {
        uint32_t bucket_id;
        float score;
    };
    struct IvfStepTiming {
        double filter_eval_ms = 0.0;
        double centroid_score_ms = 0.0;
        double result_pack_ms = 0.0;
    };
    std::vector<std::vector<CandidateBucket>> thread_bucket_results_;
    std::vector<IvfStepTiming> thread_ivf_step_timings_;

    struct ThreadDocResult {
        std::vector<QueryResult::Item> items;
    };
    std::vector<ThreadDocResult> thread_doc_results_;

    std::vector<std::vector<uint64_t, AlignedAllocator<uint64_t>>> tls_scratch_pools_;
    std::vector<std::vector<uint64_t>> tls_ivf_masks_;
    std::vector<std::vector<uint64_t>> tls_batch_masks_;

    // Dynamic work distribution: atomic counter for balanced work across workers
    alignas(64) std::atomic<uint32_t> next_work_idx_{0};

    bool use_npu_ = true;

    static constexpr const char* kTlsIvfMasksLabel = "WorkerGroup.h:tls_ivf_masks_";
    static constexpr const char* kTlsScratchPoolsLabel = "WorkerGroup.h:tls_scratch_pools_";
    static constexpr const char* kExecuteIvfHeapLabel = "WorkerGroup.h:ExecuteIVF.local_pq";
    static constexpr const char* kThreadBucketResultsLabel = "WorkerGroup.h:thread_bucket_results_";
    static constexpr const char* kCandidateMergeBufferLabel = "WorkerGroup.h:CandidateBucketMerge.candidates";
    static constexpr const char* kSortedBucketsLabel = "WorkerGroup.h:sorted_l2_buckets_";
    static constexpr const char* kTlsBatchMasksLabel = "WorkerGroup.h:tls_batch_masks_";
    static constexpr const char* kTempDocMaskLabel = "WorkerGroup.h:temp_doc_mask";
    static constexpr const char* kThreadDocResultsItemsLabel = "WorkerGroup.h:thread_doc_results_.items";
    static constexpr const char* kAllItemsLabel = "WorkerGroup.h:MergeBatchResults.all_items";

    static std::vector<CandidateBucket>& ThreadLocalCandidateHeapStorage() {
        static thread_local std::vector<CandidateBucket> storage;
        return storage;
    }

    static std::vector<CandidateBucket>& ThreadLocalCandidateMergeBuffer() {
        static thread_local std::vector<CandidateBucket> buffer;
        return buffer;
    }

    static std::vector<QueryResult::Item>& ThreadLocalAllItemsBuffer() {
        static thread_local std::vector<QueryResult::Item> buffer;
        return buffer;
    }

    static std::vector<uint64_t>& ThreadLocalTempDocMask() {
        static thread_local std::vector<uint64_t> buffer;
        return buffer;
    }

    static void WarmUpThreadLocalBuffers(bool is_leader) {
        WarmReserveVectorStorage(ThreadLocalCandidateHeapStorage(),
                                 static_cast<size_t>(group_local_pq_reserve_items_per_rank));
        WarmReserveVectorStorage(ThreadLocalTempDocMask(),
                                 static_cast<size_t>(group_temp_doc_mask_reserve_u64_per_rank));
        if (is_leader) {
            WarmReserveVectorStorage(ThreadLocalCandidateMergeBuffer(),
                                     static_cast<size_t>(group_candidate_merge_reserve_items));
            WarmReserveVectorStorage(ThreadLocalAllItemsBuffer(), static_cast<size_t>(group_all_items_reserve_items));
        }
    }

    // --- Leader Logic ---
    void PublishStage(State stage) {
        worker_counter_.store(0, std::memory_order_release);
        next_work_idx_.store(0, std::memory_order_release);
        const uint64_t epoch = ++next_stage_epoch_;
        stage_signal_.store(PackStageSignal(stage, epoch), std::memory_order_release);
    }

    [[noreturn]] void AbortWithInvariantError(const std::string& msg) const {
        std::cerr << "[WorkerGroup] " << msg << " group_id=" << group_id_
                  << " query_id=" << (current_query_ == nullptr ? 0 : current_query_->query_id) << std::endl;
        std::abort();
    }

    void ClearActiveBatchContext() {
        active_round_ = nullptr;
        active_batch_ = nullptr;
        active_batch_scores_ = nullptr;
        current_batch_size_ = 0;
    }

    void ResetRoundPlan(RoundPlan& round) {
        round.valid = false;
        round.l2_count = 0;
        round.batches.clear();
        round.mask_storage.clear();
        round.mask_stride = 0;
    }

    void InitializeQueryPrefetchState() {
        if (use_npu_) {
            for (int slot_id = 0; slot_id < npuAPI::kGroupFlagSlotCount; ++slot_id) {
                npuAPI::ClearGroupFlag(group_id_, slot_id);
            }
        }
        for (auto& round : round_plan_buffers_) {
            ResetRoundPlan(round);
        }
        ClearActiveBatchContext();
        for (int i = 0; i < static_cast<int>(thread_ivf_step_timings_.size()); ++i) {
            thread_ivf_step_timings_[i] = IvfStepTiming{};
        }
        // Reset L0 cache for new query
        if (l0_enabled_) {
            l0_cache_.Init(static_cast<size_t>(total_bucket_num_level_2), static_cast<size_t>(db_->get_l0_count()));
            if (use_npu_) {
                npuAPI::ResetL0CacheOffset(group_id_);
            }
        }
    }

    bool BuildRoundPlan(RoundPlan& round, int round_index, int& l2_offset) {
        ResetRoundPlan(round);

        if (l2_offset >= static_cast<int>(sorted_l2_buckets_.size())) {
            return false;
        }

        const int requested_l2_count =
            (round_index == 0) ? valid_bucket_num_base_level_2 : valid_bucket_num_incremental_level_2;
        const int l2_count = std::min(requested_l2_count, static_cast<int>(sorted_l2_buckets_.size()) - l2_offset);
        if (l2_count <= 0) {
            return false;
        }

        const size_t estimated_batch_count =
            (static_cast<size_t>(l2_count) + static_cast<size_t>(max_process_bucket_num_level_2) - 1) /
            static_cast<size_t>(max_process_bucket_num_level_2);
        round.batches.reserve(estimated_batch_count);

        for (int i = 0; i < l2_count; ++i) {
            const uint32_t l2_id = sorted_l2_buckets_[static_cast<size_t>(l2_offset + i)];

            if (round.batches.empty() ||
                round.batches.back().bucket_ids.size() >= static_cast<size_t>(max_process_bucket_num_level_2)) {
                round.batches.emplace_back();
                RoundBatch& new_batch = round.batches.back();
                new_batch.bucket_ids.reserve(static_cast<size_t>(group_batch_bucket_ids_reserve_items));
            }

            RoundBatch& batch = round.batches.back();
            batch.bucket_ids.push_back(l2_id);

            const Bucket& bucket = db_->get_bucket(l2_id);
            batch.score_count_floats += static_cast<size_t>(bucket.get_doc_num()) * kScoreCols;
        }

        l2_offset += l2_count;
        round.valid = !round.batches.empty();
        round.l2_count = round.valid ? l2_count : 0;
        if (!round.valid) {
            return false;
        }

        // Compute max stride across all buckets for shared mask buffer
        size_t max_stride = 0;
        for (const RoundBatch& batch : round.batches) {
            for (uint32_t bid : batch.bucket_ids) {
                max_stride = std::max(max_stride, static_cast<size_t>(db_->get_bucket(bid).get_stride()));
            }
        }
        round.mask_stride = max_stride;

        // Precompute score offsets and allocate shared mask buffer per batch
        size_t total_mask_words = 0;
        for (RoundBatch& batch : round.batches) {
            const size_t batch_size = batch.bucket_ids.size();
            batch.score_offsets_per_bucket.resize(batch_size);
            size_t score_off = 0;
            for (size_t i = 0; i < batch_size; ++i) {
                batch.score_offsets_per_bucket[i] = score_off;
                score_off += static_cast<size_t>(db_->get_bucket(batch.bucket_ids[i]).get_doc_num()) * kScoreCols;
            }
            batch.mask_offset = total_mask_words;
            total_mask_words += batch_size * max_stride;
        }

        // Allocate shared mask storage for ALL batches
        if (total_mask_words > 0) {
            round.mask_storage.resize(total_mask_words, 0);
        } else {
            round.mask_storage.clear();
        }

        return true;
    }

    // Build a round plan that covers ALL L2 buckets under the given L0 bucket IDs.
    // Used for L0-based NPU scoring: score entire mesoclusters, then cache results.
    bool BuildL0RoundPlan(RoundPlan& round, const std::vector<uint32_t>& l0_ids) {
        ResetRoundPlan(round);
        if (l0_ids.empty())
            return false;

        // Enumerate all L2 buckets under the selected L0 buckets
        size_t total_l2_count = 0;
        for (uint32_t l0_id : l0_ids) {
            total_l2_count += static_cast<size_t>(db_->get_l2_end_for_l0(l0_id) - db_->get_l2_begin_for_l0(l0_id));
        }

        const size_t estimated_batch_count =
            (total_l2_count + static_cast<size_t>(max_process_bucket_num_level_2) - 1) /
            static_cast<size_t>(max_process_bucket_num_level_2);
        round.batches.reserve(estimated_batch_count);

        for (uint32_t l0_id : l0_ids) {
            const uint32_t l2_begin = db_->get_l2_begin_for_l0(l0_id);
            const uint32_t l2_end = db_->get_l2_end_for_l0(l0_id);
            for (uint32_t l2_id = l2_begin; l2_id < l2_end; ++l2_id) {
                if (round.batches.empty() ||
                    round.batches.back().bucket_ids.size() >= static_cast<size_t>(max_process_bucket_num_level_2)) {
                    round.batches.emplace_back();
                    round.batches.back().bucket_ids.reserve(static_cast<size_t>(group_batch_bucket_ids_reserve_items));
                }

                RoundBatch& batch = round.batches.back();
                batch.bucket_ids.push_back(l2_id);
                batch.score_count_floats += static_cast<size_t>(db_->get_bucket(l2_id).get_doc_num()) * kScoreCols;
            }
        }

        round.valid = !round.batches.empty();
        round.l2_count = 0;  // L0 plan doesn't track L2 count

        if (!round.valid)
            return false;

        // Compute max stride and score offsets per batch
        size_t max_stride = 0;
        for (const RoundBatch& batch : round.batches) {
            for (uint32_t bid : batch.bucket_ids) {
                max_stride = std::max(max_stride, static_cast<size_t>(db_->get_bucket(bid).get_stride()));
            }
        }
        round.mask_stride = max_stride;

        size_t total_mask_words = 0;
        for (RoundBatch& batch : round.batches) {
            const size_t batch_size = batch.bucket_ids.size();
            batch.score_offsets_per_bucket.resize(batch_size);
            size_t score_off = 0;
            for (size_t i = 0; i < batch_size; ++i) {
                batch.score_offsets_per_bucket[i] = score_off;
                score_off += static_cast<size_t>(db_->get_bucket(batch.bucket_ids[i]).get_doc_num()) * kScoreCols;
            }
            batch.mask_offset = total_mask_words;
            total_mask_words += batch_size * max_stride;
        }

        if (total_mask_words > 0)
            round.mask_storage.resize(total_mask_words, 0);
        else
            round.mask_storage.clear();

        return true;
    }

    // After NPU completes an L0 round, record which L0s have been scored in HBM.
    void MarkL0Scored(const std::vector<uint32_t>& l0_ids) {
        for (uint32_t l0_id : l0_ids) {
            l0_cache_.l0_scored[l0_id] = true;
        }
    }

    // Collect results from a custom score buffer (not PrefetchSlot).
    // Used for L0-cached scores where no NPU wait is needed.
    void CollectFromScoreBuffer(RoundPlan& round, float* score_base, bool& satisfied) {
        if (!round.valid)
            return;

        for (RoundBatch& batch : round.batches) {
            if (satisfied)
                break;

            active_round_ = &round;
            active_batch_ = &batch;
            active_batch_scores_ = score_base + batch.score_offset_floats;
            current_batch_size_ = static_cast<uint32_t>(batch.bucket_ids.size());

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

    // Find unique L0 bucket IDs for a set of L2 bucket IDs.
    std::vector<uint32_t> GetL0SetForL2Range(int l2_offset, int l2_count) const {
        std::vector<uint32_t> l0_set;
        for (int i = 0; i < l2_count; ++i) {
            uint32_t l2_id = sorted_l2_buckets_[static_cast<size_t>(l2_offset + i)];
            uint32_t l0_id = db_->GetL0ForL2(l2_id);
            if (std::find(l0_set.begin(), l0_set.end(), l0_id) == l0_set.end()) {
                l0_set.push_back(l0_id);
            }
        }
        return l0_set;
    }

    // Split L0 set into "full commit" (above threshold) and "individual L2" (below threshold).
    // selected_l0_set: the L0 IDs currently selected (from sorted_l2_buckets_[offset..offset+count-1])
    struct L0CommitDecision {
        std::vector<uint32_t> full_l0_ids;           // commit entire L0 to NPU
        std::vector<uint32_t> non_threshold_l2_ids;  // L2s in non-threshold L0s
    };

    L0CommitDecision SplitL0ByThreshold(int l2_offset, int l2_count) const {
        L0CommitDecision decision;

        // Build set of selected L2 IDs and map them to L0
        std::unordered_map<uint32_t, uint32_t> l0_selected_count;  // l0_id → count of selected L2s
        for (int i = 0; i < l2_count; ++i) {
            uint32_t l2_id = sorted_l2_buckets_[static_cast<size_t>(l2_offset + i)];
            uint32_t l0_id = db_->GetL0ForL2(l2_id);
            ++l0_selected_count[l0_id];
        }

        for (const auto& [l0_id, selected] : l0_selected_count) {
            uint32_t total_l2_in_l0 = db_->get_l2_end_for_l0(l0_id) - db_->get_l2_begin_for_l0(l0_id);
            float ratio = static_cast<float>(selected) / static_cast<float>(total_l2_in_l0);
            if (ratio >= l0_commit_threshold_) {
                decision.full_l0_ids.push_back(l0_id);
            }
        }

        // Build set of threshold L0s for fast lookup
        std::unordered_set<uint32_t> full_l0_set(decision.full_l0_ids.begin(), decision.full_l0_ids.end());

        // Collect L2s in non-threshold L0s
        for (int i = 0; i < l2_count; ++i) {
            uint32_t l2_id = sorted_l2_buckets_[static_cast<size_t>(l2_offset + i)];
            uint32_t l0_id = db_->GetL0ForL2(l2_id);
            if (full_l0_set.count(l0_id) == 0) {
                decision.non_threshold_l2_ids.push_back(l2_id);
            }
        }

        return decision;
    }

    float* EnsurePinnedScratch(size_t needed_floats) {
        if (needed_floats > pinned_scratch_floats_) {
            if (pinned_scratch_)
                npuAPI::FreeHostPinned(pinned_scratch_);
            npuAPI::AllocateHostPinned(reinterpret_cast<void**>(&pinned_scratch_), needed_floats * sizeof(float));
            pinned_scratch_floats_ = needed_floats;
        }
        return pinned_scratch_;
    }

    void EnsureMaskFilterBuffers(size_t max_floats, size_t max_counts) {
        if (max_floats > mf_buf_capacity_) {
            if (mf_scores_buf_)
                npuAPI::FreeHostPinned(mf_scores_buf_);
            if (mf_indices_buf_)
                npuAPI::FreeHostPinned(mf_indices_buf_);
            npuAPI::AllocateHostPinned(reinterpret_cast<void**>(&mf_scores_buf_), max_floats * sizeof(float));
            npuAPI::AllocateHostPinned(reinterpret_cast<void**>(&mf_indices_buf_), max_floats * sizeof(uint32_t));
            mf_buf_capacity_ = max_floats;
        }
        if (max_counts > mf_counts_capacity_) {
            if (mf_counts_buf_)
                npuAPI::FreeHostPinned(mf_counts_buf_);
            npuAPI::AllocateHostPinned(reinterpret_cast<void**>(&mf_counts_buf_), max_counts * sizeof(uint32_t));
            mf_counts_capacity_ = max_counts;
        }
    }

    bool MaxProbeL2BucketLimitEnabled() const {
        return max_probe_l2_bucket_num_enable == 1;
    }

    bool ReachesMaxProbeL2BucketLimit(int searched_l2_count, const RoundPlan& round) const {
        return MaxProbeL2BucketLimitEnabled() && searched_l2_count + round.l2_count >= max_probe_l2_bucket_num;
    }

    void AccountConsumedRound(const RoundPlan& round, int& searched_l2_count, int& consumed_round_count,
                              int& process_round_count_level_2) {
        ++consumed_round_count;
        searched_l2_count += round.l2_count;
        process_round_count_level_2 += static_cast<int>(round.batches.size());
        int searched_l2_in_round = 0;
        for (const RoundBatch& batch : round.batches) {
            searched_l2_in_round += static_cast<int>(batch.bucket_ids.size());
            current_query_->RecordSearchedLevel2Buckets(batch.bucket_ids);
        }
        current_query_->searched_bucket_count_level_2 += searched_l2_in_round;
    }

    void AssignRoundToSlot(RoundPlan& round, int slot_id) {
        size_t used_floats = 0;
        for (RoundBatch& batch : round.batches) {
            batch.score_offset_floats = used_floats;
            used_floats += batch.score_count_floats;
            if (used_floats > prefetch_slot_capacity_floats_[slot_id]) {
                throw std::runtime_error(
                    "[WorkerGroup] score slot capacity exceeded: slot_id=" + std::to_string(slot_id) +
                    ", required_floats=" + std::to_string(used_floats) +
                    ", capacity_floats=" + std::to_string(prefetch_slot_capacity_floats_[slot_id]));
            }
        }
    }

    void LaunchRoundToSlot(RoundPlan& round, int slot_id) {
        if (!round.valid) {
            return;
        }

        AssignRoundToSlot(round, slot_id);

        auto start = std::chrono::high_resolution_clock::now();
        if (use_npu_) {
            npuAPI::ResetGroupFlag(group_id_, slot_id);
            // Merge all batches into a single launch to eliminate per-batch driver overhead.
            // AssignRoundToSlot computes batch offsets as a cumulative sum:
            //   batch[0].score_offset_floats = 0,
            //   batch[1].score_offset_floats = sum(batch[0].doc_count), ...
            // LaunchBatchKernel's internal offset_C accumulation produces the identical layout.
            std::vector<uint32_t> all_bucket_ids;
            for (const RoundBatch& batch : round.batches) {
                all_bucket_ids.insert(all_bucket_ids.end(), batch.bucket_ids.begin(), batch.bucket_ids.end());
            }
            if (!all_bucket_ids.empty()) {
                float* d2h_target = use_mask_filter_kernel_ ? nullptr : prefetch_slots_[slot_id].score_buffer;
                npuAPI::LaunchBatchKernel(npu_stream_, all_bucket_ids, d2h_target, group_id_);
            }
            // Always enqueue completion after LaunchBatchKernel.
            // When mask filter is active, a second completion is enqueued after mask filter
            // (in LaunchMaskFilterForRound). WaitForSlot waits for the LATEST event.
            npuAPI::EnqueueGroupCompletion(npu_stream_, group_id_, slot_id);
        } else {
            for (const RoundBatch& batch : round.batches) {
                ComputeBatchScoresOnCpu(batch, prefetch_slots_[slot_id].score_buffer + batch.score_offset_floats);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        current_query_->timing_metrics.npu_async_launch_ms += elapsed.count();
    }

    void WaitForSlot(int slot_id) {
        if (!use_npu_) {
            return;
        }
        auto start = std::chrono::high_resolution_clock::now();
        // Pure spin on ACL event completion. This keeps the low-overhead wait strategy
        // from the compact-output path, but removes the extra D2H flag-ready copy from
        // the critical path.
        while (!npuAPI::IsGroupCompletionReady(group_id_, slot_id)) {
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

    void PrepareRoundMasks(RoundPlan& round, int slot_id = -1) {
        auto cpu_start = std::chrono::high_resolution_clock::now();

        if (!current_query_->filter_exp.bucket_plan.valid) {
            // No filter: all masks are 0xFF — skip per-batch barrier syncs entirely.
            // mask_storage was already sized in BuildRoundPlan for all batches.
            if (!round.mask_storage.empty()) {
                std::memset(round.mask_storage.data(), 0xFF, round.mask_storage.size() * sizeof(uint64_t));
            }
        } else {
            for (size_t bi = 0; bi < round.batches.size(); ++bi) {
                RoundBatch& batch = round.batches[bi];
                active_round_ = &round;
                active_batch_ = &batch;
                active_batch_scores_ = nullptr;
                current_batch_size_ = static_cast<uint32_t>(batch.bucket_ids.size());

                // Zero this batch's mask region (allocated in BuildRoundPlan)
                const size_t needed = static_cast<size_t>(current_batch_size_) * round.mask_stride;
                std::memset(round.mask_storage.data() + batch.mask_offset, 0, needed * sizeof(uint64_t));

                PublishStage(BATCH_ATTR_FILTER_MASK);
                InBucketAttrFilter(0);
                WaitFollowers();
            }
        }
        ClearActiveBatchContext();

        auto cpu_end = std::chrono::high_resolution_clock::now();
        current_query_->timing_metrics.inbucket_attr_filter_overlapped_ms +=
            std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

        // If mask filter kernel is enabled, launch it now (after CPU masks are ready)
        if (use_mask_filter_kernel_ && slot_id >= 0) {
            LaunchMaskFilterForRound(round, slot_id);
        }
    }

    // Build score_offsets_bytes and mask_infos for all buckets across all batches in a round.
    // Returns merged bucket_ids, score_offsets_bytes, and mask_infos.
    void BuildMaskFilterParams(const RoundPlan& round, std::vector<uint32_t>& all_bucket_ids,
                               std::vector<size_t>& score_offsets_bytes,
                               std::vector<npuAPI::BucketMaskInfo>& mask_infos) {
        for (const RoundBatch& batch : round.batches) {
            for (size_t i = 0; i < batch.bucket_ids.size(); ++i) {
                uint32_t bid = batch.bucket_ids[i];
                all_bucket_ids.push_back(bid);

                // Score offset: absolute byte offset into d_result_ws
                score_offsets_bytes.push_back((batch.score_offset_floats + batch.score_offsets_per_bucket[i]) *
                                              sizeof(float));

                // Mask offset: position in mask_storage for this bucket
                npuAPI::BucketMaskInfo mi;
                mi.mask_offset_u64 = batch.mask_offset + i * round.mask_stride;
                mi.mask_words = (db_->get_bucket(bid).get_doc_num() + 63) / 64;
                mask_infos.push_back(mi);
            }
        }
    }

    // Launch mask filter kernel after masks are prepared.
    // Writes compact results to mf_scores_buf_/mf_indices_buf_/mf_counts_buf_.
    void LaunchMaskFilterForRound(RoundPlan& round, int slot_id) {
        if (!use_npu_ || !use_mask_filter_kernel_ || round.batches.empty())
            return;

        std::vector<uint32_t> all_bucket_ids;
        std::vector<size_t> score_offsets_bytes;
        std::vector<npuAPI::BucketMaskInfo> mask_infos;
        BuildMaskFilterParams(round, all_bucket_ids, score_offsets_bytes, mask_infos);

        if (all_bucket_ids.empty())
            return;

        // Ensure pinned buffers are large enough
        size_t total_doc_capacity = 0;
        for (uint32_t bid : all_bucket_ids)
            total_doc_capacity += db_->get_bucket(bid).get_doc_num();
        EnsureMaskFilterBuffers(total_doc_capacity, all_bucket_ids.size());

        auto mf_start = std::chrono::high_resolution_clock::now();
        npuAPI::MaskFilterTimingMs mf_timing;
        npuAPI::LaunchMaskFilter(npu_stream_, group_id_, all_bucket_ids, score_offsets_bytes, round.mask_storage.data(),
                                 mask_infos, round.mask_stride, mf_scores_buf_, mf_indices_buf_, mf_counts_buf_,
                                 &mf_timing);
        // LaunchMaskFilter is fully synchronous (syncs + D2H internally).
        // Record a completion event so WaitForSlot won't spin forever.
        npuAPI::EnqueueGroupCompletion(npu_stream_, group_id_, slot_id);
        auto mf_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> mf_elapsed = mf_end - mf_start;
        current_query_->timing_metrics.npu_mask_filter_launch_ms += mf_elapsed.count();
        current_query_->timing_metrics.npu_mask_filter_h2d_ms += mf_timing.h2d_ms;
        current_query_->timing_metrics.npu_mask_filter_kernel_exec_ms += mf_timing.kernel_exec_ms;
        current_query_->timing_metrics.npu_mask_filter_d2h_ms += mf_timing.d2h_ms;
    }

    // Collect results from mask filter kernel compact output.
    // Reads per-bucket counts, scores, and local doc indices.
    // Maps local indices to global IDs via bucket.get_global_ids().
    void CollectMaskFilterResults(RoundPlan& round, bool& satisfied) {
        auto& topk = current_query_->result.topk_results;
        const int target_k = current_query_->expanded_k;

        size_t global_bucket_idx = 0;
        size_t score_offset = 0;

        for (const RoundBatch& batch : round.batches) {
            if (satisfied)
                break;

            auto coll_start = std::chrono::high_resolution_clock::now();

            for (size_t i = 0; i < batch.bucket_ids.size(); ++i) {
                uint32_t bid = batch.bucket_ids[i];
                const Bucket& bucket = db_->get_bucket(bid);
                const auto& gids = bucket.get_global_ids();
                int doc_num = bucket.get_doc_num();

                uint32_t count = mf_counts_buf_[global_bucket_idx];

                global_bucket_idx++;
                current_query_->RecordBucketMaskStats(static_cast<int>(count), doc_num);

                for (uint32_t m = 0; m < count; ++m) {
                    float score = mf_scores_buf_[score_offset + m];
                    uint32_t local_idx = mf_indices_buf_[score_offset + m];

                    if (local_idx < static_cast<uint32_t>(doc_num)) {
                        topk.emplace_back(gids[local_idx], score);
                    }
                }
                score_offset += doc_num;
            }

            current_query_->timing_metrics.result_collection_ms +=
                std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - coll_start)
                    .count();

            auto merge_start = std::chrono::high_resolution_clock::now();
            QueryResult::KeepTopK(topk, target_k);
            satisfied = current_query_->result.check_satisfaction(target_k);
            current_query_->timing_metrics.final_merge_ms +=
                std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - merge_start)
                    .count();
        }
    }

    void CollectRoundResults(RoundPlan& round, int slot_id, bool& satisfied) {
        if (!round.valid) {
            return;
        }

        WaitForSlot(slot_id);

        if (use_mask_filter_kernel_ && use_npu_) {
            // Mask filter path: read compact results from mf_scores_buf_/mf_indices_buf_/mf_counts_buf_
            CollectMaskFilterResults(round, satisfied);
            return;
        }

        for (RoundBatch& batch : round.batches) {
            if (satisfied) {
                break;
            }

            active_round_ = &round;
            active_batch_ = &batch;
            active_batch_scores_ = prefetch_slots_[slot_id].score_buffer + batch.score_offset_floats;
            current_batch_size_ = static_cast<uint32_t>(batch.bucket_ids.size());

            if (use_npu_) {
                npuAPI::DebugVerifyBatchResults(batch.bucket_ids, current_query_->query_vector.data(),
                                                active_batch_scores_, group_id_);
            }

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

    void LeaderLoop() {
        // 1. Fetch Query
        current_query_ = sched_->Pop();
        if (!current_query_) {
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
        double max_filter_eval_ms = 0.0;
        double max_centroid_score_ms = 0.0;
        double max_result_pack_ms = 0.0;
        for (const IvfStepTiming& timing : thread_ivf_step_timings_) {
            max_filter_eval_ms = std::max(max_filter_eval_ms, timing.filter_eval_ms);
            max_centroid_score_ms = std::max(max_centroid_score_ms, timing.centroid_score_ms);
            max_result_pack_ms = std::max(max_result_pack_ms, timing.result_pack_ms);
        }
        current_query_->timing_metrics.bucket_level_ivf_filter_eval_ms = max_filter_eval_ms;
        current_query_->timing_metrics.bucket_level_ivf_centroid_score_ms = max_centroid_score_ms;
        current_query_->timing_metrics.bucket_level_ivf_result_pack_ms = max_result_pack_ms;

        start = std::chrono::high_resolution_clock::now();
        CandidateBucketMerge();
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        current_query_->timing_metrics.candidate_bucket_merge_ms = elapsed.count();

        // 3. Stage 2 — Upload query to NPU once per query
        if (use_npu_) {
            npuAPI::UploadQuery(current_query_->query_vector.data(),
                                current_query_->mmad_query_padded_fp16.empty()
                                    ? nullptr
                                    : current_query_->mmad_query_padded_fp16.data(),
                                group_id_);
        }

        int l2_offset = 0;
        int next_round_index = 0;
        bool satisfied = false;
        int searched_l2_count = 0;
        int consumed_round_count = 0;
        int process_round_count_level_2 = 0;

        RoundPlan* first_round = &round_plan_buffers_[0];
        RoundPlan* current_round = &round_plan_buffers_[1];
        RoundPlan* future_round = &round_plan_buffers_[2];

        // Decide: use L0 scoring for first round, or original path?
        // L0 scoring only when: enabled, has filter, first round has L2 buckets.
        const bool use_l0_first_round =
            l0_enabled_ && current_query_->filter_exp.bucket_plan.valid && !sorted_l2_buckets_.empty();

        if (use_l0_first_round) {
            // === L0-optimized first round ===
            if (BuildRoundPlan(*first_round, next_round_index, l2_offset)) {
                ++next_round_index;

                // Split L0s by commit threshold
                const int fr_l2_start = l2_offset - first_round->l2_count;
                L0CommitDecision decision = SplitL0ByThreshold(fr_l2_start, first_round->l2_count);

                // If no L0 passes threshold, fall through to original path
                if (decision.full_l0_ids.empty()) {
                    // No L0 worth committing → use original LaunchRoundToSlot
                    // (same as non-L0 path, which is known to work correctly)
                    LaunchRoundToSlot(*first_round, kFirstRoundSlotId);

                    PrepareRoundMasks(*first_round, kFirstRoundSlotId);
                    CollectRoundResults(*first_round, kFirstRoundSlotId, satisfied);
                    AccountConsumedRound(*first_round, searched_l2_count, consumed_round_count,
                                         process_round_count_level_2);
                    if (!satisfied && MaxProbeL2BucketLimitEnabled() && searched_l2_count >= max_probe_l2_bucket_num) {
                        satisfied = true;
                    }
                    ResetRoundPlan(*first_round);
                } else {
                    // Dual-path: threshold L0s → commit entire L0 + GatherL0Scores
                    //            non-threshold L2s → LaunchBatchKernel (L2 granularity)
                    // Per-L2 split avoids cross-L0 batch boundary bugs.

                    AssignRoundToSlot(*first_round, kFirstRoundSlotId);
                    std::unordered_set<uint32_t> full_l0_set(decision.full_l0_ids.begin(), decision.full_l0_ids.end());

                    // Per-L2 classification
                    std::vector<uint32_t> path_a_l2_ids;  // threshold L0 → GatherL0Scores
                    std::vector<uint32_t> path_b_l2_ids;  // non-threshold → LaunchBatchKernel
                    size_t path_a_floats = 0, path_b_floats = 0;
                    // batch_l2_is_a[bi][li] = true if L2 belongs to a threshold L0
                    std::vector<std::vector<bool>> batch_l2_is_a(first_round->batches.size());

                    for (size_t bi = 0; bi < first_round->batches.size(); ++bi) {
                        const RoundBatch& batch = first_round->batches[bi];
                        batch_l2_is_a[bi].resize(batch.bucket_ids.size());
                        for (size_t li = 0; li < batch.bucket_ids.size(); ++li) {
                            uint32_t l2_id = batch.bucket_ids[li];
                            uint32_t l0_id = db_->GetL0ForL2(l2_id);
                            bool is_a = full_l0_set.count(l0_id) > 0;
                            batch_l2_is_a[bi][li] = is_a;
                            size_t fl = static_cast<size_t>(db_->get_bucket(l2_id).get_doc_num()) * kScoreCols;
                            if (is_a) {
                                path_a_l2_ids.push_back(l2_id);
                                path_a_floats += fl;
                            } else {
                                path_b_l2_ids.push_back(l2_id);
                                path_b_floats += fl;
                            }
                        }
                    }

                    auto l0_launch_start = std::chrono::high_resolution_clock::now();
                    if (use_npu_)
                        npuAPI::ResetGroupFlag(group_id_, kFirstRoundSlotId);

                    // Build unified offset map: each L2 gets its absolute position
                    // in d_result_ws matching the score_buffer layout.
                    // absolute_offset = batch.score_offset_floats + batch.score_offsets_per_bucket[i]
                    std::vector<size_t> l2_unified_offsets(
                        first_round->batches.size() == 0
                            ? 0
                            : std::accumulate(first_round->batches.begin(), first_round->batches.end(), size_t(0),
                                              [](size_t s, const RoundBatch& b) { return s + b.bucket_ids.size(); }));
                    {
                        size_t idx = 0;
                        for (size_t bi = 0; bi < first_round->batches.size(); ++bi)
                            for (size_t li = 0; li < first_round->batches[bi].bucket_ids.size(); ++li)
                                l2_unified_offsets[idx++] = (first_round->batches[bi].score_offset_floats +
                                                             first_round->batches[bi].score_offsets_per_bucket[li]) *
                                                            sizeof(float);
                    }

                    // Build Path A dst offsets and Path B offsets in unified layout
                    std::vector<size_t> path_a_dst_offsets;
                    std::vector<size_t> path_b_offsets_bytes;
                    std::vector<uint32_t> path_b_indices;  // index into l2_unified_offsets
                    {
                        size_t idx = 0;
                        for (size_t bi = 0; bi < first_round->batches.size(); ++bi)
                            for (size_t li = 0; li < first_round->batches[bi].bucket_ids.size(); ++li, ++idx) {
                                if (batch_l2_is_a[bi][li])
                                    path_a_dst_offsets.push_back(l2_unified_offsets[idx]);
                                else {
                                    path_b_offsets_bytes.push_back(l2_unified_offsets[idx]);
                                    path_b_indices.push_back(idx);
                                }
                            }
                    }

                    if (use_mask_filter_kernel_ && use_npu_) {
                        // === Mask filter path: both paths write to d_result_ws, no D2H+sync ===

                        // Path A: commit threshold L0s → dev_l0_cache → D2D gather to d_result_ws
                        if (!path_a_l2_ids.empty()) {
                            RoundPlan l0_round;
                            if (BuildL0RoundPlan(l0_round, decision.full_l0_ids)) {
                                std::vector<uint32_t> l0_plan_l2_ids;
                                for (const RoundBatch& l0_batch : l0_round.batches)
                                    l0_plan_l2_ids.insert(l0_plan_l2_ids.end(), l0_batch.bucket_ids.begin(),
                                                          l0_batch.bucket_ids.end());
                                npuAPI::LaunchL0BatchKernel(npu_stream_, l0_plan_l2_ids, group_id_,
                                                            l0_cache_.l2_locations);

                                // D2D gather to d_result_ws at unified offsets, no D2H
                                npuAPI::GatherTimingMs gather_timing;
                                npuAPI::GatherL0Scores(npu_stream_, group_id_, path_a_l2_ids, l0_cache_.l2_locations,
                                                       nullptr, &gather_timing, &path_a_dst_offsets);

                                MarkL0Scored(decision.full_l0_ids);
                            }
                        }

                        // Path B: non-threshold L2s → LaunchBatchKernel at unified offsets
                        if (!path_b_l2_ids.empty()) {
                            npuAPI::LaunchBatchKernel(npu_stream_, path_b_l2_ids, nullptr, group_id_,
                                                      &path_b_offsets_bytes);
                        }
                    } else {
                        // === Original dual-path: D2H+sync + scatter to score_buffer ===

                        // Path A: commit threshold L0s → dev_l0_cache, gather scores to host
                        if (!path_a_l2_ids.empty()) {
                            RoundPlan l0_round;
                            if (BuildL0RoundPlan(l0_round, decision.full_l0_ids)) {
                                if (use_npu_) {
                                    std::vector<uint32_t> l0_plan_l2_ids;
                                    for (const RoundBatch& l0_batch : l0_round.batches)
                                        l0_plan_l2_ids.insert(l0_plan_l2_ids.end(), l0_batch.bucket_ids.begin(),
                                                              l0_batch.bucket_ids.end());
                                    npuAPI::LaunchL0BatchKernel(npu_stream_, l0_plan_l2_ids, group_id_,
                                                                l0_cache_.l2_locations);

                                    if (path_a_floats > rearrange_buf_.size())
                                        rearrange_buf_.resize(path_a_floats);

                                    npuAPI::GatherTimingMs gather_timing;
                                    npuAPI::GatherL0Scores(npu_stream_, group_id_, path_a_l2_ids,
                                                           l0_cache_.l2_locations, rearrange_buf_.data(),
                                                           &gather_timing);

                                } else {
                                    if (path_a_floats > rearrange_buf_.size())
                                        rearrange_buf_.resize(path_a_floats);
                                    size_t off = 0;
                                    for (size_t bi = 0; bi < first_round->batches.size(); ++bi) {
                                        const RoundBatch& batch = first_round->batches[bi];
                                        for (size_t li = 0; li < batch.bucket_ids.size(); ++li) {
                                            if (batch_l2_is_a[bi][li]) {
                                                RoundBatch single_l2;
                                                single_l2.bucket_ids = {batch.bucket_ids[li]};
                                                single_l2.score_offsets_per_bucket = {0};
                                                single_l2.score_count_floats =
                                                    static_cast<size_t>(
                                                        db_->get_bucket(batch.bucket_ids[li]).get_doc_num()) *
                                                    kScoreCols;
                                                ComputeBatchScoresOnCpu(single_l2, rearrange_buf_.data() + off);
                                                off += single_l2.score_count_floats;
                                            }
                                        }
                                    }
                                }
                                MarkL0Scored(decision.full_l0_ids);
                            }
                        }

                        // Path B: non-threshold L2s → LaunchBatchKernel
                        if (!path_b_l2_ids.empty()) {
                            if (use_npu_) {
                                EnsurePinnedScratch(path_b_floats);
                                npuAPI::LaunchBatchKernel(npu_stream_, path_b_l2_ids, pinned_scratch_, group_id_);
                            } else {
                                if (path_b_floats > 0 && path_b_l2_ids.size() > 0) {
                                    size_t total = path_a_floats + path_b_floats;
                                    if (total > rearrange_buf_.size())
                                        rearrange_buf_.resize(total);
                                    size_t off = path_a_floats;
                                    for (size_t bi = 0; bi < first_round->batches.size(); ++bi) {
                                        const RoundBatch& batch = first_round->batches[bi];
                                        for (size_t li = 0; li < batch.bucket_ids.size(); ++li) {
                                            if (!batch_l2_is_a[bi][li]) {
                                                RoundBatch single_l2;
                                                single_l2.bucket_ids = {batch.bucket_ids[li]};
                                                single_l2.score_offsets_per_bucket = {0};
                                                single_l2.score_count_floats =
                                                    static_cast<size_t>(
                                                        db_->get_bucket(batch.bucket_ids[li]).get_doc_num()) *
                                                    kScoreCols;
                                                ComputeBatchScoresOnCpu(single_l2, rearrange_buf_.data() + off);
                                                off += single_l2.score_count_floats;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    auto l0_launch_end = std::chrono::high_resolution_clock::now();
                    current_query_->timing_metrics.npu_async_launch_ms +=
                        std::chrono::duration<double, std::milli>(l0_launch_end - l0_launch_start).count();

                    if (use_mask_filter_kernel_ && use_npu_) {
                        // Mask filter path: masks → mask filter kernel → compact results
                        PrepareRoundMasks(*first_round, kFirstRoundSlotId);
                        WaitForSlot(kFirstRoundSlotId);
                        CollectMaskFilterResults(*first_round, satisfied);
                    } else {
                        // Original path: D2H+sync → scatter to score_buffer → CollectFromScoreBuffer
                        if (use_npu_ && (!path_a_l2_ids.empty() || !path_b_l2_ids.empty()))
                            npuAPI::EnqueueGroupCompletion(npu_stream_, group_id_, kFirstRoundSlotId);

                        PrepareRoundMasks(*first_round);

                        if (use_npu_)
                            WaitForSlot(kFirstRoundSlotId);

                        // Scatter per-L2 into score_buffer at correct offsets
                        float* slot_base = prefetch_slots_[kFirstRoundSlotId].score_buffer;
                        size_t a_off = 0, b_off = 0;
                        for (size_t bi = 0; bi < first_round->batches.size(); ++bi) {
                            const RoundBatch& batch = first_round->batches[bi];
                            for (size_t li = 0; li < batch.bucket_ids.size(); ++li) {
                                size_t fl = static_cast<size_t>(db_->get_bucket(batch.bucket_ids[li]).get_doc_num()) *
                                            kScoreCols;
                                size_t dst = batch.score_offset_floats + batch.score_offsets_per_bucket[li];
                                if (batch_l2_is_a[bi][li]) {
                                    std::memcpy(slot_base + dst, rearrange_buf_.data() + a_off, fl * sizeof(float));
                                    a_off += fl;
                                } else {
                                    const float* src =
                                        use_npu_ ? pinned_scratch_ : rearrange_buf_.data() + path_a_floats;
                                    std::memcpy(slot_base + dst, src + b_off, fl * sizeof(float));
                                    b_off += fl;
                                }
                            }
                        }

                        CollectFromScoreBuffer(*first_round, slot_base, satisfied);
                    }

                    AccountConsumedRound(*first_round, searched_l2_count, consumed_round_count,
                                         process_round_count_level_2);
                    if (!satisfied && MaxProbeL2BucketLimitEnabled() && searched_l2_count >= max_probe_l2_bucket_num) {
                        satisfied = true;
                    }
                    ResetRoundPlan(*first_round);
                }  // else (dual-path L0 scoring)
            }
        } else {
            // === Original first round path ===
            if (BuildRoundPlan(*first_round, next_round_index, l2_offset)) {
                ++next_round_index;
                LaunchRoundToSlot(*first_round, kFirstRoundSlotId);

                int pre_slot_id = kIncrementalSlotAId;
                bool has_pre_round = false;
                // When mask filter is active, skip pre-round pipeline to avoid d_result_ws overwrite
                if (!use_mask_filter_kernel_ && current_query_->filter_exp.bucket_plan.valid &&
                    !ReachesMaxProbeL2BucketLimit(searched_l2_count, *first_round)) {
                    has_pre_round = !satisfied && BuildRoundPlan(*current_round, next_round_index, l2_offset);
                    if (has_pre_round) {
                        ++next_round_index;
                        LaunchRoundToSlot(*current_round, pre_slot_id);
                    }
                }

                PrepareRoundMasks(*first_round, kFirstRoundSlotId);
                CollectRoundResults(*first_round, kFirstRoundSlotId, satisfied);
                AccountConsumedRound(*first_round, searched_l2_count, consumed_round_count,
                                     process_round_count_level_2);
                if (!satisfied && MaxProbeL2BucketLimitEnabled() && searched_l2_count >= max_probe_l2_bucket_num) {
                    satisfied = true;
                }
                ResetRoundPlan(*first_round);

                if (has_pre_round && !satisfied) {
                    PrepareRoundMasks(*current_round, pre_slot_id);
                    CollectRoundResults(*current_round, pre_slot_id, satisfied);
                    AccountConsumedRound(*current_round, searched_l2_count, consumed_round_count,
                                         process_round_count_level_2);
                    if (!satisfied && MaxProbeL2BucketLimitEnabled() && searched_l2_count >= max_probe_l2_bucket_num) {
                        satisfied = true;
                    }
                    ResetRoundPlan(*current_round);
                }
            }
        }

        int current_slot_id = kIncrementalSlotAId;
        int future_slot_id = kIncrementalSlotBId;
        bool has_current_round = false;

        // For incremental rounds with L0 cache: score uncached L0s then use cache
        // Skip when mask filter kernel is active (unified d_result_ws path)
        if (!satisfied && l0_enabled_ && current_query_->filter_exp.bucket_plan.valid) {
            while (!satisfied && l2_offset < static_cast<int>(sorted_l2_buckets_.size())) {
                if (MaxProbeL2BucketLimitEnabled() && searched_l2_count >= max_probe_l2_bucket_num) {
                    satisfied = true;
                    break;
                }

                // Build a normal incremental round plan for next batch of L2 buckets
                const int l2_start = l2_offset;
                RoundPlan inc_round;
                if (!BuildRoundPlan(inc_round, next_round_index, l2_offset))
                    break;
                ++next_round_index;

                // Get ALL L0 IDs in this round
                std::vector<uint32_t> l0_set = GetL0SetForL2Range(l2_start, inc_round.l2_count);

                // Separate into uncached (need scoring) and cached (already in dev_l0_cache)
                std::vector<uint32_t> uncached_l0s;
                std::unordered_set<uint32_t> cached_l0_set;
                for (uint32_t l0_id : l0_set) {
                    if (!l0_cache_.IsL0Scored(l0_id))
                        uncached_l0s.push_back(l0_id);
                    else
                        cached_l0_set.insert(l0_id);
                }

                // If no L0 has been cached yet and no uncached L0s exist, use original path
                if (uncached_l0s.empty() && cached_l0_set.empty()) {
                    LaunchRoundToSlot(inc_round, current_slot_id);
                    PrepareRoundMasks(inc_round, current_slot_id);
                    CollectRoundResults(inc_round, current_slot_id, satisfied);
                    AccountConsumedRound(inc_round, searched_l2_count, consumed_round_count,
                                         process_round_count_level_2);
                    if (!satisfied && MaxProbeL2BucketLimitEnabled() && searched_l2_count >= max_probe_l2_bucket_num) {
                        satisfied = true;
                    }
                    continue;
                }

                // Among uncached L0s, split by threshold
                L0CommitDecision inc_decision = SplitL0ByThreshold(l2_start, inc_round.l2_count);
                std::vector<uint32_t> uncached_full_l0s;
                for (uint32_t l0_id : inc_decision.full_l0_ids) {
                    if (!l0_cache_.IsL0Scored(l0_id))
                        uncached_full_l0s.push_back(l0_id);
                }
                // L0s available for GatherL0Scores: cached + threshold-passing uncached
                std::unordered_set<uint32_t> gather_l0_set(cached_l0_set.begin(), cached_l0_set.end());
                gather_l0_set.insert(uncached_full_l0s.begin(), uncached_full_l0s.end());

                // If no L0 is available for gather and no cached L0s, use original path
                if (gather_l0_set.empty()) {
                    LaunchRoundToSlot(inc_round, current_slot_id);
                    PrepareRoundMasks(inc_round, current_slot_id);
                    CollectRoundResults(inc_round, current_slot_id, satisfied);
                    AccountConsumedRound(inc_round, searched_l2_count, consumed_round_count,
                                         process_round_count_level_2);
                    if (!satisfied && MaxProbeL2BucketLimitEnabled() && searched_l2_count >= max_probe_l2_bucket_num) {
                        satisfied = true;
                    }
                    continue;
                }

                AssignRoundToSlot(inc_round, current_slot_id);

                // Per-L2 classification
                std::vector<uint32_t> path_a_l2_ids;  // in cached or threshold L0 → GatherL0Scores
                std::vector<uint32_t> path_b_l2_ids;  // in non-threshold uncached L0 → LaunchBatchKernel
                size_t path_a_floats = 0, path_b_floats = 0;
                std::vector<std::vector<bool>> batch_l2_is_a(inc_round.batches.size());

                for (size_t bi = 0; bi < inc_round.batches.size(); ++bi) {
                    const RoundBatch& batch = inc_round.batches[bi];
                    batch_l2_is_a[bi].resize(batch.bucket_ids.size());
                    for (size_t li = 0; li < batch.bucket_ids.size(); ++li) {
                        uint32_t l2_id = batch.bucket_ids[li];
                        uint32_t l0_id = db_->GetL0ForL2(l2_id);
                        bool is_a = gather_l0_set.count(l0_id) > 0;
                        batch_l2_is_a[bi][li] = is_a;
                        size_t fl = static_cast<size_t>(db_->get_bucket(l2_id).get_doc_num()) * kScoreCols;
                        if (is_a) {
                            path_a_l2_ids.push_back(l2_id);
                            path_a_floats += fl;
                        } else {
                            path_b_l2_ids.push_back(l2_id);
                            path_b_floats += fl;
                        }
                    }
                }

                auto inc_l0_start = std::chrono::high_resolution_clock::now();
                if (use_npu_)
                    npuAPI::ResetGroupFlag(group_id_, current_slot_id);

                // Build unified offset map for d_result_ws
                std::vector<size_t> inc_l2_unified_offsets(
                    std::accumulate(inc_round.batches.begin(), inc_round.batches.end(), size_t(0),
                                    [](size_t s, const RoundBatch& b) { return s + b.bucket_ids.size(); }));
                {
                    size_t idx = 0;
                    for (size_t bi = 0; bi < inc_round.batches.size(); ++bi)
                        for (size_t li = 0; li < inc_round.batches[bi].bucket_ids.size(); ++li)
                            inc_l2_unified_offsets[idx++] = (inc_round.batches[bi].score_offset_floats +
                                                             inc_round.batches[bi].score_offsets_per_bucket[li]) *
                                                            sizeof(float);
                }

                std::vector<size_t> inc_path_a_dst_offsets;
                std::vector<size_t> inc_path_b_offsets_bytes;
                {
                    size_t idx = 0;
                    for (size_t bi = 0; bi < inc_round.batches.size(); ++bi)
                        for (size_t li = 0; li < inc_round.batches[bi].bucket_ids.size(); ++li, ++idx) {
                            if (batch_l2_is_a[bi][li])
                                inc_path_a_dst_offsets.push_back(inc_l2_unified_offsets[idx]);
                            else
                                inc_path_b_offsets_bytes.push_back(inc_l2_unified_offsets[idx]);
                        }
                }

                // Score uncached threshold L0s → dev_l0_cache
                if (!uncached_full_l0s.empty()) {
                    RoundPlan l0_inc_round;
                    if (BuildL0RoundPlan(l0_inc_round, uncached_full_l0s)) {
                        if (use_npu_) {
                            std::vector<uint32_t> l0_plan_l2_ids;
                            for (const RoundBatch& l0_batch : l0_inc_round.batches) {
                                l0_plan_l2_ids.insert(l0_plan_l2_ids.end(), l0_batch.bucket_ids.begin(),
                                                      l0_batch.bucket_ids.end());
                            }
                            npuAPI::LaunchL0BatchKernel(npu_stream_, l0_plan_l2_ids, group_id_, l0_cache_.l2_locations);
                        }
                        MarkL0Scored(uncached_full_l0s);
                    }
                }

                if (use_mask_filter_kernel_ && use_npu_) {
                    // === Mask filter path: both paths write to d_result_ws ===

                    // Path A: Gather scores from cache → D2D to d_result_ws at unified offsets
                    if (!path_a_l2_ids.empty()) {
                        npuAPI::GatherTimingMs gather_timing;
                        npuAPI::GatherL0Scores(npu_stream_, group_id_, path_a_l2_ids, l0_cache_.l2_locations, nullptr,
                                               &gather_timing, &inc_path_a_dst_offsets);
                    }

                    // Path B: non-threshold L2s → LaunchBatchKernel at unified offsets
                    if (!path_b_l2_ids.empty()) {
                        npuAPI::LaunchBatchKernel(npu_stream_, path_b_l2_ids, nullptr, group_id_,
                                                  &inc_path_b_offsets_bytes);
                    }
                } else {
                    // === Original dual-path: D2H+sync + scatter ===

                    // Path A: Gather scores from cache → D2H to host
                    if (!path_a_l2_ids.empty()) {
                        if (use_npu_) {
                            if (path_a_floats > rearrange_buf_.size())
                                rearrange_buf_.resize(path_a_floats);

                            npuAPI::GatherTimingMs gather_timing;
                            npuAPI::GatherL0Scores(npu_stream_, group_id_, path_a_l2_ids, l0_cache_.l2_locations,
                                                   rearrange_buf_.data(), &gather_timing);

                        } else {
                            if (path_a_floats > rearrange_buf_.size())
                                rearrange_buf_.resize(path_a_floats);
                            size_t off = 0;
                            for (size_t bi = 0; bi < inc_round.batches.size(); ++bi) {
                                const RoundBatch& batch = inc_round.batches[bi];
                                for (size_t li = 0; li < batch.bucket_ids.size(); ++li) {
                                    if (batch_l2_is_a[bi][li]) {
                                        RoundBatch single_l2;
                                        single_l2.bucket_ids = {batch.bucket_ids[li]};
                                        single_l2.score_offsets_per_bucket = {0};
                                        single_l2.score_count_floats =
                                            static_cast<size_t>(db_->get_bucket(batch.bucket_ids[li]).get_doc_num()) *
                                            kScoreCols;
                                        ComputeBatchScoresOnCpu(single_l2, rearrange_buf_.data() + off);
                                        off += single_l2.score_count_floats;
                                    }
                                }
                            }
                        }
                    }

                    // Path B: non-threshold uncached L0s → LaunchBatchKernel
                    if (!path_b_l2_ids.empty()) {
                        if (use_npu_) {
                            EnsurePinnedScratch(path_b_floats);
                            npuAPI::LaunchBatchKernel(npu_stream_, path_b_l2_ids, pinned_scratch_, group_id_);
                        } else {
                            size_t total = path_a_floats + path_b_floats;
                            if (total > rearrange_buf_.size())
                                rearrange_buf_.resize(total);
                            size_t off = path_a_floats;
                            for (size_t bi = 0; bi < inc_round.batches.size(); ++bi) {
                                const RoundBatch& batch = inc_round.batches[bi];
                                for (size_t li = 0; li < batch.bucket_ids.size(); ++li) {
                                    if (!batch_l2_is_a[bi][li]) {
                                        RoundBatch single_l2;
                                        single_l2.bucket_ids = {batch.bucket_ids[li]};
                                        single_l2.score_offsets_per_bucket = {0};
                                        single_l2.score_count_floats =
                                            static_cast<size_t>(db_->get_bucket(batch.bucket_ids[li]).get_doc_num()) *
                                            kScoreCols;
                                        ComputeBatchScoresOnCpu(single_l2, rearrange_buf_.data() + off);
                                        off += single_l2.score_count_floats;
                                    }
                                }
                            }
                        }
                    }
                }

                auto inc_l0_end = std::chrono::high_resolution_clock::now();
                current_query_->timing_metrics.npu_async_launch_ms +=
                    std::chrono::duration<double, std::milli>(inc_l0_end - inc_l0_start).count();

                if (use_mask_filter_kernel_ && use_npu_) {
                    // Mask filter path: masks → mask filter → compact results
                    PrepareRoundMasks(inc_round, current_slot_id);
                    WaitForSlot(current_slot_id);
                    CollectMaskFilterResults(inc_round, satisfied);
                } else {
                    // Original path: D2H+sync → scatter → CollectFromScoreBuffer
                    if (use_npu_ && (!path_a_l2_ids.empty() || !path_b_l2_ids.empty()))
                        npuAPI::EnqueueGroupCompletion(npu_stream_, group_id_, current_slot_id);

                    PrepareRoundMasks(inc_round);

                    if (use_npu_)
                        WaitForSlot(current_slot_id);

                    float* slot_buf = prefetch_slots_[current_slot_id].score_buffer;
                    size_t a_off = 0, b_off = 0;
                    for (size_t bi = 0; bi < inc_round.batches.size(); ++bi) {
                        const RoundBatch& batch = inc_round.batches[bi];
                        for (size_t li = 0; li < batch.bucket_ids.size(); ++li) {
                            size_t fl =
                                static_cast<size_t>(db_->get_bucket(batch.bucket_ids[li]).get_doc_num()) * kScoreCols;
                            size_t dst = batch.score_offset_floats + batch.score_offsets_per_bucket[li];
                            if (batch_l2_is_a[bi][li]) {
                                std::memcpy(slot_buf + dst, rearrange_buf_.data() + a_off, fl * sizeof(float));
                                a_off += fl;
                            } else {
                                const float* src = use_npu_ ? pinned_scratch_ : rearrange_buf_.data() + path_a_floats;
                                std::memcpy(slot_buf + dst, src + b_off, fl * sizeof(float));
                                b_off += fl;
                            }
                        }
                    }

                    CollectFromScoreBuffer(inc_round, slot_buf, satisfied);
                }
                AccountConsumedRound(inc_round, searched_l2_count, consumed_round_count, process_round_count_level_2);
                if (!satisfied && MaxProbeL2BucketLimitEnabled() && searched_l2_count >= max_probe_l2_bucket_num) {
                    satisfied = true;
                }
            }
        } else {
            // Original incremental round path (no L0 cache or no filter)
            has_current_round = false;
            if (!satisfied && BuildRoundPlan(*current_round, next_round_index, l2_offset)) {
                ++next_round_index;
                LaunchRoundToSlot(*current_round, current_slot_id);
                has_current_round = true;
            }

            int round_iter = 0;
            while (!satisfied && has_current_round) {
                // When mask filter kernel is active, scores stay in d_result_ws (no D2H).
                // A future_round launch would overwrite d_result_ws before mask filter reads it.
                // Disable pipeline pre-launch when mask filter is active.
                bool has_future_round = false;
                if (!use_mask_filter_kernel_ && !ReachesMaxProbeL2BucketLimit(searched_l2_count, *current_round) &&
                    BuildRoundPlan(*future_round, next_round_index, l2_offset)) {
                    ++next_round_index;
                    LaunchRoundToSlot(*future_round, future_slot_id);
                    has_future_round = true;
                }

                PrepareRoundMasks(*current_round, current_slot_id);
                CollectRoundResults(*current_round, current_slot_id, satisfied);
                AccountConsumedRound(*current_round, searched_l2_count, consumed_round_count,
                                     process_round_count_level_2);
                if (!satisfied && MaxProbeL2BucketLimitEnabled() && searched_l2_count >= max_probe_l2_bucket_num) {
                    satisfied = true;
                }
                ResetRoundPlan(*current_round);
                ++round_iter;

                if (satisfied) {
                    if (has_future_round) {
                        WaitForSlot(future_slot_id);
                        ResetRoundPlan(*future_round);
                    }
                    break;
                }

                if (!has_future_round) {
                    break;
                }

                std::swap(current_round, future_round);
                std::swap(current_slot_id, future_slot_id);
            }
        }

        current_query_->timing_metrics.probe_expand_rounds = consumed_round_count;
        current_query_->process_round_count_l2 = consumed_round_count;
        current_query_->process_round_count_level_2 = process_round_count_level_2;
        current_query_->searched_bucket_count_l2 = searched_l2_count;
        current_query_->finalize_results();
        current_query_->MarkProcessingEnd();

        sched_->PushResult(current_query_);
        current_query_ = nullptr;
        ClearActiveBatchContext();
    }

    // --- Follower Logic ---
    void FollowerLoop(int rank, uint64_t& observed_stage_epoch) {
        const uint64_t signal = stage_signal_.load(std::memory_order_acquire);
        const uint64_t stage_epoch = UnpackStageEpoch(signal);
        if (stage_epoch == observed_stage_epoch) {
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
        if (s == IDLE) {
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

        if (s == IVF_FILTER) {
            ScopedMemoryEventSession memory_event_scope(
                current_query_ == nullptr ? nullptr : current_query_->memory_event_session());
            ExecuteIVF(rank);
            observed_stage_epoch = stage_epoch;
            SignalDone();
        } else if (s == BATCH_ATTR_FILTER_MASK) {
            ScopedMemoryEventSession memory_event_scope(
                current_query_ == nullptr ? nullptr : current_query_->memory_event_session());
            InBucketAttrFilter(rank);
            observed_stage_epoch = stage_epoch;
            SignalDone();
        } else if (s == BATCH_COLLECT_RESULTS) {
            ScopedMemoryEventSession memory_event_scope(
                current_query_ == nullptr ? nullptr : current_query_->memory_event_session());
            ExecuteResultCollection(rank);
            observed_stage_epoch = stage_epoch;
            SignalDone();
        }
    }

    void SignalDone() {
        worker_counter_.fetch_add(1, std::memory_order_release);
    }
    void WaitFollowers() {
        while (worker_counter_.load(std::memory_order_acquire) < (cores_per_group - 1)) {
#if defined(__aarch64__) || defined(__arm__)
            __asm__ volatile("yield");
#else
            _mm_pause();
#endif
        }
    }

    // --- Execution Logic ---
    void ExecuteIVF(int rank) {
        MemoryEventSession* session = current_query_ == nullptr ? nullptr : current_query_->memory_event_session();
        std::vector<uint64_t>& local_mask = tls_ivf_masks_[rank];
        const size_t local_mask_old_capacity = session == nullptr ? 0 : local_mask.capacity();
        const size_t scratch_old_capacity = session == nullptr ? 0 : tls_scratch_pools_[rank].capacity();
        auto step_start = std::chrono::high_resolution_clock::now();
        local_mask.clear();
        if (bucket_level_ivf_enable) {
            uint32_t u64_count = db_->get_bucket_level_ivf().get_aligned_stride();
            local_mask.resize(u64_count);
            std::fill(local_mask.begin(), local_mask.end(), ~0ULL);
        } else {
            current_query_->search_ivf_plan(db_->get_bucket_level_ivf(), local_mask, tls_scratch_pools_[rank], rank);
        }
        RecordCapacityGrowth<uint64_t>(session, kTlsIvfMasksLabel, local_mask_old_capacity, local_mask.capacity());
        RecordCapacityGrowth<uint64_t>(session, kTlsScratchPoolsLabel, scratch_old_capacity,
                                       tls_scratch_pools_[rank].capacity());
        auto step_end = std::chrono::high_resolution_clock::now();
        thread_ivf_step_timings_[rank].filter_eval_ms =
            std::chrono::duration<double, std::milli>(step_end - step_start).count();
        const int local_candidate_limit = db_->get_bucket_level_ivf().get_buckets_per_core();

        step_start = std::chrono::high_resolution_clock::now();

        auto cmp = [](const CandidateBucket& a, const CandidateBucket& b) { return a.score > b.score; };
        std::vector<CandidateBucket>& heap = ThreadLocalCandidateHeapStorage();
        const size_t heap_old_capacity = session == nullptr ? 0 : heap.capacity();
        heap.clear();

        const float* centroids = db_->get_centroids();
        const float* q_vec = current_query_->query_vector.data();
        uint32_t buckets_per_core = db_->get_bucket_level_ivf().get_buckets_per_core();
        uint32_t base_global_bucket_idx = rank * buckets_per_core;
        uint32_t mask_size = local_mask.size();

        for (uint32_t i = 0; i < mask_size; ++i) {
            uint64_t word = local_mask[i];
            if (word == 0)
                continue;
            uint32_t word_base_offset = i * 64;
            // Use ctzll to skip zero bits: iterate only over set bits
            while (word) {
                int bit = __builtin_ctzll(word);
                word &= word - 1;  // clear lowest set bit

                uint32_t local_bid = word_base_offset + bit;
                if (local_bid >= buckets_per_core)
                    continue;
                uint32_t bucket_id = base_global_bucket_idx + local_bid;
                if (bucket_id >= static_cast<uint32_t>(total_bucket_num_level_2))
                    continue;

                const float* c_vec = centroids + bucket_id * vector_dim;
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
                for (; d + 7 < vector_dim; d += 8) {
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
                if (heap.size() < static_cast<size_t>(local_candidate_limit)) {
                    heap.push_back({bucket_id, dot_product});
                    std::push_heap(heap.begin(), heap.end(), cmp);
                } else if (!heap.empty() && dot_product > heap.front().score) {
                    std::pop_heap(heap.begin(), heap.end(), cmp);
                    heap.back() = {bucket_id, dot_product};
                    std::push_heap(heap.begin(), heap.end(), cmp);
                }
            }
        }
        step_end = std::chrono::high_resolution_clock::now();
        thread_ivf_step_timings_[rank].centroid_score_ms =
            std::chrono::duration<double, std::milli>(step_end - step_start).count();
        auto& my_res = thread_bucket_results_[rank];
        const size_t my_res_old_capacity = session == nullptr ? 0 : my_res.capacity();
        my_res.clear();
        RecordCapacityGrowth<CandidateBucket>(session, kExecuteIvfHeapLabel, heap_old_capacity, heap.capacity());
        step_start = std::chrono::high_resolution_clock::now();
        while (!heap.empty()) {
            std::pop_heap(heap.begin(), heap.end(), cmp);
            my_res.push_back(heap.back());
            heap.pop_back();
        }
        step_end = std::chrono::high_resolution_clock::now();
        thread_ivf_step_timings_[rank].result_pack_ms =
            std::chrono::duration<double, std::milli>(step_end - step_start).count();
        RecordCapacityGrowth<CandidateBucket>(session, kThreadBucketResultsLabel, my_res_old_capacity,
                                              my_res.capacity());
    }

    void ComputeBatchScoresOnCpu(const RoundBatch& batch, float* score_buffer) {
        if (score_buffer == nullptr) {
            return;
        }

        const float* query_vec = current_query_->query_vector.data();
        const float* doc_vectors = db_->get_doc_vectors();
        for (size_t batch_idx = 0; batch_idx < batch.bucket_ids.size(); ++batch_idx) {
            const Bucket& bucket = db_->get_bucket(batch.bucket_ids[batch_idx]);
            const auto& global_ids = bucket.get_global_ids();
            float* bucket_scores = score_buffer + batch.score_offsets_per_bucket[batch_idx];
            for (size_t local_idx = 0; local_idx < global_ids.size(); ++local_idx) {
                const float* doc_vec =
                    doc_vectors + static_cast<size_t>(global_ids[local_idx]) * static_cast<size_t>(vector_dim);
                float dot_product = 0.0f;
#if defined(__aarch64__) || defined(__arm__)
                float32x4_t sum0 = vdupq_n_f32(0.0f);
                float32x4_t sum1 = vdupq_n_f32(0.0f);
                int d = 0;
                for (; d + 7 < vector_dim; d += 8) {
                    float32x4_t q0 = vld1q_f32(query_vec + d);
                    float32x4_t q1 = vld1q_f32(query_vec + d + 4);
                    float32x4_t v0 = vld1q_f32(doc_vec + d);
                    float32x4_t v1 = vld1q_f32(doc_vec + d + 4);
                    sum0 = vfmaq_f32(sum0, q0, v0);
                    sum1 = vfmaq_f32(sum1, q1, v1);
                }
                dot_product = vaddvq_f32(sum0) + vaddvq_f32(sum1);
                for (; d < vector_dim; ++d) {
                    dot_product += query_vec[d] * doc_vec[d];
                }
#else
                for (int d = 0; d < vector_dim; ++d) {
                    dot_product += query_vec[d] * doc_vec[d];
                }
#endif
                bucket_scores[local_idx] = dot_product;
            }
        }
    }

    void CandidateBucketMerge() {
        MemoryEventSession* session = current_query_ == nullptr ? nullptr : current_query_->memory_event_session();
        std::vector<CandidateBucket>& candidates = ThreadLocalCandidateMergeBuffer();
        const size_t candidates_old_capacity = session == nullptr ? 0 : candidates.capacity();
        candidates.clear();
        for (const auto& bucket_list : thread_bucket_results_)
            candidates.insert(candidates.end(), bucket_list.begin(), bucket_list.end());

        auto cmp_desc = [](const CandidateBucket& a, const CandidateBucket& b) { return a.score > b.score; };
        std::sort(candidates.begin(), candidates.end(), cmp_desc);
        RecordCapacityGrowth<CandidateBucket>(session, kCandidateMergeBufferLabel, candidates_old_capacity,
                                              candidates.capacity());

        // Direct L2 bucket IDs (IVF operates at L2 granularity)
        const size_t sorted_buckets_old_capacity = session == nullptr ? 0 : sorted_l2_buckets_.capacity();
        sorted_l2_buckets_.resize(candidates.size());
        for (size_t i = 0; i < candidates.size(); ++i)
            sorted_l2_buckets_[i] = candidates[i].bucket_id;
        RecordCapacityGrowth<uint32_t>(session, kSortedBucketsLabel, sorted_buckets_old_capacity,
                                       sorted_l2_buckets_.capacity());
    }

    void InBucketAttrFilter(int rank) {
        if (active_round_ == nullptr || active_batch_ == nullptr) {
            return;
        }

        MemoryEventSession* session = current_query_ == nullptr ? nullptr : current_query_->memory_event_session();
        const int total = static_cast<int>(current_batch_size_);
        const std::vector<uint32_t>& batch_bucket_ids = active_batch_->bucket_ids;

        std::vector<uint64_t>& temp_doc_mask = ThreadLocalTempDocMask();
        const size_t temp_doc_mask_old_capacity = session == nullptr ? 0 : temp_doc_mask.capacity();
        const size_t scratch_old_capacity = session == nullptr ? 0 : tls_scratch_pools_[rank].capacity();
        temp_doc_mask.clear();

        // Dynamic work distribution: each worker atomically grabs the next bucket
        while (true) {
            int i = static_cast<int>(next_work_idx_.fetch_add(1, std::memory_order_relaxed));
            if (i >= total)
                break;

            uint32_t bid = batch_bucket_ids[static_cast<size_t>(i)];
            const Bucket& bucket = db_->get_bucket(bid);
            uint32_t bucket_stride = bucket.get_stride();

            const auto& plan = current_query_->filter_exp.bucket_plan;
            uint8_t plan_result = 0;
            if (plan.valid) {
                plan_result = current_query_->search_bucket_plan(bucket, bid, &db_->get_bucket_level_ivf(),
                                                                 temp_doc_mask, tls_scratch_pools_[rank]);
            } else {
                plan_result = 2;
            }
            if (plan_result != 2 && temp_doc_mask.size() < static_cast<size_t>(bucket_stride)) {
                AbortWithInvariantError("search_bucket_plan returned undersized mask buffer: bucket_id=" +
                                        std::to_string(bid) + ", mask_size=" + std::to_string(temp_doc_mask.size()) +
                                        ", bucket_stride=" + std::to_string(bucket_stride));
            }
            // Write mask to shared buffer at batch-relative position
            size_t write_offset = active_batch_->mask_offset + static_cast<size_t>(i) * active_round_->mask_stride;
            std::memcpy(active_round_->mask_storage.data() + write_offset, temp_doc_mask.data(),
                        static_cast<size_t>(bucket_stride) * sizeof(uint64_t));
        }
        RecordCapacityGrowth<uint64_t>(session, kTempDocMaskLabel, temp_doc_mask_old_capacity,
                                       temp_doc_mask.capacity());
        RecordCapacityGrowth<uint64_t>(session, kTlsScratchPoolsLabel, scratch_old_capacity,
                                       tls_scratch_pools_[rank].capacity());
    }

    void ExecuteResultCollection(int rank) {
        if (active_round_ == nullptr || active_batch_ == nullptr || active_batch_scores_ == nullptr) {
            return;
        }

        MemoryEventSession* session = current_query_ == nullptr ? nullptr : current_query_->memory_event_session();
        const int total = static_cast<int>(current_batch_size_);
        const std::vector<uint32_t>& batch_bucket_ids = active_batch_->bucket_ids;
        const std::vector<size_t>& score_offsets = active_batch_->score_offsets_per_bucket;

        auto& my_res = thread_doc_results_[rank];
        const size_t my_items_old_capacity = session == nullptr ? 0 : my_res.items.capacity();
        my_res.items.clear();

        // Per-thread min-heap: keep only the top expanded_k items per thread.
        const int heap_k = current_query_->expanded_k;
        auto heap_worse = [](const QueryResult::Item& a, const QueryResult::Item& b) { return a.score > b.score; };
        bool heap_full = false;

        const bool no_filter = !current_query_->filter_exp.bucket_plan.valid;

        // Dynamic work distribution: each worker atomically grabs the next bucket
        while (true) {
            int i = static_cast<int>(next_work_idx_.fetch_add(1, std::memory_order_relaxed));
            if (i >= total)
                break;

            uint32_t bid = batch_bucket_ids[static_cast<size_t>(i)];
            const Bucket& bucket = db_->get_bucket(bid);
            size_t score_read_offset = score_offsets[static_cast<size_t>(i)];
            float* score_ptr = active_batch_scores_ + score_read_offset;
            int doc_num = bucket.get_doc_num();
            const auto& gids = bucket.get_global_ids();

            if (no_filter) {
                // Fast path: all docs pass filter — skip mask scanning entirely.
                // Direct linear scan over compact scores (stride-1, cache-friendly).
                for (int id = 0; id < doc_num; ++id) {
                    float score = score_ptr[id];
                    if (!heap_full) {
                        my_res.items.emplace_back(gids[id], score);
                        if (static_cast<int>(my_res.items.size()) == heap_k) {
                            std::make_heap(my_res.items.begin(), my_res.items.end(), heap_worse);
                            heap_full = true;
                        }
                    } else if (score > my_res.items.front().score) {
                        std::pop_heap(my_res.items.begin(), my_res.items.end(), heap_worse);
                        my_res.items.back() = QueryResult::Item(gids[id], score);
                        std::push_heap(my_res.items.begin(), my_res.items.end(), heap_worse);
                    }
                }
            } else {
                // Filter path: scan mask bits to find passing docs.
                uint32_t bucket_stride = bucket.get_stride();
                if (score_read_offset + static_cast<size_t>(doc_num) * kScoreCols > active_batch_->score_count_floats) {
                    AbortWithInvariantError(
                        "score_read_offset exceeded stored batch score range: bucket_id=" + std::to_string(bid) +
                        ", score_read_offset=" + std::to_string(score_read_offset) +
                        ", doc_num=" + std::to_string(doc_num) +
                        ", batch_score_count_floats=" + std::to_string(active_batch_->score_count_floats));
                }

                const uint64_t* pmask64 = active_round_->mask_storage.data() + active_batch_->mask_offset +
                                          static_cast<size_t>(i) * active_round_->mask_stride;

                int passed_count = 0;
                for (uint32_t d = 0; d < bucket_stride; d++) {
                    uint64_t cur_mask = pmask64[d];
                    int base_id = static_cast<int>(d) << 6;
                    passed_count += __builtin_popcountll(cur_mask);
                    while (cur_mask) {
                        int bit = __builtin_ctzll(cur_mask);
                        int id = base_id + bit;
                        if (id < doc_num) {
                            float score = score_ptr[id];
                            if (!heap_full) {
                                my_res.items.emplace_back(gids[id], score);
                                if (static_cast<int>(my_res.items.size()) == heap_k) {
                                    std::make_heap(my_res.items.begin(), my_res.items.end(), heap_worse);
                                    heap_full = true;
                                }
                            } else if (score > my_res.items.front().score) {
                                std::pop_heap(my_res.items.begin(), my_res.items.end(), heap_worse);
                                my_res.items.back() = QueryResult::Item(gids[id], score);
                                std::push_heap(my_res.items.begin(), my_res.items.end(), heap_worse);
                            }
                        }
                        cur_mask &= cur_mask - 1;
                    }
                }
                current_query_->RecordBucketMaskStats(passed_count, doc_num);
            }
        }

        RecordCapacityGrowth<QueryResult::Item>(session, kThreadDocResultsItemsLabel, my_items_old_capacity,
                                                my_res.items.capacity());
    }

    void MergeBatchResults() {
        MemoryEventSession* session = current_query_ == nullptr ? nullptr : current_query_->memory_event_session();
        std::vector<QueryResult::Item>& all_items = ThreadLocalAllItemsBuffer();
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
            auto& thread_items = thread_doc_results_[t].items;
            all_items.insert(all_items.end(), thread_items.begin(), thread_items.end());
        }

        // Leader single KeepTopK sort
        QueryResult::KeepTopK(all_items, k);
        if (static_cast<int>(all_items.size()) > k) {
            all_items.resize(static_cast<size_t>(k));
        }

        current_query_->merge_batch_results(all_items);
        RecordCapacityGrowth<QueryResult::Item>(session, kAllItemsLabel, all_items_old_capacity, all_items.capacity());
    }
};
