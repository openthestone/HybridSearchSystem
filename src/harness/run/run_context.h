/*
 * run_context.h -- the state and the operations the five scheduling schemes share.
 *
 * All five (serial, round-robin, shard groups, pipeline) read the same loaded index and the same
 * query/filter inputs, and write the same result and timing buffers. They also all go through the
 * same three things: one shard's search, the host merge of the per-shard top-Ks, and the
 * --slow_cards sleep -- which has to sit at ONE place, or the schemes would not face an identical
 * disturbance and the A/B would mean nothing.
 *
 * Not copyable: the worker pool's threads are joined in the destructor.
 */
#pragma once

#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "../query/filter_setup.h"
#include "setup.h"
#include "src/full_recall/core/full_recall_result.h"
#include "src/full_recall/index/data_table.h"

namespace npur_harness {

class RunContext {
   public:
    RunContext(const ShardLayout& layout, std::vector<std::shared_ptr<NpuRetrieval::DataTable>>& tables,
               const RunInputs& inputs);
    ~RunContext();
    RunContext(const RunContext&) = delete;
    RunContext& operator=(const RunContext&) = delete;

    // ---- fixed for the run --------------------------------------------------
    const std::vector<int32_t>& shardDevices;
    std::vector<std::shared_ptr<NpuRetrieval::DataTable>>& shardTables;
    const std::vector<std::vector<float>>& queries;
    const FilterSet& filters;
    const bool haveFilters;
    const std::vector<uint32_t>& topks;
    const uint32_t nShards;
    const size_t nq;  // queries to run
    const size_t Q;   // queries in the file; --num_queries past it makes them cycle
    const size_t bs;  // --batch_size

    // ---- written as the run goes -------------------------------------------
    std::vector<std::vector<int64_t>> resultDocIds;
    std::vector<double> latencies_ms;  // each query's TRUE end-to-end, for the percentiles
    std::vector<double> perQueryMs;
    std::vector<std::vector<double>> shardMs;  // [shard][query], only under --shard_latency_dump
    std::vector<double> mergeMs;               // host merge ms per query
    const bool dumpShardLat;
    double wallMs = 0.0;  // the timed pass, which is what qps is computed from

    // ---- --target_qps open-loop pacing -------------------------------------
    const double paceQps;
    const bool paced;
    std::chrono::steady_clock::time_point paceT0;  // set when the timed pass starts
    std::chrono::steady_clock::time_point arrivalOf(size_t queryIdx) const;

    // ---- what every scheme goes through ------------------------------------
    // The simulated NPU load fluctuation, applied inside the measured latency.
    void ApplyFluctuation(int32_t deviceId, double serviceMs);

    // One shard's whole batch on the calling thread: pin, bind the device, build the trees, search.
    void RunShardBody(uint32_t s, size_t start, size_t end, bool record);

    // One batch across every shard, then the host merge into resultDocIds.
    bool RunBatch(size_t start, size_t end, bool record);

    // --shard_worker_pool: persistent per-shard threads instead of spawning per batch. Must be
    // called before the first RunBatch; the destructor stops and joins them.
    void StartWorkerPool();

   private:
    struct MergeEntry {  // 16 bytes: 8-byte id first avoids the padding a leading float would force
        int64_t id;
        float score;
        uint32_t shard;
    };

    void FoldShardIntoAcc(uint32_t s, size_t start, size_t end);
    void PoolWorker(uint32_t s);

    std::vector<int32_t> slowCards;
    bool fluctuate = false;
    std::chrono::steady_clock::time_point fluctT0;

    std::vector<std::vector<std::shared_ptr<NpuRetrieval::FullRecallResult>>> shardResults;
    std::vector<char> shardOk;  // char, not bool: vector<bool> elements share a word and would race

    std::vector<std::vector<MergeEntry>> acc;  // --stream_merge running merge, one per batch slot
    std::mutex accMtx;
    std::vector<MergeEntry> mergeScratch;  // reused across folds (folds are serialized under accMtx)

    std::mutex poolMtx;
    std::condition_variable poolGo, poolDone;
    uint64_t poolGen = 0;
    uint32_t poolBusy = 0;
    bool poolStop = false;
    size_t jobStart = 0, jobEnd = 0;
    bool jobRecord = false;
    std::vector<std::thread> poolThreads;
};

}  // namespace npur_harness
