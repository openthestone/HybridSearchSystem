// fr_search: drive FullRecallSearcher directly, bypassing the online-serving layer.
//   fr_search --index_dir DIR --query_file hw_queries.fvecs --filter_file filter_expr_600.txt \
//             --dataset_hw dataset_HW.bin --recall_queries 20 --topk 100 --num_queries 10000

#include <gflags/gflags.h>

#include "run/flags.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "host/mem_report.h"
#include "recall/filter_report.h"
#include "run/report.h"
#include "run/schedule.h"
#include "run/setup.h"
#include "src/full_recall/index/data_table.h"
#include "src/utils/logger.h"

using namespace npur_harness;  // NOLINT: leaf TU, these modules are its own

DEFINE_string(index_dir, "", "index directory produced by fr_builder");
DEFINE_string(query_file, "", "query vectors (.fvecs; headered [n][dim][floats] or per-vector)");
DEFINE_int32(device_id, 0, "NPU device id (single-card mode)");
DEFINE_string(shard_index_dirs, "",
              "comma-separated per-shard index dirs for multi-card mode; each is a disjoint corpus "
              "slice built with converter --doc_offset. Empty = single-card (--index_dir).");
DEFINE_string(device_ids, "", "comma-separated NPU device ids, one per --shard_index_dirs entry");
DEFINE_string(shard_latency_dump, "",
              "multi-card only: TSV of per-query per-shard timing (idx, shard0_ms..shardN_ms, "
              "max_shard_ms, merge_ms, total_ms). Meaningful at --batch_size 1.");
DEFINE_bool(shard_worker_pool, false,
            "multi-card only: persistent per-shard worker threads instead of spawning per batch. "
            "Applies to plain N-way sharding and, per group, to --shard_group_size.");
DEFINE_string(card_cpus, "",
              "per-card CPU affinity, one taskset-style core list per --device_ids entry, in the "
              "same order, separated by ';' -- e.g. \"96-119;48-71\". Get the lists from "
              "`npu-smi info -t topo` (CPU Affinity column); they differ per machine. "
              "Empty = no pinning.");
DEFINE_bool(shard_allow_partial, false,
            "multi-card only: drop a failed shard and merge the survivors. Off = strict, any "
            "shard failure fails the batch, which keeps recall measurement exact.");
DEFINE_bool(stream_merge, false,
            "multi-card only: fold each shard's sorted top-K into a running merge as that shard "
            "finishes, so only the last fold is on the critical path. Byte-identical results.");
DEFINE_int32(topk, 100, "top-K to return");
DEFINE_int32(num_queries, 0, "number of queries to run (0 = all in file; may exceed it, queries cycle)");
DEFINE_int32(warmup, 5, "warmup batches run before timing (results/latency discarded)");
DEFINE_int32(batch_size, 1, "queries per FullRecallSearcher batch");
DEFINE_bool(round_robin, false,
            "round-robin baseline: every --device_ids card must hold the FULL corpus (pass the "
            "same full-index dir for each --shard_index_dirs entry, or recall collapses). Queries "
            "are work-stolen onto idle cards in batch_size chunks, each searched on ONE card.");
DEFINE_int32(shard_group_size, 0,
             "split the --device_ids cards into groups of this many, each holding a FULL corpus "
             "split group-size ways (pass the group's shard dirs once; run.sh replicates them). A "
             "query fans out inside ONE group, so latency is the group's while groups work-steal "
             "independently. group-size == card count means a single group. 0 = off.");
DEFINE_string(slow_cards, "",
              "simulated NPU load fluctuation (contract: 模拟的 NPU 负载波动): comma-separated DEVICE "
              "ids that periodically share their card with other work. Empty = no disturbance.");
DEFINE_double(slow_factor, 2.0, "a slowed card's work takes this many times longer. <=1 disables.");
DEFINE_double(slow_period_ms, 1000.0, "period of the slow/normal cycle in ms (the fluctuation rate)");
DEFINE_double(slow_duty, 0.5,
              "fraction of each period a --slow_cards card spends slowed. 1.0 would be a "
              "permanently slower card, i.e. heterogeneity rather than fluctuation.");
DEFINE_bool(static_assign, false,
            "FIXED task assignment -- chunk k is bound up front (card k % N under --round_robin, "
            "group k % G under --shard_group_size) instead of going to whichever worker is idle. "
            "This is the contract's '固定任务分配' baseline. Off = dynamic work stealing.");
DEFINE_double(target_qps, 0.0,
              "open-loop arrival pacing: query i arrives at t0 + i/rate and its latency is measured "
              "FROM that arrival, so queueing counts. 0 = off (closed loop, service time only).");
DEFINE_bool(parallel_load, false,
            "load the shards' DataTables on one thread each instead of one after another. The "
            "per-stage [Load] lines then interleave between shards; the per-shard summaries are "
            "printed in order after the join.");
DEFINE_bool(group_prefetch, false,
            "shard groups only: build the NEXT chunk's query trees while the cards are still busy "
            "with the current one. Needs --pipeline and --shard_worker_pool. Costs one more chunk "
            "held per group.");
DEFINE_bool(group_extract_parallel, false,
            "shard groups only: run a group's per-shard Extract on persistent workers instead of "
            "one shard after another on the group thread. Takes time off chunk LATENCY, not qps.");
DEFINE_bool(pipeline, false,
            "sharded throughput pipeline: overlap batch N's host Extract with batch N+1's device "
            "stage. Requires >1 device; latency is not meaningful in this mode.");
DEFINE_string(latency_dump, "",
              "if set, write per-query 'idx, ms, topk' TSV lines here (only meaningful at "
              "--batch_size 1)");
DEFINE_string(vec_field, "content", "vector field name (must match schema)");
DEFINE_string(posting_field, "content", "posting field name (must match schema)");
DEFINE_string(filter_file, "", "filter expressions, one per line; empty = match-all");
DEFINE_uint64(filter_seed, 42, "seed for randomly assigning filters to queries (matches old/'s scheme)");
DEFINE_string(topk_file, "", "optional top-K file, one integer per query line; overrides --topk per query");
DEFINE_string(dataset_hw, "", "dataset_HW.bin, for the CPU brute-force recall reference");
DEFINE_int32(recall_queries, 0, "number of queries to CPU-verify recall on (0 = off)");
DEFINE_uint64(cpu_doc_offset, 0,
              "absolute source-doc offset of this shard index in dataset_HW.bin; the reference "
              "scans [cpu_doc_offset, cpu_doc_offset+docNum) and returns absolute ids, matching an "
              "index built with converter --doc_offset. 0 = whole/prefix.");
DEFINE_bool(recall_fp32, false,
            "compute the CPU recall reference in full FP32, without rounding the vectors through "
            "FP16 first, so it also charges the FP16 quantization itself. Use a DIFFERENT "
            "--recall_ref_file: the mode is part of the cache fingerprint.");
DEFINE_string(recall_ref_file, "",
              "cache file for the ground-truth top-K sets, keyed by a size+mtime fingerprint of the "
              "dataset/query/filter/topk inputs. A stale fingerprint triggers a recompute.");
DEFINE_int32(recall_ref_threads, 0,
             "CPU threads for the one-time brute-force reference (0 = all cores). Only speeds up a "
             "cache miss; never changes the result.");
DEFINE_string(effective_filter_file, "", "optional output file for the first 5 effective filter expressions");
DEFINE_string(converted_tag_freq_file, "",
              "optional tag_doc_freq.txt emitted by fr_converter; used only for effective-filter diagnostics");
DEFINE_bool(mem_report, false, "after the search, print host peak RSS and per-device NPU HBM usage ([MEM] lines)");

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    if ((FLAGS_index_dir.empty() && FLAGS_shard_index_dirs.empty()) || FLAGS_query_file.empty()) {
        std::fprintf(stderr,
                     "usage: fr_search --index_dir DIR --query_file q.fvecs [--filter_file f.txt] [...]\n"
                     "   or: fr_search --shard_index_dirs d0,d1 --device_ids 0,1 --query_file q.fvecs [...]\n");
        return 1;
    }

    aclError aclRet = aclInit(nullptr);
    if (aclRet != ACL_SUCCESS) {
        LOG_ERROR("aclInit failed: " << aclRet);
        return 1;
    }

    ShardLayout layout;
    if (!ResolveShards(layout)) {
        aclFinalize();
        return 1;
    }
    const std::vector<std::string>& shardDirs = layout.dirs;
    const std::vector<int32_t>& shardDevices = layout.devices;
    const uint32_t nShards = layout.count();

    MemSnap memBase, memLoad;
    if (FLAGS_mem_report)
        memBase = SampleMem(shardDevices);

    std::vector<std::shared_ptr<NpuRetrieval::DataTable>> shardTables;
    uint64_t totalDocNum = 0;
    if (!LoadShards(layout, shardTables, totalDocNum) || !InitStreamPool(layout)) {
        aclFinalize();
        return 1;
    }
    auto& dataTable = shardTables[0];  // alias for the single-shard diagnostics below
    const uint64_t docNum = RecallCorpusDocNum(layout, shardTables, totalDocNum);
    if (FLAGS_mem_report)
        memLoad = SampleMem(shardDevices);

    RunInputs in;
    if (!LoadRunInputs(in) ||
        !ReportEffectiveFilters(dataTable, in.filters, in.haveFilters, in.topks, in.nq, shardDirs[0])) {
        aclFinalize();
        return 1;
    }
    LOG_INFO("running " << in.nq << " queries (batch_size=" << FLAGS_batch_size << ", default_topk=" << FLAGS_topk
                        << ", filters=" << (in.haveFilters ? FLAGS_filter_file : std::string("none")) << ")");

    RunContext rc(layout, shardTables, in);
    rc.StartWorkerPool();

    // Exactly one scheme runs. Groups subsume --pipeline: the group path pipelines inside a group.
    bool ok;
    if (FLAGS_round_robin && nShards > 1) {
        ok = RunRoundRobin(rc);
    } else if (FLAGS_shard_group_size > 0 && nShards >= static_cast<uint32_t>(FLAGS_shard_group_size) &&
               nShards % static_cast<uint32_t>(FLAGS_shard_group_size) == 0) {
        ok = RunShardGroups(rc);
    } else if (FLAGS_pipeline && nShards > 1) {
        ok = RunPipeline(rc);
    } else {
        ok = RunSerial(rc);
    }
    if (!ok) {
        aclFinalize();
        return 1;
    }

    DumpQueryLatencies(rc.perQueryMs, rc.topks, rc.nq);
    DumpShardLatencies(rc.shardMs, rc.mergeMs, rc.perQueryMs, rc.nShards, rc.nq);
    PrintLatencySummary(rc.latencies_ms, rc.wallMs, rc.bs);

    if (FLAGS_mem_report)
        PrintMemReport(shardDevices, memBase, memLoad, SampleMem(shardDevices), ReadHostKb("VmHWM:"));

    CheckRecall(rc.queries, rc.filters, rc.haveFilters, rc.topks, rc.resultDocIds, docNum, rc.nq);

    aclFinalize();
    return 0;
}
