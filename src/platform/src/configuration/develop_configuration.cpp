/*
 * Definitions for the config-module owned gflags (see develop_configuration.h).
 *
 * Defaults are bring-up values. The *_block_dim flags map to AI-core counts and
 * MUST be revisited for the target SoC (e.g. Ascend910B has ~20-24 AI cores);
 * the topk ratio/threshold flags gate the NPU pre-filter and affect recall.
 * Treat these as placeholders until validated on hardware (TODO Phase 7).
 */
#include "src/configuration/develop_configuration.h"

DEFINE_int32(full_recall_batch_search_thread_num, 8, "worker threads for batch search");
DEFINE_int32(full_recall_batch_accumulation_max_size, 16, "max queries accumulated per batch");
DEFINE_int32(full_recall_batch_accumulation_time_interval, 1, "batch accumulation window (ms)");
DEFINE_int32(full_recall_aggregator_block_dim, 8, "AI-core block dim for the aggregator kernel");
DEFINE_int32(full_recall_scorer_block_dim, 8, "AI-core block dim for the vector MMad kernel");
DEFINE_int32(full_recall_text_filter_block_dim, 8, "AI-core block dim for the bitmap filter kernel");
DEFINE_int32(full_recall_npu_topk_loop_count, 1, "NPU topk kernel loop count");
DEFINE_int32(full_recall_search_limit, 100, "default topK / result limit");
DEFINE_int32(full_recall_stream_init_size, 8, "initial ACL stream pool size per device");
DEFINE_double(full_recall_bitlist_denseness_threshold, 0.05,
              "posting is stored as bitlist below this hit-density, else bitset");
DEFINE_double(full_recall_device_load_rate, 0.9, "fraction of device docs to load");
DEFINE_double(full_recall_npu_topk_enters_threshold_ratio, 0.5,
              "trigger NPU topk pre-filter when candidates exceed this fraction");
DEFINE_double(full_recall_npu_topk_finish_buffer_ratio, 1.5,
              "topk early-quit tolerance: stop narrowing once down to topK*ratio candidates and return "
              "them all for the host to trim. Not a buffer size -- it cannot truncate below topK. <=1.0 "
              "disables early-quit (the check is unsatisfiable), so the kernel runs the full loop_count.");
DEFINE_bool(full_recall_send_batcher_only_score, false, "score-only path (skip filter) for benchmarking");
DEFINE_string(npuretrieval_log_level, "DEBUG", "log threshold: DEBUG/INFO/WARN/ERROR");
