/*
 * Stub for NpuRetrieval `src/configuration/develop_configuration.h`.
 *
 * Declares the gflags tunables the full-recall path reads. The config module
 * OWNS (DEFINEs, in develop_configuration.cpp) the `full_recall_*` flags plus
 * `npuretrieval_log_level`. The builder-owned flags (data_dir, data_out,
 * build_from_memory, ipc_socket_path, server_log_properties) are DEFINEd by
 * build/builder/main.cpp; they are only DECLAREd here so shared code can read
 * them without pulling that main in.
 *
 * Defaults for block_dim / ratio / threshold flags are BRING-UP guesses and
 * affect kernel tiling + correctness — tune in TODO Phase 7 against the real
 * NPU. Adjust freely via --flag on the command line.
 */
#pragma once

#include <gflags/gflags.h>

// ---- config-module owned (DEFINEd in develop_configuration.cpp) ------------
DECLARE_int32(full_recall_batch_search_thread_num);
DECLARE_int32(full_recall_batch_accumulation_max_size);
DECLARE_int32(full_recall_batch_accumulation_time_interval);
DECLARE_int32(full_recall_aggregator_block_dim);
DECLARE_int32(full_recall_scorer_block_dim);
DECLARE_int32(full_recall_text_filter_block_dim);
DECLARE_int32(full_recall_npu_topk_loop_count);
DECLARE_int32(full_recall_search_limit);
DECLARE_int32(full_recall_stream_init_size);
DECLARE_double(full_recall_bitlist_denseness_threshold);
DECLARE_double(full_recall_device_load_rate);
DECLARE_double(full_recall_npu_topk_enters_threshold_ratio);
DECLARE_double(full_recall_npu_topk_finish_buffer_ratio);
DECLARE_bool(full_recall_send_batcher_only_score);
DECLARE_string(npuretrieval_log_level);

// ---- builder-main owned (DEFINEd in build/builder/main.cpp) -----------------
DECLARE_bool(build_from_memory);
DECLARE_string(data_dir);
DECLARE_string(data_out);
DECLARE_string(ipc_socket_path);
DECLARE_string(server_log_properties);
