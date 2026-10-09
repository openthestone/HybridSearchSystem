/*
 * flags.h -- the fr_search gflags, declared for the modules that read them. They are DEFINEd in
 * main.cpp; main.cpp includes this header too, so a DECLARE whose type drifts from its DEFINE is a
 * compile error there rather than something that shows up at runtime.
 *
 * Generated from main.cpp's DEFINE_ list; keep the two in step.
 */
#pragma once

#include <gflags/gflags.h>

DECLARE_string(index_dir);
DECLARE_string(query_file);
DECLARE_int32(device_id);
DECLARE_string(shard_index_dirs);
DECLARE_string(device_ids);
DECLARE_string(shard_latency_dump);
DECLARE_bool(shard_worker_pool);
DECLARE_string(card_cpus);
DECLARE_bool(shard_allow_partial);
DECLARE_bool(stream_merge);
DECLARE_int32(topk);
DECLARE_int32(num_queries);
DECLARE_int32(warmup);
DECLARE_int32(batch_size);
DECLARE_bool(round_robin);
DECLARE_int32(shard_group_size);
DECLARE_string(slow_cards);
DECLARE_double(slow_factor);
DECLARE_double(slow_period_ms);
DECLARE_double(slow_duty);
DECLARE_bool(static_assign);
DECLARE_double(target_qps);
DECLARE_bool(parallel_load);
DECLARE_bool(group_prefetch);
DECLARE_bool(group_extract_parallel);
DECLARE_bool(pipeline);
DECLARE_string(latency_dump);
DECLARE_string(vec_field);
DECLARE_string(posting_field);
DECLARE_string(filter_file);
DECLARE_uint64(filter_seed);
DECLARE_string(topk_file);
DECLARE_string(dataset_hw);
DECLARE_int32(recall_queries);
DECLARE_uint64(cpu_doc_offset);
DECLARE_bool(recall_fp32);
DECLARE_string(recall_ref_file);
DECLARE_int32(recall_ref_threads);
DECLARE_string(effective_filter_file);
DECLARE_string(converted_tag_freq_file);
DECLARE_bool(mem_report);
