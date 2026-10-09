/*
 * schedule.h -- the four ways fr_search feeds the cards. Each takes the shared RunContext, fills
 * its result and timing buffers, sets ctx.wallMs for the timed pass, and returns false having
 * logged why if a search failed.
 *
 * Which one runs is decided in main() from the flags, and only one ever does.
 */
#pragma once

#include "run_context.h"

namespace npur_harness {

// One batch at a time: a single card, or an N-way fan-out batch by batch. The default, and the
// only mode where --batch_size 1 latency means what it says.
bool RunSerial(RunContext& rc);

// --round_robin: every card holds the FULL corpus and a whole chunk is searched on ONE card, taken
// from a shared queue (or bound to card k%N under --static_assign). The contract's baseline.
bool RunRoundRobin(RunContext& rc);

// --shard_group_size: the cards split into groups, each group holding a full corpus. A query fans
// out inside ONE group while groups take chunks independently. Subsumes --pipeline.
bool RunShardGroups(RunContext& rc);

// --pipeline: overlap batch N's host Extract with batch N+1's device stage. Throughput only --
// latency is not meaningful in this mode.
bool RunPipeline(RunContext& rc);

}  // namespace npur_harness
