/*
 * cpu_affinity.h -- the comma/semicolon flag lists that name cards (--shard_index_dirs,
 * --device_ids, --slow_cards, --card_cpus) and the per-card CPU pinning --card_cpus installs.
 *
 * The cards do not share a NUMA node (card 0 off node4, card 1 off node2 on the 2-card box), so a
 * process-wide `taskset` is remote to at least one of them and its H2D/D2H crosses QPI.
 * Linux only; everything here is a no-op elsewhere.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace npur_harness {

// Splits on ',' and trims spaces/tabs; empty fields are dropped.
std::vector<std::string> SplitCsv(const std::string& s);

// Installs --card_cpus: one ';'-separated taskset-style core list per entry of `deviceIds`, in the
// same order. Pins the calling thread when there is exactly one card, and prints the [AFFINITY]
// line. Returns false with `error` set when the count or a list does not parse.
bool SetCardCpus(const std::string& spec, const std::vector<int32_t>& deviceIds, std::string* error);

// No-ops when --card_cpus was not given, or when the index is past the last card.
void PinThreadToCard(size_t cardIdx);
void PinThreadToCards(size_t firstCard, size_t count);

}  // namespace npur_harness
