/*
 * filter_stats.h -- TextFilter's instrumentation: everything that only runs when an NPUR_*_STATS
 * switch asks for it, plus the two debug dumps. None of it is on the query path with the switches
 * off, and all of it used to sit inline in text_filter.cpp.
 *
 * The switches and what they cost:
 *   NPUR_BITLIST_STATS=1   [BITSTAT-QUERY] every NPUR_BITLIST_STATS_EVERY calls. Cheap.
 *   NPUR_BITLIST_STATS=2   also [BITSTAT-DEEP] for the first NPUR_BITLIST_STATS_CALLS calls --
 *                          ONE D2H PER POSTING. Never leave this on for a timing run.
 *   NPUR_EMPTY_STATS=1     [EMPTYSTAT]. Walks every (posting, segment) pair a second time.
 *   NPUR_EXPR_STATS=1      [EXPRSTAT] post-order expression shape. Cheap.
 */
#pragma once

#include <cstdint>
#include <vector>

namespace NpuRetrieval {

// 0 = off, 1 = per-query summary, 2 = also the per-posting deep dump.
uint32_t BitlistStatsLevel();

// One bitlist->bitset conversion call, as the [BITSTAT] lines describe it.
struct BitlistCallShape {
    int32_t deviceId = 0;
    uint32_t segmentsNum = 0;
    uint32_t postingsNum = 0;
    uint64_t tableCount = 0;  // (segments x postings) pairs the query referenced
    uint64_t bitlistNum = 0;  // of those, the ones that needed conversion
    uint32_t segmentByteSize = 0;
    uint32_t blockDim = 0;
    uint32_t maxPostingLengthByte = 0;
    uint32_t formerNum = 0;  // the kernel's core split, which is what coreImbalance is about
    uint32_t formerLength = 0;
    uint32_t tailLength = 0;
    uint8_t* const* srcList = nullptr;  // level 2 reads each converted posting's 8-byte header back
};

// Accumulates, and every NPUR_BITLIST_STATS_EVERY calls prints [BITSTAT-QUERY].
void NoteBitlistCall(const BitlistCallShape& s);

// Prints [BITSTAT-DEEP] for the first NPUR_BITLIST_STATS_CALLS calls and returns after that. Call
// it only at level >= 2 with bitlistNum > 0: the call budget is counted here, so calling it
// otherwise would burn the budget on nothing.
void NoteBitlistDeep(const BitlistCallShape& s);

bool EmptyStatsEnabled();
void NoteEmptyOperands(const std::vector<std::vector<uint8_t>*>& postingTypes,
                       const std::vector<std::vector<uint16_t>*>& postingWeights,
                       const std::vector<uint8_t>& sparseDirect, uint32_t postingsNum, uint32_t segmentsNum);

bool ExprStatsEnabled();
// Call only after CheckPostExpr returned true; it trusts the node encodings.
void NoteExprShape(const std::vector<uint32_t>& postExpr);

}  // namespace NpuRetrieval
