/*
 * recall_ref.h -- the CPU brute-force ground truth the --recall_queries check measures against,
 * and its on-disk cache.
 *
 * The cache is keyed by a fingerprint of everything the ground truth depends on, so a changed
 * input file or flag makes it recompute instead of silently comparing against the wrong answer.
 */
#pragma once

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "../../io/dataset_hw.h"
#include "../query/filter_expr.h"

namespace npur_harness {

// Everything the ground truth depends on. Two runs that differ in any field are two different
// ground truths, which is exactly what the fingerprint has to cover.
struct RecallRefInputs {
    std::string datasetPath;  // each path contributes its (size, mtime), not its content
    std::string queryPath;
    std::string filterPath;
    std::string topkPath;
    uint64_t docNum = 0;      // docs the reference scans
    uint64_t nq = 0;          // queries actually run
    uint64_t docOffset = 0;   // --cpu_doc_offset
    uint64_t filterSeed = 0;  // --filter_seed
    uint32_t dim = 0;
    int32_t topk = 0;            // --topk
    int32_t numQueriesFlag = 0;  // --num_queries as given (0 = all)
    bool fp32 = false;           // --recall_fp32: a different ground truth, not a tighter one
};

// Scans [docOffset, docOffset + docNum) of `hw`. fp32 = false rounds the query and every document
// vector through FP16 first; fp32 = true keeps full precision, so it also charges the engine for
// the FP16 quantization itself.
std::unordered_set<int64_t> CpuTopK(const std::vector<float>& query, const FilterNode* filter,
                                    const npur_port::HwDataset& hw, uint64_t docNum, int topk, uint64_t docOffset,
                                    bool fp32);

std::string RecallFingerprint(const RecallRefInputs& in);

// Both return false on a missing, corrupt or stale file; a false Load just means recompute.
bool LoadRecallRef(const std::string& path, const std::string& fp, uint64_t nq,
                   std::vector<std::unordered_set<int64_t>>& refs);
bool WriteRecallRef(const std::string& path, const std::string& fp,
                    const std::vector<std::unordered_set<int64_t>>& refs);

}  // namespace npur_harness
