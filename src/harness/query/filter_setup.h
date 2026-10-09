/*
 * filter_setup.h -- turn the run's filter inputs into what the engine takes: the expression file
 * into a per-query AST (FilterSet), and one AST into an engine QueryNode tree.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "filter_expr.h"
#include "src/full_recall/retrieval/query/query_node_imp.h"

namespace npur_harness {

struct FilterSet {
    std::vector<std::string> exprs;                   // raw lines
    std::vector<std::unique_ptr<FilterNode>> parsed;  // parsed per expr (nullptr = match-all/parse-fail)
    std::vector<uint32_t> assign;                     // per-query -> expr index (or UINT32_MAX = none)

    const FilterNode* astFor(size_t q) const {
        if (q >= assign.size() || assign[q] == UINT32_MAX)
            return nullptr;
        return parsed[assign[q]].get();
    }
};

// Parses every line of `path`. When the file holds exactly numQueries expressions they are used
// one-to-one; otherwise each query draws one at random from `seed`.
bool LoadFilters(const std::string& path, size_t numQueries, uint64_t seed, FilterSet& fs);

// One positive integer per line, exactly numQueries of them.
bool LoadTopKFile(const std::string& path, size_t numQueries, std::vector<uint32_t>& topks);

// Builds the engine tree for one AST. The nodes are owned by `owned`, which must outlive the
// returned pointer; a null `n` yields an empty AND, i.e. match-all.
NpuRetrieval::QueryNode* BuildQueryNode(const FilterNode* n, const std::string& field,
                                        std::vector<std::unique_ptr<NpuRetrieval::QueryNode>>& owned);

}  // namespace npur_harness
