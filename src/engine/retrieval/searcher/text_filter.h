#pragma once

#include <cstdint>
#include <vector>
#include "src/full_recall/retrieval/device/device_common_external.h"
#include "src/full_recall/retrieval/query/query_node.h"
#include "src/full_recall/index/data_table.h"
#include "configuration/develop_configuration.h"
#include "runtime/gm_memory_manager.h"
#include "src/workflow/async/executor.h"

namespace NpuRetrieval {

// One query's flattened query tree for one shard: the postfix token stream the
// BitmapTextFilter kernel interprets, plus the postings it refers to. Shard-specific,
// because a token missing from this shard's DataTable drops its node.
//
// Building this is pure host work -- the post-order walk in query_node_imp.cpp
// includes nothing but its own header and the logger, so it issues no ACL calls at
// all. That is what makes it safe to overlap with a running kernel; see
// NPUR_OVERLAP_POSTEXPR in full_recall_searcher.cpp.
struct FilterExpr {
    std::vector<uint32_t> postExpr;
    std::vector<std::vector<uint8_t>*> postingTypes;         // posting type of each posting
    std::vector<std::vector<uint8_t*>*> postingDeviceAddrs;  // device address of each posting
    uint32_t opNum{0};
    // Query tree has no children on this shard: nothing to filter, every doc passes.
    // The all-ones memset that expresses this is device work, so it stays out of the
    // build and happens in Compute.
    bool emptyTree{false};
    bool valid{false};
    // Optionally pre-built by BatchPreparePostings during the scorer's execution window
    // (NPUR_OVERLAP_POSTING): the posting pointer table transpose + H2D, so Compute can skip
    // PostingBitListToSet. These chunks are then owned by the caller (FreePreparedPostings),
    // not by Compute.
    bool postingsPrepared{false};
    GmBlock bitsetPostingsChunk{};
    GmBlock devicePostingsChunk{};
};

class TextFilter {
   public:
    explicit TextFilter(const int32_t deviceId, const std::shared_ptr<DataTable> dataTable)
        : m_deviceId(deviceId), m_dataTable(dataTable) {
        m_executor = CreateExecutor();
        m_segmentLength = m_dataTable->GetSegmentLength();
        m_segmentByteSize = m_dataTable->GetSegmentByteSize();
    }

    ~TextFilter() {}

    // `prepared` (optional) supplies expressions already built by BatchPrepareExpr,
    // one per query tree; nullptr means build them inline as usual.
    bool BatchCompute(const std::vector<QueryNode*> queryTrees, uint8_t* resultInDevice,
                      const std::vector<FilterExpr>* prepared = nullptr);
    bool Compute(QueryNode* queryTree, uint8_t* resultInDevice, uint32_t resultByteSize,
                 const FilterExpr* prepared = nullptr);

    // Host-only, ACL-free, single-threaded: safe to run while a kernel occupies the
    // device. Runs on the caller's thread by design -- the point is to fill a window
    // the caller is otherwise spending blocked.
    bool BatchPrepareExpr(const std::vector<QueryNode*> queryTrees, std::vector<FilterExpr>& out);

    // Build each expr's posting pointer table (PostingBitListToSet) ahead of the filter
    // kernel, so it can run in the scorer's execution window. Nearly pure host + a small
    // H2D, no big HBM kernel. FreePreparedPostings releases the chunks after BatchCompute.
    bool BatchPreparePostings(std::vector<FilterExpr>& exprs);
    void FreePreparedPostings(std::vector<FilterExpr>& exprs);

   private:
    bool PrepareExpr(QueryNode* queryTree, FilterExpr& out);
    bool PostingBitListToSet(const std::vector<std::vector<uint8_t>*>& postingTypes,
                             const std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs, uint32_t postingsNum,
                             GmBlock& bitsetPostingsChunk, GmBlock& devicePostingsChunk);
    bool PostingBitListTransposeAndOffset(const std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs,
                                          uint32_t segmentsNum, uint32_t postingsNum, GmBlock& devicePostingsChunk);
    // A/B toggle (env NPUR_POOL_POSTINGS): allocate/free the devicePostings pointer
    // table either via TEXT_FILTER_POSTINGS_POOL (default) or, when the env is 0, a
    // raw per-query aclrtMalloc/aclrtFree (the pre-pooling baseline). Both write into
    // / read from the GmBlock so every call site stays identical between the modes.
    void AllocateDevicePostings(size_t postingsByteSize, GmBlock& devicePostingsChunk);
    void FreeDevicePostings(GmBlock& devicePostingsChunk);
    bool BitmapTextFilter(const std::vector<uint32_t>& postExpr, TextFilterContextData& context,
                          uint8_t*& devicePostings, uint8_t* resultInDevice);
    // theoretical max posting length within one segment for the bitlist type
    uint32_t GetMaxPostingLength(const uint32_t docNumPerSegment, const double denseness);
    uint32_t GetBitlistTypeNum(const std::vector<std::vector<uint8_t>*>& postingTypes);

    void CreateContextData(const uint32_t exprLen, const uint32_t opNum, const int32_t postingNum,
                           TextFilterContextData& context);

    bool CheckPostingTypes(const std::vector<std::vector<uint8_t>*>& postingTypes, uint32_t segmentsNum,
                           uint32_t postingsNum);
    bool CheckPostExpr(const std::vector<uint32_t>& postExpr, const uint32_t postingNums);
    bool CheckPostExprNotNode(const std::vector<uint32_t>& postExpr, uint32_t& idx, uint32_t& postingRemained,
                              uint32_t& numInStack);
    bool CheckPostExprAndOrNode(const std::vector<uint32_t>& postExpr, uint32_t& idx, uint32_t& postingRemained,
                                uint32_t& numInStack);
    bool CheckPostExprConjNode(const std::vector<uint32_t>& postExpr, uint32_t& idx, uint32_t& postingRemained,
                               uint32_t& numInStack);

    void PrintDevicePostings(uint8_t*& devicePostings, size_t postingsByteSize, uint32_t postingsNum,
                             uint32_t segmentsNum, std::vector<uint8_t*> postings);
    void PrintFilterRes(const uint8_t* resultInDevice);

   private:
    const int32_t m_deviceId{};
    const std::shared_ptr<DataTable> m_dataTable{};
    std::shared_ptr<Executor> m_executor;
    uint32_t m_segmentLength;
    uint32_t m_segmentByteSize;
};
}  // namespace NpuRetrieval
