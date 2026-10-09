#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "src/utils/performance_recorder.h"
#include "src/full_recall/retrieval/device/device_common_external.h"
#include "src/full_recall/retrieval/query/query_node.h"
#include "src/full_recall/index/data_table.h"
#include "configuration/develop_configuration.h"
#include "../runtime/gm_memory_manager.h"
#include "src/workflow/async/executor.h"

namespace NpuRetrieval {

// One query's flattened query tree for one shard: the postfix token stream the BitmapTextFilter
// kernel interprets, plus the postings it refers to. Shard-specific -- a token missing from this
// shard's DataTable drops its node.
struct FilterExpr {
    std::vector<uint32_t> postExpr;
    std::vector<std::vector<uint8_t>*> postingTypes;  // posting type of each posting
    // Per-segment weight of each posting: 0 = all-zero, else K for a sparse posting or
    // kDensePostingWeight. See PostingFieldData::GetPostingWeights.
    std::vector<std::vector<uint16_t>*> postingWeights;
    std::vector<std::vector<uint8_t*>*> postingDeviceAddrs;  // device address of each posting
    uint32_t opNum{0};
    // Nothing to filter on this shard: every doc passes, via an all-ones memset in Compute.
    bool emptyTree{false};
    bool valid{false};
    // NPUR_OVERLAP_POSTING: pre-built during the scorer's window, and then owned by the caller
    // (FreePreparedPostings), not by Compute.
    bool postingsPrepared{false};
    GmBlock bitsetPostingsChunk{};
    GmBlock devicePostingsChunk{};
    // NPUR_DEFER_CONV_SYNC: must be drained (DrainPreparedConversions) before any filter kernel
    // reads the bitsets, and before FreePreparedPostings releases what it reads and writes.
    aclrtStream pendingConvStream{nullptr};
};

struct FilterSlot {
    TextFilterContextData context{};
    const std::vector<uint32_t>* postExpr{nullptr};
    GmBlock bitsetPostingsChunk{};
    GmBlock devicePostingsChunk{GmPoolName::TEXT_FILTER_POSTINGS_POOL, nullptr, 0};
    uint8_t* devicePostings{nullptr};
    uint8_t* result{nullptr};
    uint32_t* devicePostExpr{nullptr};  // slice of the batch's single postExpr allocation
    uint16_t* resultStack{nullptr};     // slice of the batch's single stack block
    uint32_t exprOffset{0};
    uint32_t stackOffset{0};
    bool ownPostings{false};
    bool active{false};  // has a kernel to launch (emptyTree queries do not)
};

// One batch of the fused filter, held between its three phases so they can run apart:
//   FusedPrepare  host setup, one postExpr allocation + H2D, one resultStack block
//   FusedLaunch   every query's kernel onto one stream, no sync
//   FusedFinish   the sync (unless the caller already has it) and the teardown
struct FusedFilterBatch {
    std::vector<FilterSlot> slot;
    size_t activeCount{0};
    uint32_t* deviceExprBase{nullptr};
    GmBlock exprChunk{GmPoolName::TEXT_FILTER_EXPR_POOL, nullptr, 0};
    GmBlock stackChunk{GmPoolName::TEXT_FILTER_STACK_POOL, nullptr, 0};
    aclrtStream stream{nullptr};
    bool deviceSet{false};  // FusedPrepare's aclrtSetDevice is held; FusedFinish releases it
    bool launched{false};   // kernels are queued on `stream`
    std::unique_ptr<RecordGuard> allGuard;
    std::unique_ptr<RecordGuard> kernelGuard;
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

    // `prepared` (optional): expressions already built by BatchPrepareExpr; nullptr builds inline.
    bool BatchCompute(const std::vector<QueryNode*> queryTrees, uint8_t* resultInDevice,
                      const std::vector<FilterExpr>* prepared = nullptr);
    bool Compute(QueryNode* queryTree, uint8_t* resultInDevice, uint32_t resultByteSize,
                 const FilterExpr* prepared = nullptr);

    bool BatchPrepareExpr(const std::vector<QueryNode*> queryTrees, std::vector<FilterExpr>& out);

    // deferConvSync (NPUR_DEFER_CONV_SYNC) leaves the conversion's stream in pendingConvStream.
    bool BatchPreparePostings(std::vector<FilterExpr>& exprs, bool deferConvSync = false);
    bool DrainPreparedConversions(std::vector<FilterExpr>& exprs);
    void FreePreparedPostings(std::vector<FilterExpr>& exprs);

    // After a successful FusedPrepare, FusedFinish must be called exactly once, on every path.
    // streamSynced says the caller already synchronised `stream`, so FusedFinish skips its wait.
    static bool FusedPathApplies(const std::vector<FilterExpr>* prepared);
    bool FusedPrepare(const std::vector<QueryNode*>& queryTrees, uint8_t* resultInDevice,
                      const std::vector<FilterExpr>& prepared, FusedFilterBatch& batch);
    void FusedLaunch(FusedFilterBatch& batch, bool timeKernel);
    bool FusedFinish(FusedFilterBatch& batch, bool streamSynced);

   private:
    // NPUR_BATCH_FILTER: only runs when BatchCompute was handed a `prepared` vector.
    bool BatchComputeFused(const std::vector<QueryNode*>& queryTrees, uint8_t* resultInDevice,
                           const std::vector<FilterExpr>& prepared);
    // The batch's postExpr block back to wherever it came from; idempotent.
    void ReleaseFusedExpr(FusedFilterBatch& batch);
    bool PrepareExpr(QueryNode* queryTree, FilterExpr& out);
    bool PostingBitListToSet(const std::vector<std::vector<uint8_t>*>& postingTypes,
                             const std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs, uint32_t postingsNum,
                             GmBlock& bitsetPostingsChunk, GmBlock& devicePostingsChunk,
                             const std::vector<uint8_t>* sparseDirect,
                             const std::vector<std::vector<uint16_t>*>* postingWeights,
                             aclrtStream* deferSync = nullptr);
    bool PostingBitListTransposeAndOffset(const std::vector<std::vector<uint8_t*>*>& postingDeviceAddrs,
                                          uint32_t segmentsNum, uint32_t postingsNum, GmBlock& devicePostingsChunk);
    // NPUR_POOL_POSTINGS=0 swaps the pool for a raw per-query aclrtMalloc/aclrtFree.
    void AllocateDevicePostings(size_t postingsByteSize, GmBlock& devicePostingsChunk);
    void FreeDevicePostings(GmBlock& devicePostingsChunk);
    bool BitmapTextFilter(const std::vector<uint32_t>& postExpr, TextFilterContextData& context,
                          uint8_t*& devicePostings, uint8_t* resultInDevice);
    // theoretical max posting length within one segment for the bitlist type
    uint32_t GetMaxPostingLength(const uint32_t docNumPerSegment, const double denseness);
    // Both of GetBitlistTypeNum's answers in one walk; see NPUR_ONE_COUNT_PASS.
    static void CountBitlistTypes(const std::vector<std::vector<uint8_t>*>& postingTypes,
                                  const std::vector<uint8_t>* sparseDirect, uint32_t& withMask, uint32_t& all);
    uint32_t GetBitlistTypeNum(const std::vector<std::vector<uint8_t>*>& postingTypes,
                               const std::vector<uint8_t>* sparseDirect = nullptr);

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
