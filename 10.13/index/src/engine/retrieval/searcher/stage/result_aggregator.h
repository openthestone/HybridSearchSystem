#pragma once

#include <cstdint>
#include <vector>
#include "acl/acl.h"
#include "src/full_recall/index/docid/doc_id_mapping.h"
#include "src/full_recall/core/full_recall_result.h"
#include "full_recall/retrieval/searcher/runtime/gm_memory_pool.h"

namespace NpuRetrieval {

// Handoff between the two halves of AggrAndTopK:
//   AggrAndTopKDevice  -> kernels + D2H, leaves the raw top-K in these host buffers (owns them)
//   AggrAndTopKExtract -> sort + resolve global ids from them, then frees them (pure CPU)
struct AggrDeviceResult {
    uint8_t* hostTOPKResult{nullptr};
    uint8_t* hostDocLocation{nullptr};
    uint32_t effectiveCount{0};
    uint32_t effectiveCountSmaller{0};
    bool empty{true};
};

class ResultAggregator {
   public:
    ResultAggregator() = default;

    ~ResultAggregator() = default;

    // Applies NPUR_SHARD_TOPK_RATIO, so this may be smaller than the final top-K.
    void SetTopK(uint32_t topK);

    void SetDocIdMapping(const DocIdMapping* docIdMapping) {
        m_docIdMapping = docIdMapping;
    }

    void SetDeviceId(const int32_t deviceId) {
        m_deviceId = deviceId;
    }

    bool AggrAndTopK(uint8_t* filterResultInDevice, uint8_t* docScoreInDevice, size_t docAllByteSize,
                     uint8_t*& docLocationInDeviceConst, std::shared_ptr<FullRecallResult>& fullRecallResult);

    // Device, on false, has already freed everything; on true it either fills `out` (Extract must
    // run and free it) or leaves out.empty.
    bool AggrAndTopKDevice(uint8_t* filterResultInDevice, uint8_t* docScoreInDevice, size_t docAllByteSize,
                           uint8_t*& docLocationInDeviceConst, AggrDeviceResult& out);
    bool AggrAndTopKExtract(AggrDeviceResult& dr, std::shared_ptr<FullRecallResult>& fullRecallResult);

    // The whole batch's device stage in one call, phased ACROSS queries so the stream syncs and
    // count read-backs are paid once per batch. `out` is sized to topKs.size(); a query that
    // fails or finds no candidates is left empty.
    //
    // sharedStream (NPUR_SHARE_FILTER_STREAM): a caller-owned stream with the batch's filter
    // kernels still on it, never returned to the pool here. *sharedStreamSynced is set once
    // phase 1's sync has succeeded -- the caller's licence to tear the filter down without its own.
    bool AggrAndTopKDeviceBatch(const std::vector<uint8_t*>& filterResults, const std::vector<uint8_t*>& docScores,
                                size_t docAllByteSize, uint8_t*& docLocationInDeviceConst,
                                const std::vector<uint32_t>& topKs, std::vector<AggrDeviceResult>& out,
                                aclrtStream sharedStream = nullptr, bool* sharedStreamSynced = nullptr);

    // NPUR_BATCH_AGGREGATE=1 selects the batched path above. Default off.
    static bool BatchAggregateMode();

   private:
    bool Aggregator(uint8_t* filterResultInDevice, uint8_t*& docScoreInDevice, size_t docAllByteSize,
                    uint8_t*& docLocationInDeviceConst, uint8_t*& docLocationInDeviceResult,
                    uint8_t*& resultNextInDevice, uint8_t*& resultLocationNextInDevice, uint32_t topK,
                    uint32_t& effectiveCount, uint32_t& topKEffectiveCountSmaller);

    void FreeAggregator(uint8_t* hostTOPKResult, uint8_t* hostDocLocation, GmBlock& docLocationInDeviceResultChunk,
                        GmBlock& resultNextInDeviceChunk, GmBlock& resultLocationNextInDeviceChunk);

    bool AggregatorResultFill(uint8_t* docScoreInDevice, uint8_t* docLocationInDeviceResult,
                              uint8_t* resultNextInDevice, uint8_t* resultLocationNextInDevice, uint8_t* hostTOPKResult,
                              uint8_t* hostDocLocation, uint32_t effectiveCount, uint32_t effectiveCountSmaller,
                              std::shared_ptr<FullRecallResult>& fullRecallResult);
    bool AggregatorResultFillD2H(uint8_t* docScoreInDevice, uint8_t* docLocationInDeviceResult,
                                 uint8_t* resultNextInDevice, uint8_t* resultLocationNextInDevice,
                                 uint8_t* hostTOPKResult, uint8_t* hostDocLocation, uint32_t effectiveCount,
                                 uint32_t effectiveCountSmaller);
    bool AggregatorResultFillExtract(uint8_t* hostTOPKResult, uint8_t* hostDocLocation, uint32_t effectiveCount,
                                     uint32_t effectiveCountSmaller,
                                     std::shared_ptr<FullRecallResult>& fullRecallResult);

    bool TopK(uint8_t*& docScoreInDevice, uint8_t*& docLocationInDeviceResult, uint8_t*& resultNextInDevice,
              uint8_t*& resultLocationNextInDevice, uint32_t blockNumberAggregator, uint32_t blockDimAggregator,
              uint32_t* docNumberShards, uint32_t topK, uint32_t& effectiveCount, uint32_t& topKEffectiveCountSmaller);

    bool Merge(uint8_t*& docScoreInDevice, uint8_t*& docLocationInDeviceResult, uint32_t blockDimAggregator,
               uint32_t* docNumberShards, uint32_t shardAllByte);

   private:
    int32_t m_deviceId{};
    const DocIdMapping* m_docIdMapping = nullptr;
    uint32_t m_topK{1};
};
}  // namespace NpuRetrieval
