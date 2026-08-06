#pragma once

#include <cstdint>
#include <vector>
#include "src/full_recall/index/docid/doc_id_mapping.h"
#include "src/full_recall/core/full_recall_result.h"
#include "full_recall/retrieval/searcher/runtime/gm_memory_pool.h"

namespace NpuRetrieval {

class ResultAggregator {
   public:
    ResultAggregator() = default;

    ~ResultAggregator() = default;

    // Applies NPUR_SHARD_TOPK_RATIO: stores the per-shard aggregation top-K, which may be
    // smaller than the final top-K (the host merge still assembles the full top-K from all
    // shards). Defined in the .cpp so it can read the ratio env.
    void SetTopK(uint32_t topK);

    void SetDocIdMapping(const DocIdMapping* docIdMapping) {
        m_docIdMapping = docIdMapping;
    }

    void SetDeviceId(const int32_t deviceId) {
        m_deviceId = deviceId;
    }

    bool AggrAndTopK(uint8_t* filterResultInDevice, uint8_t* docScoreInDevice, size_t docAllByteSize,
                     uint8_t*& docLocationInDeviceConst, std::shared_ptr<FullRecallResult>& fullRecallResult);

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
