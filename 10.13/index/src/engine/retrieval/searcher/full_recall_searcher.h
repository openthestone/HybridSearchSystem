#pragma once

#include <cstdint>
#include <vector>
#include "src/full_recall/index/data_table.h"
#include "stage/text_filter.h"
#include "stage/vector_scorer_mmad.h"
#include "stage/result_aggregator.h"
#include "src/workflow/async/executor.h"

namespace NpuRetrieval {
class FullRecallSearcher {
   public:
    FullRecallSearcher(const int32_t deviceId, const std::shared_ptr<DataTable> dataTable)
        : m_deviceId(deviceId),
          m_dataTable(dataTable),
          m_textFilter(deviceId, dataTable),
          m_vectorScorerMmad(deviceId, dataTable) {
        if (dataTable != nullptr) {
            m_docIdMapping = dataTable->GetDocIdMapping();
            m_docLocationInDevice = dataTable->GetDocLocationInDevice();
        }
        m_executor = CreateExecutor();
    }

    ~FullRecallSearcher() {
        m_docIdMapping = nullptr;
    }

    void AddQuery(QueryNode* queryTrees, const std::vector<float>& queryVectors, uint32_t topK);

    bool BatchSearch(std::vector<std::shared_ptr<FullRecallResult>>& fullRecallResults,
                     const std::string& vectorFieldName, bool isMultiShard);

    // BatchSearch split into a device stage and a pure-CPU host stage, so a caller can overlap
    // batch N's Extract with batch N+1's Device. deviceResults must be passed unchanged between them.
    bool BatchSearchDevice(const std::string& vectorFieldName, bool isMultiShard,
                           std::vector<AggrDeviceResult>& deviceResults);
    bool BatchSearchExtract(std::vector<AggrDeviceResult>& deviceResults,
                            std::vector<std::shared_ptr<FullRecallResult>>& fullRecallResults, bool isMultiShard);

    // the following combination is used for the vector-scoring-only path
    bool OneFilter(GmBlock& filterResultChunk);
    bool BatchScore(const std::string& vectorFieldName, GmBlock& resultChunk);

    bool OneAggregator(GmBlock& resultChunk, GmBlock& filterResultChunk, uint32_t resultInDeviceOffset,
                       bool isMultiShard, std::vector<std::shared_ptr<FullRecallResult>>& fullRecallResults);

   private:
    bool RunAggregation(uint8_t* filterResultInDevice, uint8_t* resultInDevice,
                        std::vector<std::shared_ptr<FullRecallResult>>& fullRecallResults, bool isMultiShard);
    // sharedStream / sharedStreamSynced: see ResultAggregator::AggrAndTopKDeviceBatch.
    bool RunAggregationDevice(uint8_t* filterResultInDevice, uint8_t* resultInDevice,
                              std::vector<AggrDeviceResult>& deviceResults, aclrtStream sharedStream = nullptr,
                              bool* sharedStreamSynced = nullptr);
    bool RunAggregationExtract(std::vector<AggrDeviceResult>& deviceResults,
                               std::vector<std::shared_ptr<FullRecallResult>>& fullRecallResults, bool isMultiShard);

    const int32_t m_deviceId{};
    const std::shared_ptr<DataTable> m_dataTable{};
    const DocIdMapping* m_docIdMapping{};
    uint8_t* m_docLocationInDevice{};
    TextFilter m_textFilter;
    VectorScorerMmad m_vectorScorerMmad;
    std::vector<QueryNode*> m_queries{};
    std::vector<uint16_t> m_queryVectors{};
    std::vector<uint32_t> m_topKs{};
    uint32_t m_dimension{};
    std::shared_ptr<Executor> m_executor = nullptr;
};
}  // namespace NpuRetrieval
