#pragma once
#include <cstdint>
#include <vector>
#include "src/full_recall/index/data_table.h"
#include "src/full_recall/retrieval/searcher/runtime/gm_memory_manager.h"

namespace NpuRetrieval {

// A scorer kernel that has been launched but not yet waited on. It occupies the NPU for the whole
// gap between Launch and Sync, so the caller may run host work in between.
struct ScorerLaunch {
    aclrtStream stream{nullptr};
    uint8_t* queryInDevice{nullptr};
    // Set when queryInDevice came from SCORER_QUERY_POOL; Sync hands it back instead of freeing.
    GmBlock queryChunk{GmPoolName::SCORER_QUERY_POOL, nullptr, 0};
    // false when there is nothing to wait for -- launch failed and cleaned up after itself, or this
    // is an ASCENDC_CPU_DEBUG build where the kernel runs synchronously.
    bool launched{false};
};

class VectorScorerMmad {
   public:
    explicit VectorScorerMmad(const int32_t deviceId, const std::shared_ptr<DataTable> dataTable)
        : m_deviceId(deviceId), m_dataTable(dataTable) {}

    ~VectorScorerMmad() {}

    bool BatchCompute(const std::vector<uint16_t>& queryVectors, uint32_t queryDimension, const std::string& fieldName,
                      GmBlock& resultChunk);

    // Call Sync exactly once per Launch, including on the failure path: Launch reports failure but
    // Sync still has to release whatever it managed to take.
    bool BatchComputeLaunch(const std::vector<uint16_t>& queryVectors, uint32_t queryDimension,
                            const std::string& fieldName, GmBlock& resultChunk, ScorerLaunch& launch);
    bool BatchComputeSync(ScorerLaunch& launch);

   private:
    uint16_t GetB1N(const uint32_t dimension);

    const int32_t m_deviceId;
    const std::shared_ptr<DataTable> m_dataTable{};
};
}  // namespace NpuRetrieval
