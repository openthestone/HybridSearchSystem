#pragma once
#include <cstdint>
#include <vector>
#include "src/full_recall/index/data_table.h"
#include "src/full_recall/retrieval/searcher/runtime/gm_memory_manager.h"

namespace NpuRetrieval {

// A scorer kernel that has been launched but not yet waited on. The kernel occupies
// the NPU for the whole gap between BatchComputeLaunch and BatchComputeSync, so the
// caller may run host work in between. Holds the resources Sync has to release.
struct ScorerLaunch {
    aclrtStream stream{nullptr};
    uint8_t* queryInDevice{nullptr};
    // false when there is nothing to wait for -- launch failed and already cleaned
    // up after itself, or this is an ASCENDC_CPU_DEBUG build where the kernel runs
    // synchronously. Sync is then a no-op.
    bool launched{false};
};

class VectorScorerMmad {
   public:
    explicit VectorScorerMmad(const int32_t deviceId, const std::shared_ptr<DataTable> dataTable)
        : m_deviceId(deviceId), m_dataTable(dataTable) {}

    ~VectorScorerMmad() {}

    // Launch + Sync back to back; the NPU is idle for none of it and the host waits.
    bool BatchCompute(const std::vector<uint16_t>& queryVectors, uint32_t queryDimension, const std::string& fieldName,
                      GmBlock& resultChunk);

    // The same work split so host code can fill the kernel's execution window. Call
    // Sync exactly once per Launch, including on the failure path: Launch reports
    // failure but Sync still has to release whatever it managed to take.
    bool BatchComputeLaunch(const std::vector<uint16_t>& queryVectors, uint32_t queryDimension,
                            const std::string& fieldName, GmBlock& resultChunk, ScorerLaunch& launch);
    bool BatchComputeSync(ScorerLaunch& launch);

   private:
    uint16_t GetB1N(const uint32_t dimension);

    const int32_t m_deviceId;
    const std::shared_ptr<DataTable> m_dataTable{};
};
}  // namespace NpuRetrieval
