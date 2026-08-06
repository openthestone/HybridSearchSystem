#pragma once
#include "src/utils/logger.h"

namespace NpuRetrieval {
class MockDataTable {
   public:
    MockDataTable() = default;

    ~MockDataTable() = default;

    bool LoadData(const int32_t deviceId, const std::string& dataDir) {
        LOG_WARN("Mock function, deviceId:" << deviceId << " dataDir:" << dataDir);
        return true;
    };
};

using DataTable = MockDataTable;
}  // namespace NpuRetrieval