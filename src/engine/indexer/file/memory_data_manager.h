#pragma once

#include <functional>
#include <string>
#include <memory>
#include "src/full_recall/core/constant_definition.h"

namespace NpuRetrieval {
class MemoryDataManager {
   public:
    static MemoryDataManager* Instance() {
        static MemoryDataManager value;
        return &value;
    }
    void AddFieldData(const std::string& fieldName, uint32_t segmentId, const std::vector<uint8_t>& data);
    std::shared_ptr<std::vector<uint8_t>> GetFieldData(const std::string& fieldName, uint32_t segmentId) const;
    void Clear();

   private:
    std::unordered_map<std::string, std::shared_ptr<std::vector<uint8_t>>> m_fieldDataMap;
};
}  // namespace NpuRetrieval
