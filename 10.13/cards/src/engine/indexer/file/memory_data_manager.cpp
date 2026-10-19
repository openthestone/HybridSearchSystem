#include "memory_data_manager.h"
#include <fstream>
#include "src/utils/logger.h"

namespace NpuRetrieval {
void MemoryDataManager::AddFieldData(const std::string& fieldName, uint32_t segmentId,
                                     const std::vector<uint8_t>& data) {
    std::string key = fieldName + "_" + std::to_string(segmentId);
    m_fieldDataMap[key] = std::make_shared<std::vector<uint8_t>>(data);
    LOG_INFO("Added field data: " << key << ", size: " << data.size());
}

std::shared_ptr<std::vector<uint8_t>> MemoryDataManager::GetFieldData(const std::string& fieldName,
                                                                      uint32_t segmentId) const {
    std::string key = fieldName + "_" + std::to_string(segmentId);
    auto it = m_fieldDataMap.find(key);
    if (it != m_fieldDataMap.end()) {
        return it->second;
    }
    return nullptr;
}

void MemoryDataManager::Clear() {
    m_fieldDataMap.clear();
}
}  // namespace NpuRetrieval
