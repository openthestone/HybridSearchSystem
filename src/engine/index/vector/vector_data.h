#pragma once

#include <memory>
#include <unordered_map>
#include "vector_field_data.h"

namespace NpuRetrieval {
class VectorData {
   public:
    VectorData(uint32_t version, uint32_t segmentNum, const std::string& dataDir)
        : m_version(version), m_segmentNum(segmentNum), m_dataDir(dataDir) {}

    ~VectorData() = default;

    /*
     * Load all segments of one field's vectors.
     */
    bool AddFieldData(const std::string& fieldName);

    bool GetFieldData(const std::string& fieldName, VectorFieldData*& fieldData) const;

    bool CheckData(uint32_t splitDocNumZn) const;

   private:
    bool ResolveFieldDir(const std::string& fieldName, std::string& realInputDir);
    bool ResolveSegmentPath(const std::string& inputDir, const std::string& fieldName, uint32_t segmentId,
                            std::string& filePath);
    std::unordered_map<std::string, std::unique_ptr<VectorFieldData>> m_fields{};
    uint32_t m_version{};
    uint32_t m_segmentNum{};
    std::string m_dataDir{};
};
}  // namespace NpuRetrieval