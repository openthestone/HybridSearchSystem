#pragma once

#include <memory>
#include <unordered_map>
#include "posting_field_data.h"

namespace NpuRetrieval {
class PostingData {
   public:
    PostingData(uint32_t version, uint32_t segmentNum, const std::string& dataDir, uint32_t docNum,
                uint32_t docNumPerSegment)
        : m_version(version),
          m_segmentNum(segmentNum),
          m_dataDir(dataDir),
          m_docNum(docNum),
          m_docNumPerSegment(docNumPerSegment) {}

    ~PostingData() = default;

    bool AddFieldData(const std::string& fieldName);

    bool GetFieldData(const std::string& fieldName, PostingFieldData*& fieldData) const;

   private:
    bool ResolveFieldDir(const std::string& fieldName, std::string& realInputDir);
    bool ResolveSegmentPath(const std::string& inputDir, const std::string& fieldName, uint32_t segmentId,
                            std::string& filePath);
    std::unordered_map<std::string, std::unique_ptr<PostingFieldData>> m_fields{};
    uint32_t m_version{};
    uint32_t m_segmentNum{};
    std::string m_dataDir{};
    uint32_t m_docNum{};
    uint32_t m_docNumPerSegment{};
};
}  // namespace NpuRetrieval