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

    // Load-time attribution for DataTable's one-line [Load] summary, summed over fields.
    long long LoadReadMs() const {
        return m_loadReadMs;
    }
    long long LoadDictMs() const {
        return m_loadDictMs;
    }
    long long LoadBulkMs() const {
        return m_loadBulkMs;
    }
    long long LoadH2dMs() const {
        return m_loadH2dMs;
    }
    uint64_t LoadTokens() const {
        return m_loadTokens;
    }

   private:
    long long m_loadReadMs{0};
    long long m_loadDictMs{0};
    long long m_loadBulkMs{0};
    long long m_loadH2dMs{0};
    uint64_t m_loadTokens{0};
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