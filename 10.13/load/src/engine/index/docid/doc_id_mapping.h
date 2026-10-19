#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>

namespace NpuRetrieval {
class DocIdMapping {
   public:
    DocIdMapping(uint32_t version, uint32_t segmentNum, uint32_t docNum, uint32_t docNumPerSegment,
                 const std::string& dataDir)
        : m_version(version),
          m_segmentNum(segmentNum),
          m_docNum(docNum),
          m_docNumPerSegment(docNumPerSegment),
          m_dataDir(dataDir) {}

    ~DocIdMapping() = default;

    bool Load();

    bool GetGDocId(uint32_t lDocId, uint64_t& gDocid) const;

    // The global-id array itself, for a caller that knows its lookups in advance and wants to
    // prefetch them. Reads still go through GetGDocId, which is the bounds-checked accessor.
    const uint64_t* GDocIdData() const {
        return m_gDocIds.data();
    }
    size_t GDocIdCount() const {
        return m_gDocIds.size();
    }

    std::string GetSDocId(uint32_t lDocId) const;

    bool GetLDocId(uint64_t gDocId, uint32_t& lDocid) const;

    bool GetLDocId(const std::string& sDocId, uint32_t& lDocid) const;

   private:
    bool ResolveIdMappingDir(std::string& realInputDir);
    void WarmGDocIds() const;
    bool ResolveSegmentPath(const std::string& inputDir, uint32_t segmentId, std::string& filePath);
    bool ReadGdocidSegment(std::ifstream& ifs, uint32_t& ldocid);
    bool ReadSdocidSegment(std::ifstream& ifs, uint32_t segmentId, uint32_t& ldocid);
    std::vector<uint64_t> m_gDocIds;
    std::vector<std::string> m_sDocIds;
    std::unordered_map<uint64_t, uint32_t> m_gDocId2LDocId;
    std::unordered_map<std::string, uint32_t> m_sDocId2LDocId;
    uint32_t m_version{};
    uint32_t m_segmentNum{};
    uint32_t m_docNum{};
    uint32_t m_docNumPerSegment{};
    std::string m_dataDir{};
    uint32_t m_headerLength{};
};
}  // namespace NpuRetrieval
