#pragma once
#include <string>
#include <unordered_map>
#include "src/full_recall/indexer/proto/index_builder.pb.h"
namespace NpuRetrieval {
struct DocIdInfo {
    std::string name;
    bool isSdocid = false;
};

class BuilderSchema {
   public:
    bool Initialize(const std::string& data);

    const DocIdInfo& GetDocid() {
        return m_docIdInfo;
    }

    const std::unordered_map<std::string, Building::SectionInvertedInfo>& GetInvertedFiled() {
        return m_inverted;
    }

    const std::unordered_map<std::string, std::uint16_t>& GetEmbeddingFiled() {
        return m_embedding;
    }

    uint8_t GetVersion() {
        return m_version;
    }

    uint32_t GetSegmentNum() {
        return m_segmentNum;
    }

    uint32_t GetDocNum() {
        return m_docNum;
    }

    uint32_t GetDocNumPerSegment() {
        return m_docNumPerSegment;
    }

    uint32_t GetSplitDocNumZn() {
        return m_splitDocNumZn;
    }

   private:
    bool ParseAttachment(const Building::IndexBuilderConfig& config);
    bool ParseSection(const Building::IndexBuilderConfig& config);
    DocIdInfo m_docIdInfo;
    uint8_t m_version = 0;
    std::unordered_map<std::string, Building::SectionInvertedInfo> m_inverted;  // posting field name
    std::unordered_map<std::string, std::uint16_t> m_embedding;                 // vector <name, dimension>
    uint32_t m_segmentNum = 0;                                                  // number of segments
    uint32_t m_docNum = 0;                                                      // total number of docs
    uint32_t m_docNumPerSegment = 0;                                            // docs per segment
    uint32_t m_splitDocNumZn = 0;
};
}  // namespace NpuRetrieval
