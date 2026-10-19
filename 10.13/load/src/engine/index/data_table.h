#pragma once

#include "posting/posting_data.h"
#include "vector/vector_data.h"
#include "docid/doc_id_mapping.h"
#include "common/uncopyable.h"
#include "src/full_recall/indexer/proto/index_meta.pb.h"

namespace NpuRetrieval {
class DataTable {
   public:
    DataTable() = default;

    ~DataTable() {
        if (m_docLocationInDevice != nullptr) {
            CHECK_ACL_ONLY_LOG(aclrtFree(m_docLocationInDevice));
        }
    }

    bool LoadData(const int32_t deviceId, const std::string& dataDir);

    bool GetPostingFieldData(const std::string& fieldName, PostingFieldData*& fieldData) const;

    bool GetVectorFieldData(const std::string& fieldName, VectorFieldData*& fieldData) const;

    DocIdMapping* GetDocIdMapping() const {
        return m_docIdMapping.get();
    }

    uint32_t GetSegmentNum() const {
        return m_segmentNum;
    }

    uint32_t GetSegmentLength() const {
        return m_segmentLength;
    }

    uint32_t GetSegmentByteSize() const {
        return m_segmentByteSize;
    }

    uint32_t GetDocNum() const {
        return m_docNum;
    }

    uint32_t GetScoreExtendDocNum() const {
        return m_scoreExtendDocNum;
    }

    uint32_t GetDocNumPerSegment() const {
        return m_docNumPerSegment;
    }

    uint8_t* GetDocLocationInDevice() const {
        return m_docLocationInDevice;
    }

    uint32_t GetSplitDocNumZn() const {
        return m_splitDocNumZn;
    }

    const std::unordered_set<std::string>& GetPostingFields() const {
        return m_postingFields;
    }

    int32_t GetDeviceId() const {
        return m_deviceId;
    }

    NPURETRIEVAL_DECLARE_UNCOPYABLE(DataTable);

   private:
    bool ResolveDataDir(const std::string& dataDir, std::string& realDataDir);
    bool LoadIndexMeta(const std::string& dataDir, Building::Meta::IndexMeta& indexMeta);
    bool UploadDocLocation();
    std::unique_ptr<PostingData> m_postingData{};
    std::unique_ptr<VectorData> m_vectorData{};
    std::unique_ptr<DocIdMapping> m_docIdMapping{};
    uint32_t m_segmentNum{};
    uint32_t m_segmentLength{};
    uint32_t m_segmentByteSize{};
    uint32_t m_docNum{};
    // scoring byte-alignment padding must be >= the aggregator padding; currently satisfied
    uint32_t m_scoreExtendDocNum{};
    uint32_t m_docNumPerSegment{};
    uint32_t m_splitDocNumZn{};
    std::unordered_set<std::string> m_postingFields;
    std::vector<std::string> m_vectorFields;
    uint32_t m_version{};
    uint8_t* m_docLocationInDevice{};
    int32_t m_deviceId{-1};
};
}  // namespace NpuRetrieval