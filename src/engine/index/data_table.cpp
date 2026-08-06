#include "data_table.h"
#include <filesystem>
#include "google/protobuf/json/json.h"
#include "src/utils/file_manager.h"
#include "src/utils/logger.h"
#include "src/utils/string_util.h"
#include "src/full_recall/core/constant_definition.h"
#include "src/full_recall/core/number/trans_number.h"

namespace NpuRetrieval {
bool DataTable::ResolveDataDir(const std::string& dataDir, std::string& realDataDir) {
    if (!GetRealFilePath(dataDir, realDataDir)) {
        LOG_ERROR("GetRealFilePath check failed.");
        return false;
    }
    std::error_code ec;
    if (!std::filesystem::exists(realDataDir, ec)) {
        LOG_ERROR("input path not exists. file path:" << realDataDir);
        return false;
    }
    return true;
}

bool DataTable::LoadData(const int32_t deviceId, const std::string& dataDir) {
    std::string realDataDir;
    if (!ResolveDataDir(dataDir, realDataDir)) {
        return false;
    }
    auto ret = aclrtSetDevice(deviceId);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtSetDevice fail, deviceId is:" << deviceId << " error code is:" << ret);
        return false;
    }
    m_deviceId = deviceId;
    // load the meta file
    Building::Meta::IndexMeta indexMeta;
    if (!LoadIndexMeta(realDataDir, indexMeta)) {
        LOG_ERROR("LoadIndexMeta fail");
        return false;
    }
    m_segmentNum = indexMeta.statistics().segment_num();
    m_docNum = indexMeta.statistics().doc_num();
    m_docNumPerSegment = indexMeta.config().doc_num_per_segment();
    if (m_docNumPerSegment % 8 != 0) {  // data must be 8-byte aligned
        LOG_ERROR("docNumPerSegment invalid, shoule be a multiple of 8, current value:" << m_docNumPerSegment);
        return false;
    }
    m_splitDocNumZn = indexMeta.config().split_doc_num_zn();
    m_scoreExtendDocNum = ((m_docNum - 1) / m_splitDocNumZn + 1) * m_splitDocNumZn;
    m_segmentLength = m_docNumPerSegment / 16;   // 16 = bits per uint16
    m_segmentByteSize = m_docNumPerSegment / 8;  // 8 docs per byte
    if (m_splitDocNumZn == 0) {
        LOG_ERROR("split_doc_num_zn is 0, please check");
        return false;
    }
    // load the id mapping
    m_docIdMapping = std::make_unique<DocIdMapping>(m_version, m_segmentNum, m_docNum, m_docNumPerSegment, realDataDir);
    if (!m_docIdMapping->Load()) {
        LOG_ERROR("Load idmapping fail");
        return false;
    }
    // load the vectors
    m_vectorData = std::make_unique<VectorData>(m_version, m_segmentNum, realDataDir);
    for (auto& fieldName : indexMeta.schema().vector()) {
        if (!m_vectorData->AddFieldData(fieldName)) {
            LOG_ERROR("Load vector fail");
            return false;
        }
    }
    // validate the vector data
    if (!m_vectorData->CheckData(m_splitDocNumZn)) {
        LOG_ERROR("check vector data fail");
        return false;
    }
    // load the postings (inverted index)
    m_postingData = std::make_unique<PostingData>(m_version, m_segmentNum, realDataDir, m_docNum, m_docNumPerSegment);
    for (auto& fieldName : indexMeta.schema().posting()) {
        if (!m_postingData->AddFieldData(fieldName)) {
            LOG_ERROR("Load posting fail");
            return false;
        }
        m_postingFields.insert(fieldName);
    }

    // upload the identity doc-location array used by the aggregator kernel
    if (!UploadDocLocation()) {
        return false;
    }
    LOG_INFO("data table load success. dataDir="
             << dataDir << ", deviceId=" << deviceId << ", m_segmentNum=" << m_segmentNum
             << ",m_segmentLength=" << m_segmentLength << ",m_segmentByteSize=" << m_segmentByteSize
             << ",m_docNum=" << m_docNum << ",m_docNumPerSegment=" << m_docNumPerSegment
             << ",m_splitDocNumZn=" << m_splitDocNumZn << ",m_version=" << m_version);
    return true;
}

bool DataTable::UploadDocLocation() {
    uint32_t aggregatorExtendDocNum = GetNpuDocNumber(m_docNum);
    uint32_t docLocationByteSize = aggregatorExtendDocNum * sizeof(float);
    std::vector<uint32_t> docLocationHost(aggregatorExtendDocNum);
    for (uint32_t i = 0; i < aggregatorExtendDocNum; ++i) {
        docLocationHost[i] = i;
    }
    auto ret = aclrtMalloc((void**)&m_docLocationInDevice, docLocationByteSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("docLocationInDevice aclrtMalloc fail, error code is:" << ret);
        return false;
    }
    ret = aclrtMemcpy(m_docLocationInDevice, docLocationByteSize, (void*)docLocationHost.data(), docLocationByteSize,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("docLocationInDevice aclrtMemcpy  fail, error code is:" << ret);
        return false;
    }
    return true;
}

bool DataTable::LoadIndexMeta(const std::string& dataDir, Building::Meta::IndexMeta& indexMeta) {
    // locate the single meta file in the data dir
    bool metaFound = false;
    std::filesystem::path metaPath{};
    std::string fileNameWithoutExt;
    for (const auto& entry : std::filesystem::directory_iterator(dataDir)) {
        if (entry.path().extension().string() == META_FILE_SUFFIX) {
            metaPath = entry.path();
            fileNameWithoutExt = entry.path().stem().string();
            metaFound = true;
            break;
        }
    }
    if (!metaFound) {
        LOG_ERROR("meta file not exist");
        return false;
    }
    // parse the version
    std::vector<std::string> nameSplit;
    StringSplit(fileNameWithoutExt, '_', nameSplit);
    if (nameSplit.size() < 2 || !StringToNumber(nameSplit[1], m_version)) {  // nameSplit must have >= 2 parts
        LOG_ERROR("meta filename invalid");
        return false;
    }

    // parse the file
    std::ifstream ifs(metaPath);
    if (!ifs.is_open()) {
        LOG_ERROR("read meta file fail");
        return false;
    }
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    google::protobuf::json::ParseOptions deserializeOptions;
    auto parseStatus = google::protobuf::json::JsonStringToMessage(buffer.str(), &indexMeta, deserializeOptions);
    if (!parseStatus.ok()) {
        LOG_ERROR("fail to parse, errorCode=" << parseStatus.code() << ", errorMsg=" << parseStatus.message()
                                              << ", data: " << buffer.str());
        return false;
    }
    LOG_INFO("parse meta result: " << indexMeta.ShortDebugString());
    return true;
}

bool DataTable::GetPostingFieldData(const std::string& fieldName, PostingFieldData*& fieldData) const {
    if (m_postingData == nullptr) {
        LOG_ERROR("m_postingData is null");
        return false;
    }
    return m_postingData->GetFieldData(fieldName, fieldData);
}

bool DataTable::GetVectorFieldData(const std::string& fieldName, VectorFieldData*& fieldData) const {
    if (m_vectorData == nullptr) {
        LOG_ERROR("m_vectorData is null");
        return false;
    }
    return m_vectorData->GetFieldData(fieldName, fieldData);
}
}  // namespace NpuRetrieval
