#include "posting_data.h"
#include <filesystem>
#include "src/utils/file_manager.h"
#include "src/utils/logger.h"
#include "src/full_recall/core/constant_definition.h"

namespace NpuRetrieval {
bool PostingData::ResolveFieldDir(const std::string& fieldName, std::string& realInputDir) {
    // input directory
    std::string fieldDir = m_dataDir + "/" + fieldName;
    if (!GetRealFilePath(fieldDir, realInputDir)) {
        LOG_ERROR("GetRealFilePath of field:" << fieldName << " failed.");
        return false;
    }
    std::error_code ec;
    if (!std::filesystem::exists(realInputDir, ec) || !std::filesystem::is_directory(realInputDir, ec)) {
        LOG_ERROR("fieldDir path not exists or is not a directory. file path:" << realInputDir);
        return false;
    }
    return true;
}

bool PostingData::ResolveSegmentPath(const std::string& inputDir, const std::string& fieldName, uint32_t segmentId,
                                     std::string& filePath) {
    filePath = inputDir + "/" + POISSONENGINE + UNDERLINE + std::to_string(m_version) + UNDERLINE +
               std::to_string(segmentId) + UNDERLINE + fieldName + INVERTED_FILE_SUFFIX;
    // check the file exists
    std::error_code ec;
    if (!std::filesystem::exists(filePath, ec)) {
        LOG_ERROR("posting file not exists. file path:" << filePath);
        return false;
    }
    return true;
}

bool PostingData::AddFieldData(const std::string& fieldName) {
    LOG_DEBUG("posting data start to add field :" << fieldName);
    // read the segment files in order
    std::string fieldDir;
    if (!ResolveFieldDir(fieldName, fieldDir)) {
        return false;
    }
    auto fieldData = std::make_unique<PostingFieldData>(m_segmentNum, m_docNum, m_docNumPerSegment, fieldName);
    for (uint32_t i = 0; i < m_segmentNum; i++) {
        std::string filePath;
        if (!ResolveSegmentPath(fieldDir, fieldName, i, filePath)) {
            return false;
        }
        std::ifstream ifs(filePath, std::ios::binary | std::ios::in);
        if (!ifs.is_open()) {
            LOG_ERROR("open posting file failed, file:" << filePath);
            return false;
        }
        if (!fieldData->AddSegment(ifs, i)) {
            LOG_ERROR("posting add segment failed, file:" << filePath);
            return false;
        }
    }
    if (!fieldData->FinishAdd()) {
        LOG_ERROR("posting data finish add failed, fieldName:" << fieldName);
        return false;
    }
    m_fields[fieldName] = std::move(fieldData);
    return true;
}

bool PostingData::GetFieldData(const std::string& fieldName, PostingFieldData*& fieldData) const {
    auto iter = m_fields.find(fieldName);
    if (iter == m_fields.end()) {
        LOG_ERROR("GetFieldData failed, fieldName:" << fieldName << " not found");
        return false;
    }
    fieldData = iter->second.get();
    return true;
}
}  // namespace NpuRetrieval