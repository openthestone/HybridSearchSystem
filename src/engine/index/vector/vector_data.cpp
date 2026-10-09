#include "vector_data.h"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include "src/utils/file_manager.h"
#include "src/utils/logger.h"
#include "src/full_recall/core/constant_definition.h"

namespace NpuRetrieval {
bool VectorData::ResolveFieldDir(const std::string& fieldName, std::string& realInputDir) {
    std::string fieldDir = m_dataDir + "/" + fieldName;
    if (!GetRealFilePath(fieldDir, realInputDir)) {
        LOG_ERROR("GetRealFilePath of fieldDir failed. file path:" << fieldDir);
        return false;
    }
    std::error_code ec;
    if (!std::filesystem::exists(realInputDir, ec) || !std::filesystem::is_directory(realInputDir, ec)) {
        LOG_ERROR("fieldDir path not exists or is not a directory. file path:" << realInputDir);
        return false;
    }
    return true;
}

bool VectorData::ResolveSegmentPath(const std::string& inputDir, const std::string& fieldName, uint32_t segmentId,
                                    std::string& filePath) {
    filePath = inputDir + "/" + POISSONENGINE + UNDERLINE + std::to_string(m_version) + UNDERLINE +
               std::to_string(segmentId) + UNDERLINE + fieldName + VECTOR_FILE_SUFFIX;
    std::error_code ec;
    if (!std::filesystem::exists(filePath, ec)) {
        LOG_ERROR("vector file not exists. file path:" << filePath);
        return false;
    }
    return true;
}

bool VectorData::AddFieldData(const std::string& fieldName) {
    LOG_INFO("vector data start to add field :" << fieldName);
    std::string fieldDir;
    if (!ResolveFieldDir(fieldName, fieldDir)) {
        return false;
    }
    auto fieldData = std::make_unique<VectorFieldData>();
    // AddSegment grows one host buffer per segment; FinishAdd is the one big H2D.
    auto readT0 = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < m_segmentNum; i++) {
        std::string filePath;
        if (!ResolveSegmentPath(fieldDir, fieldName, i, filePath)) {
            return false;
        }
        std::ifstream ifs(filePath, std::ios::binary | std::ios::in);
        if (!ifs.is_open()) {
            LOG_ERROR("open vector file failed, file:" << filePath);
            return false;
        }
        std::error_code ec;
        auto fileSize = std::filesystem::file_size(filePath, ec);
        if (!fieldData->AddSegment(ifs, fileSize)) {
            LOG_ERROR("vector add segment failed, file:" << filePath);
            return false;
        }
    }
    const long long readMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - readT0).count();
    auto upT0 = std::chrono::steady_clock::now();
    if (!fieldData->FinishAdd()) {
        LOG_ERROR("vector finish add failed, fieldName:" << fieldName);
        return false;
    }
    const long long uploadMs =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - upT0).count();
    m_loadReadMs += readMs;
    m_loadH2dMs += uploadMs;
    m_fields[fieldName] = std::move(fieldData);
    LOG_INFO("vector data end to add field :" << fieldName);
    return true;
}

bool VectorData::GetFieldData(const std::string& fieldName, VectorFieldData*& fieldData) const {
    auto iter = m_fields.find(fieldName);
    if (iter == m_fields.end()) {
        LOG_ERROR("GetFieldData failed, fieldName:" << fieldName << " not found");
        return false;
    }
    fieldData = iter->second.get();
    return true;
}

bool VectorData::CheckData(uint32_t splitDocNumZn) const {
    if (m_fields.empty()) {
        LOG_ERROR("field data is empty.");
        return false;
    }
    uint32_t dimension{};
    for (const auto& [key, value] : m_fields) {
        dimension = value->GetDimension();
        break;
    }
    if (dimension == 0) {
        LOG_ERROR("invalid vector data, dimension is zero");
        return false;
    }
    /*
        {vector dimension} * {split_doc_num_zn} * {buffer count} * sizeof(half) <= {L0B byte size}
        910B4 L0B byte size = 64k, so:
        32-dim:   32*512*2*2 = 65536
        64-dim:   64*256*2*2 = 65536
    */
    static constexpr uint32_t MAX_L0B_AVAILABLE_MEM_BYTES = 65536 / 2 / 2;
    const uint32_t maxSplitDocNumZn = MAX_L0B_AVAILABLE_MEM_BYTES / dimension;
    if ((splitDocNumZn == 0) || (splitDocNumZn > maxSplitDocNumZn)) {
        LOG_ERROR("invalid vector data, splitDocNumZn=" << splitDocNumZn << ", maxSplitDocNumZn=" << maxSplitDocNumZn);
        return false;
    }
    return true;
}
}  // namespace NpuRetrieval