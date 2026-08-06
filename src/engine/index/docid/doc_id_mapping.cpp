#include "doc_id_mapping.h"
#include <filesystem>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#endif
#include "src/utils/file_manager.h"
#include "src/utils/logger.h"
#include "src/full_recall/core/constant_definition.h"
#include "src/full_recall/format/file_header.h"

namespace NpuRetrieval {
bool DocIdMapping::ResolveIdMappingDir(std::string& realInputDir) {
    // input directory
    std::string idMappingDir = m_dataDir + "/" + ID_MAPPING_DIR;
    if (!GetRealFilePath(idMappingDir, realInputDir)) {
        LOG_ERROR("GetRealFilePath of idMappingDir failed.");
        return false;
    }
    std::error_code ec;
    if (!std::filesystem::exists(realInputDir, ec) || !std::filesystem::is_directory(realInputDir, ec)) {
        LOG_ERROR("idMappingDir path not exists or is not a directory. file path:" << realInputDir);
        return false;
    }
    return true;
}

bool DocIdMapping::ResolveSegmentPath(const std::string& inputDir, uint32_t segmentId, std::string& filePath) {
    filePath = inputDir + "/" + POISSONENGINE + UNDERLINE + std::to_string(m_version) + UNDERLINE +
               std::to_string(segmentId) + ID_MAPPING_FILE_SUFFIX;
    // check the file exists
    std::error_code ec;
    if (!std::filesystem::exists(filePath, ec)) {
        LOG_ERROR("idMapping file not exists. file path:" << filePath);
        return false;
    }
    return true;
}

bool DocIdMapping::Load() {
    // read the idm segment files in order
    std::string idMappingDir;
    if (!ResolveIdMappingDir(idMappingDir)) {
        return false;
    }
    uint32_t ldocid = 0;
    for (uint32_t i = 0; i < m_segmentNum; i++) {
        std::string filePath;
        if (!ResolveSegmentPath(idMappingDir, i, filePath)) {
            return false;
        }
        // open the file in binary mode
        std::ifstream ifs(filePath, std::ios::binary | std::ios::in);
        if (!ifs.is_open()) {
            LOG_ERROR("open idmapping file failed, file:" << filePath);
            return false;
        }
        uint8_t version = 0;
        std::string extended;
        if (!DecodeFileHeader(ifs, version, extended)) {
            ifs.close();
            return false;
        }
        m_headerLength = static_cast<uint32_t>(ifs.tellg());
        // version check
        if (version != m_version) {
            LOG_ERROR("m_version:" << m_version << " version:" << version << " not same, please check");
            ifs.close();
            return false;
        }
        // the extended field holds the gdocid/sdocid tag
        if (extended == SDOCID_TYPE) {
            if (!ReadSdocidSegment(ifs, i, ldocid)) {
                ifs.close();
                return false;
            }
        } else {
            if (!ReadGdocidSegment(ifs, ldocid)) {
                ifs.close();
                return false;
            }
        }
        ifs.close();
    }
    WarmGDocIds();
    LOG_DEBUG("Load success");
    return true;
}

void DocIdMapping::WarmGDocIds() const {
    // create_result (result_aggregator.cpp) does ~topK random GetGDocId lookups
    // into m_gDocIds (docNum * 8 bytes, ~80MB at 10M). Under memory pressure those
    // pages get reclaimed, so the FIRST query after load faults them back in one at
    // a time -> a cold-start p99 spike (seen: create_result x61, max ~470ms). Hint
    // the kernel to keep/prefetch the whole array resident so lookups stay O(1).
#if defined(MADV_WILLNEED)
    if (!m_gDocIds.empty()) {
        madvise(const_cast<uint64_t*>(m_gDocIds.data()), m_gDocIds.size() * sizeof(uint64_t), MADV_WILLNEED);
    }
#endif
}

bool DocIdMapping::ReadGdocidSegment(std::ifstream& ifs, uint32_t& ldocid) {
    GlobalDocID gdocid = 0;
    while (ifs.read(reinterpret_cast<char*>(&gdocid), sizeof(gdocid))) {
        if (ifs.gcount() != sizeof(gdocid)) {
            LOG_ERROR("gdocid invalid");
            return false;
        }
        m_gDocIds.emplace_back(gdocid);
        m_gDocId2LDocId[gdocid] = ldocid;
        ldocid++;
    }
    LOG_DEBUG("ReadGdocidSegment success");
    return true;
}

bool DocIdMapping::ReadSdocidSegment(std::ifstream& ifs, uint32_t segmentId, uint32_t& ldocid) {
    // number of docs in this segment
    uint32_t count = m_docNumPerSegment;
    // the last segment may be partial
    if (segmentId == m_segmentNum - 1) {
        count = m_docNum - (m_docNumPerSegment * segmentId);
    }
    for (uint32_t i = 0; i < count; i++) {
        // read the offset (relative to the start of the encoding, excluding the file header)
        uint32_t sdocIdOffset = 0;
        ifs.read(reinterpret_cast<char*>(&sdocIdOffset), sizeof(sdocIdOffset));
        if (ifs.gcount() != sizeof(sdocIdOffset)) {
            LOG_ERROR("read sdocIdOffset fail");
            return false;
        }
        uint32_t realOffset = sdocIdOffset + m_headerLength;
        // read the length
        uint32_t sdocIdLength = 0;
        ifs.read(reinterpret_cast<char*>(&sdocIdLength), sizeof(sdocIdLength));
        if (ifs.gcount() != sizeof(sdocIdLength)) {
            LOG_ERROR("read sdocIdLength fail");
            return false;
        }
        std::streampos originalPos = ifs.tellg();
        // read the sdocid at realOffset
        ifs.seekg(realOffset, std::ios::beg);
        if (!ifs) {
            LOG_ERROR("seekg to offset fail");
            return false;
        }
        std::string sdocid;
        sdocid.resize(sdocIdLength);
        ifs.read(reinterpret_cast<char*>(sdocid.data()), sdocIdLength);
        if (ifs.gcount() != sdocIdLength) {
            LOG_ERROR("read sdocId fail");
            return false;
        }
        // restore the position
        ifs.clear();
        ifs.seekg(originalPos, std::ios::beg);
        if (!ifs) {
            LOG_ERROR("seekg to original_pos fail");
            return false;
        }
        LOG_DEBUG("sdocid is:" << sdocid << " ldocid is:" << ldocid);
        m_sDocIds.emplace_back(sdocid);
        m_sDocId2LDocId[sdocid] = ldocid;
        ldocid++;
    }
    return true;
}

bool DocIdMapping::GetGDocId(uint32_t lDocId, uint64_t& gDocid) const {
    if (lDocId >= m_gDocIds.size()) {
        LOG_ERROR("invalid ldocid:" << lDocId);
        return false;
    }
    gDocid = m_gDocIds[lDocId];
    return true;
}

std::string DocIdMapping::GetSDocId(uint32_t lDocId) const {
    if (lDocId >= m_sDocIds.size()) {
        LOG_ERROR("invalid ldocid:" << lDocId);
        return "";
    }
    return m_sDocIds[lDocId];
}

bool DocIdMapping::GetLDocId(uint64_t gDocId, uint32_t& lDocid) const {
    auto iter = m_gDocId2LDocId.find(gDocId);
    if (iter == m_gDocId2LDocId.end()) {
        LOG_ERROR("invalid gDocId:" << gDocId);
        return false;
    }
    lDocid = iter->second;
    return true;
}

bool DocIdMapping::GetLDocId(const std::string& sDocId, uint32_t& lDocid) const {
    auto iter = m_sDocId2LDocId.find(sDocId);
    if (iter == m_sDocId2LDocId.end()) {
        LOG_ERROR("invalid sDocId:" << sDocId);
        return false;
    }
    lDocid = iter->second;
    return true;
}
}  // namespace NpuRetrieval