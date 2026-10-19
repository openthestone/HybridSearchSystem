#include <gflags/gflags.h>
#include "idmapping_builder.h"
#include "src/utils/logger.h"
#include "src/full_recall/indexer/file/file_writer.h"
#include "src/full_recall/indexer/file/file_reader.h"
#include "src/utils/file_manager.h"

DECLARE_bool(build_from_memory);
namespace NpuRetrieval {

std::string IDMappingBuilder::BuildHeaderExtension() {
    std::string extended;
    uint16_t extendedByteNum = 1;
    extended.append(reinterpret_cast<const char*>(&extendedByteNum), sizeof(extendedByteNum));
    if (m_isSdocid) {
        LOG_INFO("is sdocid");
        extended.append(SDOCID_TYPE);
    } else {
        LOG_INFO("is gdocid");
        extended.append(GDOCID_TYPE);
    }
    return extended;
}

bool IDMappingBuilder::ResolveInputPath(const std::string& inputDir, const std::string& name, uint32_t segmentid,
                                        std::string& docidFilePath) {
    docidFilePath = inputDir + "/" + name + "/" + ATTACHMENT_FIELD + name + DOT + std::to_string(segmentid);
    std::error_code ec;
    if (!std::filesystem::exists(docidFilePath, ec)) {
        LOG_ERROR("docid file not exists. file name:" << docidFilePath);
        return false;
    }
    return true;
}

bool IDMappingBuilder::ResolveOutputPath(uint32_t segmentid, const std::string& outputDir,
                                         std::string& outputDocidFilePath) {
    std::string filename = POISSONENGINE + UNDERLINE + std::to_string(m_version) + UNDERLINE +
                           std::to_string(segmentid) + ID_MAPPING_FILE_SUFFIX;
    std::string outputDocDir = outputDir + "/" + ID_MAPPING_DIR;
    outputDocidFilePath = outputDocDir + "/" + filename;
    std::error_code ecOut;
    if (!std::filesystem::exists(outputDocDir, ecOut)) {
        if (!AddDirRecursively(outputDocDir)) {
            LOG_ERROR("docid output dir create fail. file path:" << outputDocDir);
            return false;
        }
    }
    return true;
}

// For sdocid the docid file stores an (offset,length) index up front and the raw bytes are
// appended after all docs; for a plain gdocid the 8-byte id is written inline.
bool IDMappingBuilder::WriteDocId(uint32_t docCount, uint32_t ldocid, GlobalDocID gdocid, const uint8_t* data,
                                  size_t dataSize, std::ofstream& idMappingFile) {
    m_globalToLocal[gdocid] = ldocid;
    if (m_isSdocid) {
        uint32_t sdocIdLength = static_cast<uint32_t>(dataSize);
        // offset is relative to the start of the encoded body (excluding the header)
        uint32_t sdocIdOffset = docCount * (sizeof(sdocIdOffset) + sizeof(sdocIdLength)) + m_lastSdocidOffset;
        m_lastSdocidOffset += sdocIdLength;
        if (!idMappingFile.write(reinterpret_cast<const char*>(&sdocIdOffset), sizeof(sdocIdOffset))) {
            LOG_ERROR("write sdocid offset fail");
            return false;
        }
        if (!idMappingFile.write(reinterpret_cast<const char*>(&sdocIdLength), sizeof(sdocIdLength))) {
            LOG_ERROR("write sdocid length fail");
            return false;
        }
        m_sdocidDetails.append(reinterpret_cast<const char*>(data), dataSize);
        LOG_DEBUG("sdocid length:" << dataSize << " ldocid is:" << ldocid);
    } else {
        if (!idMappingFile.write(reinterpret_cast<const char*>(&gdocid), sizeof(gdocid))) {
            LOG_ERROR("write gdocid length fail");
            return false;
        }
        LOG_DEBUG("gdocid is:" << gdocid << " ldocid is:" << ldocid);
    }
    return true;
}

bool IDMappingBuilder::Build(BuilderSchema& schema, const std::string& inputPath, uint32_t segmentid,
                             const std::string& outputPath) {
    LOG_INFO("ID mapping builder start, segment:" << segmentid
                                                  << "mode:" << (FLAGS_build_from_memory ? "Memory" : "File"));
    m_isSdocid = schema.GetDocid().isSdocid;
    m_version = schema.GetVersion();
    const std::string& fieldName = schema.GetDocid().name;
    std::string outputDocidFile;
    if (!ResolveOutputPath(segmentid, outputPath, outputDocidFile)) {
        return false;
    }
    std::ofstream idMappingFile(outputDocidFile, std::ios::binary | std::ios::out);
    if (!idMappingFile.is_open()) {
        LOG_ERROR("open output file failed, segmentid:" << segmentid);
        return false;
    }
    // write the file header
    std::string extended = BuildHeaderExtension();
    if (!WriteHeader(idMappingFile, schema.GetVersion(), extended)) {
        LOG_ERROR("write header fail");
        idMappingFile.close();
        return false;
    }

    bool ret = false;
    if (FLAGS_build_from_memory) {
        ret =
            ReadAndDoTaskFromMemory(fieldName, segmentid,
                                    [this, &idMappingFile](GlobalDocID gdocid, uint32_t localDocid, uint32_t docCount,
                                                           const uint8_t* data, size_t dataSize) -> bool {
                                        if (!WriteDocId(docCount, localDocid, gdocid, data, dataSize, idMappingFile)) {
                                            LOG_ERROR("WriteDocId fail");
                                            return false;
                                        }
                                        return true;
                                    });
    } else {
        std::string inputDocidFile;
        if (!ResolveInputPath(inputPath, fieldName, segmentid, inputDocidFile)) {
            idMappingFile.close();
            return false;
        }
        ret = ReadAndDoTask(
            inputDocidFile,
            [this, &idMappingFile](GlobalDocID gdocid, uint32_t localDocid, uint32_t docCount,
                                   const std::string& byteDataBuffer) -> bool {
                if (!WriteDocId(docCount, localDocid, gdocid, reinterpret_cast<const uint8_t*>(byteDataBuffer.data()),
                                byteDataBuffer.length(), idMappingFile)) {
                    LOG_ERROR("WriteDocId fail");
                    return false;
                }
                return true;
            });
    }
    if (!ret) {
        LOG_ERROR("ReadAndDoTask fail");
        idMappingFile.close();
        return false;
    }
    // flush the accumulated sdocid bytes after the offset/length index
    if (!m_sdocidDetails.empty()) {
        if (!idMappingFile.write(m_sdocidDetails.c_str(), m_sdocidDetails.length())) {
            LOG_ERROR("write sdocid offset fail");
            idMappingFile.close();
            return false;
        }
    }

    idMappingFile.close();
    LOG_INFO("ID mapping builder end, segment:" << segmentid);
    return true;
}

}  // namespace NpuRetrieval
