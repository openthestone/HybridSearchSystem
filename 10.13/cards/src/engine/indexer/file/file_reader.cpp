#include "file_reader.h"
#include <fstream>
#include "src/utils/logger.h"
#include "memory_data_manager.h"

namespace NpuRetrieval {
bool ReadAndDoTask(const std::string& inputPath,
                   std::function<bool(GlobalDocID, uint32_t, uint32_t, const std::string&)> task) {
    std::ifstream ifs(inputPath, std::ios::binary | std::ios::in);
    if (!ifs.is_open()) {
        LOG_ERROR("open input file failed, file:" << inputPath);
        return false;
    }
    uint32_t count = 0;
    ifs.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (ifs.gcount() != sizeof(count)) {
        LOG_ERROR("invalid file header, file:" << inputPath);
        return false;
    }
    LOG_DEBUG("input file:" << inputPath << " doc count is:" << count);
    // loop to read the remaining content: length gid bytedata
    for (uint32_t i = 0; i < count; i++) {
        uint32_t len = 0;
        GlobalDocID gdocid = 0;
        std::string byteDataBuffer;
        ifs.read(reinterpret_cast<char*>(&len), sizeof(len));
        if (ifs.gcount() != sizeof(len)) {
            LOG_ERROR("invalid len, file:" << inputPath);
            return false;
        }

        ifs.read(reinterpret_cast<char*>(&gdocid), sizeof(gdocid));
        if (ifs.gcount() != sizeof(gdocid)) {
            LOG_ERROR("invalid gdocid, file:" << inputPath);
            return false;
        }
        uint32_t pbLen = len - sizeof(gdocid);
        byteDataBuffer.resize(pbLen);
        ifs.read(reinterpret_cast<char*>(byteDataBuffer.data()), pbLen);
        if (ifs.gcount() != pbLen) {
            LOG_ERROR("invalid pb, file:" << inputPath);
            return false;
        }
        if (!task(gdocid, i, count, byteDataBuffer)) {
            LOG_ERROR("do task fail");
            return false;
        }
    }
    return true;
}

bool ReadAndDoTaskFromMemory(const std::string& fieldName, uint32_t segmentId,
                             std::function<bool(GlobalDocID, uint32_t, uint32_t, const uint8_t*, size_t)> task) {
    auto dataPtr = MemoryDataManager::Instance()->GetFieldData(fieldName, segmentId);
    if (!dataPtr) {
        LOG_ERROR("Field data not found in memory: " << fieldName << "_" << segmentId);
        return false;
    }
    const std::vector<uint8_t>& data = *dataPtr;
    size_t offset = 0;
    if (data.size() < sizeof(uint32_t)) {
        LOG_ERROR("Invalid data format - missing count header");
        return false;
    }
    uint32_t count = *reinterpret_cast<const uint32_t*>(data.data());
    offset += sizeof(uint32_t);
    LOG_DEBUG("read memory data field:" << fieldName << " segment:" << segmentId << " doc count:" << count);
    for (uint32_t i = 0; i < count; i++) {
        if (offset + sizeof(uint32_t) > data.size()) {
            LOG_ERROR("Invalid data format - missing length");
            return false;
        }
        uint32_t len = *reinterpret_cast<const uint32_t*>(data.data() + offset);
        offset += sizeof(uint32_t);
        if (offset + sizeof(GlobalDocID) > data.size()) {
            LOG_ERROR("Invalid data format - missing gdocid");
            return false;
        }
        GlobalDocID gdocid = *reinterpret_cast<const GlobalDocID*>(data.data() + offset);
        offset += sizeof(GlobalDocID);
        uint32_t pbLen = len - sizeof(GlobalDocID);
        if (offset + pbLen > data.size()) {
            LOG_ERROR("Invalid data format - missing pb data");
            return false;
        }
        // pass a pointer to avoid copying the string
        if (!task(gdocid, i, count, data.data() + offset, pbLen)) {
            LOG_ERROR("ReadAndDoTask execution failed");
            return false;
        }
        offset += pbLen;
    }
    LOG_DEBUG("Successfully processed " << count << " docs from memory");
    return true;
}
}  // namespace NpuRetrieval
