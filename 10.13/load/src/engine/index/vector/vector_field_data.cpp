#include "vector_field_data.h"
#include "src/full_recall/format/file_header.h"
#include "src/utils/logger.h"
#include "src/configuration/develop_configuration.h"
#include "src/full_recall/core/number/trans_number.h"
#include <math.h>

namespace NpuRetrieval {

bool VectorFieldData::AddSegment(std::ifstream& inputStream, uint64_t size) {
    if (!inputStream.is_open()) {
        LOG_ERROR("inputStream is not open");
        return false;
    }
    uint8_t version = 0;
    std::string extended;
    if (!DecodeFileHeader(inputStream, version, extended)) {
        LOG_ERROR("DecodeFileHeader fail");
        return false;
    }
    // dimension is stored as 2 bytes in the header extension
    if (extended.length() != sizeof(uint16_t)) {
        LOG_ERROR("dimension get fail");
        return false;
    }
    uint16_t dimension = *reinterpret_cast<const uint16_t*>(extended.data());
    if (dimension == 0) {
        LOG_ERROR("invalid dimension");
        return false;
    }
    m_dimension = dimension;
    LOG_DEBUG("dimension is:" << m_dimension);

    // append the segment body (everything after the header) to the host buffer
    uint32_t headerLength = static_cast<uint32_t>(inputStream.tellg());
    if (headerLength >= size) {
        LOG_ERROR("invalid file size:" << size);
        return false;
    }
    uint64_t priorSize = m_hostBytes.size();
    uint64_t bodyLength = size - headerLength;
    m_hostBytes.resize(priorSize + bodyLength);
    inputStream.read(m_hostBytes.data() + priorSize, bodyLength);
    if (static_cast<uint64_t>(inputStream.gcount()) != bodyLength) {
        LOG_ERROR("read vector value fail");
        return false;
    }
    LOG_DEBUG("read vector byte size is:" << bodyLength);
    return true;
}

bool VectorFieldData::FinishAdd() {
    return UploadToDevice();
}

bool VectorFieldData::UploadToDevice() {
    uint64_t dataSize = m_hostBytes.size();
    LOG_INFO("vector all data size is:" << dataSize << " byte");
    if (dataSize == 0) {
        LOG_ERROR("vector all data size is 0, please check");
        return false;
    }
    auto ret = aclrtMalloc((void**)&m_deviceData, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtMalloc fail, error code is:" << ret);
        return false;
    }
    ret = aclrtMemcpy((void*)m_deviceData, dataSize, (void*)m_hostBytes.data(), dataSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtMemcpy host to device fail, error code is:" << ret);
        return false;
    }
    m_totalLength = dataSize;
    return true;
}
}  // namespace NpuRetrieval
