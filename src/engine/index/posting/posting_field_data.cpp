#include "posting_field_data.h"
#include "src/full_recall/format/file_header.h"
#include "src/utils/logger.h"
#include "src/full_recall/core/constant_definition.h"
#include "src/full_recall/format/posting_serializer.h"

namespace NpuRetrieval {
bool PostingFieldData::AddSegment(std::ifstream& inputStream, uint32_t segmentId) {
    LOG_DEBUG("add posting segment file, sgementid:" << segmentId);
    if (!inputStream.is_open()) {
        LOG_ERROR("inputStream is not open");
        return false;
    }
    uint8_t version = 0;
    std::string extended;
    // file header
    if (!DecodeFileHeader(inputStream, version, extended)) {
        LOG_ERROR("DecodeFileHeader fail");
        return false;
    }
    if (extended == MATRIX_TYPE) {
        m_isMatrixCodec = true;
        LOG_DEBUG("posting codec type is matrix");
    } else {
        m_isMatrixCodec = false;
        LOG_DEBUG("posting codec type is not matrix");
    }
    uint32_t tokenNum = 0;
    inputStream.read(reinterpret_cast<char*>(&tokenNum), sizeof(tokenNum));
    if (inputStream.gcount() != sizeof(tokenNum)) {
        LOG_ERROR("tokenNum invalid");
        return false;
    }
    LOG_DEBUG("tokenNum is:" << tokenNum);
    // walk the dictionary; the first token entry gives the posting/token offsets
    uint64_t postingDataLength = 0;
    if (!ParseTokenDict(inputStream, tokenNum, segmentId, postingDataLength)) {
        LOG_ERROR("load token dictionary fail");
        return false;
    }
    // read the postings
    if (!ReadPostings(inputStream, postingDataLength)) {
        LOG_ERROR("tokenOffset invalid");
        return false;
    }
    return true;
}

bool PostingFieldData::ParseTokenDict(std::ifstream& inputStream, uint32_t tokenNum, uint32_t segmentId,
                                      uint64_t& postingDataLength) {
    // length from the file header to the postings region; subtracting it from a
    // postingOffset yields the address relative to the start of the postings
    uint32_t preOffset = static_cast<uint32_t>(inputStream.tellg());
    uint32_t tokenDictBufferSize = (8 + 4 + 4) * tokenNum;
    preOffset += tokenDictBufferSize;
    for (uint32_t i = 0; i < tokenNum; i++) {
        uint64_t tokenId = 0;
        uint32_t postingOffset = 0;  // offset relative to the file start
        uint32_t tokenOffset = 0;    // offset relative to the file start
        inputStream.read(reinterpret_cast<char*>(&tokenId), sizeof(tokenId));
        if (inputStream.gcount() != sizeof(tokenId)) {
            LOG_ERROR("tokenId invalid");
            return false;
        }
        inputStream.read(reinterpret_cast<char*>(&postingOffset), sizeof(postingOffset));
        if (inputStream.gcount() != sizeof(postingOffset)) {
            LOG_ERROR("postingOffset invalid");
            return false;
        }
        inputStream.read(reinterpret_cast<char*>(&tokenOffset), sizeof(tokenOffset));
        if (inputStream.gcount() != sizeof(tokenOffset)) {
            LOG_ERROR("tokenOffset invalid");
            return false;
        }
        if (postingOffset >= tokenOffset) {
            LOG_ERROR("invalid offset, postingOffset:" << postingOffset << " tokenOffset:" << tokenOffset);
            return false;
        }
        // the first token entry gives the length of the posting data region
        if (i == 0) {
            postingDataLength = tokenOffset - postingOffset;
        }
        // assemble this token's per-segment entries
        auto iter = m_tokenToDeviceByteOffsets.find(tokenId);
        if (iter == m_tokenToDeviceByteOffsets.end()) {
            // first time we see this token: one slot per segment, initialized to
            // -1 (meaning the token is absent from that segment)
            std::vector<int64_t> valueVec(m_segmentNum, -1);
            m_tokenToDeviceByteOffsets[tokenId] = std::move(valueVec);
        }
        // token present in this segment: store the offset relative to the start of m_devicePostingData
        uint64_t dataSize = 0;
        for (auto& oneSegment : m_perSegmentPostingBytes) {
            dataSize += oneSegment.size();
        }
        m_tokenToDeviceByteOffsets[tokenId][segmentId] = postingOffset - preOffset + dataSize;

        if (m_tokenToSegmentByteOffsets.find(tokenId) == m_tokenToSegmentByteOffsets.end()) {
            std::vector<int64_t> valueVec(m_segmentNum, -1);
            m_tokenToSegmentByteOffsets[tokenId] = std::move(valueVec);
        }
        m_tokenToSegmentByteOffsets[tokenId][segmentId] = postingOffset - preOffset;
    }
    return true;
}

bool PostingFieldData::ReadPostings(std::ifstream& inputStream, uint64_t dataLength) {
    // load the postings
    std::vector<char> postingData(dataLength);
    inputStream.read(postingData.data(), dataLength);
    if (static_cast<uint64_t>(inputStream.gcount()) != dataLength) {
        LOG_ERROR("read posting value fail");
        return false;
    }
    LOG_DEBUG("read posting byte size is:" << postingData.size());
    m_perSegmentPostingBytes.emplace_back(std::move(postingData));
    return true;
}

bool PostingFieldData::BuildZeroPosting() {
    std::string buffer;
    std::unordered_set<uint32_t> emptyDocIds;
    PostingSerializer serializer{};
    // allocate every segment's zero posting at the same size
    if (!serializer.Serialize(emptyDocIds, m_docNumPerSegment, m_isMatrixCodec, -1.0f, buffer)) {
        LOG_ERROR("encode 0 posting fail");
        return false;
    }
    if (buffer.empty()) {
        LOG_ERROR("encode 0 posting buffer is empty");
        return false;
    }
    auto ret = aclrtMalloc((void**)&m_zeroPostingInDevice, buffer.size(), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtMalloc m_zeroPostingInDevice fail, error code is:" << ret);
        return false;
    }
    ret = aclrtMemcpy((void*)m_zeroPostingInDevice, buffer.size(), (void*)buffer.data(), buffer.size(),
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtMemcpy 0posting to m_zeroPostingInDevice fail, error code is:" << ret);
        return false;
    }
    // the same zero posting for every segment
    m_zeroPostingAddrs = std::make_shared<std::vector<uint8_t*>>(m_segmentNum, m_zeroPostingInDevice);
    uint64_t oneTokenValueSize = sizeof(uint8_t*) * m_segmentNum;
    ret = aclrtMalloc((void**)&m_zeroPostingArrayInDevice, oneTokenValueSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtMalloc token value fail, error code is:" << ret);
        return false;
    }
    // copy from host to device
    ret = aclrtMemcpy((void*)m_zeroPostingArrayInDevice, oneTokenValueSize, (void*)(*m_zeroPostingAddrs).data(),
                      oneTokenValueSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtMemcpy token value host to device fail, error code is:" << ret);
        return false;
    }
    return true;
}

bool PostingFieldData::FinishAdd() {
    // compute the total byte size
    uint64_t dataSize = 0;
    for (auto& oneSegment : m_perSegmentPostingBytes) {
        dataSize += oneSegment.size();
    }
    LOG_INFO("host to device, posting all data size is:" << dataSize << " byte, fieldName=" << m_fieldName);
    // allocate device memory
    if (dataSize == 0) {
        LOG_ERROR("posting all data size is 0, please check");
        return false;
    }
    auto ret = aclrtMalloc((void**)&m_devicePostingData, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtMalloc fail, error code is:" << ret);
        return false;
    }

    uint64_t lastDataSize = 0;
    for (auto& oneSegment : m_perSegmentPostingBytes) {
        // copy from host to device
        ret = aclrtMemcpy((void*)(m_devicePostingData + lastDataSize), oneSegment.size(), (void*)oneSegment.data(),
                          oneSegment.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("aclrtMemcpy host to device fail, error code is:" << ret);
            return false;
        }
        lastDataSize += oneSegment.size();
    }
    // allocate the special all-zero posting
    if (!BuildZeroPosting()) {
        LOG_ERROR("BuildZeroPosting fail");
        return false;
    }

    m_zeroPostingTypes = std::make_shared<std::vector<uint8_t>>(m_segmentNum, 0);
    for (auto& oneToken : m_tokenToSegmentByteOffsets) {
        const uint64_t& tokenId = oneToken.first;
        if (m_tokenToPostingTypes.find(tokenId) == m_tokenToPostingTypes.end()) {
            m_tokenToPostingTypes[tokenId] = std::make_shared<std::vector<uint8_t>>(m_segmentNum, 0);
        }
        for (uint32_t segmentId = 0; segmentId < m_segmentNum; segmentId++) {
            auto& offsetByte = oneToken.second[segmentId];
            if (offsetByte >= 0) {
                if ((offsetByte + sizeof(uint32_t)) > m_perSegmentPostingBytes[segmentId].size()) {
                    LOG_ERROR("offsetByte out of range, offsetByte:" << offsetByte << " range:"
                                                                     << m_perSegmentPostingBytes[segmentId].size());
                    continue;
                }
                uint32_t postingHead = *(uint32_t*)(&(m_perSegmentPostingBytes[segmentId][offsetByte]));
                uint8_t postingType = (postingHead & 0xF0000000) >> 28;
                (*m_tokenToPostingTypes[tokenId])[segmentId] = postingType;
            }
        }
    }
    LogTokenPostingTypes();
    // using the offsets in m_tokenToDeviceByteOffsets, resolve device addresses
    // within m_devicePostingData and copy each token's address array to the device
    for (auto& oneToken : m_tokenToDeviceByteOffsets) {
        auto postingPtr = std::make_shared<std::vector<uint8_t*>>(m_segmentNum, nullptr);
        for (uint32_t i = 0; i < m_segmentNum; i++) {
            auto& oneSeg = oneToken.second[i];
            if (oneSeg != -1) {
                (*postingPtr)[i] = m_devicePostingData + oneSeg;
            } else {
                // fall back to the all-zero posting
                (*postingPtr)[i] = m_zeroPostingInDevice;
            }
        }
        // copy the address array
        uint8_t* tokenValue = nullptr;
        uint64_t oneTokenValueSize = sizeof(uint8_t*) * m_segmentNum;
        ret = aclrtMalloc((void**)&tokenValue, oneTokenValueSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("aclrtMalloc token value fail, error code is:" << ret);
            return false;
        }
        // copy from host to device
        ret = aclrtMemcpy((void*)tokenValue, oneTokenValueSize, (void*)(*postingPtr).data(), oneTokenValueSize,
                          ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("aclrtMemcpy token value host to device fail, error code is:" << ret);
            return false;
        }
        m_tokenToDeviceAddrArray[oneToken.first] = tokenValue;
        m_tokenToSegmentDeviceAddrs[oneToken.first] = postingPtr;
        LOG_DEBUG("host to device, token: " << oneToken.first << " seg value data size is:" << oneTokenValueSize
                                            << " byte");
    }

    return true;
}

void PostingFieldData::LogTokenPostingTypes() {
    if (NpuRetrieval::Logger::GetLogLevel() <= log4cplus::DEBUG_LOG_LEVEL) {
        for (auto& oneToken : m_tokenToPostingTypes) {
            std::string str = " tokenId:";
            const uint64_t& tokenId = oneToken.first;
            str.append(std::to_string(tokenId)).append(", postingType:[");
            for (auto type : *oneToken.second) {
                str.append(std::to_string((int)type)).append(",");
            }
            LOG_DEBUG("field:" << m_fieldName << str << "]");
        }
    }
}

uint8_t* PostingFieldData::GetData(uint64_t tokenId, bool& tokenExist) const {
    auto iter = m_tokenToDeviceAddrArray.find(tokenId);
    if (iter == m_tokenToDeviceAddrArray.end()) {
        tokenExist = false;
        return m_zeroPostingArrayInDevice;
    }
    tokenExist = true;
    return iter->second;
}

std::vector<uint8_t>* PostingFieldData::GetPostingTypes(uint64_t tokenId) const {
    auto iter = m_tokenToPostingTypes.find(tokenId);
    if (iter == m_tokenToPostingTypes.end()) {
        return m_zeroPostingTypes.get();
    }
    return iter->second.get();
}

std::vector<uint8_t*>* PostingFieldData::GetPostingDeviceAddrs(uint64_t tokenId) const {
    auto iter = m_tokenToSegmentDeviceAddrs.find(tokenId);
    if (iter == m_tokenToSegmentDeviceAddrs.end()) {
        return m_zeroPostingAddrs.get();
    }
    return iter->second.get();
}
}  // namespace NpuRetrieval
