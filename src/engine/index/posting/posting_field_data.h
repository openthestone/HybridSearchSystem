#pragma once

#include <cstdint>
#include <vector>
#include <fstream>
#include <unordered_map>
#include <memory>
#include "acl/acl.h"
#include "src/full_recall/core/constant_definition.h"
#include "src/full_recall/core/log_definition.h"

namespace NpuRetrieval {
class PostingFieldData {
   public:
    explicit PostingFieldData(uint32_t segmentNum, uint32_t docNum, uint32_t docNumPerSegment,
                              const std::string& fieldName)
        : m_segmentNum(segmentNum), m_docNum(docNum), m_docNumPerSegment{docNumPerSegment}, m_fieldName{fieldName} {}

    ~PostingFieldData() {
        if (m_devicePostingData != nullptr) {
            CHECK_ACL_ONLY_LOG(aclrtFree(m_devicePostingData));
        }
        for (const auto& tokenData : m_tokenToDeviceAddrArray) {
            if (tokenData.second != nullptr) {
                CHECK_ACL_ONLY_LOG(aclrtFree(tokenData.second));
            }
        }
        if (m_zeroPostingInDevice != nullptr) {
            CHECK_ACL_ONLY_LOG(aclrtFree(m_zeroPostingInDevice));
        }
        if (m_zeroPostingArrayInDevice != nullptr) {
            CHECK_ACL_ONLY_LOG(aclrtFree(m_zeroPostingArrayInDevice));
        }
    }

    bool AddSegment(std::ifstream& inputStream, uint32_t segmentId);

    bool FinishAdd();

    uint8_t* GetData(uint64_t tokenId, bool& tokenExist) const;

    std::vector<uint8_t>* GetPostingTypes(uint64_t tokenId) const;

    std::vector<uint8_t*>* GetPostingDeviceAddrs(uint64_t tokenId) const;

   private:
    bool ReadPostings(std::ifstream& inputStream, uint64_t dataLength);
    bool ParseTokenDict(std::ifstream& inputStream, uint32_t tokenNum, uint32_t segmentId, uint64_t& postingDataLength);
    bool BuildZeroPosting();
    void LogTokenPostingTypes();
    // one entry per segment, holding that segment's posting bytes in order
    std::vector<std::vector<char>> m_perSegmentPostingBytes{};
    // all segments concatenated and copied to the device side
    uint8_t* m_devicePostingData{};
    uint32_t m_segmentNum{};
    uint32_t m_docNum{};
    uint32_t m_docNumPerSegment{};
    std::string m_fieldName;
    bool m_isMatrixCodec{};
    // token -> device pointer to that token's per-segment posting-address array
    // (the array itself lives on the device and holds each segment's posting
    // address within m_devicePostingData)
    std::unordered_map<uint64_t, uint8_t*> m_tokenToDeviceAddrArray{};
    // token -> host-side vector of per-segment device posting pointers
    std::unordered_map<uint64_t, std::shared_ptr<std::vector<uint8_t*>>> m_tokenToSegmentDeviceAddrs{};
    // token -> per-segment byte offset relative to the start of m_devicePostingData
    std::unordered_map<uint64_t, std::vector<int64_t>> m_tokenToDeviceByteOffsets{};
    // token -> per-segment byte offset relative to the start of m_perSegmentPostingBytes
    std::unordered_map<uint64_t, std::vector<int64_t>> m_tokenToSegmentByteOffsets{};
    // token -> per-segment posting type
    std::unordered_map<uint64_t, std::shared_ptr<std::vector<uint8_t>>> m_tokenToPostingTypes{};
    // posting type of the all-zero bitset posting, one entry per segment
    std::shared_ptr<std::vector<uint8_t>> m_zeroPostingTypes{nullptr};
    // the all-zero bitset-type posting (device side)
    uint8_t* m_zeroPostingInDevice{};
    // device-side array of per-segment addresses, all pointing at m_zeroPostingInDevice
    uint8_t* m_zeroPostingArrayInDevice{};
    // host-side array of per-segment addresses, all pointing at m_zeroPostingInDevice
    std::shared_ptr<std::vector<uint8_t*>> m_zeroPostingAddrs{};
};
}  // namespace NpuRetrieval
