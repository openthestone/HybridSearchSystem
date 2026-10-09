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
// One segment's posting bytes. unique_ptr<char[]> with new[] DEFAULT-initialises, so the storage
// is not touched before the read fills it. A std::vector<char>(dataLength) VALUE-initialises, and
// at ~17GB per shard that wrote every byte twice -- 66% of an 11.7s shard load. Do not change it
// back.
class SegmentBytes {
   public:
    SegmentBytes() = default;
    explicit SegmentBytes(uint64_t byteSize) : m_data(new char[byteSize]), m_size(byteSize) {}
    char* data() {
        return m_data.get();
    }
    const char* data() const {
        return m_data.get();
    }
    uint64_t size() const {
        return m_size;
    }
    char& operator[](uint64_t i) {
        return m_data[i];
    }
    const char& operator[](uint64_t i) const {
        return m_data[i];
    }

   private:
    std::unique_ptr<char[]> m_data{};
    uint64_t m_size{0};
};

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

    // Per-segment posting weight, in the same order as GetPostingTypes. Built at load time from the
    // head word:
    //   0                     all-zero: identity for OR, annihilator for AND
    //   1 .. kMaxSparseK      a sparse posting's K, its number of set uint16 units
    //   kDensePostingWeight   dense and not empty
    std::vector<uint16_t>* GetPostingWeights(uint64_t tokenId) const;

    static constexpr uint16_t kDensePostingWeight = 0xFFFF;
    static constexpr uint16_t kMaxSparseK = 0xFFFE;

    std::vector<uint8_t*>* GetPostingDeviceAddrs(uint64_t tokenId) const;

   private:
    bool ReadPostings(std::ifstream& inputStream, uint64_t dataLength);
    bool ParseTokenDict(std::ifstream& inputStream, uint32_t tokenNum, uint32_t segmentId, uint64_t& postingDataLength);
    bool BuildZeroPosting();
    void LogTokenPostingTypes();
    std::vector<SegmentBytes> m_perSegmentPostingBytes{};
    // Load-time attribution, summed across this field's segments; see AddSegment.
    long long m_dictUs{0};     // per-token dictionary walk
    long long m_bulkUs{0};     // the one bulk read of the postings region
    uint64_t m_tokenTotal{0};  // token entries walked, to size the per-token cost
   public:
    long long DictUs() const {
        return m_dictUs;
    }
    long long BulkUs() const {
        return m_bulkUs;
    }
    uint64_t TokenTotal() const {
        return m_tokenTotal;
    }

   private:
    uint8_t* m_devicePostingData{};
    uint32_t m_segmentNum{};
    uint32_t m_docNum{};
    uint32_t m_docNumPerSegment{};
    std::string m_fieldName;
    bool m_isMatrixCodec{};
    // token -> device pointer to that token's per-segment posting-address array (the array itself
    // lives on the device)
    std::unordered_map<uint64_t, uint8_t*> m_tokenToDeviceAddrArray{};
    std::unordered_map<uint64_t, std::shared_ptr<std::vector<uint8_t*>>> m_tokenToSegmentDeviceAddrs{};
    // token -> per-segment byte offset relative to the start of m_devicePostingData
    std::unordered_map<uint64_t, std::vector<int64_t>> m_tokenToDeviceByteOffsets{};
    // token -> per-segment byte offset relative to the start of m_perSegmentPostingBytes
    std::unordered_map<uint64_t, std::vector<int64_t>> m_tokenToSegmentByteOffsets{};
    std::unordered_map<uint64_t, std::shared_ptr<std::vector<uint8_t>>> m_tokenToPostingTypes{};
    // Same shape and lifetime as m_tokenToPostingTypes; see GetPostingWeights.
    std::unordered_map<uint64_t, std::shared_ptr<std::vector<uint16_t>>> m_tokenToPostingWeights{};
    std::shared_ptr<std::vector<uint16_t>> m_zeroPostingWeights{};
    std::shared_ptr<std::vector<uint8_t>> m_zeroPostingTypes{nullptr};
    uint8_t* m_zeroPostingInDevice{};
    uint8_t* m_zeroPostingArrayInDevice{};
    std::shared_ptr<std::vector<uint8_t*>> m_zeroPostingAddrs{};
};
// True once any loaded posting carries PostingLayout::SPARSE_PACKED. Uniform across a run, but the
// readers still branch on it, so an index built before the layout existed keeps working.
bool SparsePostingsArePacked();
// NPUR_LAZY_POSTING_WEIGHTS=1: build the per-(token, segment) weights only when something reads
// them. Unwanted, GetPostingWeights returns nullptr, which every reader treats as "no weights" --
// so the failure direction is not skipping rather than skipping everything.
bool PostingWeightsWanted();
}  // namespace NpuRetrieval
