#include "posting_field_data.h"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <sstream>
#include "src/full_recall/format/file_header.h"
#include "src/utils/logger.h"
#include "src/utils/env_switch.h"
#include "src/full_recall/core/constant_definition.h"
#include "src/full_recall/format/posting_serializer.h"
#include "src/full_recall/retrieval/device/device_common_external.h"

namespace NpuRetrieval {
bool PostingFieldData::AddSegment(std::ifstream& inputStream, uint32_t segmentId) {
    LOG_DEBUG("add posting segment file, sgementid:" << segmentId);
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
    uint64_t postingDataLength = 0;
    auto dictT0 = std::chrono::steady_clock::now();
    if (!ParseTokenDict(inputStream, tokenNum, segmentId, postingDataLength)) {
        LOG_ERROR("load token dictionary fail");
        return false;
    }
    m_dictUs +=
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - dictT0).count();
    m_tokenTotal += tokenNum;
    auto bulkT0 = std::chrono::steady_clock::now();
    if (!ReadPostings(inputStream, postingDataLength)) {
        LOG_ERROR("tokenOffset invalid");
        return false;
    }
    m_bulkUs +=
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - bulkT0).count();
    return true;
}

bool PostingFieldData::ParseTokenDict(std::ifstream& inputStream, uint32_t tokenNum, uint32_t segmentId,
                                      uint64_t& postingDataLength) {
    // Subtracting this from a postingOffset yields the address relative to the postings region.
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
        if (i == 0) {
            postingDataLength = tokenOffset - postingOffset;
        }
        auto iter = m_tokenToDeviceByteOffsets.find(tokenId);
        if (iter == m_tokenToDeviceByteOffsets.end()) {
            // One slot per segment, initialized to -1 = the token is absent from that segment.
            std::vector<int64_t> valueVec(m_segmentNum, -1);
            m_tokenToDeviceByteOffsets[tokenId] = std::move(valueVec);
        }
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
    // SegmentBytes leaves the storage uninitialised, so read() is the only pass over it.
    SegmentBytes postingData(dataLength);
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
    m_zeroPostingAddrs = std::make_shared<std::vector<uint8_t*>>(m_segmentNum, m_zeroPostingInDevice);
    if (PostingWeightsWanted()) {
        m_zeroPostingWeights = std::make_shared<std::vector<uint16_t>>(m_segmentNum, 0);
    }
    uint64_t oneTokenValueSize = sizeof(uint8_t*) * m_segmentNum;
    ret = aclrtMalloc((void**)&m_zeroPostingArrayInDevice, oneTokenValueSize, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtMalloc token value fail, error code is:" << ret);
        return false;
    }
    ret = aclrtMemcpy((void*)m_zeroPostingArrayInDevice, oneTokenValueSize, (void*)(*m_zeroPostingAddrs).data(),
                      oneTokenValueSize, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtMemcpy token value host to device fail, error code is:" << ret);
        return false;
    }
    return true;
}

// NPUR_BITLIST_STATS=1: on-disk posting-layout distribution at load time. K = postingLength / 6
// is the number of non-zero uint16 units stored (a u32 byte offset plus the u16 unit itself).
static uint32_t BitlistStatsLevel() {
    static const uint32_t level = npur_env::U32("NPUR_BITLIST_STATS", 0);
    return level;
}

// Level 2 walks each sampled sparse posting's offset array: whether the units are consecutive
// decides whether a run form (6R + 2K against 6K) could pay. It also reports the largest unit
// index seen -- a segment holds 8192 units, 13 bits, so a u16 offset would be 4 bytes not 6.
static uint32_t BitlistStatsSample() {
    static const uint32_t every = static_cast<uint32_t>(npur_env::PositiveOr("NPUR_BITLIST_STATS_SAMPLE", 32));
    return every;
}

namespace {
struct LayoutStats {
    uint64_t byType[16] = {0};
    uint64_t sparseCount = 0;
    uint64_t sparseKSum = 0;
    uint64_t sparseKMax = 0;
    uint64_t denseCount = 0;
    // log2 buckets of K: [0], [1], [2,3], [4,7], ... last bucket is everything >= 2^14
    uint64_t kHist[16] = {0};
    // Exact count per hit count: SelectLayout compares hitCount/docNum against the threshold and
    // hitCount is an integer, so the reachable index sizes are one per integer. kHist buckets by
    // K in powers of two, and K is not what the comparison tests.
    static constexpr uint32_t kHitHistMax = 2048;
    uint64_t hitHist[kHitHistMax + 1] = {0};  // [kHitHistMax] is the overflow bin
    uint64_t hitOverflow = 0;

    void AddHits(uint32_t hitCount) {
        if (hitCount >= kHitHistMax) {
            hitOverflow += 1;
            hitHist[kHitHistMax] += 1;
            return;
        }
        hitHist[hitCount] += 1;
    }

    uint64_t sampled = 0;
    uint64_t sampledK = 0;
    uint64_t sampledRuns = 0;
    uint64_t runHist[9] = {0};  // run length 1,2,3,4,5,6,7,8,>8
    uint64_t maxUnitIndex = 0;
    uint64_t unsorted = 0;  // offsets not in ascending order, which would break the run count

    void AddRuns(const uint32_t* offsets, uint32_t k) {
        sampled += 1;
        sampledK += k;
        uint32_t runLen = 1;
        for (uint32_t i = 0; i < k; i++) {
            const uint64_t unitIdx = offsets[i] / sizeof(uint16_t);
            if (unitIdx > maxUnitIndex) {
                maxUnitIndex = unitIdx;
            }
            if (i == 0) {
                continue;
            }
            if (offsets[i] < offsets[i - 1]) {
                unsorted += 1;
            }
            if (offsets[i] == offsets[i - 1] + sizeof(uint16_t)) {
                runLen += 1;
                continue;
            }
            sampledRuns += 1;
            runHist[runLen > 8 ? 8 : runLen - 1] += 1;
            runLen = 1;
        }
        if (k > 0) {
            sampledRuns += 1;
            runHist[runLen > 8 ? 8 : runLen - 1] += 1;
        }
    }

    void Add(uint8_t type, uint32_t k) {
        byType[type & 0xF] += 1;
        if (type == 0) {
            denseCount += 1;
            return;
        }
        sparseCount += 1;
        sparseKSum += k;
        if (k > sparseKMax) {
            sparseKMax = k;
        }
        uint32_t bucket = 0;
        while (bucket < 15 && (k >> bucket) != 0) {
            bucket += 1;
        }
        kHist[bucket] += 1;  // bucket b holds 2^(b-1) <= k < 2^b; bucket 0 holds k == 0
    }
};
}  // namespace

static_assert(static_cast<uint32_t>(PostingLayout::SPARSE_PACKED) == SPARSE_PACKED_LAYOUT_CODE,
              "the device-side copy of the packed layout code has drifted from PostingLayout");

static std::atomic<bool> g_sparsePostingsPacked{false};

bool PostingWeightsWanted() {
    static const bool wanted = []() {
        if (!npur_env::On("NPUR_LAZY_POSTING_WEIGHTS")) {
            return true;  // the switch is what makes them optional; without it, build as before
        }
        return npur_env::On("NPUR_OR_SKIP_EMPTY") || npur_env::On("NPUR_EMPTY_STATS");
    }();
    return wanted;
}

bool SparsePostingsArePacked() {
    return g_sparsePostingsPacked.load(std::memory_order_relaxed);
}

bool PostingFieldData::FinishAdd() {
    uint64_t dataSize = 0;
    for (auto& oneSegment : m_perSegmentPostingBytes) {
        dataSize += oneSegment.size();
    }
    LOG_INFO("host to device, posting all data size is:" << dataSize << " byte, fieldName=" << m_fieldName);
    if (dataSize == 0) {
        LOG_ERROR("posting all data size is 0, please check");
        return false;
    }
    // 32 bytes of slack: FILTER_FLAG_OR_ALIGNED_COPY rounds a sparse operand's CopyIn up to a whole
    // 32-byte block, so the last posting in the region can be read up to 31 bytes past its end.
    auto ret = aclrtMalloc((void**)&m_devicePostingData, dataSize + BLOCK_SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("aclrtMalloc fail, error code is:" << ret);
        return false;
    }

    uint64_t lastDataSize = 0;
    for (auto& oneSegment : m_perSegmentPostingBytes) {
        ret = aclrtMemcpy((void*)(m_devicePostingData + lastDataSize), oneSegment.size(), (void*)oneSegment.data(),
                          oneSegment.size(), ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("aclrtMemcpy host to device fail, error code is:" << ret);
            return false;
        }
        lastDataSize += oneSegment.size();
    }
    if (!BuildZeroPosting()) {
        LOG_ERROR("BuildZeroPosting fail");
        return false;
    }

    m_zeroPostingTypes = std::make_shared<std::vector<uint8_t>>(m_segmentNum, 0);
    const bool weightsWanted = PostingWeightsWanted();
    const uint32_t statsLevel = BitlistStatsLevel();
    const bool statsOn = statsLevel > 0;
    const uint32_t sampleEvery = BitlistStatsSample();
    uint64_t sampleCounter = 0;
    LayoutStats stats;
    for (auto& oneToken : m_tokenToSegmentByteOffsets) {
        const uint64_t& tokenId = oneToken.first;
        if (m_tokenToPostingTypes.find(tokenId) == m_tokenToPostingTypes.end()) {
            m_tokenToPostingTypes[tokenId] = std::make_shared<std::vector<uint8_t>>(m_segmentNum, 0);
            if (weightsWanted) {
                m_tokenToPostingWeights[tokenId] = std::make_shared<std::vector<uint16_t>>(m_segmentNum, 0);
            }
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
                // Head word low 28 bits = hit count; zero means all-zero and the filter skips it
                // outright. For a sparse posting the weight is K, from the length word next to it.
                const uint32_t hitCount = postingHead & 0x0FFFFFFF;
                const bool packed = postingType == static_cast<uint8_t>(PostingLayout::SPARSE_PACKED);
                if (packed) {
                    g_sparsePostingsPacked.store(true, std::memory_order_relaxed);
                }
                uint16_t weight = 0;
                if (hitCount != 0) {
                    weight = kDensePostingWeight;
                    if (postingType != 0 &&
                        (offsetByte + 2 * sizeof(uint32_t)) <= m_perSegmentPostingBytes[segmentId].size()) {
                        // SPARSE_PACKED is one uint32 per unit, SPARSE_BITMAP a u32 + a u16.
                        const uint32_t bytesPerUnit = packed ? 4u : 6u;
                        const uint32_t k =
                            *(uint32_t*)(&(m_perSegmentPostingBytes[segmentId][offsetByte + 4])) / bytesPerUnit;
                        weight = (k == 0 || k > kMaxSparseK) ? kDensePostingWeight : static_cast<uint16_t>(k);
                    }
                }
                if (weightsWanted) {
                    (*m_tokenToPostingWeights[tokenId])[segmentId] = weight;
                }
                if (statsOn) {
                    stats.AddHits(hitCount);
                }
                // Header is 8 bytes: word 0 the type, word 1 the length, so the length read needs
                // 4 more bytes in range than the type read above.
                if (statsOn && (offsetByte + 2 * sizeof(uint32_t)) <= m_perSegmentPostingBytes[segmentId].size()) {
                    uint32_t postingLength = *(uint32_t*)(&(m_perSegmentPostingBytes[segmentId][offsetByte + 4]));
                    const uint32_t k = postingLength / 6;
                    stats.Add(postingType, k);
                    // The sparse body is K u32 offsets then K u16 values, after the 8-byte header.
                    const uint64_t bodyEnd = static_cast<uint64_t>(offsetByte) + 8 + postingLength;
                    if (statsLevel >= 2 && postingType != 0 && k > 0 &&
                        bodyEnd <= m_perSegmentPostingBytes[segmentId].size() && (sampleCounter++ % sampleEvery) == 0) {
                        stats.AddRuns(
                            reinterpret_cast<const uint32_t*>(&(m_perSegmentPostingBytes[segmentId][offsetByte + 8])),
                            k);
                    }
                }
            }
        }
    }
    if (statsOn) {
        std::ostringstream oss;
        const uint64_t total = stats.denseCount + stats.sparseCount;
        oss << "[BITSTAT-INDEX] field=" << m_fieldName << " segments=" << m_segmentNum << " postings=" << total
            << " dense=" << stats.denseCount << " sparse=" << stats.sparseCount << " sparseRatio="
            << (total == 0 ? 0.0 : static_cast<double>(stats.sparseCount) / static_cast<double>(total))
            << " kSum=" << stats.sparseKSum << " kMean="
            << (stats.sparseCount == 0 ? 0.0
                                       : static_cast<double>(stats.sparseKSum) / static_cast<double>(stats.sparseCount))
            << " kMax=" << stats.sparseKMax;
        const uint64_t sparseBytes = stats.sparseKSum * 6 + stats.sparseCount * 8;
        const uint64_t denseEquivBytes = stats.sparseCount * static_cast<uint64_t>(m_docNumPerSegment / 8);
        oss << " sparseBytes=" << sparseBytes << " denseEquivBytes=" << denseEquivBytes;
        // Cumulative, so entry h is the count a threshold of h/docNumPerSegment compresses.
        uint64_t cum = 0;
        uint32_t last = 0;
        for (uint32_t h = 0; h < LayoutStats::kHitHistMax; h++) {
            if (stats.hitHist[h] != 0) {
                last = h;
            }
        }
        oss << " hitOverflow=" << stats.hitOverflow << " hitCum=[";
        for (uint32_t h = 0; h <= last; h++) {
            cum += stats.hitHist[h];
            oss << (h == 0 ? "" : ",") << cum;
        }
        oss << "]";
        oss << " kHist=[";
        for (uint32_t b = 0; b < 16; b++) {
            oss << stats.kHist[b] << (b == 15 ? "" : ",");
        }
        oss << "] byType=[";
        for (uint32_t t = 0; t < 5; t++) {
            oss << stats.byType[t] << (t == 4 ? "" : ",");
        }
        oss << "]";
        if (statsLevel >= 2 && stats.sampled > 0) {
            const double meanRun = stats.sampledRuns == 0
                                       ? 0.0
                                       : static_cast<double>(stats.sampledK) / static_cast<double>(stats.sampledRuns);
            oss << " | sampled=" << stats.sampled << " sampledK=" << stats.sampledK << " runs=" << stats.sampledRuns
                << " meanRunLen=" << meanRun << " maxUnitIndex=" << stats.maxUnitIndex << " unsorted=" << stats.unsorted
                << " bytesNow=" << stats.sampledK * 6 << " bytesRun=" << stats.sampledRuns * 6 + stats.sampledK * 2
                << " bytesU16off=" << stats.sampledK * 4
                << " bytesRunU16=" << stats.sampledRuns * 4 + stats.sampledK * 2 << " runHist=[";
            for (uint32_t b = 0; b < 9; b++) {
                oss << stats.runHist[b] << (b == 8 ? "" : ",");
            }
            oss << "]";
        }
        fprintf(stdout, "%s\n", oss.str().c_str());
        fflush(stdout);
    }
    LogTokenPostingTypes();
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
        uint8_t* tokenValue = nullptr;
        uint64_t oneTokenValueSize = sizeof(uint8_t*) * m_segmentNum;
        ret = aclrtMalloc((void**)&tokenValue, oneTokenValueSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("aclrtMalloc token value fail, error code is:" << ret);
            return false;
        }
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

std::vector<uint16_t>* PostingFieldData::GetPostingWeights(uint64_t tokenId) const {
    if (!PostingWeightsWanted()) {
        return nullptr;  // nothing was built; readers take null as "no weights"
    }
    auto iter = m_tokenToPostingWeights.find(tokenId);
    if (iter == m_tokenToPostingWeights.end()) {
        return m_zeroPostingWeights.get();
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
