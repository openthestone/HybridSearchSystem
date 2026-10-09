#include "posting_serializer.h"
#include <cstdlib>
#include "src/utils/logger.h"
#include "src/utils/env_switch.h"

namespace NpuRetrieval {
namespace {
constexpr uint32_t kDocsPerUnit = 16;      // one uint16 bitmap unit covers 16 doc positions
constexpr uint32_t kHitCountBits = 28;     // the low 28 bits of the head word hold the hit count
constexpr uint32_t kMaxLayoutCode = 0x0F;  // the layout code occupies the top 4 bits
constexpr uint32_t kMaxHitCount = 0x0FFFFFFF;

// The posting head packs a 4-bit layout code and a 28-bit hit count into a
// single uint32 word: [layout:4][hitCount:28].
bool PackHeadWord(uint32_t layoutCode, uint32_t hitCount, uint32_t& headWord) {
    if (layoutCode > kMaxLayoutCode) {
        LOG_ERROR("posting layout code must be <= 15");
        return false;
    }
    if (hitCount > kMaxHitCount) {
        LOG_ERROR("posting hit count must be <= 268,435,455");
        return false;
    }
    headWord = (layoutCode << kHitCountBits) | hitCount;
    return true;
}

// The two 4-byte header words that begin every posting layout: head word, then payload length.
bool AppendHeader(std::string& buffer, PostingLayout layout, uint32_t hitCount, uint32_t payloadLength) {
    uint32_t headWord = 0;
    if (!PackHeadWord(static_cast<uint32_t>(layout), hitCount, headWord)) {
        LOG_ERROR("PackHeadWord fail, please check data");
        return false;
    }
    buffer.append(reinterpret_cast<const char*>(&headWord), sizeof(headWord));
    buffer.append(reinterpret_cast<const char*>(&payloadLength), sizeof(payloadLength));
    return true;
}

// Number of 16-doc bitmap units covering docNum positions (the final unit is zero-padded).
uint32_t BitmapUnitCount(uint32_t docNum) {
    return (docNum + kDocsPerUnit - 1) / kDocsPerUnit;
}

// Bit i is set when the doc at that position is a hit; low bits map to low doc ids.
uint16_t PackBitmapUnit(const std::unordered_set<uint32_t>& docIds, uint32_t unitIndex) {
    uint16_t mask = 0;
    for (uint32_t bit = 0; bit < kDocsPerUnit; bit++) {
        uint32_t docId = unitIndex * kDocsPerUnit + bit;
        if (docIds.count(docId) > 0) {
            mask |= (1 << bit);
        }
    }
    return mask;
}
// NPUR_SPARSE_PACKED=1 makes sparse bitmap postings come out packed. Write side only -- every
// reader handles both layouts off the head word's tag -- so the arms are two indexes, not two
// binaries, and an index built without it stays readable.
bool SparsePackedEnabled() {
    static const bool on = npur_env::On("NPUR_SPARSE_PACKED");
    return on;
}
}  // namespace

PostingLayout PostingSerializer::SelectLayout(bool isMatrixType, uint32_t targetDocNum, uint32_t docNum,
                                              float densityThreshold) {
    if (docNum == 0) {
        LOG_ERROR("docNum is 0, please check, set posting layout to dense bitmap");
        return PostingLayout::DENSE_BITMAP;
    }
    float density = targetDocNum * 1.0 / docNum;
    if (density < densityThreshold) {
        if (isMatrixType) {
            return PostingLayout::SPARSE_IDLIST;
        }
        return SparsePackedEnabled() ? PostingLayout::SPARSE_PACKED : PostingLayout::SPARSE_BITMAP;
    }
    return isMatrixType ? PostingLayout::DENSE_BYTEMAP : PostingLayout::DENSE_BITMAP;
}

bool PostingSerializer::Serialize(const std::unordered_set<uint32_t>& docIds, uint32_t docNum, bool isMatrixType,
                                  float densityThreshold, std::string& buffer) {
    PostingLayout layout = SelectLayout(isMatrixType, docIds.size(), docNum, densityThreshold);
    LOG_DEBUG("layout:" << static_cast<int>(layout) << " isMatrixType:" << isMatrixType
                        << " densityThreshold:" << densityThreshold);
    switch (layout) {
        case PostingLayout::DENSE_BITMAP:
            return WriteDenseBitmap(docIds, docNum, buffer);
        case PostingLayout::SPARSE_BITMAP:
            return WriteSparseBitmap(docIds, docNum, buffer);
        case PostingLayout::SPARSE_PACKED:
            return WriteSparsePacked(docIds, docNum, buffer);
        case PostingLayout::DENSE_BYTEMAP:
            return WriteDenseByteMap(docIds, docNum, buffer);
        case PostingLayout::SPARSE_IDLIST:
            return WriteSparseIdList(docIds, buffer);
        default:
            LOG_ERROR("unknown posting layout");
            return false;
    }
}

bool PostingSerializer::WriteDenseBitmap(const std::unordered_set<uint32_t>& docIds, uint32_t docNum,
                                         std::string& buffer) {
    LOG_DEBUG("WriteDenseBitmap start");
    uint32_t hitCount = docIds.size();
    // payload: one uint16 bitmap unit per 16 doc positions
    uint32_t unitCount = BitmapUnitCount(docNum);
    uint32_t payloadLength = unitCount * sizeof(uint16_t);

    buffer.reserve(2 * sizeof(uint32_t) + payloadLength);
    if (!AppendHeader(buffer, PostingLayout::DENSE_BITMAP, hitCount, payloadLength)) {
        return false;
    }
    for (uint32_t unit = 0; unit < unitCount; unit++) {
        uint16_t mask = PackBitmapUnit(docIds, unit);
        buffer.append(reinterpret_cast<const char*>(&mask), sizeof(mask));
    }
    return true;
}

bool PostingSerializer::WriteDenseByteMap(const std::unordered_set<uint32_t>& docIds, uint32_t docNum,
                                          std::string& buffer) {
    LOG_DEBUG("WriteDenseByteMap start");
    uint32_t hitCount = docIds.size();
    // payload: one byte per doc position, so the byte count equals docNum
    uint32_t payloadLength = docNum;

    buffer.reserve(2 * sizeof(uint32_t) + payloadLength);
    if (!AppendHeader(buffer, PostingLayout::DENSE_BYTEMAP, hitCount, payloadLength)) {
        return false;
    }
    for (uint32_t doc = 0; doc < docNum; doc++) {
        uint8_t hit = docIds.count(doc) > 0 ? 1 : 0;
        buffer.append(reinterpret_cast<const char*>(&hit), sizeof(hit));
    }
    return true;
}

bool PostingSerializer::WriteSparseBitmap(const std::unordered_set<uint32_t>& docIds, uint32_t docNum,
                                          std::string& buffer) {
    LOG_DEBUG("WriteSparseBitmap start");
    uint32_t hitCount = docIds.size();
    // Keep only the units with at least one hit: byte offsets first, then masks. Each offset is
    // unitIndex * sizeof(uint16) within the equivalent dense bitmap.
    std::string masks;
    std::string offsets;
    uint32_t unitCount = BitmapUnitCount(docNum);
    for (uint32_t unit = 0; unit < unitCount; unit++) {
        uint16_t mask = PackBitmapUnit(docIds, unit);
        if (mask != 0) {
            uint32_t offset = unit * sizeof(uint16_t);
            masks.append(reinterpret_cast<const char*>(&mask), sizeof(mask));
            offsets.append(reinterpret_cast<const char*>(&offset), sizeof(offset));
        }
    }
    uint32_t payloadLength = masks.length() + offsets.length();
    if (!AppendHeader(buffer, PostingLayout::SPARSE_BITMAP, hitCount, payloadLength)) {
        return false;
    }
    buffer.append(offsets.data(), offsets.length());
    buffer.append(masks.data(), masks.length());
    return true;
}

bool PostingSerializer::WriteSparsePacked(const std::unordered_set<uint32_t>& docIds, uint32_t docNum,
                                          std::string& buffer) {
    LOG_DEBUG("WriteSparsePacked start");
    uint32_t hitCount = docIds.size();
    // One uint32 per non-zero unit: [unitIndex:16][mask:16]. A segment holds at most
    // docNumPerSegment/16 units -- 8192 at 131072 docs -- so 16 bits is ample. Ascending order,
    // exactly as in WriteSparseBitmap.
    std::string payload;
    uint32_t unitCount = BitmapUnitCount(docNum);
    for (uint32_t unit = 0; unit < unitCount; unit++) {
        uint16_t mask = PackBitmapUnit(docIds, unit);
        if (mask != 0) {
            if (unit > 0xFFFF) {
                LOG_ERROR("unit index " << unit << " does not fit the packed layout, needs <= 65535");
                return false;
            }
            uint32_t word = (unit << 16) | mask;
            payload.append(reinterpret_cast<const char*>(&word), sizeof(word));
        }
    }
    uint32_t payloadLength = payload.length();
    if (!AppendHeader(buffer, PostingLayout::SPARSE_PACKED, hitCount, payloadLength)) {
        return false;
    }
    buffer.append(payload.data(), payload.length());
    return true;
}

bool PostingSerializer::WriteSparseIdList(const std::unordered_set<uint32_t>& docIds, std::string& buffer) {
    LOG_DEBUG("WriteSparseIdList start");
    uint32_t hitCount = docIds.size();
    // payload: the matching local doc ids, each stored as a raw uint32
    uint32_t payloadLength = hitCount * sizeof(uint32_t);

    buffer.reserve(2 * sizeof(uint32_t) + payloadLength);
    if (!AppendHeader(buffer, PostingLayout::SPARSE_IDLIST, hitCount, payloadLength)) {
        return false;
    }
    for (auto& docId : docIds) {
        buffer.append(reinterpret_cast<const char*>(&docId), sizeof(docId));
    }
    return true;
}
}  // namespace NpuRetrieval
