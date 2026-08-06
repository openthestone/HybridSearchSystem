#pragma once
#include <unordered_map>
#include <unordered_set>
#include <string>
#include "posting_layout.h"

namespace NpuRetrieval {
// Serializes an inverted posting (the set of doc ids that match one token) into
// its compact on-disk byte layout. The concrete layout is chosen automatically
// from the posting density: dense postings use a per-position bitmap/byte-map,
// sparse postings store only the hits.
class PostingSerializer {
   public:
    bool Serialize(const std::unordered_set<uint32_t>& docIds, uint32_t docNum, bool isMatrixType,
                   float densityThreshold, std::string& buffer);

   private:
    PostingLayout SelectLayout(bool isMatrixType, uint32_t targetDocNum, uint32_t docNum, float densityThreshold);
    // Dense bitmap: one bit per doc position marks whether that doc is a hit.
    bool WriteDenseBitmap(const std::unordered_set<uint32_t>& docIds, uint32_t docNum, std::string& buffer);
    // Sparse bitmap: stores only the non-zero 16-doc units as (offset, mask) pairs.
    bool WriteSparseBitmap(const std::unordered_set<uint32_t>& docIds, uint32_t docNum, std::string& buffer);
    // Dense byte-map: one byte per doc position marks whether that doc is a hit.
    bool WriteDenseByteMap(const std::unordered_set<uint32_t>& docIds, uint32_t docNum, std::string& buffer);
    // Sparse list: stores only the matching local doc ids as raw uint32 values.
    bool WriteSparseIdList(const std::unordered_set<uint32_t>& docIds, std::string& buffer);
};
}  // namespace NpuRetrieval
