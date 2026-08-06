#pragma once
#include <fstream>
#include <ostream>
#include <unordered_map>
#include <unordered_set>
#include "src/full_recall/indexer/schema/builder_schema.h"
#include "src/full_recall/core/constant_definition.h"

namespace NpuRetrieval {
using TokenPostings = std::unordered_map<std::string, std::unordered_set<uint32_t>>;

class InvertedBuilder {
   public:
    /* build a single segment
     * inputDir: input shard root directory
     * outputDir: output shard root directory
     */
    bool Build(BuilderSchema& schema, const std::string& inputDir, uint32_t segmentid,
               const std::unordered_map<uint64_t, uint32_t>& gdocid2ldocid, const std::string& outputDir);

   private:
    std::string BuildHeaderExtension(bool isMatrix);
    bool ResolveInputPath(const std::string& inputDir, const std::string& name, uint32_t segmentid,
                          std::string& invertedFilePath);
    bool ResolveOutputPath(const std::string& name, uint32_t segmentid, const std::string& outputDir,
                           std::string& outputInvertedFilePath);
    bool BuildField(const std::string& fieldName, uint32_t segIdx,
                    const std::unordered_map<uint64_t, uint32_t>& gdocid2ldocid,
                    const Building::SectionInvertedInfo& invertedInfo, const std::string& outputDir);
    bool ConsumeDoc(const std::string& byteDataBuffer, uint32_t ldocid, TokenPostings& tokenMap);
    bool ConsumeDoc(const uint8_t* data, size_t dataSize, uint32_t ldocid, TokenPostings& tokenMap);
    bool EncodeInverted(TokenPostings& tokenMap, const Building::SectionInvertedInfo& invertedInfo, uint32_t docNum,
                        std::ofstream& outputFile);
    bool WriteTokenDict(TokenPostings& tokenMap, std::vector<uint32_t>& tokenOffsetVec,
                        std::vector<uint32_t>& postingOffsetVec, uint32_t postingEndOffset, std::ofstream& outputFile);
    uint8_t m_version = 0;
    uint32_t m_headerLength = 0;  // file header length
};
}  // namespace NpuRetrieval
