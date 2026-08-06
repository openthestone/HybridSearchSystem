#pragma once
#include <fstream>
#include <ostream>
#include <unordered_map>
#include "src/full_recall/indexer/schema/builder_schema.h"
#include "src/full_recall/core/constant_definition.h"
#include "src/full_recall/indexer/proto/document.pb.h"
namespace NpuRetrieval {

class VectorBuilder {
   public:
    /* build a single segment
     * inputDir: input shard root directory
     * outputDir: output shard root directory
     */
    bool Build(BuilderSchema& schema, const std::string& inputDir, uint32_t segmentid,
               const std::unordered_map<uint64_t, uint32_t>& gdocid2ldocid, const std::string& outputDir);

   private:
    std::string BuildHeaderExtension(uint16_t dimension);
    bool ResolveInputPath(const std::string& inputDir, const std::string& name, uint32_t segmentid,
                          std::string& inputFilePath);
    bool ResolveOutputPath(const std::string& name, uint32_t segmentid, const std::string& outputDir,
                           std::string& outputVecFilePath);
    bool WriteDocVectorBlocks(const Building::Section& section, GlobalDocID gdocid, uint32_t ldocid, uint16_t dimension,
                              uint32_t dimSplitNum, std::ofstream& vecOutputFile);
    bool ConsumeDoc(GlobalDocID gdocid, uint32_t ldocid, const std::string& byteDataBuffer, uint16_t dimension,
                    uint32_t dimSplitNum, std::ofstream& vecOutputFile);
    bool ConsumeDoc(GlobalDocID gdocid, uint32_t ldocid, const uint8_t* data, size_t dataSize, uint16_t dimension,
                    uint32_t dimSplitNum, std::ofstream& vecOutputFile);
    bool BuildVectorField(const std::string& fieldName, uint32_t segIdx, uint16_t dimension,
                          const std::unordered_map<uint64_t, uint32_t>& gdocid2ldocid,
                          const std::string& realOutputDir);
    uint8_t m_version = 0;
    bool m_isLastSegment = false;
    uint32_t m_headerLength = 0;  // file header length
    uint32_t m_paddedDocNum = 0;
    uint32_t m_splitDocNumZn = 0;
};
}  // namespace NpuRetrieval
