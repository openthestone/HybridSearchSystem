#pragma once
#include <fstream>
#include <ostream>
#include <unordered_map>
#include "src/full_recall/indexer/schema/builder_schema.h"
#include "src/full_recall/core/constant_definition.h"
namespace NpuRetrieval {

class IDMappingBuilder {
   public:
    /* build a single segment
     * inputDir: input shard root directory
     * outputDir: output shard root directory
     */
    bool Build(BuilderSchema& schema, const std::string& inputPath, uint32_t segmentid, const std::string& outputPath);
    // gdocid -> ldocid mapping produced by the last Build() call
    const std::unordered_map<uint64_t, uint32_t>& GetIdMapping() {
        return m_globalToLocal;
    }

   private:
    std::string BuildHeaderExtension();
    bool ResolveInputPath(const std::string& inputDir, const std::string& name, uint32_t segmentid,
                          std::string& docidFilePath);
    bool ResolveOutputPath(uint32_t segmentid, const std::string& outputDir, std::string& outputDocidFilePath);
    bool WriteDocId(uint32_t docCount, uint32_t ldocid, GlobalDocID gdocid, const uint8_t* data, size_t dataSize,
                    std::ofstream& idMappingFile);
    std::unordered_map<uint64_t, uint32_t> m_globalToLocal;  // mapping from gdocid to ldocid
    std::string m_sdocidDetails;                             // stores the actual sdocid content
    bool m_isSdocid = false;
    uint8_t m_version = 0;
    uint32_t m_lastSdocidOffset = 0;
};
}  // namespace NpuRetrieval
