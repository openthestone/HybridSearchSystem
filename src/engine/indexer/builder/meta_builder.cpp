#include "meta_builder.h"
#include <fstream>
#include <ostream>
#include "src/utils/logger.h"
#include "google/protobuf/json/json.h"
namespace NpuRetrieval {

bool MetaBuilder::WriteMeta(const std::string& outputDir, Building::Meta::IndexMeta& indexMeta) {
    std::string filename = POISSONENGINE + UNDERLINE + std::to_string(m_version) + META_FILE_SUFFIX;
    std::string outputPath = outputDir + "/" + filename;
    // write a text file
    std::ofstream metaFile(outputPath, std::ios::out);
    if (!metaFile.is_open()) {
        LOG_ERROR("open metaFile file failed");
        return false;
    }
    std::string metaStr;
    auto parseStatus = google::protobuf::json::MessageToJsonString(indexMeta, &metaStr);
    if (!parseStatus.ok()) {
        LOG_ERROR("fail to parse, errorCode=" << parseStatus.code() << ", errorMsg=" << parseStatus.message());
        return false;
    }
    // write the file
    if (!metaFile.write(metaStr.c_str(), metaStr.length())) {
        LOG_ERROR("write meta fail");
        return false;
    }
    return true;
}

bool MetaBuilder::Build(NpuRetrieval::BuilderSchema& inputSchema, const std::string& outputDir) {
    m_version = inputSchema.GetVersion();
    Building::Meta::IndexMeta indexMeta;
    auto config = indexMeta.mutable_config();
    config->set_doc_num_per_segment(inputSchema.GetDocNumPerSegment());
    config->set_split_doc_num_zn(inputSchema.GetSplitDocNumZn());
    auto schema = indexMeta.mutable_schema();
    for (auto& embField : inputSchema.GetEmbeddingFiled()) {
        schema->add_vector(embField.first);
    }
    for (auto& postingField : inputSchema.GetInvertedFiled()) {
        schema->add_posting(postingField.first);
    }
    auto statistics = indexMeta.mutable_statistics();
    statistics->set_doc_num(inputSchema.GetDocNum());
    statistics->set_segment_num(inputSchema.GetSegmentNum());
    // write meta
    if (!WriteMeta(outputDir, indexMeta)) {
        LOG_ERROR("write meta fail");
        return false;
    }
    LOG_INFO("build meta success");
    return true;
}
}  // namespace NpuRetrieval
