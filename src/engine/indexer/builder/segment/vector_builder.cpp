#include <gflags/gflags.h>
#include "vector_builder.h"
#include <filesystem>
#include <fstream>
#include "src/utils/file_manager.h"
#include "src/full_recall/indexer/file/file_writer.h"
#include "src/full_recall/indexer/file/file_reader.h"
#include "src/utils/logger.h"
#include "src/full_recall/core/number/trans_number.h"

DECLARE_bool(build_from_memory);
namespace NpuRetrieval {
namespace {
// dimensions are laid out in blocks of 16; a partial tail block is zero-padded
constexpr uint32_t kDimsPerBlock = 16;

std::string FormatVector(const Building::FloatEmbedding& raw, const std::vector<uint16_t>& fp16) {
    std::string out;
    for (uint32_t i = 0; i < fp16.size(); i++) {
        out += std::to_string(raw.embedding(i)) + UNDERLINE + std::to_string(Float16ToFloat32(fp16[i])) + ",";
    }
    return out;
}
}  // namespace

std::string VectorBuilder::BuildHeaderExtension(uint16_t dimension) {
    std::string extended;
    uint16_t extendedByteNum = sizeof(dimension);
    extended.append(reinterpret_cast<const char*>(&extendedByteNum), sizeof(extendedByteNum));
    extended.append(reinterpret_cast<const char*>(&dimension), sizeof(dimension));
    return extended;
}

bool VectorBuilder::WriteDocVectorBlocks(const Building::Section& section, GlobalDocID gdocid, uint32_t ldocid,
                                         uint16_t dimension, uint32_t dimSplitNum, std::ofstream& vecOutputFile) {
    // only float-type embedding is supported
    if (section.float_embedding_size() < 1 || section.float_embedding(0).embedding_size() != dimension) {
        LOG_ERROR("invalid embedding size");
        return false;
    }

    std::vector<uint16_t> fp16;
    fp16.reserve(dimension);
    for (auto value : section.float_embedding(0).embedding()) {
        fp16.emplace_back(Float32ToFloat16(value));
    }
    LOG_DEBUG("gdocid:" << gdocid << " emb info:" << FormatVector(section.float_embedding(0), fp16));

    // the doc block this doc lands in; blocks follow the Zn layout
    uint32_t docBlock = ldocid / m_splitDocNumZn;
    uint32_t offsetInDocBlock = ldocid - docBlock * m_splitDocNumZn;

    for (uint32_t dimBlock = 0; dimBlock < dimSplitNum; dimBlock++) {
        // absolute byte offset = header + preceding doc blocks + preceding dim
        // blocks inside this doc block + this doc's slot inside the dim block
        uint32_t pos = m_headerLength + (docBlock * m_splitDocNumZn) * dimSplitNum * kDimsPerBlock * sizeof(uint16_t) +
                       (m_splitDocNumZn * dimBlock + offsetInDocBlock) * kDimsPerBlock * sizeof(uint16_t);
        vecOutputFile.seekp(pos, std::ios::beg);

        const char* blockPtr = reinterpret_cast<const char*>(fp16.data() + dimBlock * kDimsPerBlock);
        bool isTailBlock = (dimBlock == dimSplitNum - 1) && (dimension % kDimsPerBlock != 0);
        if (isTailBlock) {
            uint32_t tailNum = dimension % kDimsPerBlock;
            vecOutputFile.write(blockPtr, tailNum * sizeof(uint16_t));
            for (size_t i = 0; i < (kDimsPerBlock - tailNum); ++i) {
                vecOutputFile.put('\0');  // zero-pad the tail block byte by byte
            }
        } else {
            vecOutputFile.write(blockPtr, kDimsPerBlock * sizeof(uint16_t));
        }
    }
    return true;
}

bool VectorBuilder::ConsumeDoc(GlobalDocID gdocid, uint32_t ldocid, const std::string& byteDataBuffer,
                               uint16_t dimension, uint32_t dimSplitNum, std::ofstream& vecOutputFile) {
    Building::Section section;
    if (!section.ParseFromString(byteDataBuffer)) {
        LOG_ERROR("parse vector section fail");
        return false;
    }
    return WriteDocVectorBlocks(section, gdocid, ldocid, dimension, dimSplitNum, vecOutputFile);
}

bool VectorBuilder::ConsumeDoc(GlobalDocID gdocid, uint32_t ldocid, const uint8_t* data, size_t dataSize,
                               uint16_t dimension, uint32_t dimSplitNum, std::ofstream& vecOutputFile) {
    Building::Section section;
    if (!section.ParseFromArray(data, dataSize)) {
        LOG_ERROR("parse vector section fail from memory array");
        return false;
    }
    return WriteDocVectorBlocks(section, gdocid, ldocid, dimension, dimSplitNum, vecOutputFile);
}

bool VectorBuilder::BuildVectorField(const std::string& inputKey, uint32_t segIdx, uint16_t dimension,
                                     const std::unordered_map<uint64_t, uint32_t>& gdocid2ldocid,
                                     const std::string& outputFileName) {
    LOG_INFO("vector builder start, file:" << inputKey << " dimension is :" << dimension);
    std::ofstream vecOutputFile(outputFileName, std::ios::binary | std::ios::out);
    if (!vecOutputFile.is_open()) {
        LOG_ERROR("open output file failed, file:" << outputFileName);
        return false;
    }
    // write the file header, then record its length
    std::string extended = BuildHeaderExtension(dimension);
    if (!WriteHeader(vecOutputFile, m_version, extended)) {
        LOG_ERROR("write header fail");
    }
    m_headerLength = static_cast<uint32_t>(vecOutputFile.tellp());
    uint32_t dimSplitNum = (dimension + kDimsPerBlock - 1) / kDimsPerBlock;

    // The mapping is validated dense 0..N-1: ldocid must equal the row index, or the on-disk
    // placement would be wrong.
    auto checkOrder = [&gdocid2ldocid](GlobalDocID gdocid, uint32_t localDocid, uint32_t docCount) -> bool {
        auto iter = gdocid2ldocid.find(gdocid);
        if (iter == gdocid2ldocid.end()) {
            LOG_ERROR("invalid gdocid:" << gdocid);
            return false;
        }
        if (iter->second != localDocid) {
            LOG_ERROR("invalid id order, subscript is:" << localDocid << "docCount is:" << docCount);
            return false;
        }
        return true;
    };

    bool ret = false;
    if (FLAGS_build_from_memory) {
        ret = ReadAndDoTaskFromMemory(
            inputKey, segIdx,
            [this, &checkOrder, dimension, dimSplitNum, &vecOutputFile](GlobalDocID gdocid, uint32_t localDocid,
                                                                        uint32_t docCount, const uint8_t* data,
                                                                        size_t dataSize) -> bool {
                if (!checkOrder(gdocid, localDocid, docCount)) {
                    return false;
                }
                if (!ConsumeDoc(gdocid, localDocid, data, dataSize, dimension, dimSplitNum, vecOutputFile)) {
                    LOG_ERROR("ConsumeDoc fail");
                    return false;
                }
                return true;
            });
    } else {
        ret = ReadAndDoTask(
            inputKey,
            [this, &checkOrder, dimension, dimSplitNum, &vecOutputFile](
                GlobalDocID gdocid, uint32_t localDocid, uint32_t docCount, const std::string& byteDataBuffer) -> bool {
                if (!checkOrder(gdocid, localDocid, docCount)) {
                    return false;
                }
                if (!ConsumeDoc(gdocid, localDocid, byteDataBuffer, dimension, dimSplitNum, vecOutputFile)) {
                    LOG_ERROR("ConsumeDoc fail");
                    return false;
                }
                return true;
            });
    }

    if (!ret) {
        vecOutputFile.close();
        LOG_ERROR("ReadAndDoTask fail");
        return false;
    }
    // pad the trailing block of the last segment with zeros up to the full size
    uint32_t writeSize = static_cast<uint32_t>(vecOutputFile.tellp()) - m_headerLength;
    uint32_t targetSize = m_paddedDocNum * dimension * sizeof(uint16_t);
    size_t paddingSize = targetSize - writeSize;
    if (m_isLastSegment && paddingSize > 0) {
        for (size_t i = 0; i < paddingSize; ++i) {
            vecOutputFile.put('\0');  // write 0 byte by byte
        }
    }
    vecOutputFile.close();
    LOG_INFO("vector builder end, file:" << inputKey);
    return true;
}

bool VectorBuilder::ResolveInputPath(const std::string& inputDir, const std::string& name, uint32_t segmentid,
                                     std::string& embFilePath) {
    embFilePath = inputDir + "/" + name + "/" + SECTION_FIELD + name + DOT + std::to_string(segmentid);
    std::error_code ec;
    if (!std::filesystem::exists(embFilePath, ec)) {
        LOG_ERROR("vector file not exists. file name:" << embFilePath);
        return false;
    }
    return true;
}

bool VectorBuilder::ResolveOutputPath(const std::string& name, uint32_t segmentid, const std::string& outputDir,
                                      std::string& outputVecFilePath) {
    std::string filename = POISSONENGINE + UNDERLINE + std::to_string(m_version) + UNDERLINE +
                           std::to_string(segmentid) + UNDERLINE + name + VECTOR_FILE_SUFFIX;
    std::string outputVecDir = outputDir + "/" + name;
    outputVecFilePath = outputVecDir + "/" + filename;
    std::error_code ecOut;
    if (!std::filesystem::exists(outputVecDir, ecOut)) {
        if (!AddDirRecursively(outputVecDir)) {
            LOG_ERROR("vector output dir create fail. file path:" << outputVecDir);
            return false;
        }
    }
    return true;
}

bool VectorBuilder::Build(NpuRetrieval::BuilderSchema& schema, const std::string& inputDir, uint32_t segmentid,
                          const std::unordered_map<uint64_t, uint32_t>& gdocid2ldocid, const std::string& outputDir) {
    LOG_INFO("vector builder start, segment:" << segmentid);
    m_version = schema.GetVersion();
    m_splitDocNumZn = schema.GetSplitDocNumZn();

    uint32_t docNum = schema.GetDocNumPerSegment();
    if (segmentid == schema.GetSegmentNum() - 1) {
        // tail segment
        docNum = schema.GetDocNum() - docNum * segmentid;
        m_isLastSegment = true;
    }
    // how many Zn blocks the documents split into (split_doc_num_zn per block, padded)
    if (m_splitDocNumZn < 1) {
        LOG_ERROR("split_doc_num_zn is 0, please check");
        return false;
    }
    uint32_t docSplitNum = (docNum + m_splitDocNumZn - 1) / m_splitDocNumZn;
    m_paddedDocNum = docSplitNum * m_splitDocNumZn;
    LOG_INFO("vector builder segmentid:" << segmentid << " docNum:" << segmentid
                                         << " m_paddedDocNum:" << m_paddedDocNum);

    for (const auto& oneEmbedding : schema.GetEmbeddingFiled()) {
        const std::string& name = oneEmbedding.first;
        uint16_t dimension = oneEmbedding.second;
        std::string outputVecFilePath;
        if (!ResolveOutputPath(name, segmentid, outputDir, outputVecFilePath)) {
            return false;
        }
        std::string inputKey = name;
        if (!FLAGS_build_from_memory && !ResolveInputPath(inputDir, name, segmentid, inputKey)) {
            return false;
        }
        if (!BuildVectorField(inputKey, segmentid, dimension, gdocid2ldocid, outputVecFilePath)) {
            LOG_ERROR("BuildVectorField fail. vector name" << name);
            return false;
        }
    }
    LOG_INFO("vector builder end, segment:" << segmentid);
    return true;
}
}  // namespace NpuRetrieval
