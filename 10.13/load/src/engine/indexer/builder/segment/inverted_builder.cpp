#include <gflags/gflags.h>
#include "inverted_builder.h"
#include <filesystem>
#include "src/utils/logger.h"
#include "src/utils/file_manager.h"
#include "src/full_recall/indexer/file/file_writer.h"
#include "src/full_recall/indexer/file/file_reader.h"
#include "src/full_recall/indexer/proto/document.pb.h"
#include "src/full_recall/format/posting_serializer.h"
#include "src/full_recall/core/hash/hash.h"
#include "src/workflow/async/executor.h"

DECLARE_bool(build_from_memory);
namespace NpuRetrieval {
namespace {
// Terms may arrive as uint32/uint64 numbers or as raw strings depending on the field.
bool ExtractTokens(const Building::Section& section, uint32_t ldocid, TokenPostings& tokenMap) {
    for (auto& termInfo : section.terminfos()) {
        std::string token;
        switch (termInfo.value_case()) {
            case Building::TermInfo::kUint32Value:
                token = std::to_string(termInfo.uint32value());
                break;
            case Building::TermInfo::kUint64Value:
                token = std::to_string(termInfo.uint64value());
                break;
            default:
                if (termInfo.term().empty()) {
                    LOG_ERROR("invalid token, token is empty");
                    return false;
                }
                token = termInfo.term();
        }
        tokenMap[token].emplace(ldocid);
    }
    return true;
}

// Pad each posting list to a 4-byte boundary so the offsets in the token dict stay word-aligned.
bool EncodeOnePosting(PostingSerializer& serializer, const std::unordered_set<uint32_t>& docIds, uint32_t docNum,
                      const Building::SectionInvertedInfo& invertedInfo, std::string& out) {
    if (!serializer.Serialize(docIds, docNum, invertedInfo.is_matrix_posting(), invertedInfo.density_threshold(),
                              out)) {
        return false;
    }
    uint32_t remainder = out.length() % 4;
    if (remainder != 0) {
        uint32_t padded = out.length() + (4 - remainder);
        out.resize(padded, 0);
        LOG_DEBUG("postingBuffer resize to :" << padded);
    }
    return true;
}

std::string JoinDocIds(const std::unordered_set<uint32_t>& ids) {
    std::string joined;
    for (auto& id : ids) {
        joined += std::to_string(id) + ",";
    }
    return joined;
}
}  // namespace

std::string InvertedBuilder::BuildHeaderExtension(bool isMatrix) {
    std::string extended;
    // the extended content stores whether the posting is bitmap or matrix
    uint16_t extendedByteNum = 1;
    extended.append(reinterpret_cast<const char*>(&extendedByteNum), sizeof(extendedByteNum));
    extended.append(isMatrix ? MATRIX_TYPE : BITMAP_TYPE);
    return extended;
}

bool InvertedBuilder::ResolveInputPath(const std::string& inputDir, const std::string& name, uint32_t segmentid,
                                       std::string& invertedFilePath) {
    invertedFilePath = inputDir + "/" + name + "/" + SECTION_FIELD + name + DOT + std::to_string(segmentid);
    std::error_code ec;
    if (!std::filesystem::exists(invertedFilePath, ec)) {
        LOG_ERROR("inverted file not exists. file name:" << invertedFilePath);
        return false;
    }
    return true;
}

bool InvertedBuilder::ResolveOutputPath(const std::string& name, uint32_t segmentid, const std::string& outputDir,
                                        std::string& outputInvertedFilePath) {
    std::string filename = POISSONENGINE + UNDERLINE + std::to_string(m_version) + UNDERLINE +
                           std::to_string(segmentid) + UNDERLINE + name + INVERTED_FILE_SUFFIX;
    std::string outputVecDir = outputDir + "/" + name;
    outputInvertedFilePath = outputVecDir + "/" + filename;
    std::error_code ecOut;
    if (!std::filesystem::exists(outputVecDir, ecOut)) {
        if (!AddDirRecursively(outputVecDir)) {
            LOG_ERROR("vector output dir create fail. file path:" << outputVecDir);
            return false;
        }
    }
    return true;
}

bool InvertedBuilder::ConsumeDoc(const std::string& byteDataBuffer, uint32_t ldocid, TokenPostings& tokenMap) {
    Building::Section section;
    if (!section.ParseFromString(byteDataBuffer)) {
        LOG_ERROR("parse vector section fail");
        return false;
    }
    return ExtractTokens(section, ldocid, tokenMap);
}

bool InvertedBuilder::ConsumeDoc(const uint8_t* data, size_t dataSize, uint32_t ldocid, TokenPostings& tokenMap) {
    Building::Section section;
    if (!section.ParseFromArray(data, dataSize)) {
        LOG_ERROR("parse inverted section fail from memory array");
        return false;
    }
    return ExtractTokens(section, ldocid, tokenMap);
}

bool InvertedBuilder::BuildField(const std::string& inputKey, uint32_t segIdx,
                                 const std::unordered_map<uint64_t, uint32_t>& gdocid2ldocid,
                                 const Building::SectionInvertedInfo& invertedInfo, const std::string& outputFileName) {
    LOG_INFO("inverted builder start, key:" << inputKey << " mode:" << (FLAGS_build_from_memory ? "Memory" : "File"));
    std::ofstream invertedOutputFile(outputFileName, std::ios::binary | std::ios::out);
    if (!invertedOutputFile.is_open()) {
        LOG_ERROR("open output file failed, file:" << outputFileName);
        return false;
    }
    // write the file header, then record its length
    std::string extended = BuildHeaderExtension(invertedInfo.is_matrix_posting());
    if (!WriteHeader(invertedOutputFile, m_version, extended)) {
        LOG_ERROR("write header fail");
        invertedOutputFile.close();
        return false;
    }
    m_headerLength = static_cast<uint32_t>(invertedOutputFile.tellp());

    // Reads either from the on-disk section file or from the in-memory streaming buffer.
    TokenPostings tokenMap;
    bool ret = false;
    if (FLAGS_build_from_memory) {
        ret = ReadAndDoTaskFromMemory(
            inputKey, segIdx,
            [this, &gdocid2ldocid, &tokenMap](GlobalDocID gdocid, uint32_t localDocid, uint32_t docCount,
                                              const uint8_t* data, size_t dataSize) -> bool {
                if (gdocid2ldocid.find(gdocid) == gdocid2ldocid.end()) {
                    LOG_ERROR("invalid gdocid:" << gdocid << " docCount:" << docCount);
                    return false;
                }
                return ConsumeDoc(data, dataSize, localDocid, tokenMap);
            });
    } else {
        ret =
            ReadAndDoTask(inputKey,
                          [this, &gdocid2ldocid, &tokenMap](GlobalDocID gdocid, uint32_t localDocid, uint32_t docCount,
                                                            const std::string& byteDataBuffer) -> bool {
                              if (gdocid2ldocid.find(gdocid) == gdocid2ldocid.end()) {
                                  LOG_ERROR("invalid gdocid:" << gdocid << " docCount:" << docCount);
                                  return false;
                              }
                              return ConsumeDoc(byteDataBuffer, localDocid, tokenMap);
                          });
    }
    if (!ret) {
        LOG_ERROR("Read task failed");
        invertedOutputFile.close();
        return false;
    }

    if (!EncodeInverted(tokenMap, invertedInfo, gdocid2ldocid.size(), invertedOutputFile)) {
        LOG_ERROR("EncodeInverted fail");
        invertedOutputFile.close();
        return false;
    }
    invertedOutputFile.close();
    LOG_INFO("inverted builder end, file:" << inputKey);
    return true;
}

bool InvertedBuilder::EncodeInverted(TokenPostings& tokenMap, const Building::SectionInvertedInfo& invertedInfo,
                                     uint32_t docNum, std::ofstream& outputFile) {
    // pre-filter: drop tokens with an empty posting list
    for (auto it = tokenMap.begin(); it != tokenMap.end();) {
        if (it->second.empty()) {
            LOG_DEBUG("cardinality is zero, remove tokenId: " << it->first);
            it = tokenMap.erase(it);
        } else {
            ++it;
        }
    }

    // The token count and the fixed-width token dict (8 + 4 + 4 bytes each) have a known size, so
    // reserve that prefix and write the postings first.
    uint32_t tokenNum = tokenMap.size();
    uint32_t tokenDictBufferSize = (8 + 4 + 4) * tokenNum;
    uint32_t prefixLength = sizeof(tokenNum) + tokenDictBufferSize + m_headerLength;
    outputFile.seekp(prefixLength, std::ios::beg);

    PostingSerializer serializer;
    uint32_t lastPostingOffset = prefixLength;  // offset relative to the start of the file
    uint32_t lastTokenOffset = 0;               // offset relative to the start of the token literals region
    std::vector<uint32_t> tokenOffsetVec;
    std::vector<uint32_t> postingOffsetVec;
    std::string tokensBuffer;

    for (auto& entry : tokenMap) {
        const std::string& token = entry.first;
        LOG_DEBUG("token:" << token << " target ldocid:" << JoinDocIds(entry.second));

        std::string postingBuffer;
        if (!EncodeOnePosting(serializer, entry.second, docNum, invertedInfo, postingBuffer)) {
            return false;
        }
        outputFile.write(postingBuffer.c_str(), postingBuffer.length());
        postingOffsetVec.emplace_back(lastPostingOffset);

        uint32_t tokenLength = token.length();
        tokensBuffer.append(reinterpret_cast<const char*>(&tokenLength), sizeof(tokenLength));
        tokensBuffer.append(token.c_str(), tokenLength);
        tokenOffsetVec.emplace_back(lastTokenOffset);

        lastTokenOffset += sizeof(tokenLength) + tokenLength;
        lastPostingOffset += postingBuffer.length();
    }
    // token literals go right after the postings
    outputFile.write(tokensBuffer.c_str(), tokensBuffer.length());
    // seek back to just after the file header to fill in the count + token dict
    outputFile.seekp(m_headerLength, std::ios::beg);
    outputFile.write(reinterpret_cast<const char*>(&tokenNum), sizeof(tokenNum));
    return WriteTokenDict(tokenMap, tokenOffsetVec, postingOffsetVec, lastPostingOffset, outputFile);
}

bool InvertedBuilder::WriteTokenDict(TokenPostings& tokenMap, std::vector<uint32_t>& tokenOffsetVec,
                                     std::vector<uint32_t>& postingOffsetVec, uint32_t postingEndOffset,
                                     std::ofstream& outputFile) {
    if (tokenMap.size() != postingOffsetVec.size() || tokenMap.size() != tokenOffsetVec.size()) {
        LOG_ERROR("token size not equal");
        return false;
    }
    uint32_t i = 0;
    for (auto& entry : tokenMap) {
        uint64_t tokenHash = Hash64(entry.first);
        uint32_t tokenOffset = tokenOffsetVec[i] + postingEndOffset;
        outputFile.write(reinterpret_cast<const char*>(&tokenHash), sizeof(tokenHash));
        outputFile.write(reinterpret_cast<const char*>(&postingOffsetVec[i]), sizeof(uint32_t));
        outputFile.write(reinterpret_cast<const char*>(&tokenOffset), sizeof(tokenOffset));
        LOG_DEBUG("tokenId is:" << tokenHash << " postingOffset is:" << postingOffsetVec[i]
                                << " tokenOffset is:" << tokenOffset);
        i++;
    }
    return true;
}

bool InvertedBuilder::Build(NpuRetrieval::BuilderSchema& schema, const std::string& inputDir, uint32_t segmentid,
                            const std::unordered_map<uint64_t, uint32_t>& gdocid2ldocid, const std::string& outputDir) {
    LOG_INFO("inverted builder start, segment:" << segmentid);
    m_version = schema.GetVersion();
    const auto& fields = schema.GetInvertedFiled();
    if (fields.empty()) {
        return true;
    }
    std::shared_ptr<Executor> executor = CreateExecutor();
    auto logContext = std::make_shared<LogContext>();
    auto asyncContext = executor->CreateExecuteContext(*logContext);

    for (const auto& oneField : fields) {
        asyncContext->AddTask(
            [this, oneField, segmentid, &gdocid2ldocid, inputDir, outputDir]() -> ErrorCode::ResultType {
                std::string outputFilePath;
                if (!ResolveOutputPath(oneField.first, segmentid, outputDir, outputFilePath)) {
                    return ErrorCode::ResultType::FAIL;
                }
                std::string inputKey = oneField.first;
                if (!FLAGS_build_from_memory && !ResolveInputPath(inputDir, oneField.first, segmentid, inputKey)) {
                    return ErrorCode::ResultType::FAIL;
                }
                if (!BuildField(inputKey, segmentid, gdocid2ldocid, oneField.second, outputFilePath)) {
                    return ErrorCode::ResultType::FAIL;
                }
                return ErrorCode::ResultType::SUCCESS;
            });
    }

    bool isSuccess = true;
    asyncContext->Wait([&isSuccess](ErrorCode::ResultType ret) {
        if (ret != ErrorCode::ResultType::SUCCESS) {
            isSuccess = false;
        }
    });
    if (!isSuccess) {
        LOG_ERROR("Some fields failed to process, segment:" << segmentid);
        return false;
    }
    LOG_INFO("inverted builder end, segment:" << segmentid);
    return true;
}
}  // namespace NpuRetrieval
