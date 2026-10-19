#include "filter_report.h"

#include "../run/flags.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <unordered_set>

#include "src/full_recall/core/hash/hash.h"
#include "src/full_recall/retrieval/query/query_node_imp.h"
#include "src/utils/logger.h"

namespace npur_harness {
namespace {

std::string FilterAstToString(const FilterNode* n) {
    if (n == nullptr)
        return "MATCH_ALL";
    switch (n->op) {
        case FilterOp::Term:
            return "TAG(" + std::to_string(n->tag) + ")";
        case FilterOp::Not:
            return "NOT(" + FilterAstToString(n->children.empty() ? nullptr : n->children[0].get()) + ")";
        case FilterOp::And:
        case FilterOp::Or: {
            const char* op = n->op == FilterOp::And ? " AND " : " OR ";
            std::string out = "(";
            for (size_t i = 0; i < n->children.size(); ++i) {
                if (i != 0)
                    out += op;
                out += FilterAstToString(n->children[i].get());
            }
            out += ")";
            return out;
        }
    }
    return "UNKNOWN";
}

void CollectFilterTerms(const FilterNode* n, std::vector<uint64_t>& tags) {
    if (n == nullptr)
        return;
    if (n->op == FilterOp::Term) {
        tags.push_back(n->tag);
        return;
    }
    for (const auto& child : n->children)
        CollectFilterTerms(child.get(), tags);
}

const char* FilterOpName(uint32_t op) {
    switch (op) {
        case NpuRetrieval::CONJUNCTION:
            return "CONJUNCTION";
        case NpuRetrieval::AND:
            return "AND";
        case NpuRetrieval::OR:
            return "OR";
        case NpuRetrieval::NOT:
            return "NOT";
        default:
            return "UNKNOWN";
    }
}

std::string PostExprToRawString(const std::vector<uint32_t>& postExpr) {
    std::ostringstream oss;
    for (size_t i = 0; i < postExpr.size(); ++i) {
        if (i != 0)
            oss << ' ';
        oss << postExpr[i];
    }
    return oss.str();
}

std::string PostExprToText(const std::vector<uint32_t>& postExpr) {
    std::ostringstream oss;
    bool first = true;
    for (size_t i = 0; i < postExpr.size();) {
        const uint32_t op = postExpr[i++];
        if (op == NpuRetrieval::CONJUNCTION) {
            if (i >= postExpr.size()) {
                oss << "CONJUNCTION(<truncated>)";
                break;
            }
            const uint32_t partNum = postExpr[i++];
            if (!first)
                oss << ' ';
            first = false;
            oss << "CONJUNCTION(parts=" << partNum << ",postings=[";
            for (uint32_t p = 0; p < partNum && i < postExpr.size(); ++p, ++i) {
                if (p != 0)
                    oss << ',';
                oss << postExpr[i];
            }
            oss << "])";
            continue;
        }
        if (i + 1 >= postExpr.size()) {
            if (!first)
                oss << ' ';
            first = false;
            oss << FilterOpName(op) << "(<truncated>)";
            break;
        }
        const uint32_t postingNum = postExpr[i++];
        const uint32_t stackNum = postExpr[i++];
        if (!first)
            oss << ' ';
        first = false;
        oss << FilterOpName(op) << "(postings=" << postingNum << ",stack=" << stackNum << ")";
    }
    return oss.str();
}

bool TokenExists(const std::shared_ptr<NpuRetrieval::DataTable>& dataTable, const std::string& field, uint64_t token) {
    NpuRetrieval::PostingFieldData* fieldData = nullptr;
    if (!dataTable->GetPostingFieldData(field, fieldData) || fieldData == nullptr)
        return false;
    bool tokenExist = false;
    (void)fieldData->GetData(token, tokenExist);
    return tokenExist;
}

bool ReadExact(std::ifstream& in, void* data, size_t bytes) {
    in.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(bytes));
    return static_cast<size_t>(in.gcount()) == bytes;
}

bool SkipBytes(std::ifstream& in, std::streamoff bytes) {
    in.seekg(bytes, std::ios::cur);
    return static_cast<bool>(in);
}

bool ReadPostingSegmentTokens(const std::filesystem::path& filePath, std::unordered_set<uint64_t>& tokens) {
    std::ifstream in(filePath, std::ios::binary);
    if (!in) {
        LOG_WARN("cannot open posting file for token-count diagnostics: " << filePath.string());
        return false;
    }

    uint8_t version = 0;
    if (!ReadExact(in, &version, sizeof(version)) || !SkipBytes(in, 5)) {
        LOG_WARN("invalid posting header while reading token count: " << filePath.string());
        return false;
    }
    uint16_t extendedBytes = 0;
    if (!ReadExact(in, &extendedBytes, sizeof(extendedBytes)) || !SkipBytes(in, extendedBytes)) {
        LOG_WARN("invalid posting extended header while reading token count: " << filePath.string());
        return false;
    }

    uint32_t tokenNum = 0;
    if (!ReadExact(in, &tokenNum, sizeof(tokenNum))) {
        LOG_WARN("invalid posting tokenNum while reading token count: " << filePath.string());
        return false;
    }
    for (uint32_t i = 0; i < tokenNum; ++i) {
        uint64_t tokenId = 0;
        uint32_t postingOffset = 0;
        uint32_t tokenOffset = 0;
        if (!ReadExact(in, &tokenId, sizeof(tokenId)) || !ReadExact(in, &postingOffset, sizeof(postingOffset)) ||
            !ReadExact(in, &tokenOffset, sizeof(tokenOffset))) {
            LOG_WARN("invalid posting token dictionary while reading token count: " << filePath.string());
            return false;
        }
        tokens.insert(tokenId);
    }
    return true;
}

size_t PostingTokenCountFromIndexFiles(const std::string& indexDir, const std::string& field) {
    if (indexDir.empty() || field.empty())
        return 0;
    const std::filesystem::path fieldDir = std::filesystem::path(indexDir) / field;
    std::error_code ec;
    if (!std::filesystem::exists(fieldDir, ec) || !std::filesystem::is_directory(fieldDir, ec)) {
        LOG_WARN("posting field directory not found for token-count diagnostics: " << fieldDir.string());
        return 0;
    }

    std::unordered_set<uint64_t> tokens;
    size_t filesRead = 0;
    for (const auto& entry : std::filesystem::directory_iterator(fieldDir, ec)) {
        if (ec) {
            LOG_WARN("cannot iterate posting field directory: " << fieldDir.string() << ", error=" << ec.message());
            return tokens.size();
        }
        if (!entry.is_regular_file(ec) || entry.path().extension() != ".posting")
            continue;
        if (ReadPostingSegmentTokens(entry.path(), tokens))
            ++filesRead;
    }
    if (filesRead == 0) {
        LOG_WARN("no posting segment files read for token-count diagnostics under " << fieldDir.string());
    }
    return tokens.size();
}

uint64_t LookupConvertedDocFreq(const std::vector<uint64_t>* tagDocFreq, uint64_t tag) {
    if (tagDocFreq == nullptr || tag >= tagDocFreq->size())
        return 0;
    return (*tagDocFreq)[tag];
}

bool GetEffectivePostExpr(const std::shared_ptr<NpuRetrieval::DataTable>& dataTable, NpuRetrieval::QueryNode* tree,
                          std::vector<uint32_t>& postExpr, uint32_t& opNum, size_t& postingCount, bool& matchAll,
                          std::string& error) {
    std::vector<std::vector<uint8_t>*> postingTypes;
    std::vector<std::vector<uint8_t*>*> postingDeviceAddrs;
    opNum = 0;
    postingCount = 0;
    matchAll = false;
    if (tree == nullptr) {
        matchAll = true;
        return true;
    }

    {
        std::unique_lock<std::shared_mutex> lock(tree->m_adjustNodeMutex);
        if (!tree->m_isAdjustNodeDone) {
            tree->AdjustNode(dataTable->GetPostingFields());
            tree->m_isAdjustNodeDone = true;
        }
    }
    {
        std::shared_lock<std::shared_mutex> lock(tree->m_adjustNodeMutex);
        const NpuRetrieval::QueryNodeType rootType = tree->GetNodeType();
        const bool leafRoot = rootType == NpuRetrieval::QueryNodeType::TermNode ||
                              rootType == NpuRetrieval::QueryNodeType::TermsNode ||
                              rootType == NpuRetrieval::QueryNodeType::OrTermsNode;
        if (!leafRoot && tree->GetChildrenNum() == 0) {
            matchAll = true;
            return true;
        }
        std::vector<std::vector<uint16_t>*> postingWeights;
        if (!tree->GetPostOrderExpression(*dataTable, postExpr, postingTypes, postingWeights, postingDeviceAddrs,
                                          opNum)) {
            error = "GetPostOrderExpression failed";
            return false;
        }
    }
    if (postingTypes.size() != postingDeviceAddrs.size()) {
        error = "postingTypes/postingDeviceAddrs size mismatch";
        return false;
    }
    postingCount = postingDeviceAddrs.size();
    return true;
}

}  // namespace

bool LoadConvertedTagFreqFile(const std::string& path, std::vector<uint64_t>& tagDocFreq) {
    tagDocFreq.clear();
    if (path.empty())
        return true;
    std::ifstream in(path);
    if (!in) {
        LOG_WARN("converted tag frequency file not found: "
                 << path << ". Run dataset conversion once to enable this diagnostic.");
        return true;
    }

    std::string line;
    size_t loaded = 0;
    while (std::getline(in, line)) {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        uint64_t tag = 0;
        uint64_t count = 0;
        if (!(iss >> tag >> count)) {
            continue;
        }
        if (tag >= tagDocFreq.size())
            tagDocFreq.resize(tag + 1, 0);
        tagDocFreq[tag] = count;
        ++loaded;
    }
    if (!in.eof() && in.fail()) {
        LOG_ERROR("failed while reading converted tag frequency file: " << path);
        return false;
    }
    LOG_INFO("loaded converted tag frequencies: entries=" << loaded << " from " << path);
    return true;
}

bool WriteEffectiveFilterExpressions(const std::string& path, const std::shared_ptr<NpuRetrieval::DataTable>& dataTable,
                                     const FilterSet& filters, bool haveFilters, const std::vector<uint32_t>& topks,
                                     size_t nq, const std::string& postingField,
                                     const std::vector<uint64_t>* convertedTagDocFreq,
                                     const std::string& convertedTagFreqPath, const std::string& indexDir) {
    if (path.empty())
        return true;
    std::error_code ec;
    std::filesystem::path outputPath(path);
    if (!outputPath.parent_path().empty()) {
        std::filesystem::create_directories(outputPath.parent_path(), ec);
        if (ec) {
            LOG_ERROR("cannot create effective filter directory: " << outputPath.parent_path().string()
                                                                   << ", error=" << ec.message());
            return false;
        }
    }
    std::ofstream out(path, std::ios::out | std::ios::trunc);
    if (!out) {
        LOG_ERROR("cannot open effective filter file: " << path);
        return false;
    }

    out << "project=hx_npu_new\n";
    out << "format=filter_node_ast is the harness parse result; query_node_post_expr is generated by "
           "QueryNode::AdjustNode/GetPostOrderExpression and is the expression consumed by BitmapTextFilter\n";
    out << "indexed_doc_num=" << dataTable->GetDocNum() << "\n";
    out << "segment_num=" << dataTable->GetSegmentNum() << "\n";
    out << "doc_num_per_segment=" << dataTable->GetDocNumPerSegment() << "\n";
    out << "posting_field=" << postingField << "\n";
    const size_t postingTokenCount = PostingTokenCountFromIndexFiles(indexDir, postingField);
    out << "posting_token_count=" << postingTokenCount << "\n";
    out << "converted_tag_freq_file=" << convertedTagFreqPath << "\n";
    out << "converted_tag_freq_loaded=" << (convertedTagDocFreq == nullptr ? "false" : "true") << "\n";
    if (convertedTagDocFreq != nullptr) {
        size_t convertedDistinctTags = 0;
        for (uint64_t freq : *convertedTagDocFreq) {
            if (freq != 0)
                ++convertedDistinctTags;
        }
        out << "converted_distinct_tag_count=" << convertedDistinctTags << "\n";
        out << "posting_dictionary_status="
            << (convertedDistinctTags == postingTokenCount ? "matches_converter_distinct_tags"
                                                           : "mismatch_with_converter_distinct_tags")
            << "\n";
    }
    const size_t count = std::min<size_t>(5, nq);
    out << "query_count_recorded=" << count << "\n\n";

    for (size_t q = 0; q < count; ++q) {
        const FilterNode* ast = haveFilters ? filters.astFor(q) : nullptr;
        const uint32_t assigned = (haveFilters && q < filters.assign.size()) ? filters.assign[q] : UINT32_MAX;
        std::vector<std::unique_ptr<NpuRetrieval::QueryNode>> owned;
        NpuRetrieval::QueryNode* tree = BuildQueryNode(ast, postingField, owned);

        std::vector<uint32_t> postExpr;
        uint32_t opNum = 0;
        size_t postingCount = 0;
        bool matchAll = false;
        std::string error;
        const bool ok = GetEffectivePostExpr(dataTable, tree, postExpr, opNum, postingCount, matchAll, error);

        out << "query_index=" << q << "\n";
        out << "assigned_filter_index=" << (assigned == UINT32_MAX ? std::string("none") : std::to_string(assigned))
            << "\n";
        out << "top_k=" << (q < topks.size() ? topks[q] : 0) << "\n";
        out << "input_filter="
            << ((assigned != UINT32_MAX && assigned < filters.exprs.size()) ? filters.exprs[assigned] : "") << "\n";
        out << "filter_node_ast=" << FilterAstToString(ast) << "\n";
        out << "query_node_root_after_adjust=" << tree->GetName() << "\n";
        out << "query_node_child_count_after_adjust=" << tree->GetChildrenNum() << "\n";

        std::vector<uint64_t> tags;
        CollectFilterTerms(ast, tags);
        size_t postingPresentCount = 0;
        size_t postingMissingCount = 0;
        size_t convertedPresentCount = 0;
        size_t convertedMissingCount = 0;
        size_t convertedPresentButPostingMissing = 0;
        size_t postingPresentButConvertedMissing = 0;
        std::vector<bool> postingExists(tags.size(), false);
        std::vector<uint64_t> convertedFreqs(tags.size(), 0);
        for (size_t i = 0; i < tags.size(); ++i) {
            const uint64_t token = NpuRetrieval::Hash64(std::to_string(tags[i]));
            postingExists[i] = TokenExists(dataTable, postingField, token);
            convertedFreqs[i] = LookupConvertedDocFreq(convertedTagDocFreq, tags[i]);
            if (postingExists[i]) {
                ++postingPresentCount;
            } else {
                ++postingMissingCount;
            }
            if (convertedTagDocFreq != nullptr) {
                if (convertedFreqs[i] != 0) {
                    ++convertedPresentCount;
                    if (!postingExists[i])
                        ++convertedPresentButPostingMissing;
                } else {
                    ++convertedMissingCount;
                    if (postingExists[i])
                        ++postingPresentButConvertedMissing;
                }
            }
        }
        out << "terms_summary=total=" << tags.size() << ",posting_present=" << postingPresentCount
            << ",posting_zero_or_missing=" << postingMissingCount;
        if (convertedTagDocFreq != nullptr) {
            out << ",converted_doc_freq_positive=" << convertedPresentCount
                << ",converted_doc_freq_zero=" << convertedMissingCount
                << ",converted_positive_but_posting_missing=" << convertedPresentButPostingMissing
                << ",posting_present_but_converted_zero=" << postingPresentButConvertedMissing;
        }
        out << "\n";
        if (convertedTagDocFreq != nullptr) {
            const char* diagnosis = "no_missing_postings";
            if (convertedPresentButPostingMissing != 0) {
                diagnosis = "builder_or_loader_lost_tags_after_converter";
            } else if (postingPresentButConvertedMissing != 0) {
                diagnosis = "index_and_converter_frequency_file_are_inconsistent";
            } else if (postingMissingCount != 0) {
                diagnosis = "missing_postings_match_converter_absence_or_index_subset";
            }
            out << "terms_diagnosis=" << diagnosis << "\n";
            out << "missing_posting_converted_positive_tags=";
            bool first = true;
            size_t listed = 0;
            for (size_t i = 0; i < tags.size() && listed < 64; ++i) {
                if (postingExists[i] || convertedFreqs[i] == 0)
                    continue;
                if (!first)
                    out << ' ';
                first = false;
                ++listed;
                out << "TAG(" << tags[i] << "):converted_doc_freq=" << convertedFreqs[i];
            }
            if (listed == 64)
                out << " ...";
            out << "\n";
            out << "missing_posting_converted_zero_tags=";
            first = true;
            listed = 0;
            for (size_t i = 0; i < tags.size() && listed < 64; ++i) {
                if (postingExists[i] || convertedFreqs[i] != 0)
                    continue;
                if (!first)
                    out << ' ';
                first = false;
                ++listed;
                out << "TAG(" << tags[i] << ")";
            }
            if (listed == 64)
                out << " ...";
            out << "\n";
            out << "posting_present_converted_zero_tags=";
            first = true;
            listed = 0;
            for (size_t i = 0; i < tags.size() && listed < 64; ++i) {
                if (!postingExists[i] || convertedFreqs[i] != 0)
                    continue;
                if (!first)
                    out << ' ';
                first = false;
                ++listed;
                out << "TAG(" << tags[i] << ")";
            }
            if (listed == 64)
                out << " ...";
            out << "\n";
        }
        out << "terms=";
        for (size_t i = 0; i < tags.size(); ++i) {
            const uint64_t token = NpuRetrieval::Hash64(std::to_string(tags[i]));
            if (i != 0)
                out << ' ';
            out << "TAG(" << tags[i] << "):token=" << token
                << ":posting=" << (postingExists[i] ? "present" : "zero_or_missing");
            if (convertedTagDocFreq != nullptr) {
                out << ":converted_doc_freq=" << convertedFreqs[i];
            }
        }
        out << "\n";

        out << "effective_status=" << (ok ? (matchAll ? "match_all" : "post_expr") : "error") << "\n";
        if (!ok)
            out << "effective_error=" << error << "\n";
        out << "query_node_post_expr_raw=" << PostExprToRawString(postExpr) << "\n";
        out << "query_node_post_expr=" << PostExprToText(postExpr) << "\n";
        out << "query_node_op_num=" << opNum << "\n";
        out << "query_node_posting_count=" << postingCount << "\n";
        if (ok && matchAll && ast != nullptr) {
            out << "effective_warning=non-empty filter became match-all because TextFilter treats a zero-child root as "
                   "empty\n";
        }
        out << "\n";
    }

    if (!out.good()) {
        LOG_ERROR("failed while writing effective filter file: " << path);
        return false;
    }
    LOG_INFO("effective filter expressions written to " << path);
    return true;
}

bool ReportEffectiveFilters(const std::shared_ptr<NpuRetrieval::DataTable>& dataTable, const FilterSet& filters,
                            bool haveFilters, const std::vector<uint32_t>& topks, size_t nq,
                            const std::string& indexDir) {
    if (FLAGS_effective_filter_file.empty())
        return true;
    std::vector<uint64_t> convertedTagDocFreq;
    if (!FLAGS_converted_tag_freq_file.empty() &&
        !LoadConvertedTagFreqFile(FLAGS_converted_tag_freq_file, convertedTagDocFreq))
        return false;
    const std::vector<uint64_t>* freqPtr = convertedTagDocFreq.empty() ? nullptr : &convertedTagDocFreq;
    return WriteEffectiveFilterExpressions(FLAGS_effective_filter_file, dataTable, filters, haveFilters, topks, nq,
                                           FLAGS_posting_field, freqPtr, FLAGS_converted_tag_freq_file, indexDir);
}

}  // namespace npur_harness
