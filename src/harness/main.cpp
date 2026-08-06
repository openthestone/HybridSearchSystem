// =============================================================================
// fr_search — Phase 6/7 harness: drive FullRecallSearcher directly, bypassing
// the online-serving layer (batching / search proxy / index manager / vector_recall).
//
// Runs old/'s benchmark data through engine/: query vectors from hw_queries.fvecs
// (headered format), filter expressions from filter_expr_600.txt (parsed into
// QueryNode trees), against an index built by fr_builder from dataset_HW.bin.
// Reports P99 latency and, with --dataset_hw + --recall_queries, a CPU
// brute-force recall self-check (same filter + exact FP16 inner-product top-K).
//
// Usage (old/-data run):
//   fr_search --index_dir DIR --query_file hw_queries.fvecs \
//             --filter_file filter_expr_600.txt \
//             --dataset_hw dataset_HW.bin --recall_queries 20 \
//             --topk 100 --num_queries 10000
// =============================================================================
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <gflags/gflags.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <condition_variable>
#include <mutex>
#include <random>
#include <sstream>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "acl/acl.h"
#include "filter_expr.h"
#include "query_io.h"
#include "src/full_recall/core/full_recall_result.h"
#include "src/full_recall/core/hash/hash.h"
#include "src/full_recall/core/number/trans_number.h"
#include "src/full_recall/index/data_table.h"
#include "src/full_recall/retrieval/query/query_node_imp.h"
#include "src/full_recall/retrieval/searcher/full_recall_searcher.h"
#include "src/utils/logger.h"

DEFINE_string(index_dir, "", "index directory produced by fr_builder");
DEFINE_string(query_file, "", "query vectors (.fvecs; headered [n][dim][floats] or per-vector)");
DEFINE_int32(device_id, 0, "NPU device id (single-card mode)");
DEFINE_string(shard_index_dirs, "",
              "comma-separated per-shard index dirs for multi-card mode (e.g. d0,d1); each is a "
              "disjoint corpus slice built with converter --doc_offset. Empty = single-card "
              "(--index_dir on --device_id).");
DEFINE_string(device_ids, "",
              "comma-separated NPU device ids, one per --shard_index_dirs entry (e.g. 0,1). Each "
              "shard is searched on its own card; results are merged host-side into a global top-K.");
DEFINE_string(shard_latency_dump, "",
              "multi-card only: TSV of per-query per-shard timing (idx, shard0_ms..shardN_ms, "
              "max_shard_ms, merge_ms, total_ms) for tail/straggler decomposition. Meaningful at "
              "--batch_size 1 (per-batch granularity otherwise).");
DEFINE_bool(shard_worker_pool, false,
            "multi-card only: use a persistent per-shard worker-thread pool instead of spawning "
            "threads per batch (avoids the per-query fan-out spawn cost). Off = original behavior.");
DEFINE_bool(shard_allow_partial, false,
            "multi-card only: if a shard's search fails, drop it and merge the surviving shards' "
            "results instead of failing the whole batch. Off (default) = strict: any shard failure "
            "fails the batch, keeping recall measurement exact.");
DEFINE_bool(stream_merge, false,
            "multi-card only: fold each shard's sorted top-K into a running merged top-K as soon "
            "as that shard's search finishes, so N-1 shards' merge work happens during the wait for "
            "the straggler; only the last shard's fold is on the critical path. Off (default) = "
            "barrier then O(K*N) k-way merge. Produces byte-identical results (deterministic "
            "score-desc, shard-asc tie-break).");
DEFINE_int32(topk, 100, "top-K to return");
DEFINE_int32(num_queries, 0, "number of queries to run (0 = all in file; may exceed file size, queries cycle)");
DEFINE_int32(warmup, 5, "warmup batches run before timing (results/latency discarded)");
DEFINE_int32(batch_size, 1, "queries per FullRecallSearcher batch");
DEFINE_bool(round_robin, false,
            "round-robin baseline (vs the sharded scheme): every --device_ids card holds the FULL "
            "corpus (pass the same full-index dir for each --shard_index_dirs entry). Queries are "
            "work-stolen onto idle cards in batch_size chunks and each chunk is searched entirely on "
            "ONE card (isMultiShard=false, no cross-card merge). Throughput = queries / wall-clock "
            "with all cards concurrent; latency = the single-card per-query time. Requires >1 device, "
            "and each card must hold the whole corpus or recall collapses.");
DEFINE_string(latency_dump, "",
              "if set, write per-query 'idx\\tms\\ttopk' lines to this file (for tail/outlier analysis; only "
              "meaningful at batch_size=1)");
DEFINE_string(vec_field, "content", "vector field name (must match schema)");
DEFINE_string(posting_field, "content", "posting field name (must match schema)");
DEFINE_string(filter_file, "", "filter expressions, one per line; empty = match-all");
DEFINE_uint64(filter_seed, 42, "seed for randomly assigning filters to queries (matches old/'s scheme)");
DEFINE_string(topk_file, "", "optional top-K file, one integer per query line; overrides --topk per query");
DEFINE_string(dataset_hw, "", "dataset_HW.bin, for the CPU brute-force recall reference");
DEFINE_int32(recall_queries, 0, "number of queries to CPU-verify recall on (0 = off)");
DEFINE_uint64(cpu_doc_offset, 0,
              "absolute source-doc offset of this (shard) index in dataset_HW.bin; the CPU "
              "recall reference scans [cpu_doc_offset, cpu_doc_offset+docNum) and returns "
              "absolute ids, matching an index built with converter --doc_offset. 0 = whole/prefix.");
DEFINE_string(recall_ref_file, "",
              "cache file for the CPU brute-force ground-truth top-K sets. When set, the reference is "
              "computed once for ALL queries and written here (keyed by a size+mtime fingerprint of the "
              "dataset/query/filter/topk inputs); later runs load it, skip the brute-force, and verify "
              "every query. A stale fingerprint triggers an automatic recompute.");
DEFINE_int32(recall_ref_threads, 0,
             "CPU threads for the one-time brute-force reference computation (0 = all cores). Each "
             "query is independent, so this only speeds up a cache miss; it never changes the result.");
DEFINE_string(effective_filter_file, "", "optional output file for the first 5 effective filter expressions");
DEFINE_string(converted_tag_freq_file, "",
              "optional tag_doc_freq.txt emitted by fr_converter; used only for effective-filter diagnostics");
DEFINE_bool(mem_report, false, "after the search, print host peak RSS and per-device NPU HBM usage ([MEM] lines)");

namespace {

// One row from /proc/self/status ("VmRSS:" / "VmHWM:"), in kB; -1 if unavailable.
long ReadHostKb(const char* key) {
    long v = -1;
    if (std::FILE* f = std::fopen("/proc/self/status", "r")) {
        char line[256];
        const size_t klen = std::strlen(key);
        while (std::fgets(line, sizeof(line), f) != nullptr) {
            if (std::strncmp(line, key, klen) == 0) {
                std::sscanf(line + klen, "%ld", &v);
                break;
            }
        }
        std::fclose(f);
    }
    return v;
}

// A point-in-time memory snapshot: host VmRSS + per-device NPU HBM used/total (kB),
// aligned to `devices`. -1 marks an unavailable reading.
struct MemSnap {
    long rssKb = -1;
    std::vector<long> usedKb;
    std::vector<long> totalKb;
};

MemSnap SampleMem(const std::vector<int32_t>& devices) {
    MemSnap s;
    s.rssKb = ReadHostKb("VmRSS:");
    for (int32_t d : devices) {
        long usedKb = -1, totalKb = -1;
        if (aclrtSetDevice(d) == ACL_SUCCESS) {
            size_t freeB = 0, totalB = 0;
            if (aclrtGetMemInfo(ACL_HBM_MEM, &freeB, &totalB) == ACL_SUCCESS && totalB > 0) {
                usedKb = static_cast<long>((totalB - freeB) / 1024);
                totalKb = static_cast<long>(totalB / 1024);
            }
        }
        s.usedKb.push_back(usedKb);
        s.totalKb.push_back(totalKb);
    }
    return s;
}

// Report memory at three milestones so the change is visible: startup baseline,
// after the index is loaded (delta = index footprint), and after the search
// (delta = resident pools / runtime growth, since pooled buffers persist to exit).
// `hostHwmKb` is the whole-run host peak (VmHWM). HBM used is device-wide, so the
// per-milestone DELTA is what this process actually added.
void PrintMemReport(const std::vector<int32_t>& devices, const MemSnap& base, const MemSnap& load,
                    const MemSnap& search, long hostHwmKb) {
    auto mb = [](long kb) { return kb / 1024.0; };
    std::printf("[MEM] ==== memory report (delta vs previous milestone) ====\n");
    std::printf("[MEM] host RSS MB: baseline=%.1f  after_load=%.1f (%+.1f)  after_search=%.1f (%+.1f)  peak=%.1f\n",
                mb(base.rssKb), mb(load.rssKb), mb(load.rssKb - base.rssKb), mb(search.rssKb),
                mb(search.rssKb - load.rssKb), mb(hostHwmKb));
    for (size_t i = 0; i < devices.size(); ++i) {
        std::printf(
            "[MEM] dev%d HBM MB: baseline=%.1f  after_load=%.1f (%+.1f idx)  after_search=%.1f (%+.1f run)  "
            "total=%.1f\n",
            devices[i], mb(base.usedKb[i]), mb(load.usedKb[i]), mb(load.usedKb[i] - base.usedKb[i]),
            mb(search.usedKb[i]), mb(search.usedKb[i] - load.usedKb[i]), mb(search.totalKb[i]));
    }
}

using npur_harness::FilterNode;
using npur_harness::FilterOp;
using npur_harness::ParseFilter;

// Query loading lives in query_io.h (npur_port::LoadQueries), shared with tests.

// ---- filter loading + per-query assignment ---------------------------------
struct FilterSet {
    std::vector<std::string> exprs;                   // raw lines
    std::vector<std::unique_ptr<FilterNode>> parsed;  // parsed per expr (nullptr = match-all/parse-fail)
    std::vector<uint32_t> assign;                     // per-query -> expr index (or UINT32_MAX = none)

    // Build QueryNode tree for query q; keeps nodes alive in `owned`.
    // Returns nullptr filter-AST index handling: empty tree (match-all) uses AndNode w/ 0 children.
    const FilterNode* astFor(size_t q) const {
        if (q >= assign.size() || assign[q] == UINT32_MAX)
            return nullptr;
        return parsed[assign[q]].get();
    }
};

bool LoadFilters(const std::string& path, size_t numQueries, uint64_t seed, FilterSet& fs) {
    std::ifstream f(path);
    if (!f) {
        LOG_ERROR("cannot open filter file: " << path);
        return false;
    }
    std::string line;
    while (std::getline(f, line)) {
        // trim trailing CR/space
        while (!line.empty() && (line.back() == '\r' || line.back() == ' '))
            line.pop_back();
        if (!line.empty())
            fs.exprs.push_back(line);
    }
    if (fs.exprs.empty()) {
        LOG_ERROR("filter file has no expressions: " << path);
        return false;
    }
    fs.parsed.resize(fs.exprs.size());
    for (size_t i = 0; i < fs.exprs.size(); ++i) {
        bool ok = true;
        fs.parsed[i] = ParseFilter(fs.exprs[i], ok);
        if (!ok) {
            LOG_ERROR("failed to parse filter #" << i << ": " << fs.exprs[i].substr(0, 80));
            return false;
        }
    }
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<size_t> dist(0, fs.exprs.size() - 1);
    fs.assign.resize(numQueries);
    if (fs.exprs.size() == numQueries) {
        for (size_t i = 0; i < numQueries; ++i)
            fs.assign[i] = static_cast<uint32_t>(i);
        LOG_INFO("using " << fs.exprs.size() << " filter expressions as one-to-one query filters");
    } else {
        for (size_t i = 0; i < numQueries; ++i)
            fs.assign[i] = static_cast<uint32_t>(dist(rng));
        LOG_INFO("assigned " << fs.exprs.size() << " filter expressions to " << numQueries
                             << " queries by random seed=" << seed);
    }
    return true;
}

bool LoadTopKFile(const std::string& path, size_t numQueries, std::vector<uint32_t>& topks) {
    std::ifstream f(path);
    if (!f) {
        LOG_ERROR("cannot open topk file: " << path);
        return false;
    }
    topks.clear();
    topks.reserve(numQueries);
    std::string line;
    size_t lineNo = 0;
    while (std::getline(f, line)) {
        ++lineNo;
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t'))
            line.pop_back();
        size_t begin = 0;
        while (begin < line.size() && (line[begin] == ' ' || line[begin] == '\t'))
            ++begin;
        if (begin >= line.size())
            continue;
        char* end = nullptr;
        unsigned long value = std::strtoul(line.c_str() + begin, &end, 10);
        if (end == line.c_str() + begin || value == 0 || value > UINT32_MAX) {
            LOG_ERROR("invalid topk at line " << lineNo << " in " << path);
            return false;
        }
        topks.push_back(static_cast<uint32_t>(value));
    }
    if (topks.empty()) {
        LOG_ERROR("topk file has no values: " << path);
        return false;
    }
    if (topks.size() != numQueries) {
        LOG_ERROR("topk file line count " << topks.size() << " does not match running query count " << numQueries);
        return false;
    }
    return true;
}

// FilterNode -> QueryNode tree. Empty/null AST -> empty AndNode (match all).
NpuRetrieval::QueryNode* BuildQueryNode(const FilterNode* n, const std::string& field,
                                        std::vector<std::unique_ptr<NpuRetrieval::QueryNode>>& owned);

std::unique_ptr<NpuRetrieval::QueryNode> BuildOne(const FilterNode* n, const std::string& field) {
    using namespace NpuRetrieval;
    switch (n->op) {
        case FilterOp::Term: {
            uint64_t token = Hash64(std::to_string(n->tag));
            return std::make_unique<TermNode>(field, token);
        }
        case FilterOp::And: {
            auto node = std::make_unique<AndNode>();
            for (const auto& c : n->children) {
                std::unique_ptr<QueryNode> child = BuildOne(c.get(), field);
                node->AddChild(child);
            }
            return node;
        }
        case FilterOp::Or: {
            auto node = std::make_unique<OrNode>();
            for (const auto& c : n->children) {
                std::unique_ptr<QueryNode> child = BuildOne(c.get(), field);
                node->AddChild(child);
            }
            return node;
        }
        case FilterOp::Not: {
            auto node = std::make_unique<NotNode>();
            std::unique_ptr<QueryNode> child = BuildOne(n->children[0].get(), field);
            node->AddChild(child);
            return node;
        }
    }
    return std::make_unique<AndNode>();
}

NpuRetrieval::QueryNode* BuildQueryNode(const FilterNode* n, const std::string& field,
                                        std::vector<std::unique_ptr<NpuRetrieval::QueryNode>>& owned) {
    std::unique_ptr<NpuRetrieval::QueryNode> root =
        (n == nullptr) ? std::make_unique<NpuRetrieval::AndNode>() : BuildOne(n, field);
    NpuRetrieval::QueryNode* raw = root.get();
    owned.emplace_back(std::move(root));
    return raw;
}

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
        if (!tree->GetPostOrderExpression(*dataTable, postExpr, postingTypes, postingDeviceAddrs, opNum)) {
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

// ---- dataset_HW.bin (HYDSET2) mmap reader for the CPU recall reference ------
#pragma pack(push, 1)
struct HwHeader {
    char magic[8];
    uint32_t version;
    uint32_t _pad0;
    uint64_t doc_num;
    uint32_t vector_dim;
    uint32_t tag_num;
    uint32_t reserved;
    uint32_t _pad1;
};
#pragma pack(pop)
static_assert(sizeof(HwHeader) == 40, "HYDSET2 header 40 bytes");

struct HwData {
    int fd = -1;
    void* map = nullptr;
    size_t mapLen = 0;
    const HwHeader* hdr = nullptr;
    const float* vectors = nullptr;
    const uint64_t* bitmaps = nullptr;
    uint32_t dim = 0, tag_num = 0, stride = 0;
    uint64_t doc_num = 0;

    bool Open(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0)
            return false;
        struct stat st;
        if (fstat(fd, &st) != 0)
            return false;
        mapLen = st.st_size;
        map = ::mmap(nullptr, mapLen, PROT_READ, MAP_PRIVATE, fd, 0);
        if (map == MAP_FAILED) {
            map = nullptr;
            return false;
        }
        hdr = reinterpret_cast<const HwHeader*>(map);
        if (std::memcmp(hdr->magic, "HYDSET2", 7) != 0)
            return false;
        dim = hdr->vector_dim;
        tag_num = hdr->tag_num;
        doc_num = hdr->doc_num;
        stride = (tag_num + 63u) / 64u;
        const char* base = reinterpret_cast<const char*>(map);
        vectors = reinterpret_cast<const float*>(base + sizeof(HwHeader));
        bitmaps = reinterpret_cast<const uint64_t*>(base + sizeof(HwHeader) + doc_num * dim * sizeof(float));
        return true;
    }
    ~HwData() {
        if (map)
            ::munmap(map, mapLen);
        if (fd >= 0)
            ::close(fd);
    }
};

// Split "a,b,c" into trimmed non-empty tokens (for --shard_index_dirs / --device_ids).
std::vector<std::string> SplitCsv(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        size_t b = tok.find_first_not_of(" \t");
        size_t e = tok.find_last_not_of(" \t");
        if (b != std::string::npos)
            out.push_back(tok.substr(b, e - b + 1));
    }
    return out;
}

// CPU brute-force top-K over docs [docOffset, docOffset+docNum) that pass `filter`,
// by FP16-matched inner product with `query`. Returns the set of absolute doc ids.
// docOffset != 0 mirrors a shard index built with converter --doc_offset: the engine
// returns absolute global ids, so the reference must scan the same absolute rows.
std::unordered_set<int64_t> CpuTopK(const std::vector<float>& query, const FilterNode* filter, const HwData& hw,
                                    uint64_t docNum, int topk, uint64_t docOffset = 0) {
    // FP16-round the query once (match engine/'s MMad FP16 inputs).
    std::vector<float> q16(hw.dim, 0.0f);
    for (uint32_t d = 0; d < hw.dim && d < query.size(); ++d)
        q16[d] = NpuRetrieval::Float16ToFloat32(NpuRetrieval::Float32ToFloat16(query[d]));

    std::vector<std::pair<float, int64_t>> scored;
    scored.reserve(1024);
    for (uint64_t doc = docOffset; doc < docOffset + docNum; ++doc) {
        if (!npur_harness::EvalFilterBitmap(filter, hw.bitmaps + doc * hw.stride, hw.stride))
            continue;
        const float* v = hw.vectors + doc * hw.dim;
        float s = 0.0f;
        for (uint32_t d = 0; d < hw.dim; ++d) {
            float vd = NpuRetrieval::Float16ToFloat32(NpuRetrieval::Float32ToFloat16(v[d]));
            s += q16[d] * vd;
        }
        scored.emplace_back(s, static_cast<int64_t>(doc));
    }
    size_t k = std::min<size_t>(topk, scored.size());
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    std::unordered_set<int64_t> ref;
    for (size_t i = 0; i < k; ++i)
        ref.insert(scored[i].second);
    return ref;
}

// ---- recall reference cache (ground-truth top-K sets) -----------------------
// CpuTopK depends only on the dataset, queries, filters and per-query k — never
// on engine tuning knobs — so the whole reference is computed once and cached.
// A fingerprint (size+mtime of each input file plus the scalar knobs that shape
// the run) invalidates the cache whenever any input changes.
constexpr uint32_t kRecallRefMagic = 0x46455252u;  // "RREF"
constexpr uint32_t kRecallRefVersion = 1;          // bump if CpuTopK semantics change (e.g. the fp16 codec)

std::string RecallFingerprint(uint64_t docNum, uint32_t dim, uint64_t nq) {
    std::string b;
    auto put = [&](const void* p, size_t n) { b.append(static_cast<const char*>(p), n); };
    auto putFileSig = [&](const std::string& path) {
        uint64_t sz = 0;
        int64_t mt = 0;
        struct stat st{};
        if (!path.empty() && ::stat(path.c_str(), &st) == 0) {
            sz = static_cast<uint64_t>(st.st_size);
            mt = static_cast<int64_t>(st.st_mtime);
        }
        put(&sz, sizeof(sz));
        put(&mt, sizeof(mt));
    };
    uint32_t ver = kRecallRefVersion;
    put(&ver, sizeof(ver));
    putFileSig(FLAGS_dataset_hw);
    putFileSig(FLAGS_query_file);
    putFileSig(FLAGS_filter_file);
    putFileSig(FLAGS_topk_file);
    uint64_t off = FLAGS_cpu_doc_offset, seed = FLAGS_filter_seed;
    int32_t topk = FLAGS_topk, numq = FLAGS_num_queries;
    put(&docNum, sizeof(docNum));
    put(&dim, sizeof(dim));
    put(&nq, sizeof(nq));
    put(&off, sizeof(off));
    put(&seed, sizeof(seed));
    put(&topk, sizeof(topk));
    put(&numq, sizeof(numq));
    return b;
}

// Load cached refs; returns false (caller recomputes) on missing file, bad magic,
// or a fingerprint/shape mismatch.
bool LoadRecallRef(const std::string& path, const std::string& fp, uint64_t nq,
                   std::vector<std::unordered_set<int64_t>>& refs) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;
    uint32_t magic = 0, fpLen = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&fpLen), sizeof(fpLen));
    if (!in || magic != kRecallRefMagic || fpLen != fp.size())
        return false;
    std::string diskFp(fpLen, '\0');
    in.read(diskFp.data(), fpLen);
    uint64_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!in || diskFp != fp || count != nq)
        return false;  // stale or wrong shape
    refs.assign(static_cast<size_t>(nq), {});
    for (uint64_t q = 0; q < nq; ++q) {
        uint32_t n = 0;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (!in)
            return false;
        auto& r = refs[q];
        r.reserve(n);
        for (uint32_t i = 0; i < n; ++i) {
            int64_t id = 0;
            in.read(reinterpret_cast<char*>(&id), sizeof(id));
            if (!in)
                return false;
            r.insert(id);
        }
    }
    return true;
}

bool WriteRecallRef(const std::string& path, const std::string& fp,
                    const std::vector<std::unordered_set<int64_t>>& refs) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        return false;
    uint32_t magic = kRecallRefMagic, fpLen = static_cast<uint32_t>(fp.size());
    uint64_t count = refs.size();
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&fpLen), sizeof(fpLen));
    out.write(fp.data(), fp.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& r : refs) {
        uint32_t n = static_cast<uint32_t>(r.size());
        out.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (int64_t id : r)
            out.write(reinterpret_cast<const char*>(&id), sizeof(id));
    }
    return static_cast<bool>(out);
}

}  // namespace

int main(int argc, char** argv) {
    google::ParseCommandLineFlags(&argc, &argv, true);
    if ((FLAGS_index_dir.empty() && FLAGS_shard_index_dirs.empty()) || FLAGS_query_file.empty()) {
        std::fprintf(stderr,
                     "usage: fr_search --index_dir DIR --query_file q.fvecs [--filter_file f.txt] [...]\n"
                     "   or: fr_search --shard_index_dirs d0,d1 --device_ids 0,1 --query_file q.fvecs [...]\n");
        return 1;
    }

    aclError aclRet = aclInit(nullptr);
    if (aclRet != ACL_SUCCESS) {
        LOG_ERROR("aclInit failed: " << aclRet);
        return 1;
    }

    // ---- resolve the shard list: multi-card (--shard_index_dirs + --device_ids) or single ----
    std::vector<std::string> shardDirs;
    std::vector<int32_t> shardDevices;
    if (!FLAGS_shard_index_dirs.empty()) {
        shardDirs = SplitCsv(FLAGS_shard_index_dirs);
        for (const std::string& d : SplitCsv(FLAGS_device_ids))
            shardDevices.push_back(static_cast<int32_t>(std::strtol(d.c_str(), nullptr, 10)));
        if (shardDirs.empty() || shardDirs.size() != shardDevices.size()) {
            LOG_ERROR("--shard_index_dirs (" << shardDirs.size() << ") and --device_ids (" << shardDevices.size()
                                             << ") must be non-empty and equal in count");
            aclFinalize();
            return 1;
        }
    } else {
        shardDirs = {FLAGS_index_dir};
        shardDevices = {FLAGS_device_id};
    }
    const uint32_t nShards = static_cast<uint32_t>(shardDirs.size());

    // Memory milestones (only sampled with --mem_report): baseline before loading
    // any index, and again after all shards are loaded, so the report can show the
    // index footprint and later the search growth as deltas.
    MemSnap memBase, memLoad;
    if (FLAGS_mem_report)
        memBase = SampleMem(shardDevices);

    // ---- load one DataTable per shard, each resident on its own card ----
    std::vector<std::shared_ptr<NpuRetrieval::DataTable>> shardTables(nShards);
    uint64_t totalDocNum = 0;
    auto loadT0 = std::chrono::steady_clock::now();
    for (uint32_t s = 0; s < nShards; ++s) {
        shardTables[s] = std::make_shared<NpuRetrieval::DataTable>();
        auto t0 = std::chrono::steady_clock::now();
        if (!shardTables[s]->LoadData(shardDevices[s], shardDirs[s])) {
            LOG_ERROR("DataTable::LoadData failed for shard " << s << " dir=" << shardDirs[s]
                                                              << " device=" << shardDevices[s]);
            aclFinalize();
            return 1;
        }
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        totalDocNum += shardTables[s]->GetDocNum();
        // Print to stdout, not the logger: cold-start load time is a headline number
        // and the profiled search runs at NPUR_LOG_LEVEL=ERROR, which would drop a
        // LOG_INFO. Same reason the [RESULT] lines bypass the logger.
        std::printf("[LOAD] shard %u on device %d: docNum=%lu segments=%u in %lld ms\n", s, shardDevices[s],
                    static_cast<unsigned long>(shardTables[s]->GetDocNum()), shardTables[s]->GetSegmentNum(),
                    static_cast<long long>(ms));
    }
    {
        auto totalMs =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - loadT0).count();
        std::printf("[LOAD] %u shard(s), %lu docs total, loaded in %lld ms (cold start)\n", nShards,
                    static_cast<unsigned long>(totalDocNum), static_cast<long long>(totalMs));
        std::fflush(stdout);
    }
    auto& dataTable = shardTables[0];  // alias for the single-shard diagnostics below
    // Sharded: the shards are disjoint, so the corpus is their sum. Round-robin: every card
    // holds the whole corpus, so the corpus size is one card's, not the (N-times-counted) sum.
    const uint64_t docNum = FLAGS_round_robin ? shardTables[0]->GetDocNum() : totalDocNum;
    if (FLAGS_mem_report)
        memLoad = SampleMem(shardDevices);

    std::vector<std::vector<float>> queries;
    {
        std::string qerr;
        if (!npur_port::LoadQueries(FLAGS_query_file, queries, &qerr)) {
            LOG_ERROR("failed to load queries: " << qerr);
            aclFinalize();
            return 1;
        }
    }
    // num_queries may exceed the file's query count for stable percentiles;
    // queries then cycle (q % Q). The recall self-check only touches q < recall_queries < Q.
    size_t nq = (FLAGS_num_queries > 0) ? static_cast<size_t>(FLAGS_num_queries) : queries.size();

    // filters (optional)
    FilterSet filters;
    bool haveFilters = false;
    if (!FLAGS_filter_file.empty()) {
        if (!LoadFilters(FLAGS_filter_file, nq, FLAGS_filter_seed, filters)) {
            aclFinalize();
            return 1;
        }
        haveFilters = true;
    }
    std::vector<uint32_t> topks(nq, static_cast<uint32_t>(FLAGS_topk));
    if (!FLAGS_topk_file.empty() && !LoadTopKFile(FLAGS_topk_file, nq, topks)) {
        aclFinalize();
        return 1;
    }
    std::vector<uint64_t> convertedTagDocFreq;
    if (!FLAGS_converted_tag_freq_file.empty() &&
        !LoadConvertedTagFreqFile(FLAGS_converted_tag_freq_file, convertedTagDocFreq)) {
        aclFinalize();
        return 1;
    }
    const std::vector<uint64_t>* convertedTagDocFreqPtr = convertedTagDocFreq.empty() ? nullptr : &convertedTagDocFreq;
    if (!WriteEffectiveFilterExpressions(FLAGS_effective_filter_file, dataTable, filters, haveFilters, topks, nq,
                                         FLAGS_posting_field, convertedTagDocFreqPtr, FLAGS_converted_tag_freq_file,
                                         shardDirs[0])) {
        aclFinalize();
        return 1;
    }
    LOG_INFO("running " << nq << " queries (batch_size=" << FLAGS_batch_size << ", default_topk=" << FLAGS_topk
                        << ", filters=" << (haveFilters ? FLAGS_filter_file : std::string("none")) << ")");

    std::vector<std::vector<int64_t>> resultDocIds(nq);
    std::vector<double> latencies_ms;
    latencies_ms.reserve(nq);
    // Per-query latency in QUERY order (latencies_ms gets sorted for percentiles,
    // losing the index<->ms mapping needed for tail/outlier analysis).
    std::vector<double> perQueryMs(nq, 0.0);
    size_t bs = std::max<int>(1, FLAGS_batch_size);
    const size_t Q = queries.size();

    // Per-shard / merge timing for straggler decomposition (multi-card only).
    const bool dumpShardLat = (nShards > 1) && !FLAGS_shard_latency_dump.empty();
    std::vector<std::vector<double>> shardMs;  // [shard][query] the shard's own search ms
    std::vector<double> mergeMs;               // host merge ms per query
    if (dumpShardLat) {
        shardMs.assign(nShards, std::vector<double>(nq, 0.0));
        mergeMs.assign(nq, 0.0);
    }

    // ---- per-shard search body + optional persistent worker pool (multi-card) --
    // shardResults/shardOk are reused across batches (cleared per batch). Each
    // shard's search writes only its own slot, so worker threads never race.
    std::vector<std::vector<std::shared_ptr<NpuRetrieval::FullRecallResult>>> shardResults(nShards);
    std::vector<char> shardOk(nShards, 1);

    // ---- streaming (online) k-way merge accumulator (--stream_merge) -----------
    // acc[j] is the running merged top-K for query (start+j) in the current batch,
    // kept sorted by (score desc, shard asc). Each shard folds its own sorted top-K
    // into acc the moment its search returns (foldShardIntoAcc, called from
    // runShardBody), so the merge of the first N-1 shards overlaps the wait for the
    // straggler; only the last shard's O(K) fold lands on the critical path. The
    // deterministic tie-break makes the result byte-identical to the barrier k-way
    // merge regardless of the order shards finish in.
    struct MergeEntry {  // 16 bytes: 8-byte id first avoids the padding a leading float would force
        int64_t id;
        float score;
        uint32_t shard;
    };
    std::vector<std::vector<MergeEntry>> acc(nShards > 1 ? bs : 0);
    std::mutex accMtx;
    std::vector<MergeEntry> mergeScratch;  // reused across folds (folds are serialized under accMtx)
    auto foldShardIntoAcc = [&](uint32_t s, size_t start, size_t end) {
        std::lock_guard<std::mutex> lk(accMtx);
        for (size_t q = start; q < end; ++q) {
            const size_t j = q - start;
            const size_t K = topks[q];
            const auto& rs = shardResults[s];
            if (j >= rs.size() || !rs[j])
                continue;
            const auto& sw = rs[j]->scoreWithIndexes;  // this shard's sorted top-K (ids valid in prefix)
            const size_t m = std::min(K, sw.size());
            std::vector<MergeEntry>& a = acc[j];
            // two-way merge a (sorted) with sw[0..m) (sorted), keep the top K. On equal
            // score the lower shard index wins -- identical to the k-way merge's tie-break.
            mergeScratch.clear();
            size_t ia = 0, ib = 0;
            while (mergeScratch.size() < K && (ia < a.size() || ib < m)) {
                bool takeA;
                if (ia >= a.size())
                    takeA = false;
                else if (ib >= m)
                    takeA = true;
                else if (a[ia].score != sw[ib].score)
                    takeA = a[ia].score > sw[ib].score;
                else
                    takeA = a[ia].shard < s;  // equal score: lower shard first
                if (takeA) {
                    mergeScratch.push_back(a[ia]);
                    ++ia;
                } else {
                    mergeScratch.push_back(MergeEntry{static_cast<int64_t>(sw[ib].id), sw[ib].score, s});
                    ++ib;
                }
            }
            a.swap(mergeScratch);
        }
    };

    // One shard's search for the current batch. Used by both the per-batch spawn
    // path and the persistent pool below.
    auto runShardBody = [&](uint32_t s, size_t start, size_t end, bool record) {
        auto st0 = std::chrono::steady_clock::now();
        aclrtSetDevice(shardDevices[s]);  // thread-local device context for this shard
        NpuRetrieval::FullRecallSearcher searcher(shardDevices[s], shardTables[s]);
        std::vector<std::unique_ptr<NpuRetrieval::QueryNode>> owned;
        for (size_t q = start; q < end; ++q) {
            const FilterNode* ast = haveFilters ? filters.astFor(q) : nullptr;
            NpuRetrieval::QueryNode* tree = BuildQueryNode(ast, FLAGS_posting_field, owned);
            searcher.AddQuery(tree, queries[q % Q], topks[q]);
        }
        shardOk[s] = searcher.BatchSearch(shardResults[s], FLAGS_vec_field, /*isMultiShard=*/true) ? 1 : 0;
        // Fold this shard into the running merge now (during the straggler's wait), so
        // the post-barrier path only extracts ids. Timed as part of the shard so
        // max_shard_ms reflects the true per-shard critical contribution.
        if (FLAGS_stream_merge && record && shardOk[s])
            foldShardIntoAcc(s, start, end);
        if (dumpShardLat && record)
            shardMs[s][start] =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - st0).count();
    };

    // Long-lived workers for shards 0..N-2 (the last shard runs inline on the
    // caller). Off by default; --shard_worker_pool enables it to avoid the
    // per-batch thread-spawn cost. Dispatch via a generation-counter handshake:
    // main bumps poolGen under poolMtx, workers process each new generation once.
    std::mutex poolMtx;
    std::condition_variable poolGo, poolDone;
    uint64_t poolGen = 0;
    uint32_t poolBusy = 0;
    bool poolStop = false;
    size_t jobStart = 0, jobEnd = 0;
    bool jobRecord = false;
    auto poolWorker = [&](uint32_t s) {
        uint64_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(poolMtx);
            poolGo.wait(lk, [&] { return poolStop || poolGen != seen; });
            if (poolStop)
                return;
            seen = poolGen;
            size_t st = jobStart, en = jobEnd;
            bool rec = jobRecord;
            lk.unlock();
            runShardBody(s, st, en, rec);
            lk.lock();
            if (--poolBusy == 0)
                poolDone.notify_one();
        }
    };
    std::vector<std::thread> poolThreads;
    if (FLAGS_shard_worker_pool && nShards > 1)
        for (uint32_t s = 0; s + 1 < nShards; ++s)
            poolThreads.emplace_back(poolWorker, s);
    // RAII: stop + join the workers on ANY main() exit (incl. early error returns),
    // so a joinable thread never outlives its mutex/cv or hits std::terminate.
    struct PoolJoiner {
        std::vector<std::thread>& threads;
        std::mutex& mtx;
        std::condition_variable& go;
        uint64_t& gen;
        bool& stop;
        ~PoolJoiner() {
            {
                std::unique_lock<std::mutex> lk(mtx);
                stop = true;
                ++gen;
            }
            go.notify_all();
            for (auto& t : threads)
                if (t.joinable())
                    t.join();
        }
    } poolJoiner{poolThreads, poolMtx, poolGo, poolGen, poolStop};

    // ---- batch runner: single-card (original path) or multi-card fan-out -----
    // Multi-card: search each shard on its own card in its own thread with
    // isMultiShard=true (results carry global ids), then merge host-side into a
    // global top-K (mirrors the original online proxy's multi-shard merge).
    // record=false (warmup) primes each card's pools but skips extraction/merge.
    auto runBatch = [&](size_t start, size_t end, bool record) -> bool {
        const size_t bn = end - start;
        if (nShards == 1) {
            NpuRetrieval::FullRecallSearcher searcher(shardDevices[0], shardTables[0]);
            std::vector<std::unique_ptr<NpuRetrieval::QueryNode>> owned;
            for (size_t q = start; q < end; ++q) {
                const FilterNode* ast = haveFilters ? filters.astFor(q) : nullptr;
                NpuRetrieval::QueryNode* tree = BuildQueryNode(ast, FLAGS_posting_field, owned);
                searcher.AddQuery(tree, queries[q % Q], topks[q]);
            }
            std::vector<std::shared_ptr<NpuRetrieval::FullRecallResult>> results;
            if (!searcher.BatchSearch(results, FLAGS_vec_field, /*isMultiShard=*/false))
                return false;
            if (record)
                for (size_t k = 0; k < bn; ++k)
                    if (k < results.size() && results[k])
                        resultDocIds[start + k] = results[k]->docIds;
            return true;
        }
        // multi-card: search each shard on its own card, then merge host-side. Two
        // dispatch strategies (both run the LAST shard inline on THIS caller thread so
        // wall-clock stays max-over-shards):
        //   default              -> spawn threads for shards 0..N-2, join them per batch
        //   --shard_worker_pool  -> hand the batch to the persistent per-shard workers,
        //                           avoiding the ~0.56ms per-batch thread-spawn fan-out
        // Both write shardResults[s]/shardOk[s]; the fault handling and merge are shared.
        for (uint32_t s = 0; s < nShards; ++s) {
            shardResults[s].clear();
            shardOk[s] = 1;
        }
        if (FLAGS_stream_merge && record)
            for (size_t j = 0; j < bn; ++j)
                acc[j].clear();
        if (FLAGS_shard_worker_pool && nShards > 1) {
            {
                std::unique_lock<std::mutex> lk(poolMtx);
                jobStart = start;
                jobEnd = end;
                jobRecord = record;
                poolBusy = nShards - 1;
                ++poolGen;
            }
            poolGo.notify_all();
            runShardBody(nShards - 1, start, end, record);  // last shard inline
            std::unique_lock<std::mutex> lk(poolMtx);
            poolDone.wait(lk, [&] { return poolBusy == 0; });
        } else {
            std::vector<std::thread> threads;
            threads.reserve(nShards - 1);
            for (uint32_t s = 0; s + 1 < nShards; ++s)
                threads.emplace_back(runShardBody, s, start, end, record);
            runShardBody(nShards - 1, start, end, record);  // last shard inline
            for (auto& t : threads)
                t.join();
        }
        // Fault handling: strict (default) fails the batch on any shard error so recall
        // stays exactly measured; --shard_allow_partial drops the failed shards and
        // merges the survivors (production-style graceful degradation).
        bool allShardsOk = true;
        for (uint32_t s = 0; s < nShards; ++s)
            if (!shardOk[s])
                allShardsOk = false;
        if (!FLAGS_shard_allow_partial) {
            if (!allShardsOk)
                return false;
        } else if (!allShardsOk) {
            for (uint32_t s = 0; s < nShards; ++s)
                if (!shardOk[s])
                    shardResults[s].clear();  // drop failed shards; the merge skips empty heads
        }
        if (record && FLAGS_stream_merge) {
            // Streaming path: acc[j] is already the fully merged top-K (each shard folded
            // itself in as it finished, above). Just extract the ids -- the O(K*N) merge
            // work already happened, mostly hidden behind the straggler's wait.
            for (size_t q = start; q < end; ++q) {
                const size_t j = q - start;
                std::vector<int64_t>& out = resultDocIds[q];
                out.clear();
                out.reserve(acc[j].size());
                for (const auto& e : acc[j])
                    out.push_back(e.id);
            }
            // mergeMs stays 0: the merge cost is folded into the per-shard timing.
        } else if (record) {
            // Merge each query's per-shard candidates into a global top-K. This host
            // merge is part of the multi-card per-query latency, so it stays timed.
            auto m0 = std::chrono::steady_clock::now();
            // k-way merge of the per-shard sorted heads. Each shard's scoreWithIndexes has its
            // first min(topK,size) entries sorted by score desc with valid global .id (the
            // aggregator partial_sorts to topK and resolves ids only in that prefix), and the
            // global top-K is a subset of the union of per-shard top-Ks, so merging just those
            // sorted prefixes yields the exact global top-K -- O(K*N) with no concat/full-sort.
            std::vector<size_t> pos(nShards), lim(nShards);
            for (size_t q = start; q < end; ++q) {
                const size_t j = q - start;
                const size_t K = topks[q];
                for (uint32_t s = 0; s < nShards; ++s) {
                    pos[s] = 0;
                    const auto& rs = shardResults[s];
                    lim[s] = (j < rs.size() && rs[j]) ? std::min<size_t>(K, rs[j]->scoreWithIndexes.size()) : 0;
                }
                std::vector<int64_t>& out = resultDocIds[q];
                out.clear();
                out.reserve(K);
                for (size_t o = 0; o < K; ++o) {
                    int best = -1;
                    float bestScore = 0.0f;
                    for (uint32_t s = 0; s < nShards; ++s) {
                        if (pos[s] < lim[s]) {
                            float sc = shardResults[s][j]->scoreWithIndexes[pos[s]].score;
                            if (best < 0 || sc > bestScore) {
                                bestScore = sc;
                                best = static_cast<int>(s);
                            }
                        }
                    }
                    if (best < 0)
                        break;  // all shard heads exhausted
                    out.push_back(static_cast<int64_t>(shardResults[best][j]->scoreWithIndexes[pos[best]].id));
                    pos[best]++;
                }
            }
            if (dumpShardLat)
                mergeMs[start] =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - m0).count();
        }
        return true;
    };

    // Total wall-clock (ms) for the throughput report: round-robin = the concurrent fan-out;
    // serial = the sum of the back-to-back batch times.
    double wallMs = 0.0;
    if (FLAGS_round_robin && nShards > 1) {
        // ---- round-robin baseline: N full-corpus cards, work-steal queries onto idle ones ----
        // Each card runs the ORIGINAL single-card path (isMultiShard=false) on its own full
        // corpus, so a chunk's result is already the global top-K -- no cross-card merge. A
        // shared cursor hands the next batch_size chunk to whichever card asks first, i.e. to
        // the idle card. Wall-clock is the whole fan-out; per-query latency is one card's time.
        std::atomic<bool> rrFailed{false};
        std::atomic<size_t> cursor{0};  // shared work queue: fetch_add(bs) hands the next chunk to an idle card
        auto rrChunk = [&](uint32_t c, size_t start, size_t end, bool record) {
            NpuRetrieval::FullRecallSearcher searcher(shardDevices[c], shardTables[c]);
            std::vector<std::unique_ptr<NpuRetrieval::QueryNode>> owned;
            for (size_t q = start; q < end; ++q) {
                const FilterNode* ast = haveFilters ? filters.astFor(q) : nullptr;
                NpuRetrieval::QueryNode* tree = BuildQueryNode(ast, FLAGS_posting_field, owned);
                searcher.AddQuery(tree, queries[q % Q], topks[q]);
            }
            auto ct0 = std::chrono::steady_clock::now();
            std::vector<std::shared_ptr<NpuRetrieval::FullRecallResult>> results;
            bool ok = searcher.BatchSearch(results, FLAGS_vec_field, /*isMultiShard=*/false);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ct0).count();
            if (!ok) {
                rrFailed = true;
                return;
            }
            if (record) {
                // True per-query e2e: the whole chunk is submitted and completes together, so
                // every query in it experienced the full chunk wall time (ms), not ms/bs.
                for (size_t k = 0; k < (end - start); ++k) {
                    perQueryMs[start + k] = ms;
                    if (k < results.size() && results[k])
                        resultDocIds[start + k] = results[k]->docIds;
                }
            }
        };
        auto rrWorker = [&](uint32_t c, bool record) {
            aclrtSetDevice(shardDevices[c]);
            for (;;) {
                size_t start = cursor.fetch_add(bs);
                if (start >= nq)
                    break;
                rrChunk(c, start, std::min(nq, start + bs), record);
                if (rrFailed)
                    break;
            }
        };
        // Warmup: prime every card's pools in parallel (each replays the first chunk).
        if (FLAGS_warmup > 0 && nq > 0) {
            std::vector<std::thread> wu;
            for (uint32_t c = 0; c < nShards; ++c)
                wu.emplace_back([&, c] {
                    aclrtSetDevice(shardDevices[c]);
                    for (int w = 0; w < FLAGS_warmup; ++w)
                        rrChunk(c, 0, std::min<size_t>(nq, bs), /*record=*/false);
                });
            for (auto& t : wu)
                t.join();
        }
        // Timed pass: all cards drain the shared cursor concurrently (it is still 0 -- warmup
        // never touches it).
        auto rr_t0 = std::chrono::steady_clock::now();
        std::vector<std::thread> workers;
        workers.reserve(nShards);
        for (uint32_t c = 0; c < nShards; ++c)
            workers.emplace_back(rrWorker, c, /*record=*/true);
        for (auto& t : workers)
            t.join();
        wallMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - rr_t0).count();
        if (rrFailed) {
            LOG_ERROR("round-robin search failed");
            aclFinalize();
            return 1;
        }
        for (size_t q = 0; q < nq; ++q)
            latencies_ms.push_back(perQueryMs[q]);  // single-card per-query proc time
    } else {
        // ---- warmup: prime device pools / first-touch alloc, discarded ---------
        for (int w = 0; w < std::max(0, FLAGS_warmup) && nq > 0; ++w)
            runBatch(0, std::min(nq, bs), /*record=*/false);

        for (size_t start = 0; start < nq; start += bs) {
            size_t end = std::min(nq, start + bs);
            auto t0 = std::chrono::steady_clock::now();
            bool ok = runBatch(start, end, /*record=*/true);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (!ok) {
                LOG_ERROR("search failed at batch [" << start << "," << end << ")");
                aclFinalize();
                return 1;
            }
            wallMs += ms;
            // True per-query e2e: the batch completes as a unit, so each query in it saw the
            // whole batch wall time (ms), not the amortized ms/batch_size.
            for (size_t k = 0; k < (end - start); ++k) {
                latencies_ms.push_back(ms);
                perQueryMs[start + k] = ms;
            }
        }
    }

    if (!FLAGS_latency_dump.empty()) {
        std::FILE* lf = std::fopen(FLAGS_latency_dump.c_str(), "w");
        if (lf == nullptr) {
            LOG_ERROR("cannot open latency_dump file: " << FLAGS_latency_dump);
        } else {
            std::fprintf(lf, "idx\tms\ttopk\n");
            for (size_t q = 0; q < nq; ++q)
                std::fprintf(lf, "%zu\t%.4f\t%d\n", q, perQueryMs[q], topks[q]);
            std::fclose(lf);
            LOG_INFO("wrote per-query latencies to " << FLAGS_latency_dump);
        }
    }

    if (dumpShardLat) {
        std::FILE* sf = std::fopen(FLAGS_shard_latency_dump.c_str(), "w");
        if (sf == nullptr) {
            LOG_ERROR("cannot open shard_latency_dump file: " << FLAGS_shard_latency_dump);
        } else {
            std::fprintf(sf, "idx");
            for (uint32_t s = 0; s < nShards; ++s)
                std::fprintf(sf, "\tshard%u_ms", s);
            std::fprintf(sf, "\tmax_shard_ms\tmerge_ms\ttotal_ms\n");
            for (size_t q = 0; q < nq; ++q) {
                std::fprintf(sf, "%zu", q);
                double mx = 0.0;
                for (uint32_t s = 0; s < nShards; ++s) {
                    std::fprintf(sf, "\t%.4f", shardMs[s][q]);
                    if (shardMs[s][q] > mx)
                        mx = shardMs[s][q];
                }
                std::fprintf(sf, "\t%.4f\t%.4f\t%.4f\n", mx, mergeMs[q], perQueryMs[q]);
            }
            std::fclose(sf);
            LOG_INFO("wrote per-shard latencies to " << FLAGS_shard_latency_dump);
        }
    }

    if (!latencies_ms.empty()) {
        std::sort(latencies_ms.begin(), latencies_ms.end());  // latencies_ms holds each query's TRUE e2e
        auto pct = [&](double p) {
            size_t idx = std::min(latencies_ms.size() - 1, static_cast<size_t>(p * latencies_ms.size()));
            return latencies_ms[idx];
        };
        // avg/p50/p99 over each query's end-to-end latency; throughput = queries / wall-clock
        // (round-robin = N cards concurrent; serial = the Σ of back-to-back batch times).
        // Printed to stdout (not the logger) so results show under any NPUR_LOG_LEVEL.
        double sum = 0;
        for (double v : latencies_ms)
            sum += v;
        const double avg = sum / static_cast<double>(latencies_ms.size());
        const double qps = wallMs > 0 ? 1000.0 * static_cast<double>(latencies_ms.size()) / wallMs : 0.0;
        std::printf("[RESULT] latency ms: avg=%.4f p50=%.4f p99=%.4f qps=%.1f (per-query e2e, n=%zu, batch_size=%zu)\n",
                    avg, pct(0.50), pct(0.99), qps, latencies_ms.size(), bs);
    }

    if (FLAGS_mem_report)
        PrintMemReport(shardDevices, memBase, memLoad, SampleMem(shardDevices), ReadHostKb("VmHWM:"));

    // ---- CPU brute-force recall self-check (optionally cached) --------------
    // The reference top-K is independent of engine tuning knobs, so with
    // --recall_ref_file it is computed once for ALL queries and cached; later
    // runs load it (fingerprint-checked) and verify every query for ~free.
    if (!FLAGS_dataset_hw.empty() && (FLAGS_recall_queries > 0 || !FLAGS_recall_ref_file.empty())) {
        HwData hw;
        if (!hw.Open(FLAGS_dataset_hw)) {
            LOG_ERROR("cannot open/mmap dataset_HW.bin: " << FLAGS_dataset_hw);
        } else if (hw.doc_num < FLAGS_cpu_doc_offset + docNum) {
            LOG_ERROR("dataset_HW.bin doc_num=" << hw.doc_num
                                                << " < cpu_doc_offset+docNum=" << (FLAGS_cpu_doc_offset + docNum));
        } else {
            const bool cached = !FLAGS_recall_ref_file.empty();
            // With a cache we verify every query; otherwise just the first N.
            const size_t rq = cached ? nq : std::min<size_t>(FLAGS_recall_queries, nq);
            const std::string fp = RecallFingerprint(docNum, hw.dim, nq);
            std::vector<std::unordered_set<int64_t>> refs;
            const bool loaded = cached && LoadRecallRef(FLAGS_recall_ref_file, fp, nq, refs);
            if (loaded) {
                LOG_INFO("recall ref loaded from cache " << FLAGS_recall_ref_file << " (" << nq << " queries)");
            } else {
                // Compute the reference in parallel: each query is independent and
                // reads only const shared state (mmap dataset, queries, filter ASTs),
                // and threads write disjoint refs[q], so no locking is needed.
                unsigned nThreads = FLAGS_recall_ref_threads > 0 ? static_cast<unsigned>(FLAGS_recall_ref_threads)
                                                                 : std::max(1u, std::thread::hardware_concurrency());
                if (cached)
                    LOG_INFO("recall ref cache miss/stale, computing " << rq << " brute-force refs (one-time, "
                                                                       << nThreads << " threads)...");
                refs.assign(nq, {});
                std::atomic<size_t> nextQ{0};
                auto worker = [&]() {
                    for (size_t q = nextQ.fetch_add(1); q < rq; q = nextQ.fetch_add(1)) {
                        const FilterNode* ast = haveFilters ? filters.astFor(q) : nullptr;
                        refs[q] =
                            CpuTopK(queries[q], ast, hw, docNum, static_cast<int>(topks[q]), FLAGS_cpu_doc_offset);
                    }
                };
                std::vector<std::thread> pool;
                pool.reserve(nThreads);
                for (unsigned t = 0; t < nThreads; ++t)
                    pool.emplace_back(worker);
                for (std::thread& th : pool)
                    th.join();
                if (cached && !WriteRecallRef(FLAGS_recall_ref_file, fp, refs))
                    LOG_ERROR("failed to write recall ref cache: " << FLAGS_recall_ref_file);
            }
            double recallSum = 0.0;
            size_t counted = 0;
            for (size_t q = 0; q < rq; ++q) {
                const std::unordered_set<int64_t>& ref = refs[q];
                if (ref.empty())
                    continue;  // filter matched nothing; skip
                size_t hit = 0;
                for (int64_t id : resultDocIds[q])
                    if (ref.count(id))
                        ++hit;
                double recall = static_cast<double>(hit) / static_cast<double>(ref.size());
                recallSum += recall;
                ++counted;
                // Print per-query lines only for mismatches (recall < 1), plus the
                // first few in spot-check mode as a sanity confirmation. A full
                // cached run over all queries stays quiet when everything passes.
                if (recall < 1.0 || (!cached && q < 10))
                    std::printf("[RESULT] q%zu recall=%.4f (ref=%zu, engine=%zu)\n", q, recall, ref.size(),
                                resultDocIds[q].size());
            }
            if (counted > 0)
                std::printf("[RESULT] CPU-recall over %zu queries: avg=%.2f%%\n", counted,
                            (recallSum / counted) * 100.0);
            else
                std::printf("[RESULT] recall: no queries had non-empty reference results\n");
        }
        std::fflush(stdout);
    }

    aclFinalize();
    return 0;
}
