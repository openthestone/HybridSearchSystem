#include "filter_setup.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>

#include "src/full_recall/core/hash/hash.h"
#include "src/utils/logger.h"

namespace npur_harness {

bool LoadFilters(const std::string& path, size_t numQueries, uint64_t seed, FilterSet& fs) {
    std::ifstream f(path);
    if (!f) {
        LOG_ERROR("cannot open filter file: " << path);
        return false;
    }
    std::string line;
    while (std::getline(f, line)) {
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

}  // namespace npur_harness
