#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "utils/DataReader.h"
#include "Query/Query.h"

namespace RunSupport
{
namespace fs = std::filesystem;

inline fs::path FindProjectRootFrom(const fs::path &start_dir)
{
    std::error_code ec;
    fs::path p = start_dir;
    while (!p.empty())
    {
        const bool has_cmake = fs::exists(p / "CMakeLists.txt", ec) && !ec;
        ec.clear();
        const bool has_src = fs::exists(p / "src", ec) && !ec;
        ec.clear();
        const bool has_include = fs::exists(p / "include", ec) && !ec;
        ec.clear();
        if (has_cmake && has_src && has_include)
        {
            return p;
        }

        const fs::path parent = p.parent_path();
        if (parent == p)
        {
            break;
        }
        p = parent;
    }
    return {};
}

inline fs::path ResolveConfigPath()
{
    std::error_code ec;
    const fs::path cwd = fs::current_path(ec);
    if (ec)
    {
        return "config.txt";
    }

    if (cwd.filename() == "bin" && cwd.parent_path().filename() == "out")
    {
        const fs::path root_config = cwd.parent_path().parent_path() / "config.txt";
        if (fs::exists(root_config, ec) && !ec)
        {
            return root_config;
        }
        ec.clear();
    }

    const fs::path project_root = FindProjectRootFrom(cwd);
    if (!project_root.empty())
    {
        const fs::path root_config = project_root / "config.txt";
        if (fs::exists(root_config, ec) && !ec)
        {
            return root_config;
        }
        ec.clear();
    }

    const fs::path cwd_config = cwd / "config.txt";
    if (fs::exists(cwd_config, ec) && !ec)
    {
        return cwd_config;
    }

    return "config.txt";
}

inline std::vector<QueryResult::Item> ConvertCacheItemsToQueryResults(
    const std::vector<DataReader::GroundTruthResultItem> &items)
{
    std::vector<QueryResult::Item> out;
    out.reserve(items.size());
    for (const auto &item : items)
    {
        out.push_back({item.doc_id, item.score});
    }
    return out;
}

inline std::vector<DataReader::GroundTruthResultItem> ConvertQueryResultsToCacheItems(
    const std::vector<QueryResult::Item> &items)
{
    std::vector<DataReader::GroundTruthResultItem> out;
    out.reserve(items.size());
    for (const auto &item : items)
    {
        out.push_back({item.doc_id, item.score});
    }
    return out;
}

inline BucketDocTable BuildBucketDocTableFromOffsets(const std::vector<uint64_t> &bucket_doc_offsets,
                                                     const std::vector<uint32_t> &bucket_doc_ids)
{
    if (bucket_doc_offsets.size() < 2)
    {
        throw std::runtime_error("[Clustering] bucket_doc_offsets size is invalid.");
    }
    if (bucket_doc_offsets.back() != bucket_doc_ids.size())
    {
        throw std::runtime_error("[Clustering] bucket_doc_offsets/back does not match bucket_doc_ids size.");
    }

    const size_t bucket_count = bucket_doc_offsets.size() - 1;
    BucketDocTable bucket_doc_table(bucket_count);
    for (size_t bid = 0; bid < bucket_count; ++bid)
    {
        const uint64_t begin = bucket_doc_offsets[bid];
        const uint64_t end = bucket_doc_offsets[bid + 1];
        if (begin > end || end > bucket_doc_ids.size())
        {
            throw std::runtime_error("[Clustering] bucket_doc_offsets range is invalid.");
        }
        auto &bucket_docs = bucket_doc_table[bid];
        bucket_docs.reserve(static_cast<size_t>(end - begin));
        for (uint64_t i = begin; i < end; ++i)
        {
            bucket_docs.push_back(bucket_doc_ids[static_cast<size_t>(i)]);
        }
    }

    return bucket_doc_table;
}

inline size_t CountStoredDocRefs(const BucketDocTable &bucket_doc_table)
{
    size_t total_refs = 0;
    for (const auto &bucket_docs : bucket_doc_table)
    {
        total_refs += bucket_docs.size();
    }
    return total_refs;
}

inline bool ValidatePreparedQueriesAgainstPrealloc(const std::vector<DataReader::PreparedQuery> &queries)
{
    int max_seen_top_k = 0;
    long long max_seen_expanded_k = 0;
    size_t max_seen_line_no = 0;

    for (const auto &prepared : queries)
    {
        const long long expanded_k =
            static_cast<long long>(std::max(prepared.top_k, 0)) * static_cast<long long>(k_expand_param);
        if (expanded_k > static_cast<long long>(std::numeric_limits<int>::max()))
        {
            std::cerr << "[Config] Error: query line " << prepared.line_no
                      << " has expanded_k=" << expanded_k
                      << ", which exceeds int range." << std::endl;
            return false;
        }
        if (expanded_k > static_cast<long long>(max_query_topk_prealloc))
        {
            std::cerr << "[Config] Error: query line " << prepared.line_no
                      << " has top_k=" << prepared.top_k
                      << ", expanded_k=" << expanded_k
                      << ", but max_query_topk_prealloc=" << max_query_topk_prealloc
                      << ". Increase max_query_topk_prealloc (it is the expanded_k preallocation upper bound)."
                      << std::endl;
            return false;
        }

        if (prepared.top_k > max_seen_top_k)
        {
            max_seen_top_k = prepared.top_k;
        }
        if (expanded_k > max_seen_expanded_k)
        {
            max_seen_expanded_k = expanded_k;
            max_seen_line_no = prepared.line_no;
        }
    }

    std::cout << "[Query] top_k max=" << max_seen_top_k
              << ", expanded_k max=" << max_seen_expanded_k
              << " (line " << max_seen_line_no << ")"
              << ", max_query_topk_prealloc=" << max_query_topk_prealloc << "\n";
    return true;
}

inline bool WriteMetricFile(const fs::path &output_dir,
                            const std::string &file_name,
                            const std::vector<double> &values)
{
    std::ofstream out(output_dir / file_name, std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
        std::cerr << "[Writer] Failed to open metric file: " << (output_dir / file_name) << "\n";
        return false;
    }

    out << std::fixed << std::setprecision(5);
    for (double value : values)
    {
        if (std::isinf(value))
        {
            out << "inf\n";
        }
        else
        {
            out << value << '\n';
        }
    }

    if (!out.good())
    {
        std::cerr << "[Writer] Failed while writing metric file: " << (output_dir / file_name) << "\n";
        return false;
    }

    return true;
}

inline bool WriteMetricFile(const fs::path &output_dir,
                            const std::string &file_name,
                            const std::vector<size_t> &values)
{
    std::ofstream out(output_dir / file_name, std::ios::out | std::ios::trunc);
    if (!out.is_open())
    {
        std::cerr << "[Writer] Failed to open metric file: " << (output_dir / file_name) << "\n";
        return false;
    }

    for (size_t value : values)
    {
        out << value << '\n';
    }

    if (!out.good())
    {
        std::cerr << "[Writer] Failed while writing metric file: " << (output_dir / file_name) << "\n";
        return false;
    }

    return true;
}

inline bool EnsureOutputDirectory(const fs::path &dir)
{
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec)
    {
        std::cerr << "[Writer] Failed to create output directory: " << dir
                  << ", error=" << ec.message() << "\n";
        return false;
    }
    return true;
}
} // namespace RunSupport
