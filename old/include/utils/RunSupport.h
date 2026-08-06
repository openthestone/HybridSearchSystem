#pragma once

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "utils/DataReader.h"
#include "Query/Query.h"

namespace RunSupport {
namespace fs = std::filesystem;

inline fs::path FindProjectRootFrom(const fs::path& start_dir) {
    std::error_code ec;
    fs::path p = start_dir;
    while (!p.empty()) {
        const bool has_cmake = fs::exists(p / "CMakeLists.txt", ec) && !ec;
        ec.clear();
        const bool has_src = fs::exists(p / "src", ec) && !ec;
        ec.clear();
        const bool has_include = fs::exists(p / "include", ec) && !ec;
        ec.clear();
        if (has_cmake && has_src && has_include) {
            return p;
        }

        const fs::path parent = p.parent_path();
        if (parent == p) {
            break;
        }
        p = parent;
    }
    return {};
}

inline fs::path ResolveConfigPath() {
    std::error_code ec;
    const fs::path cwd = fs::current_path(ec);
    if (ec) {
        return "config.txt";
    }

    if (cwd.filename() == "bin" && cwd.parent_path().filename() == "out") {
        const fs::path root_config = cwd.parent_path().parent_path() / "config.txt";
        if (fs::exists(root_config, ec) && !ec) {
            return root_config;
        }
        ec.clear();
    }

    const fs::path project_root = FindProjectRootFrom(cwd);
    if (!project_root.empty()) {
        const fs::path root_config = project_root / "config.txt";
        if (fs::exists(root_config, ec) && !ec) {
            return root_config;
        }
        ec.clear();
    }

    const fs::path cwd_config = cwd / "config.txt";
    if (fs::exists(cwd_config, ec) && !ec) {
        return cwd_config;
    }

    return "config.txt";
}

inline std::vector<QueryResult::Item> ConvertCacheItemsToQueryResults(
    const std::vector<DataReader::GroundTruthResultItem>& items) {
    std::vector<QueryResult::Item> out;
    out.reserve(items.size());
    for (const auto& item : items) {
        out.push_back({item.doc_id, item.score});
    }
    return out;
}

inline std::vector<DataReader::GroundTruthResultItem> ConvertQueryResultsToCacheItems(
    const std::vector<QueryResult::Item>& items) {
    std::vector<DataReader::GroundTruthResultItem> out;
    out.reserve(items.size());
    for (const auto& item : items) {
        out.push_back({item.doc_id, item.score});
    }
    return out;
}

inline BucketDocTable BuildBucketDocTableFromOffsets(const std::vector<uint64_t>& bucket_doc_offsets,
                                                     const std::vector<uint32_t>& bucket_doc_ids) {
    if (bucket_doc_offsets.size() < 2) {
        throw std::runtime_error("[Clustering] bucket_doc_offsets size is invalid.");
    }
    if (bucket_doc_offsets.back() != bucket_doc_ids.size()) {
        throw std::runtime_error("[Clustering] bucket_doc_offsets/back does not match bucket_doc_ids size.");
    }

    const size_t bucket_count = bucket_doc_offsets.size() - 1;
    BucketDocTable bucket_doc_table(bucket_count);
    for (size_t bid = 0; bid < bucket_count; ++bid) {
        const uint64_t begin = bucket_doc_offsets[bid];
        const uint64_t end = bucket_doc_offsets[bid + 1];
        if (begin > end || end > bucket_doc_ids.size()) {
            throw std::runtime_error("[Clustering] bucket_doc_offsets range is invalid.");
        }
        auto& bucket_docs = bucket_doc_table[bid];
        bucket_docs.reserve(static_cast<size_t>(end - begin));
        for (uint64_t i = begin; i < end; ++i) {
            bucket_docs.push_back(bucket_doc_ids[static_cast<size_t>(i)]);
        }
    }

    return bucket_doc_table;
}

inline TwoLevelBucketLayout BuildTwoLevelBucketLayout(BucketDocTable level_1_bucket_doc_table,
                                                      int max_doc_per_bucket_level_2, const float* vectors = nullptr,
                                                      int dim = 0, std::vector<float>* l2_centroids_out = nullptr) {
    if (max_doc_per_bucket_level_2 <= 0) {
        throw std::invalid_argument("[Clustering] max_doc_per_bucket_level_2 must be > 0.");
    }

    const bool compute_centroids = (vectors != nullptr && dim > 0 && l2_centroids_out != nullptr);
    if (compute_centroids) {
        l2_centroids_out->clear();
    }

    TwoLevelBucketLayout layout;
    layout.level_1_bucket_doc_table = std::move(level_1_bucket_doc_table);
    layout.level_1_to_level_2_offsets.reserve(layout.level_1_bucket_doc_table.size() + 1);
    layout.level_1_to_level_2_offsets.push_back(0);

    for (size_t level_1_bucket_id = 0; level_1_bucket_id < layout.level_1_bucket_doc_table.size();
         ++level_1_bucket_id) {
        const std::vector<uint32_t>& doc_ids = layout.level_1_bucket_doc_table[level_1_bucket_id];
        size_t cursor = 0;
        while (cursor < doc_ids.size()) {
            const size_t chunk_end = std::min(cursor + static_cast<size_t>(max_doc_per_bucket_level_2), doc_ids.size());
            std::vector<uint32_t> level_2_docs;
            level_2_docs.reserve(chunk_end - cursor);
            for (size_t i = cursor; i < chunk_end; ++i) {
                const uint32_t doc_id = doc_ids[i];
                if (doc_id >= static_cast<uint32_t>(total_doc_num)) {
                    throw std::out_of_range("[Clustering] level-1 bucket contains an invalid doc_id.");
                }
                level_2_docs.push_back(doc_id);
            }

            if (compute_centroids) {
                std::vector<float> centroid(static_cast<size_t>(dim), 0.0f);
                for (uint32_t doc_id : level_2_docs) {
                    const float* v = vectors + static_cast<size_t>(doc_id) * static_cast<size_t>(dim);
                    for (int d = 0; d < dim; ++d) {
                        centroid[static_cast<size_t>(d)] += v[d];
                    }
                }
                float inv = 1.0f / static_cast<float>(level_2_docs.size());
                for (float& val : centroid) {
                    val *= inv;
                }
                size_t old_size = l2_centroids_out->size();
                l2_centroids_out->resize(old_size + static_cast<size_t>(dim));
                std::memcpy(l2_centroids_out->data() + old_size, centroid.data(),
                            static_cast<size_t>(dim) * sizeof(float));
            }

            layout.level_2_bucket_doc_table.push_back(std::move(level_2_docs));
            cursor = chunk_end;
        }

        layout.level_1_to_level_2_offsets.push_back(static_cast<uint32_t>(layout.level_2_bucket_doc_table.size()));
    }

    return layout;
}

inline size_t CountStoredDocRefs(const BucketDocTable& bucket_doc_table) {
    size_t total_refs = 0;
    for (const auto& bucket_docs : bucket_doc_table) {
        total_refs += bucket_docs.size();
    }
    return total_refs;
}

inline bool ValidatePreparedQueriesAgainstPrealloc(const std::vector<DataReader::PreparedQuery>& queries) {
    int max_seen_top_k = 0;
    long long max_seen_expanded_k = 0;
    size_t max_seen_line_no = 0;

    for (const auto& prepared : queries) {
        int expanded_k_value = 0;
        if (!ComputeExpandedKFromTopK(prepared.top_k, expanded_k_value)) {
            std::cerr << "[Config] Error: query line " << prepared.line_no << " has top_k=" << prepared.top_k
                      << ", k_expand_param=" << k_expand_param << ", and expanded_k exceeds int range." << std::endl;
            return false;
        }

        const long long expanded_k = static_cast<long long>(expanded_k_value);
        if (prepared.top_k > max_seen_top_k) {
            max_seen_top_k = prepared.top_k;
        }
        if (expanded_k > max_seen_expanded_k) {
            max_seen_expanded_k = expanded_k;
            max_seen_line_no = prepared.line_no;
        }
    }

    if (max_seen_expanded_k > static_cast<long long>(max_query_topk_prealloc)) {
        const int adjusted = ClampSizeTToInt(static_cast<size_t>(max_seen_expanded_k));
        std::cout << "[Config] Auto-adjust: max_query_topk_prealloc=" << max_query_topk_prealloc << " -> " << adjusted
                  << std::endl;
        max_query_topk_prealloc = adjusted;
    }

    std::cout << "[Query] top_k max=" << max_seen_top_k << ", expanded_k max=" << max_seen_expanded_k << " (line "
              << max_seen_line_no << ")"
              << ", max_query_topk_prealloc=" << max_query_topk_prealloc << "\n";
    return true;
}

inline bool WriteMetricFile(const fs::path& output_dir, const std::string& file_name,
                            const std::vector<double>& values) {
    const fs::path path = output_dir / file_name;
    std::FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) {
        std::cerr << "[Writer] Failed to open metric file: " << path << "\n";
        return false;
    }

    std::string buf;
    buf.reserve(values.size() * 24);
    char tmp[32];
    for (double value : values) {
        if (std::isinf(value)) {
            buf.append("inf\n", 4);
        } else {
            int n = std::snprintf(tmp, sizeof(tmp), "%.5f\n", value);
            buf.append(tmp, static_cast<size_t>(n));
        }
    }
    std::fwrite(buf.data(), 1, buf.size(), fp);
    bool ok = (std::ferror(fp) == 0);
    std::fclose(fp);

    if (!ok) {
        std::cerr << "[Writer] Failed while writing metric file: " << path << "\n";
    }
    return ok;
}

inline bool WriteMetricFile(const fs::path& output_dir, const std::string& file_name,
                            const std::vector<size_t>& values) {
    const fs::path path = output_dir / file_name;
    std::FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) {
        std::cerr << "[Writer] Failed to open metric file: " << path << "\n";
        return false;
    }

    std::string buf;
    buf.reserve(values.size() * 16);
    char tmp[24];
    for (size_t value : values) {
        int n = std::snprintf(tmp, sizeof(tmp), "%zu\n", value);
        buf.append(tmp, static_cast<size_t>(n));
    }
    std::fwrite(buf.data(), 1, buf.size(), fp);
    bool ok = (std::ferror(fp) == 0);
    std::fclose(fp);

    if (!ok) {
        std::cerr << "[Writer] Failed while writing metric file: " << path << "\n";
    }
    return ok;
}

inline bool EnsureOutputDirectory(const fs::path& dir) {
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) {
        std::cerr << "[Writer] Failed to create output directory: " << dir << ", error=" << ec.message() << "\n";
        return false;
    }
    return true;
}

inline bool WriteQueryLoadDiagnosticsFiles(const fs::path& result_root,
                                           const DataReader::QueryLoadDiagnostics& diagnostics) {
    const fs::path log_dir = result_root / "log";
    return EnsureOutputDirectory(log_dir) &&
           WriteMetricFile(log_dir, "query_missing_syntax_filter.txt", diagnostics.missing_syntax_filter_queries) &&
           WriteMetricFile(log_dir, "query_invalid_syntax_filter.txt", diagnostics.invalid_syntax_filter_queries) &&
           WriteMetricFile(log_dir, "query_missing_vector.txt", diagnostics.missing_vector_queries);
}

inline bool WritePreparedQueryFilterFile(const fs::path& result_root,
                                         const std::vector<DataReader::PreparedQuery>& queries) {
    const fs::path log_dir = result_root / "log";
    if (!EnsureOutputDirectory(log_dir)) {
        return false;
    }

    const fs::path output_file = log_dir / "filter.txt";
    std::ofstream out(output_file, std::ios::out | std::ios::trunc);
    if (!out.is_open()) {
        std::cerr << "[Writer] Failed to open filter file: " << output_file << "\n";
        return false;
    }

    for (const auto& query : queries) {
        out << query.filter_expr << '\n';
    }

    if (!out.good()) {
        std::cerr << "[Writer] Failed while writing filter file: " << output_file << "\n";
        return false;
    }

    std::cout << "[Loader] Query boolean filters written before processing. file=" << output_file
              << ", count=" << queries.size() << "\n";
    return true;
}

// =========================================================
// Load-phase diagnostics: timing + RSS
// =========================================================

inline size_t GetProcessRSSBytes() {
#if defined(__linux__)
    std::ifstream f("/proc/self/status");
    if (!f)
        return 0;
    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            size_t val = 0;
            for (size_t i = 6; i < line.size(); ++i) {
                if (std::isdigit(static_cast<unsigned char>(line[i]))) {
                    val = val * 10 + static_cast<size_t>(line[i] - '0');
                }
            }
            return val * 1024ULL;  // kB → bytes
        }
    }
#endif
    return 0;
}

inline double RSSBytesToMB(size_t bytes) {
    return static_cast<double>(bytes) / (1024.0 * 1024.0);
}

class LoadStageTimer {
   public:
    explicit LoadStageTimer(std::string label)
        : label_(std::move(label)), start_(std::chrono::steady_clock::now()), start_rss_(GetProcessRSSBytes()) {
        std::cout << "[LoadStage] >>> " << label_ << " | start_rss=" << std::fixed << std::setprecision(2)
                  << RSSBytesToMB(start_rss_) << " MB\n";
    }
    ~LoadStageTimer() {
        const auto end = std::chrono::steady_clock::now();
        const size_t end_rss = GetProcessRSSBytes();
        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start_).count();
        const double delta_mb = RSSBytesToMB(end_rss) - RSSBytesToMB(start_rss_);
        std::cout << "[LoadStage] <<< " << label_ << " | elapsed=" << std::fixed << std::setprecision(2) << elapsed_ms
                  << " ms"
                  << " | end_rss=" << RSSBytesToMB(end_rss) << " MB"
                  << " | delta=" << (delta_mb >= 0 ? "+" : "") << delta_mb << " MB\n";
    }
    LoadStageTimer(const LoadStageTimer&) = delete;
    LoadStageTimer& operator=(const LoadStageTimer&) = delete;

   private:
    std::string label_;
    std::chrono::steady_clock::time_point start_;
    size_t start_rss_;
};

}  // namespace RunSupport
