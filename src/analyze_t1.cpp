#include <algorithm>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "Clustering/run_clustering.h"
#include "analyze/SelectionRateAnalyzer.h"
#include "utils/DataReader.h"
#include "utils/RunSupport.h"

namespace fs = RunSupport::fs;

namespace
{
bool LoadPreparedQueriesFromFvec(const std::vector<std::string> &query_paths,
                                 int expected_dim,
                                 int default_top_k,
                                 std::vector<DataReader::PreparedQuery> &queries,
                                 std::string &loaded_path)
{
    queries.clear();
    loaded_path.clear();

    if (expected_dim < 64 || default_top_k <= 0)
    {
        return false;
    }

    auto append_query = [&](const float *raw, int source_dim, size_t line_no) {
        DataReader::PreparedQuery q;
        q.query_vec.assign(static_cast<size_t>(expected_dim), 0.0f);
        const int copy_dim = std::min<int>(64, std::min<int>(expected_dim, source_dim));
        for (int d = 0; d < copy_dim; ++d)
        {
            q.query_vec[static_cast<size_t>(d)] = raw[static_cast<size_t>(d)];
        }
        q.filter_expr = "";
        q.top_k = default_top_k;
        q.line_no = line_no;
        queries.push_back(std::move(q));
    };

    for (const auto &path : query_paths)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
        {
            continue;
        }

        in.seekg(0, std::ios::end);
        const std::streamoff file_size = in.tellg();
        in.seekg(0, std::ios::beg);
        if (file_size < static_cast<std::streamoff>(sizeof(int32_t) * 2))
        {
            continue;
        }

        int32_t first = 0;
        int32_t second = 0;
        in.read(reinterpret_cast<char *>(&first), sizeof(int32_t));
        in.read(reinterpret_cast<char *>(&second), sizeof(int32_t));
        if (!in)
        {
            continue;
        }

        const std::streamoff headered_bytes = static_cast<std::streamoff>(sizeof(int32_t) * 2) +
                                              static_cast<std::streamoff>(first) * static_cast<std::streamoff>(second) *
                                                  static_cast<std::streamoff>(sizeof(float));

        if (first > 0 && second >= 64 && file_size == headered_bytes)
        {
            const int32_t rows = first;
            const int32_t source_dim = second;
            std::vector<float> raw(static_cast<size_t>(source_dim));
            queries.clear();
            queries.reserve(static_cast<size_t>(rows));
            for (int32_t i = 0; i < rows; ++i)
            {
                in.read(reinterpret_cast<char *>(raw.data()),
                        static_cast<std::streamsize>(raw.size() * sizeof(float)));
                if (!in)
                {
                    queries.clear();
                    break;
                }
                append_query(raw.data(), source_dim, static_cast<size_t>(i + 1));
            }
            if (!queries.empty())
            {
                loaded_path = path;
                return true;
            }
            continue;
        }

        in.clear();
        in.seekg(0, std::ios::beg);
        queries.clear();
        size_t line_no = 0;
        while (true)
        {
            int32_t dim = 0;
            in.read(reinterpret_cast<char *>(&dim), sizeof(int32_t));
            if (!in)
            {
                break;
            }
            if (dim < 64)
            {
                queries.clear();
                break;
            }

            std::vector<float> raw(static_cast<size_t>(dim));
            in.read(reinterpret_cast<char *>(raw.data()),
                    static_cast<std::streamsize>(raw.size() * sizeof(float)));
            if (!in)
            {
                queries.clear();
                break;
            }

            ++line_no;
            append_query(raw.data(), dim, line_no);
        }

        if (!queries.empty())
        {
            loaded_path = path;
            return true;
        }
    }

    return false;
}
} // namespace

int main()
{
    constexpr const char *kDatasetFile = "../../dataset_HW.bin";
    constexpr int kDefaultTopK = 100;
    const std::vector<std::string> kQueryPaths = {
        "datasets/hw_queries.fvecs",
        "../datasets/hw_queries.fvecs",
        "../../datasets/hw_queries.fvecs",
        "../../../datasets/hw_queries.fvecs"
    };

    const fs::path config_path = RunSupport::ResolveConfigPath();
    std::cout << "[Analyze] Loading config from " << config_path << "...\n";
    if (!LoadParams(config_path.string(), ResourceConfigProfile::Parallel))
    {
        std::cerr << "[Error] Failed to load config: " << config_path << "\n";
        return -1;
    }

#ifdef _OPENMP
    if (cpu_core_count > 0)
    {
        omp_set_num_threads(cpu_core_count);
    }
#endif

    const fs::path query_result_root_dir =
        query_result_root.empty() ? fs::path(".") : fs::path(query_result_root);
    const fs::path analyze_output_file = query_result_root_dir / "log" / "analyze_t1.txt";

    std::cout << "[Loader] Loading dataset cache...\n";
    DataReader::DatasetBuffers dataset_buffers;
    DataReader::LoadFailureReason dataset_failure = DataReader::LoadFailureReason::None;
    if (!DataReader::LoadDatasetCache(kDatasetFile, dataset_buffers, &dataset_failure))
    {
        if (dataset_failure != DataReader::LoadFailureReason::VectorDimMismatch)
        {
            std::cerr << "[Fatal] Failed to load dataset cache: " << kDatasetFile << "\n";
        }
        return -1;
    }
    std::cout << "[Loader] Dataset loaded. Docs=" << total_doc_num
              << ", Dim=" << vector_dim
              << ", Tags=" << total_tag_num
              << ", Buckets(Level1)=" << total_bucket_num_level_1 << "\n";

    std::cout << "[Loader] Loading queries...\n";
    std::vector<DataReader::PreparedQuery> queries;
    std::string loaded_query_path;
    if (!LoadPreparedQueriesFromFvec(kQueryPaths,
                                     vector_dim,
                                     kDefaultTopK,
                                     queries,
                                     loaded_query_path))
    {
        std::cerr << "[Fatal] Failed to read queries from deep1B_queries.fvecs\n";
        return -1;
    }
    std::cout << "[Loader] Queries loaded from " << loaded_query_path << "\n";

    if (queries.empty())
    {
        std::cout << "[Info] No valid query found in " << loaded_query_path << ".\n";
        if (!Analyze::SelectionRateAnalyzer::WriteRows(analyze_output_file, {}))
        {
            std::cerr << "[Fatal] Failed to create empty analysis file: " << analyze_output_file << "\n";
            return -1;
        }
        return 0;
    }

    std::cout << "[Loader] Prepared query count: " << queries.size() << "\n";
    std::cout << "[Analyze] analyze_t1=" << ::analyze_t1
              << ", batch0(level_1)=" << ::valid_bucket_num_base_level_1
              << ", incremental(level_1)=" << ::valid_bucket_num_incremental_level_1 << "\n";
    FilterExpCompiler::WarmUpThreadLocalBuffers();

    std::cout << "[Clustering] Syncing clustering context...\n";
    std::vector<uint64_t> bucket_doc_offsets;
    std::vector<uint32_t> bucket_doc_ids;
    const int sync_result = SyncClusteringContextFromMain(dataset_buffers.vectors,
                                                          dataset_buffers.bitmaps,
                                                          bucket_doc_offsets,
                                                          bucket_doc_ids,
                                                          dataset_buffers.centroids,
                                                          total_bucket_num_level_1,
                                                          total_tag_num,
                                                          vector_dim,
                                                          total_doc_num,
                                                          max_doc_per_bucket_level_2);
    if (sync_result != 0)
    {
        std::cerr << "[Fatal] Failed to run or load clustering outputs (centroids.bin/buckets).\n";
        return -1;
    }

    if (bucket_doc_offsets.size() < 2)
    {
        std::cerr << "[Fatal] Invalid clustering output: bucket_doc_offsets is empty.\n";
        return -1;
    }

    const size_t clustered_bucket_count = bucket_doc_offsets.size() - 1;
    if (clustered_bucket_count != static_cast<size_t>(total_bucket_num_level_1))
    {
        std::cerr << "[Fatal] Invalid clustering output: config total_bucket_num_level_1="
                  << total_bucket_num_level_1
                  << ", but clustering produced "
                  << clustered_bucket_count
                  << " level-1 buckets.\n";
        return -1;
    }

    if (dataset_buffers.centroids.size() !=
        static_cast<size_t>(total_bucket_num_level_1) * static_cast<size_t>(vector_dim))
    {
        std::cerr << "[Fatal] Invalid clustering output: centroids size mismatch.\n";
        return -1;
    }

    const InputDataset dataset = DataReader::BuildInputDataset(dataset_buffers);
    BucketDocTable level_1_bucket_doc_table;
    try
    {
        level_1_bucket_doc_table =
            RunSupport::BuildBucketDocTableFromOffsets(bucket_doc_offsets, bucket_doc_ids);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Fatal] Failed to build BucketDocTable from clustering output: " << e.what() << "\n";
        return -1;
    }

    TwoLevelBucketLayout bucket_layout;
    try
    {
        bucket_layout = RunSupport::BuildTwoLevelBucketLayout(std::move(level_1_bucket_doc_table),
                                                              max_doc_per_bucket_level_2);
    }
    catch (const std::exception &e)
    {
        std::cerr << "[Fatal] Failed to adapt clustering output into two-level buckets: "
                  << e.what() << "\n";
        return -1;
    }

    total_bucket_num_level_2 = static_cast<int>(bucket_layout.Level2BucketCount());
    if (!FinalizePreallocationParams())
    {
        return -1;
    }

    std::cout << "[Analyze] Analysis input ready. StoredDocRefs="
              << RunSupport::CountStoredDocRefs(bucket_layout.level_1_bucket_doc_table)
              << ", Level1BucketCount=" << bucket_layout.Level1BucketCount()
              << ", Level2BucketCount=" << bucket_layout.Level2BucketCount() << "\n";

    Analyze::SelectionRateAnalyzer analyzer(dataset, bucket_layout.level_1_bucket_doc_table);
    std::vector<Analyze::QuerySelectionRateRow> rows;
    rows.reserve(queries.size());

    for (size_t idx = 0; idx < queries.size(); ++idx)
    {
        rows.push_back(analyzer.AnalyzeQuery(queries[idx]));
        std::cout << (idx + 1) << "/" << queries.size() << "\n";
    }

    if (!Analyze::SelectionRateAnalyzer::WriteRows(analyze_output_file, rows))
    {
        std::cerr << "[Fatal] Failed to write analysis output: " << analyze_output_file << "\n";
        return -1;
    }

    std::cout << "[Analyze] Output written to " << analyze_output_file << "\n";
    return 0;
}
