#include "SPANN.h"
#include <atomic>

#if defined(__has_include)
#if __has_include("SuperKMeans/include/superkmeans/hierarchical_superkmeans.h") && __has_include(<Eigen/Dense>)
#define DQJ_ENABLE_SUPERKMEANS 1
extern "C" {
void sgemm_(const char* transa,
            const char* transb,
            int* m,
            int* n,
            int* k,
            const float* alpha,
            const float* a,
            int* lda,
            const float* b,
            int* ldb,
            const float* beta,
            float* c,
            int* ldc);
}
#include "SuperKMeans/include/superkmeans/hierarchical_superkmeans.h"
#else
#define DQJ_ENABLE_SUPERKMEANS 0
#endif
#else
#define DQJ_ENABLE_SUPERKMEANS 0
#endif

static int g_loaded_total_doc_num = 0;
static int g_loaded_vector_dim = 0;
static int g_loaded_total_tag_num = 0;
static int g_loaded_total_bucket_num = 0;
static int g_loaded_max_doc_per_bucket = 0;

static fs::path ResolveProjectRoot() {
    std::error_code ec;
    fs::path cur = fs::current_path(ec);
    if (ec || cur.empty()) {
        return fs::current_path();
    }

    fs::path p = cur;
    while (!p.empty()) {
        bool has_cmake = fs::exists(p / "CMakeLists.txt", ec) && !ec;
        bool has_src = fs::exists(p / "src", ec) && !ec;
        bool has_include = fs::exists(p / "include", ec) && !ec;
        if (has_cmake && has_src && has_include) {
            return p;
        }
        fs::path parent = p.parent_path();
        if (parent == p) break;
        p = parent;
    }

    return cur;
}

static bool LoadBucketDocIdsBins(const fs::path& buckets_dir,
                                 int bucket_num,
                                 int total_docs,
                                 std::vector<uint64_t>& out_bucket_doc_offsets,
                                 std::vector<uint32_t>& out_bucket_doc_ids) {
    if (bucket_num <= 0 || total_docs <= 0) return false;
    if (!fs::exists(buckets_dir) || !fs::is_directory(buckets_dir)) return false;

    out_bucket_doc_offsets.assign(static_cast<size_t>(bucket_num) + 1, 0ULL);
    out_bucket_doc_ids.clear();

    for (int bid = 0; bid < bucket_num; ++bid) {
        fs::path file = buckets_dir / ("bucket" + std::to_string(bid) + "_doc_ids.bin");
        std::ifstream in(file, std::ios::binary);
        if (!in) {
            out_bucket_doc_offsets[static_cast<size_t>(bid + 1)] = out_bucket_doc_offsets[static_cast<size_t>(bid)];
            continue;
        }

        int64_t n = 0;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (!in || n < 0) return false;

        for (int64_t i = 0; i < n; ++i) {
            int32_t doc_id1 = 0;
            in.read(reinterpret_cast<char*>(&doc_id1), sizeof(doc_id1));
            if (!in) return false;

            if (doc_id1 <= 0 || doc_id1 > total_docs) continue;
            out_bucket_doc_ids.push_back(static_cast<uint32_t>(doc_id1 - 1));
        }
        out_bucket_doc_offsets[static_cast<size_t>(bid + 1)] = static_cast<uint64_t>(out_bucket_doc_ids.size());
    }

    return true;
}

static bool LoadCentroidsDefaultBin(const fs::path& path,
                                    std::vector<float>& out_centroids,
                                    int vector_dim,
                                    int total_bucket_num,
                                    int& inferred_bucket_num) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    int32_t bucket_count = 0;
    int32_t centroid_dim = 0;
    in.read(reinterpret_cast<char*>(&bucket_count), sizeof(bucket_count));
    in.read(reinterpret_cast<char*>(&centroid_dim), sizeof(centroid_dim));
    if (!in || bucket_count <= 0 || centroid_dim <= 0) return false;

    inferred_bucket_num = std::max<int>(total_bucket_num, bucket_count);
    out_centroids.assign(static_cast<size_t>(inferred_bucket_num) * static_cast<size_t>(vector_dim), 0.0f);

    for (int32_t bid = 0; bid < bucket_count; ++bid) {
        std::vector<float> row(static_cast<size_t>(centroid_dim), 0.0f);
        in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
        if (!in) return false;
        if (bid < 0 || bid >= inferred_bucket_num) continue;
        float* dst = out_centroids.data() + static_cast<size_t>(bid) * static_cast<size_t>(vector_dim);
        int copy_dim = std::min<int>(vector_dim, centroid_dim);
        for (int d = 0; d < copy_dim; ++d) dst[d] = row[static_cast<size_t>(d)];
    }
    return true;
}

#if DQJ_ENABLE_SUPERKMEANS
static bool SaveCentroidsDefaultBin(const fs::path& path,
                                    const std::vector<float>& centroids,
                                    int bucket_num,
                                    int vector_dim) {
    if (bucket_num <= 0 || vector_dim <= 0) return false;
    if (centroids.size() < static_cast<size_t>(bucket_num) * static_cast<size_t>(vector_dim)) return false;

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;

    const int32_t bucket_count = static_cast<int32_t>(bucket_num);
    const int32_t centroid_dim = static_cast<int32_t>(vector_dim);
    out.write(reinterpret_cast<const char*>(&bucket_count), sizeof(bucket_count));
    out.write(reinterpret_cast<const char*>(&centroid_dim), sizeof(centroid_dim));

    for (int bid = 0; bid < bucket_num; ++bid) {
        const float* row = centroids.data() + static_cast<size_t>(bid) * static_cast<size_t>(vector_dim);
        out.write(reinterpret_cast<const char*>(row), static_cast<std::streamsize>(sizeof(float) * vector_dim));
        if (!out) return false;
    }

    return true;
}

static bool SaveBucketsDocIdsBins(const fs::path& buckets_dir,
                                  const std::vector<uint32_t>& assignments,
                                  const std::vector<float>& centroids,
                                  int bucket_num,
                                  int total_docs,
                                  int vector_dim) {
    if (bucket_num <= 0 || total_docs <= 0) return false;
    if (assignments.size() != static_cast<size_t>(total_docs)) return false;
    if (vector_dim <= 0) return false;
    if (centroids.size() < static_cast<size_t>(bucket_num) * static_cast<size_t>(vector_dim)) return false;

    std::error_code ec;
    fs::remove_all(buckets_dir, ec);
    ec.clear();
    fs::create_directories(buckets_dir, ec);
    if (ec) return false;

    std::vector<uint64_t> bucket_counts(static_cast<size_t>(bucket_num), 0ULL);
#if defined(_OPENMP)
    const int n_threads = omp_get_max_threads();
    std::vector<std::vector<uint64_t>> local_counts(static_cast<size_t>(n_threads),
                                                    std::vector<uint64_t>(static_cast<size_t>(bucket_num), 0ULL));
#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        auto& lc = local_counts[static_cast<size_t>(tid)];
#pragma omp for schedule(static)
        for (int i = 0; i < total_docs; ++i) {
            uint32_t bid = assignments[static_cast<size_t>(i)];
            if (bid >= static_cast<uint32_t>(bucket_num)) {
                bid %= static_cast<uint32_t>(bucket_num);
            }
            lc[static_cast<size_t>(bid)]++;
        }
    }

    for (int t = 0; t < n_threads; ++t) {
        for (int bid = 0; bid < bucket_num; ++bid) {
            bucket_counts[static_cast<size_t>(bid)] += local_counts[static_cast<size_t>(t)][static_cast<size_t>(bid)];
        }
    }
#else
    for (int i = 0; i < total_docs; ++i) {
        uint32_t bid = assignments[static_cast<size_t>(i)];
        if (bid >= static_cast<uint32_t>(bucket_num)) {
            bid %= static_cast<uint32_t>(bucket_num);
        }
        bucket_counts[static_cast<size_t>(bid)]++;
    }
#endif

    std::vector<uint64_t> bucket_offsets(static_cast<size_t>(bucket_num) + 1, 0ULL);
    for (int bid = 0; bid < bucket_num; ++bid) {
        bucket_offsets[static_cast<size_t>(bid + 1)] =
            bucket_offsets[static_cast<size_t>(bid)] + bucket_counts[static_cast<size_t>(bid)];
    }

    std::vector<int32_t> flat_doc_ids(static_cast<size_t>(total_docs), 0);
    std::vector<uint64_t> write_positions = bucket_offsets;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < total_docs; ++i) {
        uint32_t bid = assignments[static_cast<size_t>(i)];
        if (bid >= static_cast<uint32_t>(bucket_num)) {
            bid %= static_cast<uint32_t>(bucket_num);
        }

        uint64_t pos = 0;
#if defined(_OPENMP)
#pragma omp atomic capture
#endif
        {
            pos = write_positions[static_cast<size_t>(bid)];
            write_positions[static_cast<size_t>(bid)]++;
        }
        flat_doc_ids[static_cast<size_t>(pos)] = static_cast<int32_t>(i + 1);
    }
    std::atomic<bool> write_failed(false);
#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int bid = 0; bid < bucket_num; ++bid) {
        if (write_failed.load(std::memory_order_relaxed)) {
            continue;
        }

        const fs::path bucket_file = buckets_dir / ("bucket" + std::to_string(bid) + "_doc_ids.bin");
        std::ofstream out(bucket_file, std::ios::binary | std::ios::trunc);
        if (!out) {
            write_failed.store(true, std::memory_order_relaxed);
            continue;
        }

        const uint64_t begin = bucket_offsets[static_cast<size_t>(bid)];
        const uint64_t end = bucket_offsets[static_cast<size_t>(bid + 1)];
        const int64_t base_n = static_cast<int64_t>(end - begin);
        out.write(reinterpret_cast<const char*>(&base_n), sizeof(base_n));
        if (!out) {
            write_failed.store(true, std::memory_order_relaxed);
            continue;
        }

        if (base_n <= 0) {
            continue;
        }

        const int32_t* ptr = flat_doc_ids.data() + static_cast<size_t>(begin);
        out.write(reinterpret_cast<const char*>(ptr),
                  static_cast<std::streamsize>(base_n * static_cast<int64_t>(sizeof(int32_t))));
        if (!out) {
            write_failed.store(true, std::memory_order_relaxed);
        }
    }

    return !write_failed.load(std::memory_order_relaxed);
}

#endif

#if DQJ_ENABLE_SUPERKMEANS
static bool RunBalancedSuperKMeansInMain(const fs::path& project_root,
                                         const std::vector<float>& vectors,
                                         int total_doc_num,
                                         int total_bucket_num,
                                         int vector_dim,
                                         int max_doc_per_bucket) {
    if (total_doc_num <= 0 || total_bucket_num <= 0 || vector_dim <= 0) {
        std::cerr << "[Clustering] Runtime 参数非法，无法执行 SuperKMeans 聚类\n";
        return false;
    }
    if (vectors.size() != static_cast<size_t>(total_doc_num) * static_cast<size_t>(vector_dim)) {
        std::cerr << "[Clustering] 输入向量数量与维度不匹配，无法执行 SuperKMeans 聚类\n";
        return false;
    }

    std::cout << "[Clustering] 开始执行平衡版 SuperKMeans 聚类流程（HierarchicalSuperKMeans）\n";
    const auto t0 = std::chrono::steady_clock::now();

    try {
        skmeans::HierarchicalSuperKMeansConfig cfg;
        cfg.n_threads = (THREADS > 0) ? static_cast<uint32_t>(THREADS) : 0;
        cfg.seed = 42;
        cfg.verbose = false;
        cfg.suppress_warnings = true;

        // Quality-first defaults.
        cfg.sampling_fraction = 1.0f;
        cfg.iters_mesoclustering = 5;
        cfg.iters_fineclustering = 10;
        cfg.iters_refinement = 2;
        cfg.tol = 1e-5f;
        cfg.early_termination = false;

        cfg.use_blas_only = false;

        skmeans::HierarchicalSuperKMeans<> kmeans(static_cast<size_t>(total_bucket_num),
                                                  static_cast<size_t>(vector_dim),
                                                  cfg);

        std::vector<float> centroids = kmeans.Train(vectors.data(), static_cast<size_t>(total_doc_num));

        auto print_iter_stats = [&](const char* phase_name,
                                    const std::vector<skmeans::SuperKMeansIterationStats>& stats) {
            for (const auto& st : stats) {
                std::cout << "[SuperKMeans] 迭代 " << st.iteration
                          << " (" << phase_name << ")"
                          << ": loss=" << std::scientific << std::setprecision(4) << st.objective
                          << std::defaultfloat
                          << ", elapsed=" << format_min_sec(st.elapsed_seconds)
                          << "\n";
            }
        };

        print_iter_stats("meso", kmeans.hierarchical_iteration_stats.mesoclustering_iteration_stats);
        print_iter_stats("fine", kmeans.hierarchical_iteration_stats.fineclustering_iteration_stats);
        print_iter_stats("refine", kmeans.hierarchical_iteration_stats.refinement_iteration_stats);

        if (centroids.size() != static_cast<size_t>(total_bucket_num) * static_cast<size_t>(vector_dim)) {
            std::cerr << "[Clustering] SuperKMeans 输出中心点尺寸异常: " << centroids.size() << "\n";
            return false;
        }

        std::vector<uint32_t> assignments =
            kmeans.FastAssign(vectors.data(),
                              centroids.data(),
                              static_cast<size_t>(total_doc_num),
                              static_cast<size_t>(total_bucket_num));
        if (assignments.size() != static_cast<size_t>(total_doc_num)) {
            std::cerr << "[Clustering] SuperKMeans 输出分配尺寸异常: " << assignments.size() << "\n";
            return false;
        }

        const fs::path centroids_file = project_root / "centroids.bin";
        const fs::path buckets_dir = project_root / "buckets";

        if (!SaveCentroidsDefaultBin(centroids_file, centroids, total_bucket_num, vector_dim)) {
            std::cerr << "[Clustering] 写入 centroids.bin 失败: " << centroids_file << "\n";
            return false;
        }
        if (!SaveBucketsDocIdsBins(buckets_dir,
                                 assignments,
                                 centroids,
                                 total_bucket_num,
                                 total_doc_num,
                                 vector_dim)) {
            std::cerr << "[Clustering] 写入 buckets 失败: " << buckets_dir << "\n";
            return false;
        }

        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(t1 - t0).count();
        auto balance = skmeans::SuperKMeans<>::GetClustersBalanceStats(assignments.data(),
                                                                        assignments.size(),
                                                                        static_cast<size_t>(total_bucket_num));
        std::cout << "[Clustering] SuperKMeans 聚类完成, elapsed=" << format_min_sec(elapsed)
                  << ", balance_cv=" << balance.cv
                  << ", min=" << balance.min
                  << ", max=" << balance.max << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[Clustering] SuperKMeans 运行异常: " << e.what() << "\n";
        return false;
    }

    return true;
}
#else
static bool RunBalancedSuperKMeansInMain(const fs::path& project_root,
                                         const std::vector<float>&,
                                         int total_doc_num,
                                         int total_bucket_num,
                                         int vector_dim,
                                         int max_doc_per_bucket) {
    std::cout << "[Clustering] SuperKMeans 依赖不可用（缺少 Eigen 或头文件），回退 SPANN 聚类流程。\n";

    std::vector<std::string> args = {
        "SPANN",
        "--dim", std::to_string(vector_dim),
        "--total", std::to_string(total_doc_num),
        "--clusters", std::to_string(total_bucket_num),
        "--posting-limit", std::to_string(max_doc_per_bucket),
        "--chunk", "50000",
        "--seed", "42",
        "--threads", "0",
        "--iter", "30",
        "--output-root", project_root.string()
    };
    return RunClusteringStageInProcess("SPANN", std::move(args), RunSPANN) == 0;
}
#endif

static bool UpdateBucketsAndCentroidsFromClusterResult(const fs::path& project_root,
                                                        std::vector<uint64_t>& g_bucket_doc_offsets,
                                                        std::vector<uint32_t>& g_bucket_doc_ids,
                                                        std::vector<float>& g_centroids) {
    fs::path centroids_file = project_root / "centroids.bin";
    fs::path buckets_dir = project_root / "buckets";
    if (!fs::exists(centroids_file) || !fs::exists(buckets_dir)) {
        return false;
    }

    std::vector<uint64_t> bucket_doc_offsets_loaded;
    std::vector<uint32_t> bucket_doc_ids_loaded;
    std::vector<float> centroids_loaded;
    int inferred_bucket_num = g_loaded_total_bucket_num;

    if (!LoadCentroidsDefaultBin(centroids_file, centroids_loaded, g_loaded_vector_dim, g_loaded_total_bucket_num, inferred_bucket_num)) {
        std::cerr << "[Clustering] 读取 centroids.bin 失败: " << centroids_file << "\n";
        return false;
    }

    if (!LoadBucketDocIdsBins(buckets_dir,
                              inferred_bucket_num,
                              g_loaded_total_doc_num,
                              bucket_doc_offsets_loaded,
                              bucket_doc_ids_loaded)) {
        std::cerr << "[Clustering] 读取 buckets 目录失败: " << buckets_dir << "\n";
        return false;
    }

    g_bucket_doc_offsets = std::move(bucket_doc_offsets_loaded);
    g_bucket_doc_ids = std::move(bucket_doc_ids_loaded);
    g_centroids = std::move(centroids_loaded);
    g_loaded_total_bucket_num = inferred_bucket_num;

    std::cout << "[Clustering] 已加载并应用聚类结果: bucket_doc_offsets=" << g_bucket_doc_offsets.size()
              << ", bucket_doc_ids=" << g_bucket_doc_ids.size()
              << ", centroids=" << g_centroids.size() << "\n";
    return true;
}

static int SyncClusteringContextFromMain(const std::vector<float>& g_vectors,
                                         const std::vector<uint64_t>& g_bitmaps,
                                         std::vector<uint64_t>& g_bucket_doc_offsets,
                                         std::vector<uint32_t>& g_bucket_doc_ids,
                                         std::vector<float>& g_centroids,
                                         int total_bucket_num,
                                         int total_tag_num,
                                         int vector_dim,
                                         int total_doc_num,
                                         int max_doc_per_bucket) {
    g_loaded_max_doc_per_bucket = max_doc_per_bucket;
    g_loaded_total_bucket_num = total_bucket_num;
    g_loaded_total_tag_num = total_tag_num;
    g_loaded_vector_dim = vector_dim;
    g_loaded_total_doc_num = total_doc_num;

    auto& clustering_ctx = clustering::Clustering::Instance();
    clustering_ctx.Reset();
    clustering_ctx.EnableInMemory(true);
    clustering_ctx.SetRuntimeParams(g_loaded_total_doc_num,
                                    g_loaded_total_tag_num,
                                    g_loaded_total_bucket_num,
                                    g_loaded_vector_dim,
                                    g_loaded_max_doc_per_bucket);

    bool ok = clustering_ctx.SetInputDataset(g_vectors.data(),
                                             g_loaded_total_doc_num,
                                             g_loaded_vector_dim,
                                             g_bitmaps.data(),
                                             g_loaded_total_tag_num);
    if (!ok) {
        std::cerr << "[Clustering] SetInputDataset 失败（仅同步上下文，不影响主流程）\n";
    }

    fs::path project_root = ResolveProjectRoot();
    if (!UpdateBucketsAndCentroidsFromClusterResult(project_root, g_bucket_doc_offsets, g_bucket_doc_ids, g_centroids)) {
        std::cout << "[Clustering] 未检测到有效 centroids.bin / buckets，开始执行聚类...\n";
        if (!RunBalancedSuperKMeansInMain(project_root,
                                          g_vectors,
                                          g_loaded_total_doc_num,
                                          g_loaded_total_bucket_num,
                                          g_loaded_vector_dim,
                                          g_loaded_max_doc_per_bucket)) {
            std::cerr << "[Error] 聚类失败\n";
            return -1;
        }
        if (!UpdateBucketsAndCentroidsFromClusterResult(project_root, g_bucket_doc_offsets, g_bucket_doc_ids, g_centroids)) {
            std::cerr << "[Error] 聚类后读取 centroids.bin / buckets 失败\n";
            return -1;
        }
    }
    return 0;
}
