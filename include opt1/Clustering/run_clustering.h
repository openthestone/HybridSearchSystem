#include "SPANN.h"
#include <atomic>

static void AddClusteringCacheDiagnostic(std::vector<std::string>* diagnostics,
                                         const std::string& category,
                                         const std::string& detail) {
    if (diagnostics == nullptr) return;
    diagnostics->push_back(category + ": " + detail);
}

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
static int g_loaded_total_bucket_num_level_1 = 0;
static int g_loaded_max_doc_per_bucket = 0;
static std::vector<uint32_t> g_l1_to_l0_map;

static bool LoadL1ToL0Map(const fs::path& project_root, int expected_l1_count) {
    const fs::path map_file = project_root / "l1_to_l0_map.bin";
    std::ifstream ifs(map_file, std::ios::binary | std::ios::ate);
    if (!ifs) return false;
    auto file_size = ifs.tellg();
    if (file_size != static_cast<std::streamsize>(expected_l1_count * sizeof(uint32_t))) {
        std::cerr << "[Clustering] l1_to_l0_map.bin size mismatch: expected "
                  << expected_l1_count * sizeof(uint32_t) << " bytes, got " << file_size << "\n";
        return false;
    }
    ifs.seekg(0);
    g_l1_to_l0_map.resize(static_cast<size_t>(expected_l1_count));
    ifs.read(reinterpret_cast<char*>(g_l1_to_l0_map.data()),
             static_cast<std::streamsize>(g_l1_to_l0_map.size() * sizeof(uint32_t)));
    if (!ifs) {
        g_l1_to_l0_map.clear();
        return false;
    }
    std::cout << "[Clustering] 已加载 l1_to_l0_map.bin: " << g_l1_to_l0_map.size() << " entries\n";
    return true;
}


static fs::path ResolveProjectRoot() {
    std::error_code ec;
    const char* env_root = std::getenv("HYBRID_PROJECT_ROOT");
    if (env_root != nullptr && env_root[0] != '\0') {
        fs::path root = fs::path(env_root);
        if (fs::exists(root, ec) && !ec && fs::is_directory(root, ec) && !ec) {
            return root;
        }
        ec.clear();
    }

    fs::path cur = fs::current_path(ec);
    if (ec || cur.empty()) {
        return fs::current_path();
    }

    if (cur.filename() == "bin" && cur.parent_path().filename() == "out") {
        fs::path installed_project_root = cur.parent_path().parent_path();
        if (!installed_project_root.empty()) {
            return installed_project_root;
        }
    }
    if (cur.filename() == "out") {
        fs::path installed_project_root = cur.parent_path();
        if (!installed_project_root.empty()) {
            return installed_project_root;
        }
    }

    fs::path p = cur;
    while (!p.empty()) {
        ec.clear();
        bool has_cmake = fs::exists(p / "CMakeLists.txt", ec) && !ec;
        ec.clear();
        bool has_src = fs::exists(p / "src", ec) && !ec;
        ec.clear();
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
                                 std::vector<uint32_t>& out_bucket_doc_ids,
                                 std::vector<std::string>* diagnostics = nullptr) {
    if (bucket_num <= 0 || total_docs <= 0) {
        std::ostringstream oss;
        oss << "无法读取 buckets，运行参数非法: bucket_num=" << bucket_num
            << ", total_docs=" << total_docs;
        AddClusteringCacheDiagnostic(diagnostics, "运行参数非法", oss.str());
        return false;
    }

    std::error_code ec;
    const bool exists = fs::exists(buckets_dir, ec);
    if (ec || !exists) {
        std::ostringstream oss;
        oss << "buckets 目录不存在: " << buckets_dir;
        if (ec) oss << ", fs_error=" << ec.message();
        AddClusteringCacheDiagnostic(diagnostics, "路径缺失", oss.str());
        return false;
    }
    ec.clear();
    if (!fs::is_directory(buckets_dir, ec) || ec) {
        std::ostringstream oss;
        oss << "buckets 路径不是目录: " << buckets_dir;
        if (ec) oss << ", fs_error=" << ec.message();
        AddClusteringCacheDiagnostic(diagnostics, "路径类型错误", oss.str());
        return false;
    }

    out_bucket_doc_offsets.assign(static_cast<size_t>(bucket_num) + 1, 0ULL);
    out_bucket_doc_ids.clear();

    int missing_bucket_files = 0;
    for (int bid = 0; bid < bucket_num; ++bid) {
        fs::path file = buckets_dir / ("bucket" + std::to_string(bid) + "_doc_ids.bin");
        std::ifstream in(file, std::ios::binary);
        if (!in) {
            ++missing_bucket_files;
            out_bucket_doc_offsets[static_cast<size_t>(bid + 1)] = out_bucket_doc_offsets[static_cast<size_t>(bid)];
            continue;
        }

        int64_t n = 0;
        in.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (!in) {
            std::ostringstream oss;
            oss << "bucket 文件头读取失败: " << file
                << ", bucket_id=" << bid
                << ", expected_header_bytes=" << sizeof(n);
            AddClusteringCacheDiagnostic(diagnostics, "buckets 文件损坏", oss.str());
            return false;
        }
        if (n < 0) {
            std::ostringstream oss;
            oss << "bucket 文件声明了负数 doc 数: " << file
                << ", bucket_id=" << bid
                << ", declared_doc_count=" << n;
            AddClusteringCacheDiagnostic(diagnostics, "buckets 文件损坏", oss.str());
            return false;
        }

        for (int64_t i = 0; i < n; ++i) {
            int32_t doc_id1 = 0;
            in.read(reinterpret_cast<char*>(&doc_id1), sizeof(doc_id1));
            if (!in) {
                std::ostringstream oss;
                oss << "bucket 文件 doc_id 数据不完整: " << file
                    << ", bucket_id=" << bid
                    << ", declared_doc_count=" << n
                    << ", failed_at_doc_index=" << i;
                AddClusteringCacheDiagnostic(diagnostics, "buckets 文件损坏", oss.str());
                return false;
            }

            if (doc_id1 <= 0 || doc_id1 > total_docs) continue;
            out_bucket_doc_ids.push_back(static_cast<uint32_t>(doc_id1 - 1));
        }
        out_bucket_doc_offsets[static_cast<size_t>(bid + 1)] = static_cast<uint64_t>(out_bucket_doc_ids.size());
    }

    if (missing_bucket_files > 0) {
        std::cout << "[Clustering] buckets 目录中缺失 " << missing_bucket_files
                  << " 个 bucket*_doc_ids.bin 文件，已按空桶处理。\n";
    }

    return true;
}

static bool LoadCentroidsDefaultBin(const fs::path& path,
                                    std::vector<float>& out_centroids,
                                    int vector_dim,
                                    int expected_bucket_num,
                                    std::vector<std::string>* diagnostics = nullptr) {
    if (vector_dim <= 0 || expected_bucket_num <= 0) {
        std::ostringstream oss;
        oss << "无法读取 centroids.bin，运行参数非法: vector_dim=" << vector_dim
            << ", expected_bucket_num=" << expected_bucket_num;
        AddClusteringCacheDiagnostic(diagnostics, "运行参数非法", oss.str());
        return false;
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) {
        std::ostringstream oss;
        oss << "centroids.bin 打开失败: " << path;
        AddClusteringCacheDiagnostic(diagnostics, "路径不可读", oss.str());
        return false;
    }

    int32_t bucket_count = 0;
    int32_t centroid_dim = 0;
    in.read(reinterpret_cast<char*>(&bucket_count), sizeof(bucket_count));
    in.read(reinterpret_cast<char*>(&centroid_dim), sizeof(centroid_dim));
    if (!in) {
        std::ostringstream oss;
        oss << "centroids.bin 头部读取失败: " << path
            << ", expected_header_bytes=" << (sizeof(bucket_count) + sizeof(centroid_dim));
        AddClusteringCacheDiagnostic(diagnostics, "centroids 头部损坏", oss.str());
        return false;
    }
    if (bucket_count <= 0 || centroid_dim <= 0) {
        std::ostringstream oss;
        oss << "centroids.bin 头部字段非法: " << path
            << ", bucket_count=" << bucket_count
            << ", centroid_dim=" << centroid_dim;
        AddClusteringCacheDiagnostic(diagnostics, "centroids 头部损坏", oss.str());
        return false;
    }

    if (bucket_count != expected_bucket_num) {
        std::cout << "[Clustering] Ignore cached centroids.bin because bucket_count="
                  << bucket_count
                  << " differs from config total_bucket_num_level_1="
                  << expected_bucket_num
                  << ".\n";
        std::ostringstream oss;
        oss << "centroids.bin bucket_count 与配置不一致: " << path
            << ", file_bucket_count=" << bucket_count
            << ", config_total_bucket_num_level_1=" << expected_bucket_num;
        AddClusteringCacheDiagnostic(diagnostics, "配置不匹配", oss.str());
        return false;
    }

    out_centroids.assign(static_cast<size_t>(expected_bucket_num) * static_cast<size_t>(vector_dim), 0.0f);

    for (int32_t bid = 0; bid < bucket_count; ++bid) {
        std::vector<float> row(static_cast<size_t>(centroid_dim), 0.0f);
        in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(row.size() * sizeof(float)));
        if (!in) {
            std::ostringstream oss;
            oss << "centroids.bin 数据区不完整: " << path
                << ", failed_at_bucket_id=" << bid
                << ", centroid_dim=" << centroid_dim
                << ", expected_row_bytes=" << (static_cast<size_t>(centroid_dim) * sizeof(float));
            AddClusteringCacheDiagnostic(diagnostics, "centroids 数据损坏", oss.str());
            return false;
        }
        if (bid < 0 || bid >= expected_bucket_num) continue;
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
                                         int total_bucket_num_level_1,
                                         int vector_dim,
                                         int max_doc_per_bucket) {
    if (total_doc_num <= 0 || total_bucket_num_level_1 <= 0 || vector_dim <= 0) {
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

        skmeans::HierarchicalSuperKMeans<> kmeans(static_cast<size_t>(total_bucket_num_level_1),
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

        if (centroids.size() != static_cast<size_t>(total_bucket_num_level_1) * static_cast<size_t>(vector_dim)) {
            std::cerr << "[Clustering] SuperKMeans 输出中心点尺寸异常: " << centroids.size() << "\n";
            return false;
        }

        std::vector<uint32_t> assignments =
            kmeans.FastAssign(vectors.data(),
                              centroids.data(),
                              static_cast<size_t>(total_doc_num),
                              static_cast<size_t>(total_bucket_num_level_1));
        if (assignments.size() != static_cast<size_t>(total_doc_num)) {
            std::cerr << "[Clustering] SuperKMeans 输出分配尺寸异常: " << assignments.size() << "\n";
            return false;
        }

        const fs::path centroids_file = project_root / "centroids.bin";
        const fs::path buckets_dir = project_root / "buckets";

        if (!SaveCentroidsDefaultBin(centroids_file, centroids, total_bucket_num_level_1, vector_dim)) {
            std::cerr << "[Clustering] 写入 centroids.bin 失败: " << centroids_file << "\n";
            return false;
        }
        if (!SaveBucketsDocIdsBins(buckets_dir,
                                 assignments,
                                 centroids,
                                 total_bucket_num_level_1,
                                 total_doc_num,
                                 vector_dim)) {
            std::cerr << "[Clustering] 写入 buckets 失败: " << buckets_dir << "\n";
            return false;
        }

        // Save L1→L0 mapping (mesocluster membership)
        {
            std::vector<uint32_t> l1_to_l0_map = kmeans.GetL1ToL0Map();
            if (static_cast<int>(l1_to_l0_map.size()) != total_bucket_num_level_1) {
                std::cerr << "[Clustering] L1→L0 map size mismatch: " << l1_to_l0_map.size()
                          << " vs expected " << total_bucket_num_level_1 << "\n";
                return false;
            }
            const fs::path l1_l0_map_file = project_root / "l1_to_l0_map.bin";
            std::ofstream ofs(l1_l0_map_file, std::ios::binary);
            if (!ofs) {
                std::cerr << "[Clustering] 无法创建 l1_to_l0_map.bin: " << l1_l0_map_file << "\n";
                return false;
            }
            ofs.write(reinterpret_cast<const char*>(l1_to_l0_map.data()),
                      static_cast<std::streamsize>(l1_to_l0_map.size() * sizeof(uint32_t)));
            if (!ofs) {
                std::cerr << "[Clustering] 写入 l1_to_l0_map.bin 失败\n";
                return false;
            }
            std::cout << "[Clustering] L1→L0 mapping saved: " << l1_to_l0_map.size()
                      << " entries → " << l1_l0_map_file << "\n";
        }

        const auto t1 = std::chrono::steady_clock::now();
        const double elapsed = std::chrono::duration<double>(t1 - t0).count();
        auto balance = skmeans::SuperKMeans<>::GetClustersBalanceStats(assignments.data(),
                                                                        assignments.size(),
                                                                        static_cast<size_t>(total_bucket_num_level_1));
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
                                         int total_bucket_num_level_1,
                                         int vector_dim,
                                         int max_doc_per_bucket) {
    std::cout << "[Clustering] SuperKMeans 依赖不可用（缺少 Eigen 或头文件），回退 SPANN 聚类流程。\n";

    std::vector<std::string> args = {
        "SPANN",
        "--dim", std::to_string(vector_dim),
        "--total", std::to_string(total_doc_num),
        "--clusters", std::to_string(total_bucket_num_level_1),
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
                                                        std::vector<float>& g_centroids,
                                                        std::vector<std::string>* diagnostics = nullptr) {
    fs::path centroids_file = project_root / "centroids.bin";
    fs::path buckets_dir = project_root / "buckets";
    std::error_code ec;
    bool centroids_path_ok = true;
    bool buckets_path_ok = true;
    const bool centroids_exists = fs::exists(centroids_file, ec);
    if (ec || !centroids_exists) {
        std::ostringstream oss;
        oss << "centroids.bin 不存在: " << centroids_file;
        if (ec) oss << ", fs_error=" << ec.message();
        AddClusteringCacheDiagnostic(diagnostics, "路径缺失", oss.str());
        centroids_path_ok = false;
    } else {
        ec.clear();
        if (fs::is_directory(centroids_file, ec) || ec) {
            std::ostringstream oss;
            oss << "centroids.bin 路径不是普通文件: " << centroids_file;
            if (ec) oss << ", fs_error=" << ec.message();
            AddClusteringCacheDiagnostic(diagnostics, "路径类型错误", oss.str());
            centroids_path_ok = false;
        }
    }
    ec.clear();
    const bool buckets_exists = fs::exists(buckets_dir, ec);
    if (ec || !buckets_exists) {
        std::ostringstream oss;
        oss << "buckets 目录不存在: " << buckets_dir;
        if (ec) oss << ", fs_error=" << ec.message();
        AddClusteringCacheDiagnostic(diagnostics, "路径缺失", oss.str());
        buckets_path_ok = false;
    } else {
        ec.clear();
        if (!fs::is_directory(buckets_dir, ec) || ec) {
            std::ostringstream oss;
            oss << "buckets 路径不是目录: " << buckets_dir;
            if (ec) oss << ", fs_error=" << ec.message();
            AddClusteringCacheDiagnostic(diagnostics, "路径类型错误", oss.str());
            buckets_path_ok = false;
        }
    }
    if (!centroids_path_ok || !buckets_path_ok) {
        return false;
    }

    std::vector<uint64_t> bucket_doc_offsets_loaded;
    std::vector<uint32_t> bucket_doc_ids_loaded;
    std::vector<float> centroids_loaded;
    const bool centroids_ok = LoadCentroidsDefaultBin(centroids_file,
                                                      centroids_loaded,
                                                      g_loaded_vector_dim,
                                                      g_loaded_total_bucket_num_level_1,
                                                      diagnostics);
    if (!centroids_ok) {
        std::cout << "[Clustering] Cached centroids.bin is missing or does not match config: "
                  << centroids_file << "\n";
    }

    const bool buckets_ok = LoadBucketDocIdsBins(buckets_dir,
                                                 g_loaded_total_bucket_num_level_1,
                                                 g_loaded_total_doc_num,
                                                 bucket_doc_offsets_loaded,
                                                 bucket_doc_ids_loaded,
                                                 diagnostics);
    if (!buckets_ok) {
        std::cerr << "[Clustering] 读取 buckets 目录失败: " << buckets_dir << "\n";
    }

    if (!centroids_ok || !buckets_ok) {
        return false;
    }

    g_bucket_doc_offsets = std::move(bucket_doc_offsets_loaded);
    g_bucket_doc_ids = std::move(bucket_doc_ids_loaded);
    g_centroids = std::move(centroids_loaded);

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
                                         int total_bucket_num_level_1,
                                         int total_tag_num,
                                         int vector_dim,
                                         int total_doc_num,
                                         int max_doc_per_bucket) {
    g_loaded_max_doc_per_bucket = max_doc_per_bucket;
    g_loaded_total_bucket_num_level_1 = total_bucket_num_level_1;
    g_loaded_total_tag_num = total_tag_num;
    g_loaded_vector_dim = vector_dim;
    g_loaded_total_doc_num = total_doc_num;

    auto& clustering_ctx = clustering::Clustering::Instance();
    clustering_ctx.Reset();
    clustering_ctx.EnableInMemory(true);
    clustering_ctx.SetRuntimeParams(g_loaded_total_doc_num,
                                    g_loaded_total_tag_num,
                                    g_loaded_total_bucket_num_level_1,
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
    std::cout << "[Clustering] Cache diagnostics enabled. project_root=" << project_root << "\n"
              << "[Clustering] Expect cache files: " << (project_root / "centroids.bin")
              << " and " << (project_root / "buckets") << "\n";
    std::vector<std::string> cache_diagnostics;
    if (!UpdateBucketsAndCentroidsFromClusterResult(project_root,
                                                    g_bucket_doc_offsets,
                                                    g_bucket_doc_ids,
                                                    g_centroids,
                                                    &cache_diagnostics)) {
        std::cout << "[Clustering] 未检测到有效 centroids.bin / buckets，开始执行聚类...\n";
        std::cout << "[Clustering] 缓存诊断条目数: " << cache_diagnostics.size() << "\n";
        if (!cache_diagnostics.empty()) {
            std::cout << "[Clustering] 缓存不可用原因如下（分类输出）:\n";
            for (size_t i = 0; i < cache_diagnostics.size(); ++i) {
                std::cout << "  [" << (i + 1) << "] " << cache_diagnostics[i] << "\n";
            }
        } else {
            std::cout << "[Clustering] 缓存诊断为空：这通常表示当前二进制没有包含完整诊断逻辑，"
                      << "请确认已重新编译并运行的是刚生成的目标。\n";
        }
        std::cout.flush();
        if (!RunBalancedSuperKMeansInMain(project_root,
                                          g_vectors,
                                          g_loaded_total_doc_num,
                                          g_loaded_total_bucket_num_level_1,
                                          g_loaded_vector_dim,
                                          g_loaded_max_doc_per_bucket)) {
            std::cerr << "[Error] 聚类失败\n";
            return -1;
        }
        std::vector<std::string> post_cluster_diagnostics;
        if (!UpdateBucketsAndCentroidsFromClusterResult(project_root,
                                                        g_bucket_doc_offsets,
                                                        g_bucket_doc_ids,
                                                        g_centroids,
                                                        &post_cluster_diagnostics)) {
            std::cerr << "[Error] 聚类后读取 centroids.bin / buckets 失败\n";
            if (!post_cluster_diagnostics.empty()) {
                std::cerr << "[Clustering] 聚类后读取失败原因如下（分类输出）:\n";
                for (size_t i = 0; i < post_cluster_diagnostics.size(); ++i) {
                    std::cerr << "  [" << (i + 1) << "] " << post_cluster_diagnostics[i] << "\n";
                }
            }
            return -1;
        }
    }

    // Load L1→L0 mapping (generated during clustering)
    if (!LoadL1ToL0Map(project_root, g_loaded_total_bucket_num_level_1)) {
        std::cerr << "[Clustering] l1_to_l0_map.bin 不存在或加载失败，将重新聚类\n";
        if (!RunBalancedSuperKMeansInMain(project_root,
                                          g_vectors,
                                          g_loaded_total_doc_num,
                                          g_loaded_total_bucket_num_level_1,
                                          g_loaded_vector_dim,
                                          g_loaded_max_doc_per_bucket)) {
            std::cerr << "[Error] 聚类失败（l1_to_l0_map 重新生成）\n";
            return -1;
        }
        if (!LoadL1ToL0Map(project_root, g_loaded_total_bucket_num_level_1)) {
            std::cerr << "[Error] 聚类后加载 l1_to_l0_map.bin 仍失败\n";
            return -1;
        }
    }

    return 0;
}
