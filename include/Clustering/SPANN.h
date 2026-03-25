#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <array>
#include <limits>
#ifdef _WIN32
#include <windows.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

#include "Clustering.h"

namespace fs = std::filesystem;

static int SPANN_DIM = 64;
static int64_t SPANN_EXPECTED_TOTAL = 10'000'000;
static int SPANN_N_CLUSTERS = 10'000;
static int SPANN_CHUNK_SIZE = 50'000;
static int SPANN_RANDOM_STATE = 42;
static int SPANN_MAX_ITER = 30;

struct SPANNMatrixF32 {
    int64_t rows{0};
    int cols{0};
    std::vector<float> data;

    float* row_ptr(int64_t r) { return data.data() + r * cols; }
    const float* row_ptr(int64_t r) const { return data.data() + r * cols; }
};

static fs::path spann_resolve_project_root_for_outputs() {
    std::error_code ec;
    fs::path cur = fs::current_path(ec);
    if (ec || cur.empty()) return fs::current_path();
    fs::path p = cur;
    while (!p.empty()) {
        bool has_cmake = fs::exists(p / "CMakeLists.txt", ec) && !ec;
        bool has_src = fs::exists(p / "src", ec) && !ec;
        bool has_include = fs::exists(p / "include", ec) && !ec;
        if (has_cmake && has_src && has_include) return p;
        fs::path parent = p.parent_path();
        if (parent == p) break;
        p = parent;
    }
    return cur;
}

static double spann_sq_l2(const float* a, const float* b, int d) {
    double s = 0.0;
    for (int i = 0; i < d; ++i) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        s += diff * diff;
    }
    return s;
}

static SPANNMatrixF32 spann_kmeans_plus_plus_init(const SPANNMatrixF32& X, int k, std::mt19937& rng, int threads) {
    SPANNMatrixF32 centers;
    centers.rows = k;
    centers.cols = X.cols;
    centers.data.resize(static_cast<size_t>(k) * static_cast<size_t>(X.cols));

    std::uniform_int_distribution<int64_t> uni_first(0, X.rows - 1);
    int64_t first = uni_first(rng);
    std::copy(X.row_ptr(first), X.row_ptr(first) + X.cols, centers.row_ptr(0));

    std::vector<double> closest(static_cast<size_t>(X.rows), 0.0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(threads > 0 ? threads : 1)
#endif
    for (int64_t i = 0; i < X.rows; ++i) {
        closest[static_cast<size_t>(i)] = spann_sq_l2(X.row_ptr(i), centers.row_ptr(0), X.cols);
    }

    std::uniform_real_distribution<double> uni01(0.0, 1.0);
    for (int c = 1; c < k; ++c) {
        double total = std::accumulate(closest.begin(), closest.end(), 0.0);
        double r = uni01(rng) * total;

        double cum = 0.0;
        int64_t idx = 0;
        for (; idx < X.rows; ++idx) {
            cum += closest[static_cast<size_t>(idx)];
            if (cum >= r) break;
        }
        if (idx >= X.rows) idx = X.rows - 1;

        std::copy(X.row_ptr(idx), X.row_ptr(idx) + X.cols, centers.row_ptr(c));

#if defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(threads > 0 ? threads : 1)
#endif
        for (int64_t i = 0; i < X.rows; ++i) {
            double d = spann_sq_l2(X.row_ptr(i), centers.row_ptr(c), X.cols);
            if (d < closest[static_cast<size_t>(i)]) closest[static_cast<size_t>(i)] = d;
        }
    }

    return centers;
}

static std::pair<SPANNMatrixF32, std::vector<int>> spann_kmeans_lloyd(const SPANNMatrixF32& X,
                                                                       SPANNMatrixF32 centers,
                                                                       int max_iter,
                                                                       int threads,
                                                                       const std::chrono::steady_clock::time_point& program_start) {
    std::vector<int> labels(static_cast<size_t>(X.rows), -1);

    for (int it = 0; it < max_iter; ++it) {
        SPANNMatrixF32 sums;
        sums.rows = centers.rows;
        sums.cols = centers.cols;
        sums.data.assign(static_cast<size_t>(sums.rows) * static_cast<size_t>(sums.cols), 0.0f);
        std::vector<int64_t> counts(static_cast<size_t>(centers.rows), 0);
        double loss = 0.0;

#if defined(_OPENMP)
#pragma omp parallel num_threads(threads > 0 ? threads : 1)
#endif
        {
            SPANNMatrixF32 sums_local;
            sums_local.rows = centers.rows;
            sums_local.cols = centers.cols;
            sums_local.data.assign(static_cast<size_t>(sums_local.rows) * static_cast<size_t>(sums_local.cols), 0.0f);
            std::vector<int64_t> counts_local(static_cast<size_t>(centers.rows), 0);
            double loss_local = 0.0;

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
            for (int64_t r = 0; r < X.rows; ++r) {
                const float* x = X.row_ptr(r);
                int best = 0;
                double best_d = spann_sq_l2(x, centers.row_ptr(0), X.cols);
                for (int c = 1; c < centers.rows; ++c) {
                    double d = spann_sq_l2(x, centers.row_ptr(c), X.cols);
                    if (d < best_d) {
                        best_d = d;
                        best = c;
                    }
                }
                labels[static_cast<size_t>(r)] = best;
                counts_local[static_cast<size_t>(best)] += 1;
                loss_local += best_d;
                float* row_sum = sums_local.row_ptr(best);
                for (int j = 0; j < X.cols; ++j) row_sum[j] += x[j];
            }

#if defined(_OPENMP)
#pragma omp critical
#endif
            {
                for (int c = 0; c < centers.rows; ++c) {
                    counts[static_cast<size_t>(c)] += counts_local[static_cast<size_t>(c)];
                }
                for (size_t i = 0; i < sums.data.size(); ++i) sums.data[i] += sums_local.data[i];
                loss += loss_local;
            }
        }

        for (int c = 0; c < centers.rows; ++c) {
            if (counts[static_cast<size_t>(c)] <= 0) continue;
            float* dst = centers.row_ptr(c);
            const float* src = sums.row_ptr(c);
            double inv = 1.0 / static_cast<double>(counts[static_cast<size_t>(c)]);
            for (int j = 0; j < centers.cols; ++j) dst[j] = static_cast<float>(src[j] * inv);
        }

        auto elapsed_iter = std::chrono::duration<double>(std::chrono::steady_clock::now() - program_start).count();
        std::cout << "[SPANN] 迭代 " << (it + 1)
                  << ": loss=" << std::scientific << std::setprecision(4) << loss
                  << std::defaultfloat
                  << ", elapsed=" << format_min_sec(elapsed_iter) << "\n";
    }

    return {centers, labels};
}

static void spann_enforce_bucket_size_limit(const SPANNMatrixF32& X,
                                            const SPANNMatrixF32& centers,
                                            int bucket_limit,
                                            std::vector<int>& labels) {
    if (bucket_limit <= 0) return;

    const int k = static_cast<int>(centers.rows);
    const int64_t n = X.rows;

    std::vector<int> counts(static_cast<size_t>(k), 0);
    std::vector<std::vector<std::pair<double, int64_t>>> bucket_items(static_cast<size_t>(k));

    // The distance-to-own-center computation is independent per point and dominates this stage.
    std::vector<std::pair<double, int64_t>> point_dist(static_cast<size_t>(n), {0.0, -1});
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < n; ++i) {
        int b = labels[static_cast<size_t>(i)];
        if (b < 0 || b >= k) continue;
        double d = spann_sq_l2(X.row_ptr(i), centers.row_ptr(b), X.cols);
        point_dist[static_cast<size_t>(i)] = {d, i};
    }

    for (int64_t i = 0; i < n; ++i) {
        int b = labels[static_cast<size_t>(i)];
        if (b < 0 || b >= k) continue;
        counts[static_cast<size_t>(b)]++;
        bucket_items[static_cast<size_t>(b)].push_back(point_dist[static_cast<size_t>(i)]);
    }

#if defined(_OPENMP)
#pragma omp parallel for schedule(dynamic)
#endif
    for (int b = 0; b < k; ++b) {
        auto& items = bucket_items[static_cast<size_t>(b)];
        std::sort(items.begin(), items.end(), [](const auto& a, const auto& c) { return a.first > c.first; });
    }

    std::vector<int> underfull;
    underfull.reserve(static_cast<size_t>(k));
    for (int b = 0; b < k; ++b) {
        if (counts[static_cast<size_t>(b)] < bucket_limit) underfull.push_back(b);
    }

    for (int b = 0; b < k; ++b) {
        auto& items = bucket_items[static_cast<size_t>(b)];
        size_t ptr = 0;
        while (counts[static_cast<size_t>(b)] > bucket_limit && ptr < items.size()) {
            int64_t vid = items[ptr].second;
            ++ptr;

            int best_target = -1;
            double best_d = std::numeric_limits<double>::max();
            for (int ub : underfull) {
                if (counts[static_cast<size_t>(ub)] >= bucket_limit) continue;
                double d = spann_sq_l2(X.row_ptr(vid), centers.row_ptr(ub), X.cols);
                if (d < best_d) {
                    best_d = d;
                    best_target = ub;
                }
            }
            if (best_target < 0) break;

            labels[static_cast<size_t>(vid)] = best_target;
            counts[static_cast<size_t>(b)]--;
            counts[static_cast<size_t>(best_target)]++;
        }
    }
}

static SPANNMatrixF32 spann_recompute_centroids(const SPANNMatrixF32& X,
                                                const std::vector<std::vector<int64_t>>& members,
                                                const SPANNMatrixF32& fallback_centers) {
    SPANNMatrixF32 out = fallback_centers;
    const int k = static_cast<int>(fallback_centers.rows);
    const int d = fallback_centers.cols;

    for (int c = 0; c < k; ++c) {
        const auto& mem = members[static_cast<size_t>(c)];
        if (mem.empty()) continue;

        std::fill(out.row_ptr(c), out.row_ptr(c) + d, 0.0f);
        for (int64_t id : mem) {
            const float* x = X.row_ptr(id);
            float* dst = out.row_ptr(c);
            for (int j = 0; j < d; ++j) dst[j] += x[j];
        }
        float inv = 1.0f / static_cast<float>(mem.size());
        float* dst = out.row_ptr(c);
        for (int j = 0; j < d; ++j) dst[j] *= inv;
    }

    return out;
}

static std::vector<std::vector<int64_t>> spann_closure_expand_members(const SPANNMatrixF32& X,
                                                                       const SPANNMatrixF32& centers,
                                                                       const std::vector<int>& primary_labels) {
    auto closure_begin = std::chrono::steady_clock::now();
    const int k = static_cast<int>(centers.rows);
    std::vector<std::vector<int64_t>> members(static_cast<size_t>(k));

    // 目标桶容量：优先使用运行时 max_doc_per_bucket。
    int64_t target_bucket_size = static_cast<int64_t>(clustering::Clustering::Instance().GetRuntimeParams().max_doc_per_bucket);
    if (target_bucket_size <= 0) {
        target_bucket_size = X.rows / std::max<int64_t>(1, centers.rows);
    }
    if (target_bucket_size <= 0) target_bucket_size = 1;

    // 初始成员构建分为三步：并行计数 -> 预分配 -> 并行写入，避免串行 push_back 热点。
    std::vector<int64_t> bucket_counts(static_cast<size_t>(k), 0);
#if defined(_OPENMP)
    int omp_threads = omp_get_max_threads();
    std::vector<std::vector<int64_t>> local_counts(static_cast<size_t>(omp_threads),
                                                   std::vector<int64_t>(static_cast<size_t>(k), 0));
#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& lc = local_counts[static_cast<size_t>(tid)];
#pragma omp for schedule(static)
        for (int64_t i = 0; i < X.rows; ++i) {
            int primary = primary_labels[static_cast<size_t>(i)];
            if (primary < 0 || primary >= k) primary = 0;
            lc[static_cast<size_t>(primary)] += 1;
        }
    }
    for (int t = 0; t < omp_threads; ++t) {
        const auto& lc = local_counts[static_cast<size_t>(t)];
        for (int b = 0; b < k; ++b) {
            bucket_counts[static_cast<size_t>(b)] += lc[static_cast<size_t>(b)];
        }
    }
#else
    for (int64_t i = 0; i < X.rows; ++i) {
        int primary = primary_labels[static_cast<size_t>(i)];
        if (primary < 0 || primary >= k) primary = 0;
        bucket_counts[static_cast<size_t>(primary)] += 1;
    }
#endif

    for (int b = 0; b < k; ++b) {
        members[static_cast<size_t>(b)].resize(static_cast<size_t>(bucket_counts[static_cast<size_t>(b)]));
    }

    std::vector<int64_t> write_pos(static_cast<size_t>(k), 0);
#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < X.rows; ++i) {
        int primary = primary_labels[static_cast<size_t>(i)];
        if (primary < 0 || primary >= k) primary = 0;

        int64_t pos = 0;
#if defined(_OPENMP)
#pragma omp atomic capture
#endif
        {
            pos = write_pos[static_cast<size_t>(primary)];
            write_pos[static_cast<size_t>(primary)] += 1;
        }
        members[static_cast<size_t>(primary)][static_cast<size_t>(pos)] = i;
    }

    // 若没有任何小桶，直接结束扩容。
    bool has_underfull_bucket = false;
    int underfull_bucket_count = 0;
    for (int c = 0; c < k; ++c) {
        if (bucket_counts[static_cast<size_t>(c)] < target_bucket_size) {
            has_underfull_bucket = true;
            underfull_bucket_count += 1;
        }
    }

    if (!has_underfull_bucket) return members;

    // 硬补齐策略：按全局前缀分配每个桶的 round-robin 起点，可并行补齐且结果可复现。
    std::vector<int64_t> deficit_prefix(static_cast<size_t>(k) + 1, 0);
    for (int b = 0; b < k; ++b) {
        int64_t cur = bucket_counts[static_cast<size_t>(b)];
        int64_t need = (cur < target_bucket_size) ? (target_bucket_size - cur) : 0;
        deficit_prefix[static_cast<size_t>(b + 1)] = deficit_prefix[static_cast<size_t>(b)] + need;
    }
    int64_t replica_assignments = deficit_prefix[static_cast<size_t>(k)];

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (int b = 0; b < k; ++b) {
        int64_t cur = bucket_counts[static_cast<size_t>(b)];
        if (cur >= target_bucket_size) continue;

        int64_t need = target_bucket_size - cur;
        int64_t start = deficit_prefix[static_cast<size_t>(b)];
        auto& bucket = members[static_cast<size_t>(b)];
        bucket.reserve(bucket.size() + static_cast<size_t>(need));
        for (int64_t t = 0; t < need; ++t) {
            int64_t doc_id = (start + t) % X.rows;
            bucket.push_back(doc_id);
        }
        bucket_counts[static_cast<size_t>(b)] = target_bucket_size;
    }
    underfull_bucket_count = 0;

    double closure_total_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - closure_begin).count();
    std::cout << "[SPANN] closure_expand detail: fill_mode=strict_round_robin"
              << ", remaining_underfull=" << underfull_bucket_count
              << ", replica_assignments=" << replica_assignments
              << ", total=" << format_min_sec(closure_total_seconds) << "\n";

    return members;
}

static bool spann_save_centers_default_bin_sorted(const SPANNMatrixF32& centers, const fs::path& out_bin) {
    const int k = static_cast<int>(centers.rows);
    const int d = centers.cols;

    std::vector<int> order(static_cast<size_t>(k));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        const float* A = centers.row_ptr(a);
        const float* B = centers.row_ptr(b);
        for (int j = 0; j < d; ++j) {
            if (A[j] < B[j]) return true;
            if (A[j] > B[j]) return false;
        }
        return a < b;
    });

    std::ofstream ofs(out_bin, std::ios::binary);
    if (!ofs) return false;

    int32_t rows = static_cast<int32_t>(k);
    int32_t cols = static_cast<int32_t>(d);
    ofs.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    ofs.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    for (int idx : order) {
        const float* row = centers.row_ptr(idx);
        ofs.write(reinterpret_cast<const char*>(row), static_cast<std::streamsize>(sizeof(float) * d));
    }
    return static_cast<bool>(ofs);
}

static bool spann_save_bucket_doc_ids_bins(const std::vector<std::vector<int64_t>>& members,
                                           const std::vector<int>& order,
                                           const fs::path& buckets_dir) {
    std::error_code ec;
    fs::remove_all(buckets_dir, ec);
    ec.clear();
    fs::create_directories(buckets_dir, ec);
    if (ec) return false;

    const int k = static_cast<int>(members.size());
    for (int sorted_bucket_id = 0; sorted_bucket_id < k; ++sorted_bucket_id) {
        fs::path out_bin = buckets_dir /
            ("bucket" + std::to_string(sorted_bucket_id) + "_doc_ids.bin");
        std::ofstream ofs(out_bin, std::ios::binary);
        if (!ofs) return false;

        int original_bucket_id = order[static_cast<size_t>(sorted_bucket_id)];
        const auto& mem = members[static_cast<size_t>(original_bucket_id)];

        int64_t n = static_cast<int64_t>(mem.size());
        ofs.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (int64_t doc_id0 : mem) {
            int32_t doc_id1 = static_cast<int32_t>(doc_id0 + 1);
            ofs.write(reinterpret_cast<const char*>(&doc_id1), sizeof(doc_id1));
        }
        if (!ofs) return false;
    }
    return true;
}

int RunSPANN(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::cout.setf(std::ios::unitbuf);

    auto& clustering_ctx = clustering::Clustering::Instance();
    auto start_time = std::chrono::steady_clock::now();

    fs::path output_root = spann_resolve_project_root_for_outputs();
    int posting_limit = 1024;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_val = [&](const char* name) {
            if (i + 1 >= argc) {
                std::cerr << name << " 需要一个值\n";
                std::exit(1);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--output-root") output_root = fs::path(need_val("--output-root"));
        else if (arg == "--dim") SPANN_DIM = std::stoi(need_val("--dim"));
        else if (arg == "--total") SPANN_EXPECTED_TOTAL = std::stoll(need_val("--total"));
        else if (arg == "--clusters") SPANN_N_CLUSTERS = std::stoi(need_val("--clusters"));
        else if (arg == "--chunk") SPANN_CHUNK_SIZE = std::stoi(need_val("--chunk"));
        else if (arg == "--iter") SPANN_MAX_ITER = std::stoi(need_val("--iter"));
        else if (arg == "--seed") SPANN_RANDOM_STATE = std::stoi(need_val("--seed"));
        else if (arg == "--threads") THREADS = std::stoi(need_val("--threads"));
        else if (arg == "--posting-limit") posting_limit = std::stoi(need_val("--posting-limit"));
        else if (arg == "-h" || arg == "--help") {
            std::cout << "用法: SPANN.exe [选项]\n"
                      << "  --output-root <路径> 产物输出目录（默认项目根）\n"
                      << "  --dim <整数>\n"
                      << "  --total <整数>\n"
                      << "  --clusters <整数>\n"
                      << "  --chunk <整数>\n"
                      << "  --iter <整数>\n"
                      << "  --seed <整数>\n"
                      << "  --threads <整数>\n"
                      << "  --posting-limit <整数>     桶规模上限（默认自动=ceil(total/clusters)）\n";
            return 0;
        } else {
            std::cerr << "未知选项: " << arg << "\n";
            return 1;
        }
    }

#if defined(_OPENMP)
    int threads = THREADS > 0 ? THREADS : omp_get_max_threads();
#else
    int threads = 1;
#endif

    SPANNMatrixF32 X;
    clustering::Clustering::MatrixF32 input_vecs;
    if (!clustering_ctx.GetInputVectorsMatrix(input_vecs) ||
        input_vecs.rows <= 0 || input_vecs.cols <= 0 || input_vecs.data.empty()) {
        std::cerr << "[SPANN] 仅支持内存向量输入，但当前未获取到有效内存向量\n";
        return 1;
    }
    X.rows = input_vecs.rows;
    X.cols = input_vecs.cols;
    X.data = std::move(input_vecs.data);
    SPANN_DIM = X.cols;
    SPANN_EXPECTED_TOTAL = X.rows;
    std::cout << "[SPANN] 使用内存向量数据，形状=(" << X.rows << "," << X.cols << ")\n";

    if (X.rows <= 0 || X.cols <= 0 || X.data.empty()) {
        std::cerr << "[SPANN] 输入向量为空\n";
        return 1;
    }

    if (posting_limit <= 0) {
        posting_limit = static_cast<int>((X.rows + SPANN_N_CLUSTERS - 1) / SPANN_N_CLUSTERS);
    }
    std::cout << "[SPANN] 参数: clusters=" << SPANN_N_CLUSTERS
              << ", posting_limit=" << posting_limit
              << ", threads=" << threads << "\n";

    std::mt19937 rng(SPANN_RANDOM_STATE);
    SPANNMatrixF32 centers = spann_kmeans_plus_plus_init(X, SPANN_N_CLUSTERS, rng, threads);
    auto init_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    std::cout << "[SPANN] 初始质心完成，总耗时=" << format_min_sec(init_elapsed) << "\n";
    auto km = spann_kmeans_lloyd(X, centers, SPANN_MAX_ITER, threads, start_time);

    SPANNMatrixF32 primary_centers = std::move(km.first);
    std::vector<int> primary_labels = std::move(km.second);

    auto post_limit_begin = std::chrono::steady_clock::now();
    spann_enforce_bucket_size_limit(X, primary_centers, posting_limit, primary_labels);
    auto post_limit_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - post_limit_begin).count();
    std::cout << "[SPANN] bucket_size_limit elapsed=" << format_min_sec(post_limit_elapsed) << "\n";

    auto closure_begin = std::chrono::steady_clock::now();
    std::vector<std::vector<int64_t>> expanded_members =
        spann_closure_expand_members(X, primary_centers, primary_labels);
    auto closure_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - closure_begin).count();
    std::cout << "[SPANN] closure_expand elapsed=" << format_min_sec(closure_elapsed) << "\n";

    auto recompute_begin = std::chrono::steady_clock::now();
    SPANNMatrixF32 expanded_centers = spann_recompute_centroids(X, expanded_members, primary_centers);
    auto recompute_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - recompute_begin).count();
    std::cout << "[SPANN] recompute_centroids elapsed=" << format_min_sec(recompute_elapsed) << "\n";

    int k = static_cast<int>(expanded_centers.rows);
    int d = expanded_centers.cols;
    std::vector<int> order(static_cast<size_t>(k));
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        const float* A = expanded_centers.row_ptr(a);
        const float* B = expanded_centers.row_ptr(b);
        for (int j = 0; j < d; ++j) {
            if (A[j] < B[j]) return true;
            if (A[j] > B[j]) return false;
        }
        return a < b;
    });
    std::vector<int> inv_order(static_cast<size_t>(k), 0);
    for (int i = 0; i < k; ++i) inv_order[static_cast<size_t>(order[static_cast<size_t>(i)])] = i;
    for (auto& lab : primary_labels) {
        if (lab >= 0 && lab < k) lab = inv_order[static_cast<size_t>(lab)];
        else lab = 0;
    }

    SPANNMatrixF32 sorted_centers;
    sorted_centers.rows = expanded_centers.rows;
    sorted_centers.cols = expanded_centers.cols;
    sorted_centers.data.resize(static_cast<size_t>(sorted_centers.rows) * static_cast<size_t>(sorted_centers.cols));
    for (int i = 0; i < k; ++i) {
        const float* src = expanded_centers.row_ptr(order[static_cast<size_t>(i)]);
        float* dst = sorted_centers.row_ptr(i);
        std::copy(src, src + d, dst);
    }

    clustering::Clustering::MatrixF32 stage0_centers;
    stage0_centers.rows = sorted_centers.rows;
    stage0_centers.cols = sorted_centers.cols;
    stage0_centers.data = sorted_centers.data;
    clustering_ctx.SetStage0(stage0_centers, primary_labels);

    std::error_code ec;
    fs::create_directories(output_root, ec);
    if (ec) {
        std::cerr << "[SPANN] 创建输出目录失败: " << output_root << ", err=" << ec.message() << "\n";
        return 1;
    }

    fs::path centers_bin = output_root / "centroids.bin";
    fs::path buckets_dir = output_root / "buckets";
    if (!spann_save_centers_default_bin_sorted(expanded_centers, centers_bin)) {
        std::cerr << "[SPANN] 写入失败: " << centers_bin << "\n";
        return 1;
    }
    if (!spann_save_bucket_doc_ids_bins(expanded_members, order, buckets_dir)) {
        std::cerr << "[SPANN] 写入失败: " << buckets_dir << "\n";
        return 1;
    }

    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
    std::cout << "[SPANN] 已写入 " << centers_bin << " 与 " << buckets_dir
              << "，总耗时=" << format_min_sec(elapsed) << "\n";

    return 0;
}
