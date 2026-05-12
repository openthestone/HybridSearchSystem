// gen_vec_step1.cpp
// Step 1: Read REAL dataset.bin, run K-means on sampled vectors,
// compute per-cluster diagonal statistics, assign ALL documents to nearest
// centroid, write GSSV 6 split temp files (txt only).
//
// GSSV 6 format captures: cluster centroids, per-cluster diagonal stds,
// cluster member counts, and per-document cluster assignments — enough for
// step 2 to generate vectors with realistic geometric structure while
// preserving bitmap-vector correlation.
//
// All output is text-only, split into sim_temp<NNN>.txt files (~100KB each)
// under the sim_temp/ directory.
//
// Compile: g++ -O3 -march=native -ffast-math -fopenmp gen_vec_step1.cpp -o gen_vec_step1

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

namespace fs = std::filesystem;

namespace {

constexpr float kMinStd = 1e-6f;

struct DatasetHeader {
    uint64_t doc_num = 0;
    uint32_t vec_dim = 0;
    uint32_t tag_num = 0;
    uint32_t bucket_num = 0;  // not in HYDSET2; kept for GSSV temp format compat, always 0
};

struct Step1Args {
    fs::path input_path = "dataset.bin";
    fs::path temp_path  = "vec_temp";
    size_t K = 8192;
    size_t sample_count = 500000;
    uint64_t seed = 20260414ULL;
    bool seed_overridden = false;
    bool normalize_vectors = false;
    size_t split_size = 819200;  // 0 = single file, >0 = split into ~split_size byte files
    size_t chunk_docs = 65536;
    int kmeans_iters = 20;
};

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------
void PrintUsage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "\n"
        << "Step 1: Extract vector distribution statistics via large-K K-means\n"
        << "        and assign all documents to nearest centroid.\n"
        << "\n"
        << "Options:\n"
        << "  --input <path>      Real dataset file (default: dataset.bin)\n"
        << "  --temp  <path>      Output temp dir or file (default: vec_temp)\n"
        << "  --K <n>             Number of K-means clusters (default: 8192)\n"
        << "  --sample-count <n>  Number of vectors to sample (default: 500000)\n"
        << "  --seed <u64>        Random seed for step 2 (default: 20260414)\n"
        << "  --normalize         L2-normalize vectors in generated data\n"
        << "  --split-size <n>    Split temp into files of ~n bytes (default: 819200, 0 = single file)\n"
        << "  --chunk-docs <n>    Docs per I/O chunk for assignment (default: 65536)\n"
        << "  --kmeans-iters <n>  K-means iterations (default: 20)\n"
        << "  -h, --help          Show this help\n";
}

Step1Args ParseArgs(int argc, char** argv) {
    Step1Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string opt = argv[i];
        auto need = [&](const char* name) {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string("missing value for ") + name);
        };
        if (opt == "--input") {
            need("--input"); args.input_path = argv[++i];
        } else if (opt == "--temp") {
            need("--temp"); args.temp_path = argv[++i];
        } else if (opt == "--K") {
            need("--K"); args.K = std::stoull(argv[++i]);
            if (args.K == 0) throw std::runtime_error("--K must be > 0");
        } else if (opt == "--sample-count") {
            need("--sample-count"); args.sample_count = std::stoull(argv[++i]);
            if (args.sample_count == 0) throw std::runtime_error("--sample-count must be > 0");
        } else if (opt == "--seed") {
            need("--seed"); args.seed = std::stoull(argv[++i]);
            args.seed_overridden = true;
        } else if (opt == "--normalize") {
            args.normalize_vectors = true;
        } else if (opt == "--split-size") {
            need("--split-size"); args.split_size = std::stoull(argv[++i]);
        } else if (opt == "--chunk-docs") {
            need("--chunk-docs"); args.chunk_docs = std::stoull(argv[++i]);
            if (args.chunk_docs == 0) throw std::runtime_error("--chunk-docs must be > 0");
        } else if (opt == "--kmeans-iters") {
            need("--kmeans-iters"); args.kmeans_iters = std::stoi(argv[++i]);
            if (args.kmeans_iters <= 0) throw std::runtime_error("--kmeans-iters must be > 0");
        } else if (opt == "-h" || opt == "--help") {
            PrintUsage(argv[0]); std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + opt);
        }
    }
    return args;
}

// ---------------------------------------------------------------------------
// File I/O helpers
// ---------------------------------------------------------------------------
DatasetHeader ReadHeader(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open: " + path.string());

    // HYDSET2 header (40 bytes, natural alignment on 64-bit):
    //   offset  0: magic[8]   "HYDSET2\0"
    //   offset  8: version    uint32_t == 2
    //   offset 12: (pad 4)
    //   offset 16: doc_num    uint64_t
    //   offset 24: vec_dim    uint32_t
    //   offset 28: tag_num    uint32_t
    //   offset 32: reserved   uint32_t
    //   offset 36: (pad 4)
    char magic[8] = {};
    in.read(magic, 8);
    if (std::memcmp(magic, "HYDSET2", 8) != 0)
        throw std::runtime_error("invalid HYDSET2 magic in: " + path.string());

    uint32_t version = 0;
    in.read(reinterpret_cast<char*>(&version), 4);
    if (version != 2)
        throw std::runtime_error("unsupported HYDSET2 version " +
                                 std::to_string(version) + " in: " + path.string());

    in.seekg(16);  // seek to doc_num

    DatasetHeader h;
    h.bucket_num = 0;  // HYDSET2 has no bucket_num field
    in.read(reinterpret_cast<char*>(&h.doc_num), 8);
    in.read(reinterpret_cast<char*>(&h.vec_dim), 4);
    in.read(reinterpret_cast<char*>(&h.tag_num), 4);

    if (!in.good() || h.doc_num == 0 || h.vec_dim == 0 || h.tag_num == 0)
        throw std::runtime_error("invalid header values in: " + path.string());

    return h;
}

// ---------------------------------------------------------------------------
// Sample vectors from the dataset for clustering (seek-based)
// ---------------------------------------------------------------------------
std::vector<float> SampleVectors(const fs::path& path, size_t offset,
                                  size_t doc_num, size_t dim,
                                  size_t target_count) {
    const size_t rate = std::max<size_t>(1, doc_num / target_count);
    const size_t count = (doc_num + rate - 1) / rate;

    std::vector<float> samples(count * dim);

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open: " + path.string());

    const size_t vec_bytes = dim * sizeof(float);
    size_t actual = 0;
    for (size_t i = 0; i < doc_num && actual < count; i += rate) {
        in.clear();
        in.seekg(static_cast<std::streamoff>(offset + i * vec_bytes));
        in.read(reinterpret_cast<char*>(samples.data() + actual * dim),
                static_cast<std::streamsize>(vec_bytes));
        if (!in.good()) break;
        ++actual;
    }
    samples.resize(actual * dim);
    return samples;
}

// ---------------------------------------------------------------------------
// Distance helpers (float accumulation for SIMD auto-vectorization)
// ---------------------------------------------------------------------------
float SquaredDist(const float* a, const float* b, size_t dim) {
    float s = 0.0f;
    for (size_t d = 0; d < dim; ++d) {
        const float v = a[d] - b[d];
        s += v * v;
    }
    return s;
}

// Find nearest center with partial-distance early termination
inline uint32_t FindNearestCenter(const float* vec, const float* centers,
                                   size_t K, size_t dim) {
    float best = std::numeric_limits<float>::max();
    uint32_t best_k = 0;
    for (size_t k = 0; k < K; ++k) {
        const float* ck = centers + k * dim;
        float s = 0.0f;
        bool dominated = false;
        for (size_t d = 0; d < dim; ++d) {
            const float v = vec[d] - ck[d];
            s += v * v;
            if (s >= best) { dominated = true; break; }
        }
        if (!dominated) {
            best = s;
            best_k = static_cast<uint32_t>(k);
        }
    }
    return best_k;
}

// ---------------------------------------------------------------------------
// K-means clustering (K-means++ init, Lloyd iterations)
// Returns K cluster centers [K * dim].
// ---------------------------------------------------------------------------
std::vector<float> KMeansCluster(const std::vector<float>& points,
                                  size_t n, size_t dim,
                                  size_t K, int max_iters) {
    std::vector<float> centers(K * dim, 0.0f);
    std::vector<uint32_t> assignment(n, 0);
    std::mt19937_64 rng(42);  // fixed seed for reproducibility

    // K-means++ initialization
    {
        const size_t first = static_cast<size_t>(rng() % n);
        std::copy(points.begin() + first * dim,
                  points.begin() + (first + 1) * dim,
                  centers.begin());

        std::vector<double> dists(n, 1e18);
        for (size_t k = 1; k < K; ++k) {
            // Update min distances to existing centers
            const float* prev = centers.data() + (k - 1) * dim;
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                const float d = SquaredDist(points.data() + i * dim, prev, dim);
                dists[i] = std::min(dists[i], static_cast<double>(d));
            }
            // Weighted selection
            double total = 0.0;
            for (double d : dists) total += d;
            if (total <= 0.0) {
                // All points identical; just pick sequentially
                std::copy(points.begin() + (k % n) * dim,
                          points.begin() + (k % n + 1) * dim,
                          centers.data() + k * dim);
                continue;
            }
            std::uniform_real_distribution<double> u(0.0, total);
            double r = u(rng);
            double cum = 0.0;
            size_t sel = 0;
            for (size_t i = 0; i < n; ++i) {
                cum += dists[i];
                if (cum >= r) { sel = i; break; }
            }
            std::copy(points.begin() + sel * dim,
                      points.begin() + (sel + 1) * dim,
                      centers.data() + k * dim);
        }
    }

    // Lloyd iterations
    for (int iter = 0; iter < max_iters; ++iter) {
        // Assign each point to nearest center
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            assignment[i] = FindNearestCenter(points.data() + i * dim,
                                               centers.data(), K, dim);
        }

        // Recompute centers
        std::vector<double> new_centers(K * dim, 0.0);
        std::vector<size_t> counts(K, 0);
        for (size_t i = 0; i < n; ++i) {
            const uint32_t k = assignment[i];
            const float* pi = points.data() + i * dim;
            for (size_t d = 0; d < dim; ++d) {
                new_centers[k * dim + d] += static_cast<double>(pi[d]);
            }
            counts[k]++;
        }
        for (size_t k = 0; k < K; ++k) {
            if (counts[k] > 0) {
                for (size_t d = 0; d < dim; ++d) {
                    centers[k * dim + d] = static_cast<float>(
                        new_centers[k * dim + d] / static_cast<double>(counts[k]));
                }
            }
        }
    }
    return centers;
}

// ---------------------------------------------------------------------------
// Per-cluster diagonal statistics from sampled vectors
// ---------------------------------------------------------------------------
struct ClusterStats {
    std::vector<float> centroids;  // [K * dim]
    std::vector<float> stds;       // [K * dim]  (diagonal std per cluster per dim)
    std::vector<uint64_t> counts;  // [K]        (member count per cluster)
};

ClusterStats ComputeClusterStats(const std::vector<float>& points,
                                  size_t n, size_t dim,
                                  const std::vector<float>& centers,
                                  size_t K) {
    ClusterStats cs;
    cs.centroids.resize(K * dim, 0.0f);
    cs.stds.resize(K * dim, 0.0f);
    cs.counts.assign(K, 0);

    // Thread-local accumulators
    const int nthreads = omp_get_max_threads();
    std::vector<std::vector<double>> tl_sum(nthreads, std::vector<double>(K * dim, 0.0));
    std::vector<std::vector<double>> tl_sum_sq(nthreads, std::vector<double>(K * dim, 0.0));
    std::vector<std::vector<uint64_t>> tl_counts(nthreads, std::vector<uint64_t>(K, 0));

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const int tid = omp_get_thread_num();
        const float* pi = points.data() + i * dim;

        const uint32_t best_k = FindNearestCenter(pi, centers.data(), K, dim);

        tl_counts[tid][best_k]++;
        for (size_t d = 0; d < dim; ++d) {
            const double v = static_cast<double>(pi[d]);
            tl_sum[tid][best_k * dim + d] += v;
            tl_sum_sq[tid][best_k * dim + d] += v * v;
        }
    }

    // Merge thread-local
    std::vector<double> sum(K * dim, 0.0);
    std::vector<double> sum_sq(K * dim, 0.0);
    for (int t = 0; t < nthreads; ++t) {
        for (size_t k = 0; k < K; ++k)
            cs.counts[k] += tl_counts[t][k];
        for (size_t j = 0; j < K * dim; ++j) {
            sum[j] += tl_sum[t][j];
            sum_sq[j] += tl_sum_sq[t][j];
        }
    }

    // Compute per-cluster centroids and stds
    // First find median std of non-empty clusters for filling empty ones
    std::vector<size_t> non_empty;
    for (size_t k = 0; k < K; ++k)
        if (cs.counts[k] >= 2) non_empty.push_back(k);

    // Compute median of average std across dimensions for non-empty clusters
    double median_std = kMinStd;
    if (!non_empty.empty()) {
        std::vector<double> avg_stds(non_empty.size());
        for (size_t i = 0; i < non_empty.size(); ++i) {
            size_t k = non_empty[i];
            double cnt = static_cast<double>(cs.counts[k]);
            double total = 0.0;
            for (size_t d = 0; d < dim; ++d) {
                double mean_d = sum[k * dim + d] / cnt;
                double var = sum_sq[k * dim + d] / cnt - mean_d * mean_d;
                if (var < 0.0) var = 0.0;
                total += std::sqrt(var);
            }
            avg_stds[i] = total / static_cast<double>(dim);
        }
        std::nth_element(avg_stds.begin(),
                         avg_stds.begin() + avg_stds.size() / 2,
                         avg_stds.end());
        median_std = avg_stds[avg_stds.size() / 2];
        if (median_std < kMinStd) median_std = kMinStd;
    }

    #pragma omp parallel for schedule(static)
    for (size_t k = 0; k < K; ++k) {
        if (cs.counts[k] < 2) {
            // Empty or near-empty cluster: use K-means center, median std, count=1
            std::copy(centers.data() + k * dim,
                      centers.data() + (k + 1) * dim,
                      cs.centroids.data() + k * dim);
            for (size_t d = 0; d < dim; ++d)
                cs.stds[k * dim + d] = static_cast<float>(median_std);
            cs.counts[k] = 1;
            continue;
        }

        const double cnt = static_cast<double>(cs.counts[k]);
        for (size_t d = 0; d < dim; ++d) {
            const double mean_d = sum[k * dim + d] / cnt;
            cs.centroids[k * dim + d] = static_cast<float>(mean_d);

            double var = sum_sq[k * dim + d] / cnt - mean_d * mean_d;
            if (var < 0.0) var = 0.0;
            float s = static_cast<float>(std::sqrt(var));
            cs.stds[k * dim + d] = (s < kMinStd) ? kMinStd : s;
        }
    }

    return cs;
}

// ---------------------------------------------------------------------------
// Assign ALL vectors in the dataset to nearest centroid
// Returns vector of uint32 cluster assignments [doc_num]
// ---------------------------------------------------------------------------
std::vector<uint32_t> AssignAllVectors(const fs::path& input_path,
                                        const DatasetHeader& h,
                                        const std::vector<float>& centers,
                                        size_t K,
                                        size_t chunk_docs) {
    const size_t doc_num = static_cast<size_t>(h.doc_num);
    const size_t dim = static_cast<size_t>(h.vec_dim);
    const size_t vec_bytes = dim * sizeof(float);

    std::ifstream in(input_path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot open dataset for assignment: " + input_path.string());

    // Skip HYDSET2 header (40 bytes)
    in.seekg(40);

    std::vector<uint32_t> assignments(doc_num);
    std::vector<float> chunk(chunk_docs * dim);

    std::cout << "[Info] Assigning all " << doc_num << " documents to "
              << K << " centroids (chunk=" << chunk_docs << ")...\n";

    for (size_t base = 0; base < doc_num; base += chunk_docs) {
        const size_t n = std::min(chunk_docs, doc_num - base);

        // Read chunk of vectors
        in.read(reinterpret_cast<char*>(chunk.data()),
                static_cast<std::streamsize>(n * vec_bytes));
        if (!in.good())
            throw std::runtime_error("read error at doc " + std::to_string(base));

        // Assign each vector to nearest centroid
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            assignments[base + i] = FindNearestCenter(chunk.data() + i * dim,
                                                       centers.data(), K, dim);
        }

        const size_t chunk_idx = base / chunk_docs;
        if (chunk_idx % 10 == 0 || base + n >= doc_num)
            std::cout << "[Progress] assignments: " << (base + n) << " / " << doc_num << "\n";
    }

    // Print assignment distribution
    std::vector<uint64_t> counts(K, 0);
    for (auto id : assignments) counts[id]++;
    uint64_t min_c = counts[0], max_c = counts[0];
    size_t empty = 0;
    for (size_t k = 0; k < K; ++k) {
        min_c = std::min(min_c, counts[k]);
        max_c = std::max(max_c, counts[k]);
        if (counts[k] == 0) ++empty;
    }
    std::cout << "[Info] Assignment distribution: min=" << min_c
              << " max=" << max_c << " empty_clusters=" << empty << "\n";

    return assignments;
}

// ---------------------------------------------------------------------------
// Write GSSV 6 split temp files
//
// Format ("GSSV 6"):
//   GSSV 6
//   <doc_num> <vec_dim> <tag_num> <bucket_num>
//   <K> <normalize:0|1> <seed:uint64>
//   CENTROIDS
//     K rows, each dim floats
//   STDS
//     K rows, each dim floats
//   COUNTS
//     1 row, K uint64 values
//   ASSIGNMENTS
//     doc_num values encoded as 2-char printable ASCII pairs
//     (chars '!'..'~', base-94: value = (c1-33)*94 + (c2-33), max 8835)
//     32 values per line (64 chars + newline), no separators
//
// Output: sim_temp<NNN>.txt files (~split_size bytes each) in the temp dir.
// ---------------------------------------------------------------------------

// Encode uint32 [0, 8835] to 2 printable ASCII chars ('!'..'~')
inline void EncodePair(uint32_t value, char out[2]) {
    if (value >= 94u * 94u)
        throw std::runtime_error("EncodePair: value " + std::to_string(value)
                                 + " exceeds base-94 range (max 8835)");
    out[0] = static_cast<char>(value / 94 + 33);
    out[1] = static_cast<char>(value % 94 + 33);
}

template <typename T>
void WriteValues(std::ostream& out, const T* data, size_t count, size_t per_line = 8) {
    for (size_t i = 0; i < count; ++i) {
        if (i > 0) {
            if (i % per_line == 0) out << "\n";
            else out << " ";
        }
        out << data[i];
    }
    if (count > 0) out << "\n";
}

void WriteSplitTempFiles(const fs::path& dir,
                          const DatasetHeader& h, size_t K,
                          const ClusterStats& cs,
                          const std::vector<uint32_t>& assignments,
                          uint64_t seed, bool normalize_vectors,
                          size_t split_size) {
    const size_t dim = static_cast<size_t>(h.vec_dim);
    const size_t doc_num = static_cast<size_t>(h.doc_num);

    fs::create_directories(dir);

    // Build content in parts to avoid holding one giant string for assignments
    // Part 1: header + centroids + stds + counts → string stream
    std::ostringstream oss;
    std::ostream& out = oss;

    // Header
    out << "GSSV 6\n";
    out << h.doc_num << " " << h.vec_dim << " "
        << h.tag_num << " " << h.bucket_num << "\n";
    out << K << " " << (normalize_vectors ? 1 : 0) << " " << seed << "\n";

    // Centroids: one cluster per line
    out << "CENTROIDS\n";
    out << std::scientific << std::setprecision(8);
    for (size_t c = 0; c < K; ++c)
        WriteValues(out, cs.centroids.data() + c * dim, dim, dim);

    // Stds: one cluster per line
    out << "STDS\n";
    for (size_t c = 0; c < K; ++c)
        WriteValues(out, cs.stds.data() + c * dim, dim, dim);

    // Counts: single row
    out << "COUNTS\n";
    out << std::fixed << std::setprecision(0);
    WriteValues(out, cs.counts.data(), K, 16);

    // ASSIGNMENTS header
    out << "ASSIGNMENTS\n";

    std::string header_part = oss.str();

    // Now write everything to split files
    // Strategy: write header_part first, then assignments values
    // Use a buffered writer that splits at newline boundaries

    size_t file_num = 1;
    size_t bytes_in_current = 0;

    auto open_next_file = [&]() -> std::ofstream {
        char fname[32];
        std::snprintf(fname, sizeof(fname), "sim_temp%03d.txt", static_cast<int>(file_num));
        std::ofstream fout(dir / fname, std::ios::binary | std::ios::trunc);
        if (!fout)
            throw std::runtime_error("cannot create: " + (dir / fname).string());
        return fout;
    };

    auto write_chunk = [&](const char* data, size_t len, std::ofstream& fout) {
        fout.write(data, static_cast<std::streamsize>(len));
        bytes_in_current += len;
    };

    // Write header_part into split files
    {
        std::ofstream fout = open_next_file();
        size_t pos = 0;
        while (pos < header_part.size()) {
            size_t remaining = header_part.size() - pos;
            size_t capacity = (split_size > bytes_in_current) ? (split_size - bytes_in_current) : 0;

            if (remaining <= capacity) {
                // Fits in current file
                write_chunk(header_part.data() + pos, remaining, fout);
                pos += remaining;
            } else {
                // Find last newline before capacity boundary
                size_t target = pos + capacity;
                size_t nl = header_part.rfind('\n', target);
                if (nl != std::string::npos && nl > pos) {
                    size_t end = nl + 1;
                    write_chunk(header_part.data() + pos, end - pos, fout);
                    pos = end;
                } else {
                    // No newline found, just write up to capacity
                    write_chunk(header_part.data() + pos, capacity, fout);
                    pos += capacity;
                }
                // Close current file and open next
                fout.close();
                ++file_num;
                bytes_in_current = 0;
                fout = open_next_file();
            }
        }
        // fout stays open for assignments
        // Write assignments into remaining files

        // Part 2: write assignments, 2-char encoded pairs, 32 per line
        constexpr size_t pairs_per_line = 32;  // 32 × 2 = 64 chars + newline
        std::string line_buf;
        line_buf.reserve(pairs_per_line * 2 + 1);

        for (size_t i = 0; i < doc_num; ++i) {
            char pair[2];
            EncodePair(assignments[i], pair);
            line_buf.append(pair, 2);

            if ((i + 1) % pairs_per_line == 0 || i + 1 == doc_num) {
                line_buf += '\n';
                if (bytes_in_current + line_buf.size() > split_size) {
                    fout.close();
                    ++file_num;
                    bytes_in_current = 0;
                    fout = open_next_file();
                }
                write_chunk(line_buf.data(), line_buf.size(), fout);
                line_buf.clear();
            }
        }
        fout.close();
    }

    // Better total estimate
    std::cout << "[Info] Split temp files: " << dir
              << " (" << file_num << " files, ~"
              << static_cast<double>(header_part.size() + doc_num * 2) / (1024.0 * 1024.0)
              << " MB total)\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Step1Args args = ParseArgs(argc, argv);
        const DatasetHeader header = ReadHeader(args.input_path);

        const size_t doc_num    = static_cast<size_t>(header.doc_num);
        const size_t dim        = static_cast<size_t>(header.vec_dim);

        std::cout << "[Info] Input: docs=" << header.doc_num
                  << " dim=" << header.vec_dim
                  << " tags=" << header.tag_num
                  << " buckets=" << header.bucket_num << "\n";

        // File section offset for vectors (HYDSET2 header = 40 bytes)
        const size_t v_off = 40;

        size_t K = args.K;

        // Step 1: Sample vectors for clustering
        std::cout << "[Step 1/4] Sampling " << args.sample_count
                  << " vectors for clustering...\n";
        auto samples = SampleVectors(args.input_path, v_off, doc_num, dim,
                                     args.sample_count);
        const size_t actual_samples = samples.size() / dim;

        if (K > actual_samples) {
            K = actual_samples;
            std::cout << "[Warn] K capped to sample_count=" << actual_samples << "\n";
        }
        std::cout << "[Info] K=" << K << " actual_samples=" << actual_samples << "\n";

        // Step 2: K-means on sampled vectors
        std::cout << "[Step 2/4] K-means clustering (" << K << " clusters, "
                  << args.kmeans_iters << " iters)...\n";
        auto centers = KMeansCluster(samples, actual_samples, dim,
                                     K, args.kmeans_iters);

        // Step 2b: Compute per-cluster statistics from same samples
        std::cout << "[Step 2b/4] Computing per-cluster statistics...\n";
        auto cs = ComputeClusterStats(samples, actual_samples, dim, centers, K);

        // Print cluster size distribution
        {
            uint64_t min_c = cs.counts[0], max_c = cs.counts[0];
            size_t empty = 0;
            for (size_t k = 0; k < K; ++k) {
                if (cs.counts[k] == 1) ++empty;
                min_c = std::min(min_c, cs.counts[k]);
                max_c = std::max(max_c, cs.counts[k]);
            }
            std::cout << "[Info] Cluster sizes (from samples): min=" << min_c
                      << " max=" << max_c
                      << " empty/near-empty=" << empty << "\n";
        }

        // Free sample memory early
        samples.clear();
        samples.shrink_to_fit();

        // Step 3: Assign ALL vectors to nearest centroid
        std::cout << "[Step 3/4] Assigning all " << doc_num << " documents...\n";
        auto assignments = AssignAllVectors(args.input_path, header, centers, K,
                                            args.chunk_docs);

        // Free centers (stats has its own copy)
        centers.clear();
        centers.shrink_to_fit();

        // Step 4: Write GSSV 6 split temp files
        std::cout << "[Step 4/4] Writing split temp files (GSSV 6, ~100KB each)...\n";
        WriteSplitTempFiles(args.temp_path, header, K, cs, assignments,
                            args.seed, args.normalize_vectors, args.split_size);

        std::cout << "[Done] Temp files written to: " << args.temp_path << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << "\n";
        return 1;
    }
}
