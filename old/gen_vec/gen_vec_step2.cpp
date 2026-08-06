// gen_vec_step2.cpp
// Step 2: Read GSSV 5/6 split temp files from step 1, generate simulated dataset
// with assignment-based diagonal Gaussian vectors and real bitmaps copied from
// the original dataset.
//
// Compile: g++ -O3 -march=native -ffast-math -fopenmp gen_vec_step2.cpp -o gen_vec_step2

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <omp.h>

#include "include/nlohmann/json.hpp"

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

constexpr float kMinStd = 1e-6f;

struct DatasetHeader {
    uint64_t doc_num = 0;
    uint32_t vec_dim = 0;
    uint32_t tag_num = 0;
    uint32_t bucket_num = 0;  // not in HYDSET2; kept for GSSV temp format compat
};

struct Step2Args {
    fs::path temp_path = "vec_temp";
    fs::path output_path = "dataset_new.bin";
    fs::path real_bitmaps_path = "dataset.bin";  // required: real dataset to copy bitmaps from
    size_t chunk_docs = 65536;
};

void PrintUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "\n"
              << "Step 2: Generate simulated dataset from GSSV 5/6 temp file.\n"
              << "        Vectors: diagonal Gaussian per cluster (assignment-based).\n"
              << "        Bitmaps: copied from real dataset via --real-bitmaps.\n"
              << "\n"
              << "Options:\n"
              << "  --temp <path>           Temp dir from step 1 (default: vec_temp)\n"
              << "  --output <path>         Output dataset file (default: dataset_new.bin)\n"
              << "  --real-bitmaps <path>   (required) Real dataset to copy bitmaps from\n"
              << "  --chunk-docs <n>        Docs per I/O chunk (default: 65536)\n"
              << "  -h, --help              Show this help\n";
}

Step2Args ParseArgs(int argc, char** argv) {
    Step2Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string opt = argv[i];
        auto need_value = [&](const char* name) {
            if (i + 1 >= argc)
                throw std::runtime_error(std::string("missing value for ") + name);
        };
        if (opt == "--temp") {
            need_value("--temp");
            args.temp_path = argv[++i];
        } else if (opt == "--output") {
            need_value("--output");
            args.output_path = argv[++i];
        } else if (opt == "--real-bitmaps") {
            need_value("--real-bitmaps");
            args.real_bitmaps_path = argv[++i];
        } else if (opt == "--chunk-docs") {
            need_value("--chunk-docs");
            const long long v = std::stoll(argv[++i]);
            if (v <= 0)
                throw std::runtime_error("--chunk-docs must be > 0");
            args.chunk_docs = static_cast<size_t>(v);
        } else if (opt == "-h" || opt == "--help") {
            PrintUsage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + opt);
        }
    }
    return args;
}

// ---------------------------------------------------------------------------
// Read GSSV 5/6 temp file
// ---------------------------------------------------------------------------
struct TempData {
    DatasetHeader header;
    uint64_t seed = 0;
    bool normalize_vectors = false;
    size_t K = 0;

    std::vector<float> centroids;               // [K * dim]
    std::vector<float> stds;                    // [K * dim]
    std::vector<uint64_t> counts;               // [K]
    std::vector<uint32_t> cluster_assignments;  // [doc_num]
};

void ExpectSection(std::istream& in, const std::string& expected) {
    std::string token;
    in >> token;
    if (token != expected)
        throw std::runtime_error("expected section '" + expected + "', got '" + token + "'");
}

// Decode 2 printable ASCII chars ('!'..'~') to uint32
inline uint32_t DecodePair(char c1, char c2) {
    return static_cast<uint32_t>(c1 - 33) * 94 + static_cast<uint32_t>(c2 - 33);
}

template <typename T>
void ReadValues(std::istream& in, std::vector<T>& v, size_t count) {
    v.resize(count);
    for (size_t i = 0; i < count; ++i)
        in >> v[i];
}

TempData ReadTempFile(const fs::path& path) {
    TempData d;
    std::string content;

    if (fs::is_directory(path)) {
        // Split file mode: concatenate numbered files from directory
        std::vector<std::pair<int, fs::path>> files;
        for (const auto& entry : fs::directory_iterator(path)) {
            if (!entry.is_regular_file())
                continue;
            std::string name = entry.path().filename().string();
            if (name.size() > 12 && name.compare(0, 8, "sim_temp") == 0 &&
                name.compare(name.size() - 4, 4, ".txt") == 0) {
                std::string num_str = name.substr(8, name.size() - 12);
                if (!num_str.empty() &&
                    std::all_of(num_str.begin(), num_str.end(), [](unsigned char c) { return std::isdigit(c); })) {
                    files.push_back({std::stoi(num_str), entry.path()});
                }
            }
        }
        if (files.empty())
            throw std::runtime_error("no sim_temp*.txt files found in: " + path.string());
        std::sort(files.begin(), files.end());
        for (auto& [num, fpath] : files) {
            std::ifstream fin(fpath, std::ios::binary);
            if (!fin)
                throw std::runtime_error("cannot open: " + fpath.string());
            content.append(std::istreambuf_iterator<char>(fin), {});
        }
        std::cout << "[Info] Read " << files.size() << " split files from: " << path << " ("
                  << static_cast<double>(content.size()) / (1024.0 * 1024.0) << " MB total)\n";
    } else {
        std::ifstream fin(path, std::ios::binary);
        if (!fin)
            throw std::runtime_error("cannot open temp file: " + path.string());
        content.assign(std::istreambuf_iterator<char>(fin), {});
    }

    std::istringstream in(content);

    // Header
    std::string magic;
    int version = 0;
    in >> magic >> version;
    if (magic != "GSSV")
        throw std::runtime_error("invalid temp file magic: " + magic);
    if (version < 2 || version > 6)
        throw std::runtime_error("unsupported temp file version: " + std::to_string(version) + " (expected 2-6)");

    in >> d.header.doc_num >> d.header.vec_dim >> d.header.tag_num >> d.header.bucket_num;

    int32_t K32 = 0, norm32 = 0;
    if (version == 4) {
        // v4 had G parameter
        int32_t G32 = 0;
        in >> K32 >> norm32 >> d.seed >> G32;
    } else if (version == 5 || version == 6) {
        // v5/v6: no G parameter
        in >> K32 >> norm32 >> d.seed;
    } else {
        // v2/v3
        in >> K32 >> norm32 >> d.seed;
    }

    if (d.header.doc_num == 0 || d.header.vec_dim == 0 || d.header.tag_num == 0)
        throw std::runtime_error("invalid header values in temp file");

    d.K = static_cast<size_t>(K32);
    d.normalize_vectors = (norm32 != 0);

    const size_t K = d.K;
    const size_t dim = static_cast<size_t>(d.header.vec_dim);

    // Centroids
    ExpectSection(in, "CENTROIDS");
    ReadValues(in, d.centroids, K * dim);

    // Stds
    ExpectSection(in, "STDS");
    ReadValues(in, d.stds, K * dim);
    // Clamp stds
    for (auto& s : d.stds)
        s = std::max(s, kMinStd);

    // Counts
    ExpectSection(in, "COUNTS");
    ReadValues(in, d.counts, K);

    // Assignments (GSSV 5/6)
    if (version == 5 || version == 6) {
        ExpectSection(in, "ASSIGNMENTS");
        const size_t doc_num = static_cast<size_t>(d.header.doc_num);
        {
            const auto spos = in.tellg();
            const size_t start = static_cast<size_t>(spos);
            const char* ptr = content.data() + start;
            const char* end = content.data() + content.size();
            d.cluster_assignments.resize(doc_num);

            if (version == 6) {
                // GSSV 6: 2-char printable ASCII pairs, skip whitespace/newlines
                for (size_t i = 0; i < doc_num; ++i) {
                    while (ptr < end && (*ptr == '\n' || *ptr == '\r'))
                        ++ptr;
                    if (ptr + 1 >= end)
                        throw std::runtime_error("unexpected end of assignments at index " + std::to_string(i));
                    d.cluster_assignments[i] = DecodePair(ptr[0], ptr[1]);
                    ptr += 2;
                }
            } else {
                // GSSV 5: decimal text, fast parse with strtoul
                for (size_t i = 0; i < doc_num; ++i) {
                    while (ptr < end && (*ptr == ' ' || *ptr == '\n' || *ptr == '\r' || *ptr == '\t'))
                        ++ptr;
                    if (ptr >= end)
                        throw std::runtime_error("unexpected end of assignments at index " + std::to_string(i));
                    char* next = nullptr;
                    const unsigned long val = strtoul(ptr, &next, 10);
                    if (next == ptr)
                        throw std::runtime_error("invalid assignment value at index " + std::to_string(i));
                    d.cluster_assignments[i] = static_cast<uint32_t>(val);
                    ptr = next;
                }
            }
        }

        // Verify range
        uint32_t max_id = 0;
        for (auto id : d.cluster_assignments)
            max_id = std::max(max_id, id);
        if (static_cast<size_t>(max_id) >= K)
            throw std::runtime_error("ASSIGNMENTS contains cluster ID " + std::to_string(max_id) +
                                     " >= K=" + std::to_string(K));
        std::cout << "[Info] Loaded " << doc_num << " cluster assignments (K=" << K << ", max_id=" << max_id << ")\n";
    }

    // v2/v3 had CORRELATION — skip for backward compat
    if (version == 3) {
        ExpectSection(in, "CORRELATION");
        std::vector<float> corr;
        ReadValues(in, corr, dim * dim);
        std::cout << "[Info] GSSV 3 correlation matrix (ignored)\n";
    }

    if (in.fail())
        throw std::runtime_error("truncated or corrupt temp file: " + path.string());

    return d;
}

// ---------------------------------------------------------------------------
// Fast deterministic RNG (SplitMix64 variant)
// ---------------------------------------------------------------------------
static inline uint64_t DeriveDocSeed(uint64_t master, uint64_t purpose_xor, size_t doc_idx) {
    return master ^ purpose_xor ^ (static_cast<uint64_t>(doc_idx) * 0x9e3779b97f4a7c15ULL);
}

class FastRng {
   public:
    uint64_t state_;
    using result_type = uint64_t;
    static constexpr result_type min() {
        return 0;
    }
    static constexpr result_type max() {
        return UINT64_MAX;
    }
    explicit FastRng(uint64_t seed) : state_(seed) {}
    result_type operator()() {
        uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
};

// ---------------------------------------------------------------------------
// Generation functions
// ---------------------------------------------------------------------------
void WriteHeader(std::ofstream& out, const DatasetHeader& h) {
    // HYDSET2 cache header — must match DatasetCacheHeaderDisk in DataReader.h.
    // Layout (40 bytes, little-endian, natural alignment on 64-bit):
    //   offset  0: magic[8]      "HYDSET2\0"
    //   offset  8: version(u32)  2
    //   offset 12: (pad 4)
    //   offset 16: doc_num(u64)
    //   offset 24: vec_dim(u32)
    //   offset 28: tag_num(u32)
    //   offset 32: reserved(u32) 0
    //   offset 36: (pad 4)
    char buf[40];
    std::memset(buf, 0, sizeof(buf));
    std::memcpy(buf + 0, "HYDSET2", 8);
    const uint32_t version = 2;
    const uint64_t doc_num = static_cast<uint64_t>(h.doc_num);
    const uint32_t vec_dim = static_cast<uint32_t>(h.vec_dim);
    const uint32_t tag_num = static_cast<uint32_t>(h.tag_num);
    std::memcpy(buf + 8, &version, 4);
    std::memcpy(buf + 16, &doc_num, 8);
    std::memcpy(buf + 24, &vec_dim, 4);
    std::memcpy(buf + 28, &tag_num, 4);
    out.write(buf, sizeof(buf));
    if (!out.good())
        throw std::runtime_error("failed to write dataset header");
}

// Generate diagonal Gaussian vectors using cluster assignments
void WriteVectors(std::ofstream& out, const DatasetHeader& h, const TempData& temp, size_t chunk_docs,
                  uint64_t master_seed) {
    const size_t doc_num = static_cast<size_t>(h.doc_num);
    const size_t dim = static_cast<size_t>(h.vec_dim);
    const bool normalize = temp.normalize_vectors;

    std::vector<float> buf(chunk_docs * dim);

    for (size_t base = 0; base < doc_num; base += chunk_docs) {
        const size_t n = std::min(chunk_docs, doc_num - base);

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            const size_t doc_idx = base + i;
            FastRng rng(DeriveDocSeed(master_seed, 0xbf58476d1ce4e5b9ULL, doc_idx));

            const uint32_t c = temp.cluster_assignments[doc_idx];
            const float* mu = temp.centroids.data() + c * dim;
            const float* sigma = temp.stds.data() + c * dim;
            float* dst = buf.data() + i * dim;

            // Generate dim N(0,1) using Box-Muller and apply diagonal Gaussian
            for (size_t d = 0; d < dim; d += 2) {
                uint64_t u1_raw = rng();
                uint64_t u2_raw = rng();
                double u1 = (u1_raw >> 11) * (1.0 / 9007199254740992.0);
                double u2 = (u2_raw >> 11) * (1.0 / 9007199254740992.0);
                u1 = std::max(u1, 1e-15);
                const double r = std::sqrt(-2.0 * std::log(u1));
                const double theta = 2.0 * 3.14159265358979323846 * u2;
                dst[d] = mu[d] + sigma[d] * static_cast<float>(r * std::cos(theta));
                if (d + 1 < dim)
                    dst[d + 1] = mu[d + 1] + sigma[d + 1] * static_cast<float>(r * std::sin(theta));
            }

            if (normalize) {
                float nsq = 0.0f;
                for (size_t d = 0; d < dim; ++d)
                    nsq += dst[d] * dst[d];
                if (nsq > 0.0f) {
                    const float inv = 1.0f / std::sqrt(nsq);
                    for (size_t d = 0; d < dim; ++d)
                        dst[d] *= inv;
                }
            }
        }

        out.write(reinterpret_cast<const char*>(buf.data()), static_cast<std::streamsize>(n * dim * sizeof(float)));
        if (!out.good())
            throw std::runtime_error("write error: vectors");
        if ((base / chunk_docs) % 64 == 0)
            std::cout << "[Progress] vectors: " << (base + n) << " / " << doc_num << "\n";
    }
}

// ---------------------------------------------------------------------------
// Copy bitmaps directly from a real dataset file
// ---------------------------------------------------------------------------
void WriteBitmapsFromRealDataset(std::ofstream& out, const DatasetHeader& h, const fs::path& real_path,
                                 size_t chunk_docs) {
    const size_t doc_num = static_cast<size_t>(h.doc_num);
    const size_t dim = static_cast<size_t>(h.vec_dim);
    const size_t tag_num = static_cast<size_t>(h.tag_num);
    const size_t stride = (tag_num + 63) / 64;

    std::ifstream real_in(real_path, std::ios::binary);
    if (!real_in)
        throw std::runtime_error("cannot open real dataset for bitmaps: " + real_path.string());

    // Read HYDSET2 header (40 bytes)
    char magic[8] = {};
    real_in.read(magic, 8);
    if (std::memcmp(magic, "HYDSET2", 8) != 0)
        throw std::runtime_error("invalid HYDSET2 magic in bitmap source: " + real_path.string());

    uint32_t version = 0;
    real_in.read(reinterpret_cast<char*>(&version), 4);
    if (version != 2)
        throw std::runtime_error("unsupported HYDSET2 version in bitmap source: " + real_path.string());

    real_in.seekg(16);  // seek to doc_num at offset 16
    uint64_t src_doc_num = 0;
    uint32_t src_dim = 0, src_tag_num = 0;
    real_in.read(reinterpret_cast<char*>(&src_doc_num), 8);
    real_in.read(reinterpret_cast<char*>(&src_dim), 4);
    real_in.read(reinterpret_cast<char*>(&src_tag_num), 4);

    if (src_dim != static_cast<uint32_t>(dim))
        throw std::runtime_error("real dataset dim mismatch for bitmap copy");
    if (src_tag_num != static_cast<uint32_t>(tag_num))
        throw std::runtime_error("real dataset tag_num mismatch for bitmap copy: expected " + std::to_string(tag_num) +
                                 ", got " + std::to_string(src_tag_num));

    const size_t src_count = static_cast<size_t>(src_doc_num);

    // Bitmaps start after: header(40) + vectors(src_count*dim*4)
    const size_t bitmap_offset = 40 + src_count * dim * sizeof(float);
    real_in.seekg(static_cast<std::streamoff>(bitmap_offset));
    if (!real_in.good())
        throw std::runtime_error("cannot seek to bitmap section in real dataset");

    const size_t bitmap_bytes_per_doc = stride * sizeof(uint64_t);
    std::vector<uint64_t> bm(chunk_docs * stride);

    for (size_t base = 0; base < doc_num; base += chunk_docs) {
        const size_t n = std::min(chunk_docs, doc_num - base);

        // If output has more docs than source, wrap around
        const size_t src_remaining = src_count - (base % src_count);
        if (n <= src_remaining && base < src_count) {
            // Simple sequential read
            real_in.read(reinterpret_cast<char*>(bm.data()), static_cast<std::streamsize>(n * bitmap_bytes_per_doc));
            if (!real_in.good())
                throw std::runtime_error("read error: bitmaps from real dataset at doc " + std::to_string(base));
        } else {
            // Need to handle wrapping: read doc-by-doc
            for (size_t i = 0; i < n; ++i) {
                const size_t src_idx = (base + i) % src_count;
                const size_t src_pos = bitmap_offset + src_idx * bitmap_bytes_per_doc;
                real_in.clear();
                real_in.seekg(static_cast<std::streamoff>(src_pos));
                real_in.read(reinterpret_cast<char*>(bm.data() + i * stride),
                             static_cast<std::streamsize>(bitmap_bytes_per_doc));
                if (!real_in.good())
                    throw std::runtime_error("read error: bitmap for doc " + std::to_string(src_idx));
            }
        }

        out.write(reinterpret_cast<const char*>(bm.data()), static_cast<std::streamsize>(n * bitmap_bytes_per_doc));
        if (!out.good())
            throw std::runtime_error("write error: bitmaps from real dataset");
        if ((base / chunk_docs) % 64 == 0)
            std::cout << "[Progress] bitmaps(real): " << (base + n) << " / " << doc_num << "\n";
    }
}

// ---------------------------------------------------------------------------
// Post-generation stats
// ---------------------------------------------------------------------------
void ComputeAndWriteStats(const fs::path& dataset_path, const DatasetHeader& h, bool normalize,
                          const fs::path& output_path) {
    const size_t doc_num = static_cast<size_t>(h.doc_num);
    const size_t dim = static_cast<size_t>(h.vec_dim);
    const size_t tag_num = static_cast<size_t>(h.tag_num);
    const size_t stride = (tag_num + 63) / 64;

    const size_t sample_count = std::min<size_t>(doc_num, 10000);
    const size_t sample_step = std::max<size_t>(1, doc_num / sample_count);

    std::vector<size_t> indices;
    indices.reserve(sample_count);
    for (size_t i = 0; i < doc_num && indices.size() < sample_count; i += sample_step)
        indices.push_back(i);
    if (indices.empty())
        return;

    const std::streamoff hdr_bytes = 40;  // sizeof(DatasetCacheHeaderDisk)
    const std::streamoff vec_bpd = static_cast<std::streamoff>(dim * sizeof(float));
    const std::streamoff bm_bpd = static_cast<std::streamoff>(stride * sizeof(uint64_t));
    const std::streamoff vec_start = hdr_bytes;
    const std::streamoff bm_start = hdr_bytes + static_cast<std::streamoff>(doc_num * dim * sizeof(float));

    double norm_sum = 0, norm_sq_sum = 0;
    float norm_min = std::numeric_limits<float>::max(), norm_max = 0;
    std::vector<double> dim_sum(dim, 0), dim_sq(dim, 0);
    size_t vecs_sampled = 0;

    const int nthreads = omp_get_max_threads();
    std::vector<double> tl_norm_sum(nthreads, 0), tl_norm_sq_sum(nthreads, 0);
    std::vector<float> tl_norm_min(nthreads, std::numeric_limits<float>::max());
    std::vector<float> tl_norm_max(nthreads, 0);
    std::vector<std::vector<double>> tl_dim_sum(nthreads, std::vector<double>(dim, 0));
    std::vector<std::vector<double>> tl_dim_sq(nthreads, std::vector<double>(dim, 0));
    std::vector<size_t> tl_vecs_sampled(nthreads, 0);

    std::cout << "[Stats] Sampling vectors...\n";
    {
        std::vector<std::ifstream> vec_streams(nthreads);
        for (auto& f : vec_streams)
            f.open(dataset_path, std::ios::binary);

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& tin = vec_streams[tid];
            std::vector<float> lvec(dim);
#pragma omp for schedule(static)
            for (size_t si = 0; si < indices.size(); ++si) {
                const size_t idx = indices[si];
                tin.clear();
                tin.seekg(vec_start + static_cast<std::streamoff>(idx) * vec_bpd);
                tin.read(reinterpret_cast<char*>(lvec.data()), dim * sizeof(float));
                if (!tin.good())
                    continue;
                float nsq = 0;
                for (size_t d = 0; d < dim; ++d) {
                    nsq += lvec[d] * lvec[d];
                    tl_dim_sum[tid][d] += lvec[d];
                    tl_dim_sq[tid][d] += lvec[d] * lvec[d];
                }
                float nm = std::sqrt(nsq);
                tl_norm_sum[tid] += nm;
                tl_norm_sq_sum[tid] += nm * nm;
                tl_norm_min[tid] = std::min(tl_norm_min[tid], nm);
                tl_norm_max[tid] = std::max(tl_norm_max[tid], nm);
                ++tl_vecs_sampled[tid];
            }
        }
    }

    // Merge thread-local
    for (int t = 0; t < nthreads; ++t) {
        norm_sum += tl_norm_sum[t];
        norm_sq_sum += tl_norm_sq_sum[t];
        norm_min = std::min(norm_min, tl_norm_min[t]);
        norm_max = std::max(norm_max, tl_norm_max[t]);
        vecs_sampled += tl_vecs_sampled[t];
        for (size_t d = 0; d < dim; ++d) {
            dim_sum[d] += tl_dim_sum[t][d];
            dim_sq[d] += tl_dim_sq[t][d];
        }
    }

    std::vector<double> tf(tag_num, 0);
    std::vector<int> tch(257, 0);
    size_t bms = 0;
    std::cout << "[Stats] Sampling bitmaps...\n";
    {
        std::vector<std::vector<double>> tl_tf(nthreads, std::vector<double>(tag_num, 0));
        std::vector<std::vector<int>> tl_tch(nthreads, std::vector<int>(257, 0));
        std::vector<size_t> tl_bms(nthreads, 0);

        std::vector<std::ifstream> bm_streams(nthreads);
        for (auto& f : bm_streams)
            f.open(dataset_path, std::ios::binary);

#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& tin = bm_streams[tid];
            std::vector<uint64_t> lbm(stride);
#pragma omp for schedule(static)
            for (size_t si = 0; si < indices.size(); ++si) {
                const size_t idx = indices[si];
                tin.clear();
                tin.seekg(bm_start + static_cast<std::streamoff>(idx) * bm_bpd);
                tin.read(reinterpret_cast<char*>(lbm.data()), stride * sizeof(uint64_t));
                if (!tin.good())
                    continue;
                int cnt = 0;
                for (size_t w = 0; w < stride; ++w) {
                    uint64_t word = lbm[w];
                    while (word != 0) {
                        const int bit = __builtin_ctzll(word);
                        const size_t t = w * 64 + static_cast<size_t>(bit);
                        if (t >= tag_num)
                            break;
                        ++cnt;
                        tl_tf[tid][t] += 1.0;
                        word &= word - 1;
                    }
                }
                tl_tch[tid][std::min(cnt, 256)]++;
                ++tl_bms[tid];
            }
        }

        for (int t = 0; t < nthreads; ++t) {
            for (size_t j = 0; j < tag_num; ++j)
                tf[j] += tl_tf[t][j];
            for (int j = 0; j <= 256; ++j)
                tch[j] += tl_tch[t][j];
            bms += tl_bms[t];
        }
    }

    if (vecs_sampled == 0)
        return;

    json stats_out;
    const double n = static_cast<double>(vecs_sampled);
    const double nm = norm_sum / n;
    const double nv = norm_sq_sum / n - nm * nm;
    stats_out["vector_norm"]["mean"] = nm;
    stats_out["vector_norm"]["std"] = std::sqrt(std::max(0.0, nv));
    stats_out["vector_norm"]["min"] = static_cast<double>(norm_min);
    stats_out["vector_norm"]["max"] = static_cast<double>(norm_max);
    stats_out["vector_norm"]["sample_count"] = vecs_sampled;
    stats_out["normalize_vectors"] = normalize;

    json dm = json::array(), ds = json::array();
    for (size_t d = 0; d < dim; ++d) {
        double md = dim_sum[d] / n;
        dm.push_back(md);
        ds.push_back(std::sqrt(std::max(0.0, dim_sq[d] / n - md * md)));
    }
    stats_out["per_dim_mean"] = dm;
    stats_out["per_dim_std"] = ds;

    if (bms > 0) {
        const double bn = static_cast<double>(bms);
        json tfa = json::array();
        for (size_t t = 0; t < tag_num; ++t)
            tfa.push_back(tf[t] / bn);
        stats_out["tag_frequency"] = tfa;
        stats_out["tag_frequency_sample_count"] = bms;

        json hist = json::array();
        for (int i = 0; i <= 256; ++i)
            if (tch[i] > 0)
                hist.push_back({{"count", i}, {"freq", tch[i]}});
        stats_out["tag_count_histogram"] = hist;
    }

    fs::path sf = output_path;
    sf.replace_extension(".gen_stats.json");
    std::ofstream sof(sf);
    if (sof) {
        sof << stats_out.dump(2) << "\n";
        std::cout << "[Stats] Written to: " << sf << "\n";
    }

    std::cout << "[Stats] norm mean=" << nm << " std=" << std::sqrt(std::max(0.0, nv)) << " min=" << norm_min
              << " max=" << norm_max << "  vecs=" << vecs_sampled << " bitmaps=" << bms << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Step2Args args = ParseArgs(argc, argv);

        // Validate --real-bitmaps is provided
        if (args.real_bitmaps_path.empty())
            throw std::runtime_error("--real-bitmaps is required. Provide the path to the real dataset.");

        // Read GSSV temp file
        TempData temp = ReadTempFile(args.temp_path);
        const DatasetHeader& header = temp.header;

        std::cout << "[Info] Header: docs=" << header.doc_num << " dim=" << header.vec_dim << " tags=" << header.tag_num
                  << " buckets=" << header.bucket_num << "\n";
        std::cout << "[Info] K=" << temp.K << " normalize=" << (temp.normalize_vectors ? "true" : "false")
                  << " seed=" << temp.seed << "\n";

        // Validate cluster assignments were loaded from temp files
        if (temp.cluster_assignments.empty())
            throw std::runtime_error(
                "no cluster assignments found in temp files. "
                "Make sure step 1 (GSSV 5/6) has been run.");

        fs::create_directories(args.output_path.parent_path().empty() ? fs::path(".") : args.output_path.parent_path());

        std::ofstream out(args.output_path, std::ios::binary);
        if (!out)
            throw std::runtime_error("cannot create: " + args.output_path.string());

        // Write header
        WriteHeader(out, header);

        // Write vectors (diagonal Gaussian using cluster assignments)
        std::cout << "[Step] Writing vectors (assignment-based diagonal Gaussian)...\n";
        WriteVectors(out, header, temp, args.chunk_docs, temp.seed);

        // Write bitmaps (from real dataset)
        std::cout << "[Step] Writing bitmaps (copying from real dataset)...\n";
        WriteBitmapsFromRealDataset(out, header, args.real_bitmaps_path, args.chunk_docs);

        out.flush();
        if (!out.good())
            throw std::runtime_error("output stream error after flush");
        out.close();

        std::error_code ec;
        const auto bytes = fs::file_size(args.output_path, ec);
        if (!ec) {
            std::cout << "[Done] Generated: " << args.output_path << " ("
                      << static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) << " GB)\n";
        }

        std::cout << "[Step] Computing post-generation statistics...\n";
        ComputeAndWriteStats(args.output_path, header, temp.normalize_vectors, args.output_path);

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << "\n";
        return 1;
    }
}
