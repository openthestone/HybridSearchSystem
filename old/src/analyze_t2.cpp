#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "utils/DataReader.h"
#include "utils/RunSupport.h"

namespace fs = RunSupport::fs;

namespace {
struct DatasetHeader {
    int doc_num = 0;
    int vec_dim = 0;
    int tag_num = 0;
    int bucket_num = 0;
};

bool ReadDatasetHeader(std::ifstream& in, DatasetHeader& header) {
    in.read(reinterpret_cast<char*>(&header.doc_num), sizeof(header.doc_num));
    in.read(reinterpret_cast<char*>(&header.vec_dim), sizeof(header.vec_dim));
    in.read(reinterpret_cast<char*>(&header.tag_num), sizeof(header.tag_num));
    in.read(reinterpret_cast<char*>(&header.bucket_num), sizeof(header.bucket_num));

    if (!in.good()) {
        return false;
    }

    return header.doc_num > 0 && header.vec_dim > 0 && header.tag_num > 0 && header.bucket_num > 0;
}

bool ValidateHeaderAgainstConfig(const DatasetHeader& header) {
    if (!g_params_loaded) {
        total_doc_num = header.doc_num;
        vector_dim = header.vec_dim;
        total_tag_num = header.tag_num;
        total_bucket_num_level_1 = header.bucket_num;
        total_bucket_num_level_2 = header.bucket_num;
        return true;
    }

    if (vector_dim != header.vec_dim) {
        DataReader::PrintVectorDimMismatchWarning("dataset.bin", header.vec_dim, vector_dim);
        return false;
    }

    if (total_doc_num != header.doc_num || total_tag_num != header.tag_num) {
        std::cerr << "[Config] Error: dataset.bin header does not match loaded config values.\n"
                  << "  config:  docs=" << total_doc_num << ", dim=" << vector_dim << ", tags=" << total_tag_num
                  << ", buckets(level_1)=" << total_bucket_num_level_1 << "\n"
                  << "  dataset: docs=" << header.doc_num << ", dim=" << header.vec_dim << ", tags=" << header.tag_num
                  << ", buckets=" << header.bucket_num << "\n";
        return false;
    }

    if (total_bucket_num_level_1 != header.bucket_num) {
        std::cout << "[Config] Warning: total_bucket_num_level_1=" << total_bucket_num_level_1
                  << " differs from dataset.bin header bucket_num=" << header.bucket_num
                  << ". Keeping config bucket count and ignoring dataset header bucket_num.\n";
    }

    return true;
}

bool SeekToBitmapSection(std::ifstream& in, const DatasetHeader& header, uint64_t bitmap_stride) {
    const uint64_t header_bytes = static_cast<uint64_t>(sizeof(int)) * 4ULL;
    const uint64_t vector_bytes =
        static_cast<uint64_t>(header.doc_num) * static_cast<uint64_t>(header.vec_dim) * sizeof(float);
    const uint64_t bucket_id_bytes = static_cast<uint64_t>(header.doc_num) * sizeof(uint32_t);
    const uint64_t bitmap_offset = header_bytes + vector_bytes + bucket_id_bytes;

    if (bitmap_stride == 0) {
        return false;
    }

    if (bitmap_offset > static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        std::cerr << "[Loader] Error: dataset bitmap offset exceeds streamoff range.\n";
        return false;
    }

    in.seekg(static_cast<std::streamoff>(bitmap_offset), std::ios::beg);
    if (!in.good()) {
        std::cerr << "[Loader] Error: failed to seek to bitmap section.\n";
        return false;
    }

    return true;
}

bool WriteDistributionFiles(const fs::path& output_dir, const std::vector<uint64_t>& tag_doc_counts, uint64_t doc_num) {
    if (!RunSupport::EnsureOutputDirectory(output_dir)) {
        return false;
    }

    constexpr size_t kLinesPerFile = 6000;
    const size_t file_count = (tag_doc_counts.size() + kLinesPerFile - 1) / kLinesPerFile;

    for (size_t file_idx = 0; file_idx < file_count; ++file_idx) {
        const size_t begin = file_idx * kLinesPerFile;
        const size_t end = std::min(begin + kLinesPerFile, tag_doc_counts.size());

        std::ostringstream file_name_stream;
        file_name_stream << "distribution_" << std::setw(4) << std::setfill('0') << (file_idx + 1) << ".txt";
        const fs::path output_file = output_dir / file_name_stream.str();

        std::ofstream out(output_file, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            std::cerr << "[Writer] Failed to open output file: " << output_file << "\n";
            return false;
        }

        out << std::fixed << std::setprecision(12);
        for (size_t tag_id = begin; tag_id < end; ++tag_id) {
            const double ratio =
                (doc_num == 0) ? 0.0 : static_cast<double>(tag_doc_counts[tag_id]) / static_cast<double>(doc_num);
            out << ratio << '\n';
        }

        if (!out.good()) {
            std::cerr << "[Writer] Failed while writing output file: " << output_file << "\n";
            return false;
        }
    }

    return true;
}
}  // namespace

int main() {
    constexpr const char* kDatasetFile = "../../dataset_HW.bin";
    // constexpr const char *kDatasetFile = "../../dataset_DEEP.bin";
    constexpr size_t kChunkDocCount = 4096;
    constexpr uint64_t kProgressDocInterval = 1000000ULL;

    const fs::path config_path = RunSupport::ResolveConfigPath();
    std::cout << "[Analyze] Loading config from " << config_path << "...\n";
    if (!LoadParams(config_path.string(), ResourceConfigProfile::Parallel)) {
        std::cerr << "[Error] Failed to load config: " << config_path << "\n";
        return -1;
    }

    const fs::path query_result_root_dir = query_result_root.empty() ? fs::path(".") : fs::path(query_result_root);
    const fs::path distribution_output_dir = query_result_root_dir / "log";

    std::ifstream in(kDatasetFile, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "[Fatal] Failed to open dataset cache: " << kDatasetFile << "\n";
        return -1;
    }

    DatasetHeader header;
    if (!ReadDatasetHeader(in, header)) {
        std::cerr << "[Fatal] Failed to read dataset header from: " << kDatasetFile << "\n";
        return -1;
    }

    if (!ValidateHeaderAgainstConfig(header)) {
        return -1;
    }

    const uint64_t bitmap_stride = static_cast<uint64_t>(header.tag_num + 63) / 64ULL;
    const uint64_t last_word_mask = (header.tag_num % 64 == 0)
                                        ? std::numeric_limits<uint64_t>::max()
                                        : ((1ULL << static_cast<unsigned>(header.tag_num % 64)) - 1ULL);

    std::cout << "[Loader] Dataset header loaded. Docs=" << header.doc_num << ", Dim=" << header.vec_dim
              << ", Tags=" << header.tag_num << ", Buckets(Level1)=" << header.bucket_num
              << ", BitmapStride(u64)=" << bitmap_stride << "\n";

    if (!SeekToBitmapSection(in, header, bitmap_stride)) {
        return -1;
    }

    std::vector<uint64_t> tag_doc_counts(static_cast<size_t>(header.tag_num), 0);
    std::vector<uint64_t> bitmap_chunk;
    bitmap_chunk.resize(static_cast<size_t>(bitmap_stride) * kChunkDocCount);

    uint64_t docs_processed = 0;
    uint64_t next_progress_mark = kProgressDocInterval;

    std::cout << "[Analyze] Streaming bitmap section from dataset.bin...\n";
    while (docs_processed < static_cast<uint64_t>(header.doc_num)) {
        const uint64_t docs_left = static_cast<uint64_t>(header.doc_num) - docs_processed;
        const size_t docs_this_round =
            static_cast<size_t>(std::min<uint64_t>(docs_left, static_cast<uint64_t>(kChunkDocCount)));
        const size_t words_this_round = docs_this_round * static_cast<size_t>(bitmap_stride);

        in.read(reinterpret_cast<char*>(bitmap_chunk.data()),
                static_cast<std::streamsize>(words_this_round * sizeof(uint64_t)));
        if (!in.good()) {
            std::cerr << "[Fatal] Failed while reading bitmap section from dataset.bin.\n";
            return -1;
        }

        for (size_t doc_idx = 0; doc_idx < docs_this_round; ++doc_idx) {
            const uint64_t* doc_bitmap = bitmap_chunk.data() + doc_idx * static_cast<size_t>(bitmap_stride);

            for (size_t word_idx = 0; word_idx < static_cast<size_t>(bitmap_stride); ++word_idx) {
                uint64_t bits = doc_bitmap[word_idx];
                if (word_idx + 1 == static_cast<size_t>(bitmap_stride)) {
                    bits &= last_word_mask;
                }

                const size_t tag_base = word_idx * 64;
                while (bits != 0) {
                    const unsigned bit_idx = static_cast<unsigned>(__builtin_ctzll(bits));
                    tag_doc_counts[tag_base + bit_idx] += 1ULL;
                    bits &= (bits - 1);
                }
            }
        }

        docs_processed += static_cast<uint64_t>(docs_this_round);
        while (docs_processed >= next_progress_mark) {
            std::cout << "[Analyze] Progress: " << docs_processed << "/" << header.doc_num << "\n";
            next_progress_mark += kProgressDocInterval;
        }
    }

    if (!WriteDistributionFiles(distribution_output_dir, tag_doc_counts, static_cast<uint64_t>(header.doc_num))) {
        return -1;
    }

    std::cout << "[Analyze] Output written to " << distribution_output_dir
              << " as distribution_XXXX.txt (6000 lines per file).\n";
    std::cout << "[Analyze] Global line N still corresponds to tag_id = N - 1 across the split files.\n";
    return 0;
}
