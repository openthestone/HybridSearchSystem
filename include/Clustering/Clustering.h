#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <cstdint>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

static int THREADS = 0; // 0 = auto, 1 = 禁用并行
static std::string format_min_sec(double seconds) {
	if (seconds < 0) seconds = 0;
	int minutes = static_cast<int>(seconds / 60.0);
	double secs = seconds - minutes * 60.0;
	std::ostringstream oss;
	oss.setf(std::ios::fixed);
	oss.precision(2);
	oss << minutes << "分" << secs << "秒";
	return oss.str();
}

static inline bool get_bit(const uint8_t* row, int bit_idx) {
    int byte_idx = bit_idx >> 3;
    int bit_off = 7 - (bit_idx & 7);
    return (row[byte_idx] >> bit_off) & 1;
}

namespace clustering {

class Clustering {
public:
    struct RuntimeParams {
        int total_doc_num{0};
        int total_tag_num{0};
        int total_bucket_num_level_1{0};
        int vector_dim{0};
        int max_doc_per_bucket{0};
    };

    struct MatrixF32 {
        int64_t rows{0};
        int cols{0};
        std::vector<float> data;
    };

    static Clustering& Instance() {
        static Clustering inst;
        return inst;
    }

    void EnableInMemory(bool on = true) {
        std::lock_guard<std::mutex> lock(mu_);
        in_memory_ = on;
    }

    bool InMemory() const {
        std::lock_guard<std::mutex> lock(mu_);
        return in_memory_;
    }

    void SetRuntimeParams(int total_doc_num,
                          int total_tag_num,
                          int total_bucket_num_level_1,
                          int vector_dim,
                          int max_doc_per_bucket) {
        std::lock_guard<std::mutex> lock(mu_);
        runtime_params_.total_doc_num = total_doc_num;
        runtime_params_.total_tag_num = total_tag_num;
        runtime_params_.total_bucket_num_level_1 = total_bucket_num_level_1;
        runtime_params_.vector_dim = vector_dim;
        runtime_params_.max_doc_per_bucket = max_doc_per_bucket;
    }

    RuntimeParams GetRuntimeParams() const {
        std::lock_guard<std::mutex> lock(mu_);
        return runtime_params_;
    }

    void Reset() {
        std::lock_guard<std::mutex> lock(mu_);
        stage0_centroids_ = {};
        stage0_labels_.clear();
        stage1_centroids_ = {};
        stage1_labels_.clear();
        leaves_.clear();
        final_bucket_docs_.clear();
        final_bucket_centroids_.clear();
        bucket_ids_.clear();
        input_vectors_.clear();
        input_bitmaps_.clear();
        input_total_docs_ = 0;
        input_vector_dim_ = 0;
        input_total_tag_num_ = 0;
    }

    bool SetInputDataset(const float* vectors,
                         int64_t total_docs,
                         int vector_dim,
                         const uint64_t* tag_bitmaps,
                         int total_tag_num) {
        std::lock_guard<std::mutex> lock(mu_);
        if (!vectors || !tag_bitmaps || total_docs <= 0 || vector_dim <= 0 || total_tag_num <= 0) {
            return false;
        }
        int64_t vec_count = total_docs * static_cast<int64_t>(vector_dim);
        int64_t bitmap_u64_stride = (total_tag_num + 63) / 64;
        int64_t bitmap_count = total_docs * bitmap_u64_stride;

        input_vectors_.assign(vectors, vectors + vec_count);
        input_bitmaps_.assign(tag_bitmaps, tag_bitmaps + bitmap_count);
        input_total_docs_ = total_docs;
        input_vector_dim_ = vector_dim;
        input_total_tag_num_ = total_tag_num;
        return true;
    }

    bool HasInputDataset() const {
        std::lock_guard<std::mutex> lock(mu_);
        return input_total_docs_ > 0 && input_vector_dim_ > 0 && input_total_tag_num_ > 0
            && !input_vectors_.empty() && !input_bitmaps_.empty();
    }

    bool GetInputVectorsMatrix(MatrixF32& out) const {
        std::lock_guard<std::mutex> lock(mu_);
        if (!HasInputDatasetNoLock()) return false;
        out.rows = input_total_docs_;
        out.cols = input_vector_dim_;
        out.data = input_vectors_;
        return true;
    }

    bool ExportInputToDocsDirectory(const std::string& docs_dir, int shard_count = 1000) const {
        std::lock_guard<std::mutex> lock(mu_);
        if (!HasInputDatasetNoLock()) return false;
        if (shard_count <= 0) shard_count = 1000;

        namespace fs = std::filesystem;
        std::error_code ec;
        fs::remove_all(docs_dir, ec);
        ec.clear();
        fs::create_directories(docs_dir, ec);
        if (ec) return false;

        int64_t docs_per_shard = (input_total_docs_ + shard_count - 1) / shard_count;
        if (docs_per_shard <= 0) docs_per_shard = 1;

        int64_t bitmap_u64_stride = (input_total_tag_num_ + 63) / 64;
        int64_t bitmap_byte_stride = (input_total_tag_num_ + 7) / 8;

        for (int shard = 0; shard < shard_count; ++shard) {
            int64_t start = static_cast<int64_t>(shard) * docs_per_shard;
            int64_t end = std::min<int64_t>(input_total_docs_, start + docs_per_shard);

            std::ostringstream shard_name;
            shard_name << "shard-" << shard;
            fs::path shard_dir = fs::path(docs_dir) / shard_name.str();
            fs::create_directories(shard_dir, ec);
            if (ec) return false;

            std::ofstream doc_ofs(shard_dir / "doc.txt");
            std::ofstream bitmap_ofs(shard_dir / "bitmap.txt");
            if (!doc_ofs || !bitmap_ofs) return false;

            for (int64_t i = start; i < end; ++i) {
                const float* vec = input_vectors_.data() + i * input_vector_dim_;
                for (int d = 0; d < input_vector_dim_; ++d) {
                    if (d) doc_ofs << ' ';
                    doc_ofs << vec[d];
                }
                doc_ofs << '\n';

                const uint8_t* row_bytes = reinterpret_cast<const uint8_t*>(input_bitmaps_.data() + i * bitmap_u64_stride);
                bitmap_ofs << Base64Encode(row_bytes, static_cast<size_t>(bitmap_byte_stride)) << '\n';
            }
        }
        return true;
    }

    void SetStage0(const MatrixF32& centers, const std::vector<int>& labels) {
        std::lock_guard<std::mutex> lock(mu_);
        stage0_centroids_ = centers;
        stage0_labels_ = labels;
    }

    bool GetStage0(MatrixF32& centers, std::vector<int>& labels) const {
        std::lock_guard<std::mutex> lock(mu_);
        if (stage0_centroids_.rows <= 0 || stage0_centroids_.cols <= 0 || stage0_centroids_.data.empty()) {
            return false;
        }
        centers = stage0_centroids_;
        labels = stage0_labels_;
        return true;
    }

    void SetStage1(const MatrixF32& centers, const std::vector<int>& labels) {
        std::lock_guard<std::mutex> lock(mu_);
        stage1_centroids_ = centers;
        stage1_labels_ = labels;
    }

    bool GetStage1(MatrixF32& centers, std::vector<int>& labels) const {
        std::lock_guard<std::mutex> lock(mu_);
        if (stage1_centroids_.rows <= 0 || stage1_centroids_.cols <= 0 || stage1_centroids_.data.empty()) {
            return false;
        }
        centers = stage1_centroids_;
        labels = stage1_labels_;
        return true;
    }

    void StoreLeaf(const std::string& name, const std::vector<int>& doc_ids_1based) {
        std::lock_guard<std::mutex> lock(mu_);
        leaves_[name] = doc_ids_1based;
    }

    const std::unordered_map<std::string, std::vector<int>>& Leaves() const {
        return leaves_;
    }

    bool HasLeaves() const {
        std::lock_guard<std::mutex> lock(mu_);
        return !leaves_.empty();
    }

    void StoreFinalBucket(const std::string& name,
                          const std::vector<int>& doc_ids_1based,
                          const std::vector<uint8_t>& centroid_bits) {
        std::lock_guard<std::mutex> lock(mu_);
        final_bucket_docs_[name] = doc_ids_1based;
        final_bucket_centroids_[name] = centroid_bits;
    }

    const std::unordered_map<std::string, std::vector<int>>& FinalBucketDocs() const {
        return final_bucket_docs_;
    }

    const std::unordered_map<std::string, std::vector<uint8_t>>& FinalBucketCentroids() const {
        return final_bucket_centroids_;
    }

    bool BuildBucketIdsFromFinalBuckets(int64_t total_docs, int default_bucket = -1) {
        std::lock_guard<std::mutex> lock(mu_);
        if (total_docs <= 0) return false;

        bucket_ids_.assign(static_cast<size_t>(total_docs), default_bucket);
        bool assigned_any = false;

        for (const auto& kv : final_bucket_docs_) {
            const std::string& bucket_name = kv.first;
            const auto& docs_1based = kv.second;
            int bucket_id = ParseBucketId(bucket_name);
            if (bucket_id < 0) continue;

            for (int doc_id_1based : docs_1based) {
                if (doc_id_1based <= 0 || doc_id_1based > total_docs) continue;
                bucket_ids_[static_cast<size_t>(doc_id_1based - 1)] = bucket_id;
                assigned_any = true;
            }
        }
        return assigned_any;
    }

    const std::vector<int>& BucketIds() const {
        return bucket_ids_;
    }

    int64_t CountAssignedBucketIds() const {
        std::lock_guard<std::mutex> lock(mu_);
        int64_t count = 0;
        for (int v : bucket_ids_) {
            if (v >= 0) ++count;
        }
        return count;
    }

private:
    static std::string Base64Encode(const uint8_t* data, size_t len) {
        static const char* kChars =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string out;
        out.reserve(((len + 2) / 3) * 4);

        size_t i = 0;
        for (; i + 3 <= len; i += 3) {
            uint32_t n = (static_cast<uint32_t>(data[i]) << 16) |
                         (static_cast<uint32_t>(data[i + 1]) << 8) |
                         static_cast<uint32_t>(data[i + 2]);
            out.push_back(kChars[(n >> 18) & 63]);
            out.push_back(kChars[(n >> 12) & 63]);
            out.push_back(kChars[(n >> 6) & 63]);
            out.push_back(kChars[n & 63]);
        }

        size_t rem = len - i;
        if (rem == 1) {
            uint32_t n = static_cast<uint32_t>(data[i]) << 16;
            out.push_back(kChars[(n >> 18) & 63]);
            out.push_back(kChars[(n >> 12) & 63]);
            out.push_back('=');
            out.push_back('=');
        } else if (rem == 2) {
            uint32_t n = (static_cast<uint32_t>(data[i]) << 16) |
                         (static_cast<uint32_t>(data[i + 1]) << 8);
            out.push_back(kChars[(n >> 18) & 63]);
            out.push_back(kChars[(n >> 12) & 63]);
            out.push_back(kChars[(n >> 6) & 63]);
            out.push_back('=');
        }
        return out;
    }

    bool HasInputDatasetNoLock() const {
        return input_total_docs_ > 0 && input_vector_dim_ > 0 && input_total_tag_num_ > 0
            && !input_vectors_.empty() && !input_bitmaps_.empty();
    }

    static int ParseBucketId(const std::string& name) {
        size_t pos = name.find("bucket");
        if (pos == std::string::npos) return -1;
        pos += 6;
        size_t end = pos;
        while (end < name.size() && std::isdigit(static_cast<unsigned char>(name[end]))) {
            ++end;
        }
        if (end == pos) return -1;
        try {
            return std::stoi(name.substr(pos, end - pos));
        } catch (...) {
            return -1;
        }
    }

    Clustering() {
        const char* env = std::getenv("CLUSTERING_INMEM");
        in_memory_ = (env != nullptr && std::string(env) == "1");
    }

private:
    mutable std::mutex mu_;
    bool in_memory_{false};

    MatrixF32 stage0_centroids_;
    std::vector<int> stage0_labels_;

    MatrixF32 stage1_centroids_;
    std::vector<int> stage1_labels_;

    std::unordered_map<std::string, std::vector<int>> leaves_;
    std::unordered_map<std::string, std::vector<int>> final_bucket_docs_;
    std::unordered_map<std::string, std::vector<uint8_t>> final_bucket_centroids_;
    std::vector<int> bucket_ids_;

    RuntimeParams runtime_params_;

    std::vector<float> input_vectors_;
    std::vector<uint64_t> input_bitmaps_;
    int64_t input_total_docs_{0};
    int input_vector_dim_{0};
    int input_total_tag_num_{0};
};

} // namespace clustering

#endif // CLUSTERING_H