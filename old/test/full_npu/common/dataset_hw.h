// common/dataset_hw.h — header-only reader for dataset_HW.bin (HYDSET2 format).
//
// Layout:
//   40B header: magic(8) + version(4) + pad(4) + doc_num(8) +
//               vector_dim(4) + tag_num(4) + reserved(4) + tail_pad(4)
//   vectors: doc_num × vector_dim × 4B float32, row-major
//   bitmaps: doc_num × stride × 8B uint64, row-major
//            stride = ceil(tag_num / 64)
//
// mmap-only — file is up to 47GB, no eager load.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace full_npu {

#pragma pack(push, 1)
struct DatasetCacheHeaderDisk {
    char magic[8];     // "HYDSET2\0"
    uint32_t version;  // 2
    uint32_t _pad0;    // align doc_num to 8
    uint64_t doc_num;
    uint32_t vector_dim;
    uint32_t tag_num;
    uint32_t reserved;
    uint32_t _pad1;  // tail pad to 40
};
#pragma pack(pop)
static_assert(sizeof(DatasetCacheHeaderDisk) == 40, "header must be 40 bytes");

class DatasetHW {
   public:
    bool Open(const std::string& path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) {
            std::perror("open");
            return false;
        }

        struct stat st;
        if (fstat(fd_, &st) != 0) {
            std::perror("fstat");
            return false;
        }
        file_size_ = static_cast<uint64_t>(st.st_size);

        // Map just the header first to read dims.
        void* h = ::mmap(nullptr, sizeof(DatasetCacheHeaderDisk), PROT_READ, MAP_PRIVATE, fd_, 0);
        if (h == MAP_FAILED) {
            std::perror("mmap header");
            return false;
        }
        auto* hdr = static_cast<DatasetCacheHeaderDisk*>(h);

        if (std::memcmp(hdr->magic, "HYDSET2", 7) != 0) {
            std::fprintf(stderr, "[fatal] bad magic: %.8s\n", hdr->magic);
            ::munmap(h, sizeof(DatasetCacheHeaderDisk));
            return false;
        }
        doc_num_ = hdr->doc_num;
        vector_dim_ = hdr->vector_dim;
        tag_num_ = hdr->tag_num;
        stride_ = (tag_num_ + 63u) / 64u;
        ::munmap(h, sizeof(DatasetCacheHeaderDisk));

        vectors_offset_ = sizeof(DatasetCacheHeaderDisk);
        bitmaps_offset_ = vectors_offset_ + doc_num_ * vector_dim_ * sizeof(float);
        expected_size_ = bitmaps_offset_ + doc_num_ * stride_ * sizeof(uint64_t);
        if (file_size_ < expected_size_) {
            std::fprintf(stderr, "[fatal] file too small: %llu < %llu expected\n", (unsigned long long)file_size_,
                         (unsigned long long)expected_size_);
            return false;
        }

        // Map whole file read-only. 47GB virtual is fine on 64-bit.
        full_map_ = ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (full_map_ == MAP_FAILED) {
            std::perror("mmap full");
            return false;
        }
        vectors_base_ = static_cast<const char*>(full_map_) + vectors_offset_;
        bitmaps_base_ = static_cast<const char*>(full_map_) + bitmaps_offset_;
        return true;
    }

    void Close() {
        if (full_map_ && full_map_ != MAP_FAILED) {
            ::munmap(full_map_, file_size_);
            full_map_ = nullptr;
        }
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
    }

    ~DatasetHW() {
        Close();
    }

    // Accessors — subset_count = 0 means full corpus.
    uint64_t DocNum(uint64_t subset) const {
        return subset > 0 && subset < doc_num_ ? subset : doc_num_;
    }
    uint32_t VectorDim() const {
        return vector_dim_;
    }
    uint32_t TagNum() const {
        return tag_num_;
    }
    uint32_t Stride() const {
        return stride_;
    }

    // doc_idx ∈ [0, doc_num). No bounds check — caller responsible.
    const float* Vector(uint64_t doc_idx) const {
        return reinterpret_cast<const float*>(vectors_base_ + doc_idx * vector_dim_ * sizeof(float));
    }
    const uint64_t* Bitmap(uint64_t doc_idx) const {
        return reinterpret_cast<const uint64_t*>(bitmaps_base_ + doc_idx * stride_ * sizeof(uint64_t));
    }

    // Bit test: does doc have tag set?
    bool HasTag(uint64_t doc_idx, uint32_t tag_id) const {
        const uint64_t* bm = Bitmap(doc_idx);
        return (bm[tag_id >> 6] >> (tag_id & 63)) & 1ULL;
    }

   private:
    int fd_ = -1;
    void* full_map_ = nullptr;
    uint64_t file_size_ = 0;
    uint64_t doc_num_ = 0;
    uint32_t vector_dim_ = 0;
    uint32_t tag_num_ = 0;
    uint32_t stride_ = 0;
    uint64_t vectors_offset_ = 0;
    uint64_t bitmaps_offset_ = 0;
    uint64_t expected_size_ = 0;
    const char* vectors_base_ = nullptr;
    const char* bitmaps_base_ = nullptr;
};

}  // namespace full_npu
