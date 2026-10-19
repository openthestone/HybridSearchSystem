/*
 * dataset_hw.h -- dataset_HW.bin (HYDSET2) reader, shared by the converter (which turns it into
 * builder input) and the harness (which scans it for the CPU recall reference). Both used to
 * carry their own copy of the header struct and the mmap/offset arithmetic.
 *
 *   [40-byte HwHeader][docNum * dim float32][docNum * ceil(tagNum/64) uint64]
 *
 * Native little-endian, read through a private mapping.
 */
#pragma once

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <string>

namespace npur_port {

#pragma pack(push, 1)
struct HwHeader {
    char magic[8];  // "HYDSET2\0"
    uint32_t version;
    uint32_t _pad0;
    uint64_t doc_num;
    uint32_t vector_dim;
    uint32_t tag_num;
    uint32_t reserved;
    uint32_t _pad1;
};
#pragma pack(pop)
static_assert(sizeof(HwHeader) == 40, "HYDSET2 header must be 40 bytes");

// The mapping is unmapped on destruction, so this has to outlive every pointer it hands out.
struct HwDataset {
    int fd = -1;
    void* map = nullptr;
    size_t mapLen = 0;
    const HwHeader* hdr = nullptr;
    const float* vectors = nullptr;
    const uint64_t* bitmaps = nullptr;
    uint32_t dim = 0, tag_num = 0, stride = 0;  // stride = uint64 words per doc bitmap
    uint64_t doc_num = 0;

    HwDataset() = default;
    HwDataset(const HwDataset&) = delete;
    HwDataset& operator=(const HwDataset&) = delete;

    // `err` names the step that failed, so callers can keep their own wording.
    bool Open(const std::string& path, std::string* err = nullptr) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            if (err)
                *err = "cannot open";
            return false;
        }
        struct stat st;
        if (fstat(fd, &st) != 0) {
            if (err)
                *err = "fstat failed";
            return false;
        }
        mapLen = static_cast<size_t>(st.st_size);
        map = ::mmap(nullptr, mapLen, PROT_READ, MAP_PRIVATE, fd, 0);
        if (map == MAP_FAILED) {
            map = nullptr;
            if (err)
                *err = "mmap failed";
            return false;
        }
        hdr = reinterpret_cast<const HwHeader*>(map);
        if (std::memcmp(hdr->magic, "HYDSET2", 7) != 0) {
            if (err)
                *err = "bad magic";
            return false;
        }
        dim = hdr->vector_dim;
        tag_num = hdr->tag_num;
        doc_num = hdr->doc_num;
        stride = (tag_num + 63u) / 64u;
        const char* base = reinterpret_cast<const char*>(map);
        vectors = reinterpret_cast<const float*>(base + sizeof(HwHeader));
        bitmaps = reinterpret_cast<const uint64_t*>(base + sizeof(HwHeader) + doc_num * dim * sizeof(float));
        return true;
    }

    ~HwDataset() {
        if (map)
            ::munmap(map, mapLen);
        if (fd >= 0)
            ::close(fd);
    }
};

}  // namespace npur_port
