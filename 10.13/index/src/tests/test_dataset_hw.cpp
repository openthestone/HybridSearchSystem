// Writes a synthetic dataset_HW.bin and reads it back through src/io/dataset_hw.h, pinning the
// HYDSET2 offsets. The converter and the harness used to carry one copy of this arithmetic each.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "src/io/dataset_hw.h"

static int g_fail = 0, g_pass = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (cond) {                                                              \
            ++g_pass;                                                            \
        } else {                                                                 \
            ++g_fail;                                                            \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                        \
    } while (0)

int main() {
    const std::string path = "/tmp/npur_dataset_hw.bin";
    // tag_num 130 is deliberately not a multiple of 64, so stride has to round up (130 -> 3).
    const uint64_t kDocs = 5;
    const uint32_t kDim = 4, kTags = 130, kStride = 3;

    std::vector<float> vecs(kDocs * kDim);
    for (size_t i = 0; i < vecs.size(); ++i)
        vecs[i] = static_cast<float>(i) * 0.5f;
    std::vector<uint64_t> bits(kDocs * kStride, 0);
    for (uint64_t d = 0; d < kDocs; ++d) {
        bits[d * kStride + 0] = 1ULL << d;        // tag d
        bits[d * kStride + 2] = 1ULL << (d % 2);  // tags 128/129, the last partial word
    }

    {
        npur_port::HwHeader h{};
        std::memcpy(h.magic, "HYDSET2", 8);
        h.version = 2;
        h.doc_num = kDocs;
        h.vector_dim = kDim;
        h.tag_num = kTags;
        std::ofstream o(path, std::ios::binary);
        o.write(reinterpret_cast<const char*>(&h), sizeof(h));
        o.write(reinterpret_cast<const char*>(vecs.data()), static_cast<std::streamsize>(vecs.size() * 4));
        o.write(reinterpret_cast<const char*>(bits.data()), static_cast<std::streamsize>(bits.size() * 8));
    }

    npur_port::HwDataset ds;
    std::string err;
    CHECK(ds.Open(path, &err));
    CHECK(ds.doc_num == kDocs);
    CHECK(ds.dim == kDim);
    CHECK(ds.tag_num == kTags);
    CHECK(ds.stride == kStride);  // ceil(130/64) == 3, not 2

    // Vectors start right after the 40-byte header; bitmaps right after the vectors.
    for (uint64_t d = 0; d < kDocs; ++d) {
        for (uint32_t k = 0; k < kDim; ++k)
            CHECK(ds.vectors[d * ds.dim + k] == vecs[d * kDim + k]);
        for (uint32_t w = 0; w < kStride; ++w)
            CHECK(ds.bitmaps[d * ds.stride + w] == bits[d * kStride + w]);
    }

    // A file that is not HYDSET2 must be refused rather than read as garbage.
    {
        const std::string bad = "/tmp/npur_dataset_hw_bad.bin";
        std::ofstream o(bad, std::ios::binary);
        o << std::string(64, 'Z');
        o.close();
        npur_port::HwDataset d2;
        std::string e2;
        CHECK(!d2.Open(bad, &e2));
        CHECK(e2 == "bad magic");
        std::remove(bad.c_str());
    }
    {
        npur_port::HwDataset d3;
        std::string e3;
        CHECK(!d3.Open("/nonexistent/dataset_HW.bin", &e3));
        CHECK(e3 == "cannot open");
    }

    std::remove(path.c_str());
    std::fprintf(stderr, "test_dataset_hw: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
