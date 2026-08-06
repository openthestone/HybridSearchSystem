// tools/build_inverted_index.cpp — build tag-major inverted postings from
// dataset_HW.bin.
//
// Output file layout (dataset_HW_inv.bin):
//   40B header: magic + version + doc_num + tag_num + segments + reserved
//   postings: tag_num × segments × 2B uint16 (16 docs/segment, bit i = doc seg*16+i has tag)
//
// Usage:
//   ./build_inverted_index --dataset=dataset_HW.bin \
//                          --out=dataset_HW_inv.bin \
//                          --doc_subset=65536

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "../common/dataset_hw.h"

#pragma pack(push, 1)
struct InvHeader {
    char magic[8];     // "HYDINV02"
    uint32_t version;  // 2
    uint32_t _pad0;
    uint64_t doc_num;  // actual docs indexed (subset)
    uint32_t tag_num;
    uint32_t segments;  // ceil(doc_num / 16)
    uint32_t reserved;
    uint32_t _pad1;
};
#pragma pack(pop)
static_assert(sizeof(InvHeader) == 40, "inv header must be 40 bytes");

int main(int argc, char** argv) {
    std::string dataset_path = "/root/sks_hw/dataset_HW.bin";
    std::string out_path = "/root/sks_hw/test/full_npu/dataset_HW_inv.bin";
    uint64_t doc_subset = 0;  // 0 = all

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto pos = a.find('=');
        if (pos == std::string::npos)
            continue;
        std::string k = a.substr(0, pos);
        std::string v = a.substr(pos + 1);
        if (k == "--dataset")
            dataset_path = v;
        else if (k == "--out")
            out_path = v;
        else if (k == "--doc_subset")
            doc_subset = std::stoull(v);
    }

    full_npu::DatasetHW ds;
    if (!ds.Open(dataset_path)) {
        std::fprintf(stderr, "failed to open dataset\n");
        return 1;
    }

    uint64_t total_docs = ds.DocNum(0);
    uint64_t M = (doc_subset > 0 && doc_subset < total_docs) ? doc_subset : total_docs;
    uint32_t tag_num = ds.TagNum();
    uint32_t segments = (uint32_t)((M + 15) / 16);

    std::printf("dataset: %s\n", dataset_path.c_str());
    std::printf("doc_subset: %llu / %llu total\n", (unsigned long long)M, (unsigned long long)total_docs);
    std::printf("tag_num: %u  segments: %u\n", tag_num, segments);

    // postings[tag * segments + seg] = uint16_t
    size_t post_count = (size_t)tag_num * segments;
    size_t post_bytes = post_count * sizeof(uint16_t);
    std::printf("postings size: %zu bytes (%.1f MB)\n", post_bytes, post_bytes / 1048576.0);

    if (post_bytes > (size_t)32 * 1024 * 1024 * 1024ull) {
        std::fprintf(stderr, "postings > 32GB, refusing\n");
        return 2;
    }

    std::vector<uint16_t> postings(post_count, 0);

    // Build: scan docs, for each tag set bit in postings[tag][doc/16].
    // Stride over doc bitmap in 64-bit chunks for speed.
    uint32_t stride = ds.Stride();
    for (uint64_t d = 0; d < M; ++d) {
        const uint64_t* bm = ds.Bitmap(d);
        uint32_t seg = (uint32_t)(d / 16);
        uint32_t bit = (uint32_t)(d % 16);
        uint16_t bit_mask = (uint16_t)(1u << bit);

        // For each 64-bit word in doc bitmap, scan set bits → tag ids.
        for (uint32_t w = 0; w < stride; ++w) {
            uint64_t word = bm[w];
            if (word == 0)
                continue;
            uint32_t base_tag = w * 64;
            while (word) {
                uint32_t b = __builtin_ctzll(word);
                uint32_t tag_id = base_tag + b;
                if (tag_id < tag_num) {
                    postings[(size_t)tag_id * segments + seg] |= bit_mask;
                }
                word &= (word - 1);
            }
        }
        if ((d & 0xFFFF) == 0) {
            std::printf("\rprogress: %llu / %llu docs", (unsigned long long)d, (unsigned long long)M);
            std::fflush(stdout);
        }
    }
    std::printf("\nprogress: done\n");

    // Write output.
    std::ofstream out(out_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        std::fprintf(stderr, "cannot open out path\n");
        return 3;
    }

    InvHeader hdr{};
    std::memcpy(hdr.magic, "HYDINV02", 8);
    hdr.version = 2;
    hdr.doc_num = M;
    hdr.tag_num = tag_num;
    hdr.segments = segments;
    out.write(reinterpret_cast<const char*>(&hdr), sizeof(hdr));
    out.write(reinterpret_cast<const char*>(postings.data()), post_bytes);
    out.close();

    std::printf("wrote %s (%llu bytes)\n", out_path.c_str(), (unsigned long long)(sizeof(hdr) + post_bytes));

    ds.Close();
    return 0;
}
