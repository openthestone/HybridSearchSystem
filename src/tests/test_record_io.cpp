// Writes with the converter's writer (src/io/record_io.h) and reads back with engine/'s ACTUAL
// consumer, NpuRetrieval::ReadAndDoTask, pinning the on-disk contract against the real reader.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "src/io/record_io.h"
#include "src/full_recall/indexer/file/file_reader.h"

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
    const std::string path = "/tmp/npur_record_io.bin";

    // Payloads include empty and binary bytes, mirroring docid files and section files.
    std::vector<std::pair<uint64_t, std::string>> recs = {
        {0ULL, std::string()},                                     // docid-style: empty payload
        {1ULL, std::string("hello")},                              // ascii
        {12345678901ULL, std::string("\x00\x01\x02\xff\x10", 5)},  // binary w/ NULs
        {4294967296ULL, std::string(300, 'x')},                    // gdocid > u32; long payload
    };

    // Write with the converter's writer: [u32 count] then records.
    {
        std::ofstream o(path, std::ios::binary);
        npur_port::WriteU32(o, static_cast<uint32_t>(recs.size()));
        for (auto& r : recs)
            npur_port::WriteRecord(o, r.first, r.second);
    }

    // Read back with engine/'s real ReadAndDoTask.
    std::vector<std::pair<uint64_t, std::string>> got;
    uint32_t seenCount = 0;
    bool ok = NpuRetrieval::ReadAndDoTask(
        path, [&](NpuRetrieval::GlobalDocID gdocid, uint32_t i, uint32_t count, const std::string& payload) -> bool {
            seenCount = count;
            CHECK(i == got.size());  // index is sequential
            got.emplace_back(static_cast<uint64_t>(gdocid), payload);
            return true;
        });

    CHECK(ok);
    CHECK(seenCount == recs.size());
    CHECK(got.size() == recs.size());
    for (size_t i = 0; i < recs.size() && i < got.size(); ++i) {
        CHECK(got[i].first == recs[i].first);    // gdocid round-trips (incl. > u32)
        CHECK(got[i].second == recs[i].second);  // payload bytes round-trip exactly
    }

    std::fprintf(stderr, "test_record_io: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
