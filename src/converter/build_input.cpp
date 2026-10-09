// build_input: produces the offline builder's INPUT from synthetic data or a dataset_HW.bin
// subset. Standalone (protobuf only; no gflags/CANN), built via src/CMakeLists.txt.
//
//   <out>/schema.json                     JSON of Building.IndexBuilderConfig
//   <out>/docid/attachment.docid.<seg>    id attachment field
//   <out>/content/section.content.<seg>   vector + tags, one Section/doc
//
// Each field file is [u32 count], then count x [u32 len][u64 gdocid][len-8 bytes protobuf].
// The docid payload is empty (len == 8); the content payload is a Building.Section carrying
// float_embedding[0] and one TermInfo{uint64Value = tagId} per set tag.
//
// The builder hashes tags as Hash64(std::to_string(tagId)) and the harness must match, so RAW
// tag ids are emitted here.
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "../io/dataset_hw.h"
#include "../io/record_io.h"
#include "src/utils/env_switch.h"
#include "src/full_recall/indexer/proto/document.pb.h"

namespace {

using npur_port::WriteRecord;
using npur_port::WriteU32;
using npur_port::WriteU64;

struct Args {
    std::string mode = "synthetic";  // synthetic | hw
    std::string out;
    std::string dataset;      // dataset_HW.bin (hw mode)
    uint64_t docs = 0;        // 0 = all (hw mode), synthetic uses all requested docs
    uint64_t doc_offset = 0;  // first source doc of this shard; shard covers [doc_offset, doc_offset+docs)
    uint32_t dim = 64;
    uint32_t tag_num = 35840;
    uint32_t tags_per_doc = 8;            // synthetic mode
    uint32_t doc_num_per_segment = 4096;  // multiple of 256 (split_doc_num_zn) + of 16 (bitset u16 unit)
    uint32_t split_doc_num_zn = 0;        // 0 => doc_num_per_segment
    double density_threshold = 0.05;
    uint64_t seed = 42;
    unsigned threads = 0;  // 0 => hardware_concurrency
};

bool ParseArgs(int argc, char** argv, Args& a) {
    auto need = [&](int& i) -> const char* {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "missing value for %s\n", argv[i]);
            return nullptr;
        }
        return argv[++i];
    };
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        const char* v = nullptr;
        if (k == "--mode") {
            if (!(v = need(i)))
                return false;
            a.mode = v;
        } else if (k == "--out") {
            if (!(v = need(i)))
                return false;
            a.out = v;
        } else if (k == "--dataset") {
            if (!(v = need(i)))
                return false;
            a.dataset = v;
        } else if (k == "--docs") {
            if (!(v = need(i)))
                return false;
            a.docs = std::strtoull(v, nullptr, 10);
        } else if (k == "--doc_offset") {
            if (!(v = need(i)))
                return false;
            a.doc_offset = std::strtoull(v, nullptr, 10);
        } else if (k == "--dim") {
            if (!(v = need(i)))
                return false;
            a.dim = std::strtoul(v, nullptr, 10);
        } else if (k == "--tag_num") {
            if (!(v = need(i)))
                return false;
            a.tag_num = std::strtoul(v, nullptr, 10);
        } else if (k == "--tags_per_doc") {
            if (!(v = need(i)))
                return false;
            a.tags_per_doc = std::strtoul(v, nullptr, 10);
        } else if (k == "--doc_num_per_segment") {
            if (!(v = need(i)))
                return false;
            a.doc_num_per_segment = std::strtoul(v, nullptr, 10);
        } else if (k == "--split_doc_num_zn") {
            if (!(v = need(i)))
                return false;
            a.split_doc_num_zn = std::strtoul(v, nullptr, 10);
        } else if (k == "--density_threshold") {
            if (!(v = need(i)))
                return false;
            a.density_threshold = std::strtod(v, nullptr);
        } else if (k == "--seed") {
            if (!(v = need(i)))
                return false;
            a.seed = std::strtoull(v, nullptr, 10);
        } else if (k == "--threads") {
            if (!(v = need(i)))
                return false;
            a.threads = static_cast<unsigned>(std::strtoul(v, nullptr, 10));
        } else {
            std::fprintf(stderr, "unknown arg: %s\n", k.c_str());
            return false;
        }
    }
    if (a.out.empty()) {
        std::fprintf(stderr, "--out is required\n");
        return false;
    }
    if (a.mode == "synthetic" && a.docs == 0) {
        a.docs = 10000;
    }
    if (a.doc_num_per_segment % 16 != 0) {
        std::fprintf(stderr, "--doc_num_per_segment must be a multiple of 16 (bitset u16 unit)\n");
        return false;
    }
    if (a.split_doc_num_zn == 0) {
        // VectorData::CheckData requires split_doc_num_zn <= MAX_L0B_AVAILABLE_MEM_BYTES/dim =
        // 16384/dim (256 for dim=64). Default to the largest such value that still fits a segment.
        uint32_t max_zn = 16384u / (a.dim ? a.dim : 64u);
        if (max_zn == 0)
            max_zn = 1;
        a.split_doc_num_zn = std::min(a.doc_num_per_segment, max_zn);
    }
    return true;
}

// Recursive mkdir -p, so a deep --out works when the parents are absent. EEXIST is ignored.
void MkDir(const std::string& p) {
    std::string cur;
    for (size_t i = 0; i < p.size(); ++i) {
        cur += p[i];
        if (p[i] == '/' || i + 1 == p.size()) {
            if (!cur.empty() && cur != "/")
                ::mkdir(cur.c_str(), 0755);
        }
    }
}

std::string WriteSchemaJson(const Args& a, uint32_t seg_num, uint64_t doc_num) {
    char buf[2048];
    std::snprintf(buf, sizeof(buf),
                  "{\n"
                  "  \"version\": 0,\n"
                  "  \"attachmentInfos\": [ { \"name\": \"docid\", \"isDocid\": true, \"isSdocid\": false } ],\n"
                  "  \"sectionInfos\": [ {\n"
                  "    \"name\": \"content\",\n"
                  "    \"docIndex\": true,\n"
                  "    \"vecIndex\": true,\n"
                  "    \"vectorInfo\": { \"dimension\": %u },\n"
                  "    \"invertedInfo\": { \"isMatrixPosting\": false, \"densityThreshold\": %g }\n"
                  "  } ],\n"
                  "  \"segmentInfo\": {\n"
                  "    \"docNum\": %llu,\n"
                  "    \"segmentNum\": %u,\n"
                  "    \"docNumPerSegment\": %u,\n"
                  "    \"splitDocNumZn\": %u\n"
                  "  }\n"
                  "}\n",
                  a.dim, a.density_threshold, static_cast<unsigned long long>(doc_num), seg_num, a.doc_num_per_segment,
                  a.split_doc_num_zn);
    return std::string(buf);
}

}  // namespace

int main(int argc, char** argv) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    Args a;
    if (!ParseArgs(argc, argv, a))
        return 1;

    npur_port::HwDataset hw;
    const float* hw_vectors = nullptr;
    const uint64_t* hw_bitmaps = nullptr;
    uint32_t hw_stride = 0;
    uint64_t total_docs = 0;

    if (a.mode == "hw") {
        std::string err;
        // `hw` holds the mapping the pointers below point into, so it must outlive the conversion.
        if (!hw.Open(a.dataset, &err)) {
            std::fprintf(stderr, "%s: %s\n", err.c_str(), a.dataset.c_str());
            return 1;
        }
        a.dim = hw.dim;
        a.tag_num = hw.tag_num;
        hw_stride = hw.stride;
        hw_vectors = hw.vectors;
        hw_bitmaps = hw.bitmaps;
        total_docs = hw.doc_num;
    } else if (a.mode == "synthetic") {
        total_docs = a.docs;
    } else {
        std::fprintf(stderr, "unknown --mode %s\n", a.mode.c_str());
        return 1;
    }

    // This shard covers source docs [doc_offset, doc_offset + N). The written global id and the hw
    // source row both use the absolute index, so each shard's DocIdMapping returns absolute global
    // ids while its on-device local space stays [0, N).
    if (a.doc_offset >= total_docs) {
        std::fprintf(stderr, "--doc_offset %llu >= source doc_num %llu, nothing to convert\n",
                     static_cast<unsigned long long>(a.doc_offset), static_cast<unsigned long long>(total_docs));
        return 1;
    }
    uint64_t avail = total_docs - a.doc_offset;
    uint64_t N = (a.docs == 0) ? avail : std::min<uint64_t>(a.docs, avail);
    if (N == 0) {
        std::fprintf(stderr, "no docs to convert\n");
        return 1;
    }
    uint32_t P = a.doc_num_per_segment;
    uint32_t seg_num = static_cast<uint32_t>((N + P - 1) / P);

    MkDir(a.out);
    MkDir(a.out + "/docid");
    MkDir(a.out + "/content");

    // tag_map.bin only proves an id exists in the dictionary; this proves the indexed documents
    // actually contain that tag.
    std::vector<uint64_t> tagDocFreq(a.tag_num, 0);
    std::mutex tagDocFreqMtx;

    std::atomic<bool> failed{false};
    auto processSeg = [&](uint32_t seg) {
        uint64_t seg_begin = static_cast<uint64_t>(seg) * P;
        uint32_t seg_docs = static_cast<uint32_t>(std::min<uint64_t>(P, N - seg_begin));
        std::vector<uint32_t> localTagDocFreq(a.tag_num, 0);

        std::ofstream docidOs(a.out + "/docid/attachment.docid." + std::to_string(seg), std::ios::binary);
        std::ofstream contentOs(a.out + "/content/section.content." + std::to_string(seg), std::ios::binary);
        if (!docidOs || !contentOs) {
            std::fprintf(stderr, "cannot open segment %u output\n", seg);
            failed = true;
            return;
        }
        WriteU32(docidOs, seg_docs);
        WriteU32(contentOs, seg_docs);

        std::mt19937_64 rng(a.seed + static_cast<uint64_t>(seg) * 0x9e3779b97f4a7c15ULL);
        std::uniform_real_distribution<float> vdist(-1.0f, 1.0f);
        std::uniform_int_distribution<uint32_t> tdist(0, a.tag_num ? a.tag_num - 1 : 0);

        for (uint32_t i = 0; i < seg_docs; ++i) {
            uint64_t gdocid = a.doc_offset + seg_begin + i;  // absolute global id == hw source row
            WriteRecord(docidOs, gdocid, std::string());

            Building::Section section;
            Building::FloatEmbedding* emb = section.add_float_embedding();
            std::unordered_set<uint32_t> tags;
            if (a.mode == "hw") {
                const float* v = hw_vectors + gdocid * a.dim;
                for (uint32_t d = 0; d < a.dim; ++d)
                    emb->add_embedding(v[d]);
                const uint64_t* bm = hw_bitmaps + gdocid * hw_stride;
                for (uint32_t w = 0; w < hw_stride; ++w) {
                    uint64_t word = bm[w];
                    while (word) {
                        uint32_t b = static_cast<uint32_t>(__builtin_ctzll(word));
                        uint32_t tag = w * 64u + b;
                        if (tag < a.tag_num)
                            tags.insert(tag);
                        word &= (word - 1);
                    }
                }
            } else {
                for (uint32_t d = 0; d < a.dim; ++d)
                    emb->add_embedding(vdist(rng));
                while (tags.size() < a.tags_per_doc && tags.size() < a.tag_num)
                    tags.insert(tdist(rng));
            }
            for (uint32_t tag : tags) {
                if (tag < localTagDocFreq.size()) {
                    ++localTagDocFreq[tag];
                }
                Building::TermInfo* ti = section.add_terminfos();
                ti->set_index(true);
                ti->set_uint64value(tag);
            }
            std::string pb;
            section.SerializeToString(&pb);
            WriteRecord(contentOs, gdocid, pb);
        }

        {
            std::lock_guard<std::mutex> lk(tagDocFreqMtx);
            for (size_t tag = 0; tag < localTagDocFreq.size(); ++tag) {
                tagDocFreq[tag] += localTagDocFreq[tag];
            }
        }
    };

    unsigned nthreads = a.threads > 0 ? a.threads : std::max(1u, std::thread::hardware_concurrency());
    nthreads = std::min<unsigned>(nthreads, seg_num);

    // Throttled progress (at most one line / ~2s). Disable with NPUR_PROGRESS=0.
    const bool progress = npur_env::OnByDefaultNumeric("NPUR_PROGRESS") && seg_num >= 8;
    auto t0 = std::chrono::steady_clock::now();
    std::mutex printMtx;
    std::chrono::steady_clock::time_point lastPrint = t0;
    std::atomic<uint32_t> completed{0};
    auto reportSeg = [&](uint32_t done) {
        if (!progress)
            return;
        std::lock_guard<std::mutex> lk(printMtx);
        auto now = std::chrono::steady_clock::now();
        if (done == seg_num || now - lastPrint >= std::chrono::seconds(2)) {
            lastPrint = now;
            double sec = std::chrono::duration<double>(now - t0).count();
            std::fprintf(stderr, "[progress] converted %u/%u segments (%.1fs)\n", done, seg_num, sec);
        }
    };

    std::atomic<uint32_t> segIdx{0};
    auto worker = [&] {
        for (;;) {
            uint32_t s = segIdx.fetch_add(1);
            if (s >= seg_num)
                break;
            processSeg(s);
            reportSeg(completed.fetch_add(1) + 1);
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(nthreads);
    for (unsigned k = 0; k < nthreads; ++k)
        pool.emplace_back(worker);
    for (auto& t : pool)
        t.join();
    if (failed) {
        std::fprintf(stderr, "conversion failed\n");
        return 1;
    }
    uint64_t written = N;  // every doc is written

    std::string schema = WriteSchemaJson(a, seg_num, written);
    std::ofstream schemaOs(a.out + "/schema.json");
    schemaOs << schema;
    schemaOs.close();

    uint64_t distinctTagCount = 0;
    uint64_t tagAssignmentCount = 0;
    std::vector<uint32_t> firstZeroDocFreqTags;
    for (size_t tag = 0; tag < tagDocFreq.size(); ++tag) {
        const uint64_t count = tagDocFreq[tag];
        if (count != 0) {
            ++distinctTagCount;
        } else if (firstZeroDocFreqTags.size() < 32) {
            firstZeroDocFreqTags.push_back(static_cast<uint32_t>(tag));
        }
        tagAssignmentCount += count;
    }
    const uint64_t zeroDocFreqTagCount = static_cast<uint64_t>(tagDocFreq.size()) - distinctTagCount;

    {
        std::ofstream freqOs(a.out + "/tag_doc_freq.txt");
        if (!freqOs) {
            std::fprintf(stderr, "cannot open tag_doc_freq.txt for writing under %s\n", a.out.c_str());
            return 1;
        }
        freqOs << "tag_id doc_freq\n";
        for (size_t tag = 0; tag < tagDocFreq.size(); ++tag) {
            freqOs << tag << ' ' << tagDocFreq[tag] << '\n';
        }
        if (!freqOs.good()) {
            std::fprintf(stderr, "failed while writing tag_doc_freq.txt under %s\n", a.out.c_str());
            return 1;
        }
    }

    {
        std::ofstream summaryOs(a.out + "/conversion_summary.txt");
        if (!summaryOs) {
            std::fprintf(stderr, "cannot open conversion_summary.txt for writing under %s\n", a.out.c_str());
            return 1;
        }
        summaryOs << "mode=" << a.mode << '\n';
        summaryOs << "dataset=" << a.dataset << '\n';
        summaryOs << "source_doc_num=" << total_docs << '\n';
        summaryOs << "doc_offset=" << a.doc_offset << '\n';
        summaryOs << "converted_doc_num=" << written << '\n';
        summaryOs << "vector_dim=" << a.dim << '\n';
        summaryOs << "tag_num=" << a.tag_num << '\n';
        summaryOs << "distinct_tag_count=" << distinctTagCount << '\n';
        summaryOs << "zero_doc_freq_tag_count=" << zeroDocFreqTagCount << '\n';
        summaryOs << "tag_assignment_count=" << tagAssignmentCount << '\n';
        summaryOs << "segment_num=" << seg_num << '\n';
        summaryOs << "doc_num_per_segment=" << a.doc_num_per_segment << '\n';
        summaryOs << "split_doc_num_zn=" << a.split_doc_num_zn << '\n';
        summaryOs << "first_zero_doc_freq_tags=";
        for (size_t i = 0; i < firstZeroDocFreqTags.size(); ++i) {
            if (i != 0)
                summaryOs << ',';
            summaryOs << firstZeroDocFreqTags[i];
        }
        summaryOs << '\n';
    }

    std::fprintf(stderr,
                 "done: %llu docs, dim=%u, tag_num=%u, distinct_tags=%llu, zero_doc_freq_tags=%llu, segments=%u (P=%u, "
                 "Zn=%u) -> %s\n",
                 static_cast<unsigned long long>(written), a.dim, a.tag_num,
                 static_cast<unsigned long long>(distinctTagCount),
                 static_cast<unsigned long long>(zeroDocFreqTagCount), seg_num, a.doc_num_per_segment,
                 a.split_doc_num_zn, a.out.c_str());
    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
