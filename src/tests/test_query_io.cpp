// Tests for src/harness/query_io.h — the .fvecs query loader (headered format
// used by hw_queries.fvecs, and per-vector standard fvecs). Writes temp files.
#include "src/harness/query_io.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

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

static std::string tmp(const char* name) {
    return std::string("/tmp/npur_qio_") + name;
}

static void wI32(std::ofstream& o, int32_t v) {
    o.write(reinterpret_cast<const char*>(&v), 4);
}
static void wF32(std::ofstream& o, float v) {
    o.write(reinterpret_cast<const char*>(&v), 4);
}

int main() {
    using npur_port::LoadQueries;

    // ---- headered format: [i32 rows][i32 dim][rows*dim f32] (hw_queries.fvecs) ----
    {
        std::string p = tmp("headered.fvecs");
        std::ofstream o(p, std::ios::binary);
        wI32(o, 3);  // rows
        wI32(o, 4);  // dim
        for (int r = 0; r < 3; ++r)
            for (int d = 0; d < 4; ++d)
                wF32(o, static_cast<float>(r * 10 + d));
        o.close();

        std::vector<std::vector<float>> q;
        std::string err;
        CHECK(LoadQueries(p, q, &err));
        CHECK(q.size() == 3);
        CHECK(q.size() == 3 && q[0].size() == 4);
        CHECK(q.size() == 3 && q[2][3] == 23.0f);  // row2,dim3 = 2*10+3
        CHECK(q.size() == 3 && q[1][0] == 10.0f);
    }

    // ---- per-vector standard fvecs: [i32 dim][dim f32] repeated ----
    {
        std::string p = tmp("pervec.fvecs");
        std::ofstream o(p, std::ios::binary);
        for (int r = 0; r < 2; ++r) {
            wI32(o, 3);  // dim per vector
            for (int d = 0; d < 3; ++d)
                wF32(o, static_cast<float>(r * 100 + d));
        }
        o.close();

        std::vector<std::vector<float>> q;
        CHECK(LoadQueries(p, q));
        CHECK(q.size() == 2);
        CHECK(q.size() == 2 && q[0].size() == 3);
        CHECK(q.size() == 2 && q[1][2] == 102.0f);
    }

    // ---- dim=64 headered (the real case) ----
    {
        std::string p = tmp("dim64.fvecs");
        std::ofstream o(p, std::ios::binary);
        wI32(o, 5);
        wI32(o, 64);
        for (int i = 0; i < 5 * 64; ++i)
            wF32(o, static_cast<float>(i));
        o.close();
        std::vector<std::vector<float>> q;
        CHECK(LoadQueries(p, q));
        CHECK(q.size() == 5 && q[0].size() == 64);
        CHECK(q.size() == 5 && q[4][63] == static_cast<float>(5 * 64 - 1));
    }

    // ---- error cases ----
    {
        std::vector<std::vector<float>> q;
        std::string err;
        CHECK(!LoadQueries("/tmp/npur_qio_does_not_exist_xyz", q, &err));  // missing file
    }
    {
        std::string p = tmp("tiny.fvecs");
        std::ofstream o(p, std::ios::binary);
        wI32(o, 1);  // only 4 bytes -> smaller than 2 int32
        o.close();
        std::vector<std::vector<float>> q;
        CHECK(!LoadQueries(p, q));  // too small
    }
    {
        // truncated headered: header says 3 rows but only 1 present -> parse what's there, no crash
        std::string p = tmp("trunc.fvecs");
        std::ofstream o(p, std::ios::binary);
        wI32(o, 3);
        wI32(o, 4);
        for (int d = 0; d < 4; ++d)
            wF32(o, 1.0f);  // only 1 row of data
        o.close();
        std::vector<std::vector<float>> q;
        // size mismatch -> not detected as headered -> falls back to per-vector parse; must not crash.
        LoadQueries(p, q);
        CHECK(true);  // reached here without crashing
    }

    std::fprintf(stderr, "test_query_io: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
