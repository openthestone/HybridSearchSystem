/*
 * query_io.h — load query vectors from a .fvecs file. Shared by the harness and
 * its tests. Mirrors old/ LoadPreparedQueriesFromFvec: supports the headered
 * format ([i32 rows][i32 dim][rows*dim float32], used by hw_queries.fvecs) and
 * the per-vector standard fvecs format ([i32 dim][dim float32] repeated).
 */
#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace npur_port {

// Loads query vectors into `out`. Returns false if the file can't be opened or
// yields no vectors. `err` (optional) receives a short reason on failure.
inline bool LoadQueries(const std::string& path, std::vector<std::vector<float>>& out, std::string* err = nullptr) {
    out.clear();
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        if (err)
            *err = "cannot open query file: " + path;
        return false;
    }
    in.seekg(0, std::ios::end);
    const std::streamoff fileSize = in.tellg();
    in.seekg(0, std::ios::beg);
    if (fileSize < static_cast<std::streamoff>(sizeof(int32_t) * 2)) {
        if (err)
            *err = "query file too small";
        return false;
    }

    int32_t first = 0, second = 0;
    in.read(reinterpret_cast<char*>(&first), sizeof(int32_t));
    in.read(reinterpret_cast<char*>(&second), sizeof(int32_t));
    const std::streamoff headered =
        static_cast<std::streamoff>(sizeof(int32_t) * 2) +
        static_cast<std::streamoff>(first) * second * static_cast<std::streamoff>(sizeof(float));
    if (first > 0 && second >= 1 && second <= 4096 && fileSize == headered) {
        // headered: first = rows, second = dim
        out.reserve(static_cast<size_t>(first));
        for (int32_t r = 0; r < first; ++r) {
            std::vector<float> v(static_cast<size_t>(second));
            in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
            if (!in)
                break;
            out.emplace_back(std::move(v));
        }
        if (out.empty() && err)
            *err = "headered file produced no rows";
        return !out.empty();
    }

    // fall back to per-vector fvecs
    in.clear();
    in.seekg(0, std::ios::beg);
    while (true) {
        int32_t dim = 0;
        in.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
        if (!in || dim <= 0 || dim > 4096)
            break;
        std::vector<float> v(static_cast<size_t>(dim));
        in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
        if (!in)
            break;
        out.emplace_back(std::move(v));
    }
    if (out.empty() && err)
        *err = "no vectors parsed";
    return !out.empty();
}

}  // namespace npur_port
