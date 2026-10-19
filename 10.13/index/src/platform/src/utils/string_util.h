/*
 * Stub for NpuRetrieval `src/utils/string_util.h`.
 *
 * `engine/` uses two helpers (in data_table.cpp / data_table_repository.cpp):
 *   void StringSplit(const std::string& s, char delim, std::vector<std::string>& out);
 *   bool StringToNumber(const std::string& s, T& out);   // template
 */
#pragma once

#include <sstream>
#include <string>
#include <vector>

namespace NpuRetrieval {

inline void StringSplit(const std::string& input, char delim, std::vector<std::string>& out) {
    out.clear();
    std::string token;
    std::istringstream iss(input);
    while (std::getline(iss, token, delim)) {
        out.emplace_back(token);
    }
}

template <typename T>
inline bool StringToNumber(const std::string& input, T& out) {
    std::istringstream iss(input);
    iss >> out;
    return !iss.fail() && iss.eof();
}

}  // namespace NpuRetrieval
