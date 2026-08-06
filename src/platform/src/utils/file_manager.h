/*
 * Stub for NpuRetrieval `src/utils/file_manager.h`.
 *
 * `engine/` only uses one free function from this header:
 *   bool GetRealFilePath(const std::string& in, std::string& out);
 * It resolves a (possibly relative / symlinked) path to a concrete path that
 * callers then probe with std::filesystem. We back it with weakly_canonical so
 * non-existent-yet paths (build output dirs) still resolve.
 */
#pragma once

#include <filesystem>
#include <string>

namespace NpuRetrieval {

inline bool GetRealFilePath(const std::string& inputPath, std::string& realPath) {
    std::error_code ec;
    std::filesystem::path resolved = std::filesystem::weakly_canonical(inputPath, ec);
    if (ec) {
        // Fall back to the raw path; the caller does its own existence checks.
        realPath = inputPath;
        return true;
    }
    realPath = resolved.string();
    return true;
}

// Create a directory and all missing parents. Used by the builders to lay out
// the index output tree. Returns true if the directory exists afterwards.
inline bool AddDirRecursively(const std::string& dir) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) {
        return false;
    }
    return std::filesystem::is_directory(dir, ec);
}

}  // namespace NpuRetrieval
