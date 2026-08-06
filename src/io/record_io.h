/*
 * record_io.h — the builder-input "length-delimited record" envelope, shared by
 * the converter (writer) and the tests. It must match exactly what engine/'s
 * NpuRetrieval::ReadAndDoTask (engine/indexer/file/file_reader.cpp) expects:
 *
 *   [u32 count]
 *   repeat count: [u32 len][u64 gdocid][ len-8 bytes payload ]   (len == 8 + payload)
 *
 * Little-endian, native (server is aarch64 LE, matches the reader).
 */
#pragma once

#include <cstdint>
#include <ostream>
#include <string>

namespace npur_port {

inline void WriteU32(std::ostream& os, uint32_t v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

inline void WriteU64(std::ostream& os, uint64_t v) {
    os.write(reinterpret_cast<const char*>(&v), sizeof(v));
}

// One record: len = 8 (gdocid) + payload.size(); [u32 len][u64 gdocid][payload].
inline void WriteRecord(std::ostream& os, uint64_t gdocid, const std::string& payload) {
    uint32_t len = static_cast<uint32_t>(sizeof(uint64_t) + payload.size());
    WriteU32(os, len);
    WriteU64(os, gdocid);
    if (!payload.empty())
        os.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

}  // namespace npur_port
