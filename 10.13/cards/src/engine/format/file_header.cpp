#include "file_header.h"
#include "src/utils/logger.h"

namespace NpuRetrieval {
namespace {
// Head layout: [version:1][reserved:5][extendedLen:2][extended:extendedLen].
constexpr std::streamoff kReservedBytes = 5;

// Read exactly sizeof(field) bytes into `field`; log and fail on a short read.
template <typename T>
bool ReadField(std::ifstream& ifs, T& field, const char* what) {
    ifs.read(reinterpret_cast<char*>(&field), sizeof(field));
    if (ifs.gcount() != static_cast<std::streamsize>(sizeof(field))) {
        LOG_ERROR("invalid " << what << " header");
        return false;
    }
    return true;
}
}  // namespace

bool DecodeFileHeader(std::ifstream& ifs, uint8_t& version, std::string& extended) {
    if (!ReadField(ifs, version, "version")) {
        return false;
    }
    // bytes 2-6 of the head are reserved and left unparsed
    ifs.seekg(kReservedBytes, std::ios::cur);
    uint16_t extendedByte = 0;
    if (!ReadField(ifs, extendedByte, "extendedByte")) {
        return false;
    }
    extended.resize(extendedByte);
    ifs.read(extended.data(), extendedByte);
    if (ifs.gcount() != static_cast<std::streamsize>(extendedByte)) {
        LOG_ERROR("invalid extended header");
        return false;
    }
    return true;
}
}  // namespace NpuRetrieval
