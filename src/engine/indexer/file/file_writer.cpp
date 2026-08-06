#include "file_writer.h"
#include "src/utils/logger.h"
namespace NpuRetrieval {
bool WriteHeader(std::ofstream& ofs, uint8_t version, std::string& extended) {
    if (ofs.fail()) {
        LOG_ERROR("ofstream is invalid");
        return false;
    }
    // write version
    if (!ofs.write(reinterpret_cast<const char*>(&version), sizeof(version))) {
        LOG_ERROR("write version fail");
        return false;
    }
    // the file head reserves bytes 2-6, filled with 0 by default
    std::vector<uint8_t> reserved(5, 0);  // reserved field length is 5 bytes
    if (!ofs.write(reinterpret_cast<const char*>(reserved.data()), reserved.size() * sizeof(uint8_t))) {
        LOG_ERROR("write reserved fail");
        return false;
    }
    // write extended
    if (!ofs.write(extended.c_str(), extended.length())) {
        LOG_ERROR("write extended fail");
        return false;
    }
    return true;
}
}  // namespace NpuRetrieval
