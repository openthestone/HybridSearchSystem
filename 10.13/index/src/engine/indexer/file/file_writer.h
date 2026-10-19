#pragma once
#include <string>
#include <fstream>

namespace NpuRetrieval {
bool WriteHeader(std::ofstream& ofs, uint8_t version, std::string& extended);
}
