#pragma once
#include <fstream>
namespace NpuRetrieval {
bool DecodeFileHeader(std::ifstream& ifs, uint8_t& version, std::string& extended);
}
