#pragma once
#include <string>

namespace NpuRetrieval {
// docid types
const std::string GDOCID_TYPE = "0";
const std::string SDOCID_TYPE = "1";
using GlobalDocID = uint64_t;
// posting (inverted list) types
const std::string MATRIX_TYPE = "0";
const std::string BITMAP_TYPE = "1";
// builder input field names
const std::string SECTION_FIELD = "section.";
const std::string ATTACHMENT_FIELD = "attachment.";
const std::string DOT = ".";
// builder output (index) names
const std::string POISSONENGINE = "poissonengine";
const std::string UNDERLINE = "_";
const std::string ID_MAPPING_DIR = "id";
const std::string VECTOR_FILE_SUFFIX = ".vector";
const std::string ID_MAPPING_FILE_SUFFIX = ".idm";
const std::string INVERTED_FILE_SUFFIX = ".posting";
const std::string META_FILE_SUFFIX = ".meta";
}  // namespace NpuRetrieval
