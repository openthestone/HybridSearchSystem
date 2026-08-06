#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace ProtoParser {

using TermCallback = void (*)(const char* term_data, size_t term_size, void* user_data);

bool ParseFirstFloatEmbedding(const uint8_t* data, size_t size, std::vector<float>& embedding);

bool ForEachTerm(const uint8_t* data, size_t size, TermCallback callback, void* user_data);

}  // namespace ProtoParser
