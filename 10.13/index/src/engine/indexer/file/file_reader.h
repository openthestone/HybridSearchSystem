#pragma once

#include <functional>
#include <string>
#include <memory>
#include "src/full_recall/core/constant_definition.h"

namespace NpuRetrieval {
bool ReadAndDoTask(const std::string& inputPath,
                   std::function<bool(GlobalDocID, uint32_t, uint32_t, const std::string&)> task);
bool ReadAndDoTaskFromMemory(const std::string& fieldName, uint32_t segmentId,
                             std::function<bool(GlobalDocID, uint32_t, uint32_t, const uint8_t*, size_t)> task);
}  // namespace NpuRetrieval
