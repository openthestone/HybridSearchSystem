#pragma once
#include <string>
namespace NpuRetrieval {
uint16_t Float32ToFloat16(float f);
float Float16ToFloat32(uint16_t h);
uint64_t GetNpuDocNumber(uint64_t docNum);
}  // namespace NpuRetrieval
