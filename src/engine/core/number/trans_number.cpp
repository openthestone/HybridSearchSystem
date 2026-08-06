#include "trans_number.h"
#include <cstdint>
#include <iostream>
#include <limits>
#include <cmath>
#include "huawei_platform/huawei_secure_c/include/securec.h"
#include "src/utils/logger.h"
namespace NpuRetrieval {
namespace {
// float16 layout = 1 sign + 5 exponent + 10 mantissa bits
constexpr uint32_t kF16SignShift = 15;
constexpr uint32_t kF16ExpShift = 10;
// float32 layout = 1 sign + 8 exponent + 23 mantissa bits
constexpr uint32_t kF32SignShift = 31;
constexpr uint32_t kF32ExpShift = 23;
// mantissa width gap between float32 (23) and float16 (10)
constexpr uint32_t kMantissaShiftDelta = kF32ExpShift - kF16ExpShift;
// the NPU processes docs in groups of 32
constexpr uint64_t kNpuDocAlign = 32;

// bit-cast a float32 to its raw uint32 representation and back
uint32_t FloatToBits(float value) {
    uint32_t bits = 0;
    if (memcpy_s(&bits, sizeof(bits), &value, sizeof(value)) != EOK) {
        LOG_ERROR("memcpy_s fail");
    }
    return bits;
}

float BitsToFloat(uint32_t bits) {
    float value = 0.0f;
    if (memcpy_s(&value, sizeof(value), &bits, sizeof(bits)) != EOK) {
        LOG_ERROR("memcpy_s fail");
    }
    return value;
}
}  // namespace

uint16_t Float32ToFloat16(float f) {
    uint32_t bits = FloatToBits(f);

    // split the float32 into its fields
    uint32_t sign = (bits >> kF32SignShift) & 0x1;
    uint32_t exponent = (bits >> kF32ExpShift) & 0xFF;
    uint32_t mantissa = bits & 0x007FFFFF;

    // zero or subnormal -> flush to signed zero
    if (exponent == 0) {
        return static_cast<uint16_t>(sign << kF16SignShift);
    }
    // NaN input: keep part of the payload so it stays a NaN
    if (exponent == 0xFF) {
        uint16_t nanMantissa = (mantissa >> kMantissaShiftDelta) & 0x3FF;
        return static_cast<uint16_t>((sign << kF16SignShift) | 0x7C00 | (nanMantissa != 0 ? nanMantissa : 1));
    }

    int32_t f16Exponent = exponent - 127 + 15;  // rebias (float32 bias 127 -> float16 bias 15)
    if (f16Exponent >= 0x1F) {
        // overflow -> saturate to infinity
        return static_cast<uint16_t>((sign << kF16SignShift) | 0x7C00);
    }
    if (f16Exponent <= 0) {
        // underflow -> would be subnormal, flush to zero
        return static_cast<uint16_t>(sign << kF16SignShift);
    }

    // truncate the mantissa (23 -> 10 bits), then round to nearest via the
    // guard bit (bit 12) and the low sticky bits
    uint32_t f16Mantissa = mantissa >> kMantissaShiftDelta;
    uint32_t roundBit = (mantissa >> 12) & 0x1;
    uint32_t stickyBits = mantissa & 0xFFF;
    if (roundBit && ((f16Mantissa & 0x1) || stickyBits)) {
        f16Mantissa += 1;
        if (f16Mantissa > 0x3FF) {  // mantissa carried out of range
            f16Mantissa = 0;
            f16Exponent += 1;
            if (f16Exponent >= 0x1F) {  // re-check overflow after the carry
                return static_cast<uint16_t>((sign << kF16SignShift) | 0x7C00);
            }
        }
    }

    return static_cast<uint16_t>((sign << kF16SignShift) | (f16Exponent << kF16ExpShift) | f16Mantissa);
}

float Float16ToFloat32(uint16_t h) {
    uint32_t sign = (h >> kF16SignShift) & 0x1;
    uint32_t exponent = (h >> kF16ExpShift) & 0x1F;
    uint32_t mantissa = h & 0x03FF;

    // zero and subnormal both map to signed zero (subnormals are simplified)
    if (exponent == 0) {
        return BitsToFloat(sign << kF32SignShift);
    }
    // infinity or NaN
    if (exponent == 0x1F) {
        uint32_t bits = (mantissa == 0) ? ((sign << kF32SignShift) | 0x7F800000)
                                        : ((sign << kF32SignShift) | 0x7FC00000 | (mantissa << kMantissaShiftDelta));
        return BitsToFloat(bits);
    }

    // normal number: rebias the exponent and widen the mantissa
    uint32_t f32Exponent = exponent - 15 + 127;
    uint32_t f32Mantissa = mantissa << kMantissaShiftDelta;
    uint32_t bits = (sign << kF32SignShift) | (f32Exponent << kF32ExpShift) | f32Mantissa;
    return BitsToFloat(bits);
}

uint64_t GetNpuDocNumber(uint64_t docNum) {
    // round up to a multiple of 32 for the NPU (a little slack is fine)
    return (docNum - 1) / kNpuDocAlign * kNpuDocAlign + kNpuDocAlign;
}
}  // namespace NpuRetrieval
