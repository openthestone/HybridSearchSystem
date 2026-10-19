#include "hash.h"
namespace NpuRetrieval {
uint32_t Hash32(const char* data, uint32_t len) {
    uint32_t out = 0;
    uint32_t seed = 0x20200526;
    MurmurHash3_x86_32(data, len, seed, &out);
    return out;
}

uint64_t Hash64(const char* data, uint32_t len) {
    constexpr uint8_t kWordCount = 2u;
    uint64_t out[kWordCount] = {0, 0};
    uint32_t seed = 0x20160713;
    MurmurHash3_x64_128(data, len, seed, out);
    return out[0] ^ out[1];
}
}  // namespace NpuRetrieval
