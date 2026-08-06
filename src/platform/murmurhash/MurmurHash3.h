//-----------------------------------------------------------------------------
// MurmurHash3 was written by Austin Appleby, and is placed in the public
// domain. The author hereby disclaims copyright to this source code.
//
// Bundled here to satisfy NpuRetrieval common/hash which includes
// "murmurhash/MurmurHash3.h" and calls MurmurHash3_x86_32 / MurmurHash3_x64_128.
//-----------------------------------------------------------------------------
#pragma once

#include <cstdint>

void MurmurHash3_x86_32(const void* key, int len, uint32_t seed, void* out);
void MurmurHash3_x86_128(const void* key, int len, uint32_t seed, void* out);
void MurmurHash3_x64_128(const void* key, int len, uint32_t seed, void* out);
