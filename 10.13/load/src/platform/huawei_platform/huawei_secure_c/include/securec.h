/*
 * Stub for Huawei `huaweiSecureC` (securec.h).
 *
 * `engine/` uses memcpy_s (trans_number.cpp) and memset_s / strncpy_s
 * (builder_impl.cpp IPC path). These map to the standard C functions with a
 * bounds check, returning 0 (EOK) on success like the real secure C library.
 *
 * On a real deployment, prefer linking the genuine libsecurec instead of this
 * shim (it performs stricter overlap/parameter validation).
 */
#pragma once

#include <cstring>

#ifndef EOK
#define EOK 0
#endif

typedef int errno_t;

static inline errno_t memcpy_s(void* dest, size_t destMax, const void* src, size_t count) {
    if (dest == nullptr || src == nullptr || count > destMax) {
        return -1;
    }
    std::memcpy(dest, src, count);
    return EOK;
}

static inline errno_t memset_s(void* dest, size_t destMax, int c, size_t count) {
    if (dest == nullptr || count > destMax) {
        return -1;
    }
    std::memset(dest, c, count);
    return EOK;
}

static inline errno_t strncpy_s(char* dest, size_t destMax, const char* src, size_t count) {
    if (dest == nullptr || src == nullptr || destMax == 0) {
        return -1;
    }
    size_t n = (count < destMax - 1) ? count : destMax - 1;
    std::strncpy(dest, src, n);
    dest[n] = '\0';
    return EOK;
}
