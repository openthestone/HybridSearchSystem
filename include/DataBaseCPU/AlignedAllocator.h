#pragma once
#include <vector>
#include <memory>
#include <limits>
#include <cstdlib>
#include <new>

template <typename T>
struct AlignedAllocator
{
    using value_type = T;
    // 64字节对齐，适配 AVX-512 和 Cache Line
    static constexpr std::size_t ALIGNMENT = 64;

    AlignedAllocator() = default;

    template <class U>
    constexpr AlignedAllocator(const AlignedAllocator<U> &) noexcept {}

    T *allocate(std::size_t n)
    {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();

        void *ptr = nullptr;

// --- Windows 实现 ---
#if defined(_WIN32) || defined(_WIN64)
        ptr = _aligned_malloc(n * sizeof(T), ALIGNMENT);
        if (!ptr)
            throw std::bad_alloc();

// --- Linux/Unix 实现 ---
#else
        if (posix_memalign(&ptr, ALIGNMENT, n * sizeof(T)) != 0)
        {
            throw std::bad_alloc();
        }
#endif

        return static_cast<T *>(ptr);
    }

    void deallocate(T *p, std::size_t) noexcept
    {
// --- Windows 实现 ---
#if defined(_WIN32) || defined(_WIN64)
        _aligned_free(p);

// --- Linux/Unix 实现 ---
#else
        free(p);
#endif
    }
};

template <typename T, typename U>
bool operator==(const AlignedAllocator<T> &, const AlignedAllocator<U> &) { return true; }

template <typename T, typename U>
bool operator!=(const AlignedAllocator<T> &, const AlignedAllocator<U> &) { return false; }