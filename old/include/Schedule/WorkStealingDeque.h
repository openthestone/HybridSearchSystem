// Lock-free work-stealing deque (Chase-Lev) for fine-grained load balancing
// Stores values inline (not pointers) for lifetime safety.
// Owner calls PushBottom/PopBottom; any thread may call Steal.

#pragma once
#include <atomic>
#include <cstddef>
#include <cstdlib>

template <typename T>
class WorkStealingDeque {
    // Non-copyable, non-movable (contains atomics + raw pointer)
    WorkStealingDeque(const WorkStealingDeque&) = delete;
    WorkStealingDeque& operator=(const WorkStealingDeque&) = delete;
    WorkStealingDeque(WorkStealingDeque&&) = delete;
    WorkStealingDeque& operator=(WorkStealingDeque&&) = delete;

   public:
    explicit WorkStealingDeque(size_t capacity) : capacity_(NextPow2(capacity)) {
        buffer_ = static_cast<T*>(std::calloc(capacity_, sizeof(T)));
        if (!buffer_)
            throw std::bad_alloc();
        top_.store(0, std::memory_order_relaxed);
        bottom_.store(0, std::memory_order_relaxed);
    }

    ~WorkStealingDeque() {
        std::free(buffer_);
    }

    // Re-initialise to empty state for reuse (avoids free/alloc cycle)
    void Reset() {
        // Use seq_cst to ensure all threads see the reset immediately.
        // Relaxed stores can be delayed indefinitely on weakly-ordered ARM.
        bottom_.store(0, std::memory_order_seq_cst);
        top_.store(0, std::memory_order_seq_cst);
    }

    bool PushBottom(const T& item) {
        size_t b = bottom_.load(std::memory_order_acquire);
        size_t t = top_.load(std::memory_order_acquire);
        if (b - t >= capacity_)
            return false;  // full
        buffer_[b & (capacity_ - 1)] = item;
        bottom_.store(b + 1, std::memory_order_release);
        return true;
    }

    bool PopBottom(T& result) {
        size_t b = bottom_.load(std::memory_order_relaxed);
        // Guard: if deque is already empty, return false immediately.
        // Without this, b-1 wraps to SIZE_MAX on unsigned, breaking the t>b check.
        size_t t = top_.load(std::memory_order_acquire);
        if (b <= t)
            return false;

        b = b - 1;
        bottom_.store(b, std::memory_order_relaxed);  // speculative decrement
        std::atomic_thread_fence(std::memory_order_seq_cst);

        t = top_.load(std::memory_order_acquire);
        if (t > b) {  // was already empty (race with Steal)
            bottom_.store(b + 1, std::memory_order_release);
            return false;
        }
        result = buffer_[b & (capacity_ - 1)];
        if (t == b) {  // last element – race with Steal
            // CAS ensures exactly one of PopBottom/Steal gets the item
            if (!top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed)) {
                bottom_.store(b + 1, std::memory_order_release);
                return false;  // thief won
            }
            bottom_.store(b + 1, std::memory_order_release);
        }
        return true;
    }

    // Debug: get raw top and bottom values
    std::pair<size_t, size_t> DebugState() const {
        return {top_.load(std::memory_order_relaxed), bottom_.load(std::memory_order_relaxed)};
    }

    bool Steal(T& result) {
        size_t t = top_.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        size_t b = bottom_.load(std::memory_order_acquire);
        if (t >= b)
            return false;  // empty
        result = buffer_[t & (capacity_ - 1)];
        return top_.compare_exchange_strong(t, t + 1, std::memory_order_seq_cst, std::memory_order_relaxed);
    }

    size_t Size() const {
        size_t b = bottom_.load(std::memory_order_relaxed);
        size_t t = top_.load(std::memory_order_relaxed);
        return (b > t) ? (b - t) : 0;
    }

    bool Empty() const {
        return Size() == 0;
    }

   private:
    static size_t NextPow2(size_t n) {
        if (n == 0)
            return 1;
        --n;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        n |= n >> 32;
        return n + 1;
    }

    T* buffer_;
    size_t capacity_;
    alignas(64) std::atomic<size_t> top_;
    alignas(64) std::atomic<size_t> bottom_;
};
