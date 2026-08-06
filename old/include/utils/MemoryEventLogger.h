#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <queue>
#include <string>
#include <utility>
#include <vector>

class MemoryEventSession {
   public:
    MemoryEventSession() = default;
    explicit MemoryEventSession(size_t query_id) : query_id_(query_id) {}

    void Reset(size_t query_id) {
        std::lock_guard<std::mutex> lock(mu_);
        query_id_ = query_id;
        entry_count_ = 0;
        overflow_growth_bytes_ = 0;
    }

    size_t query_id() const {
        return query_id_;
    }

    void RecordGrowth(const char* label, size_t growth_bytes) {
        if (label == nullptr || growth_bytes == 0) {
            return;
        }

        std::lock_guard<std::mutex> lock(mu_);
        for (size_t i = 0; i < entry_count_; ++i) {
            if (std::strcmp(entries_[i].label, label) == 0) {
                entries_[i].total_growth_bytes += growth_bytes;
                return;
            }
        }

        if (entry_count_ < entries_.size()) {
            entries_[entry_count_++] = Entry{label, growth_bytes};
            return;
        }

        overflow_growth_bytes_ += growth_bytes;
    }

    void AppendLogLines(std::vector<std::string>& lines) const {
        std::lock_guard<std::mutex> lock(mu_);
        for (size_t i = 0; i < entry_count_; ++i) {
            lines.push_back(FormatLine(query_id_, entries_[i].label, entries_[i].total_growth_bytes));
        }

        if (overflow_growth_bytes_ != 0) {
            lines.push_back(FormatLine(query_id_, "MemoryEventLogger.h:overflow", overflow_growth_bytes_));
        }
    }

   private:
    struct Entry {
        const char* label = nullptr;
        size_t total_growth_bytes = 0;
    };

    static std::string FormatLine(size_t query_id, const char* label, size_t growth_bytes) {
        return "q" + std::to_string(query_id) + " " + label + " +" + std::to_string(growth_bytes) + "B";
    }

    size_t query_id_ = 0;
    mutable std::mutex mu_;
    std::array<Entry, 32> entries_{};
    size_t entry_count_ = 0;
    size_t overflow_growth_bytes_ = 0;
};

inline thread_local MemoryEventSession* g_active_memory_event_session = nullptr;

class ScopedMemoryEventSession {
   public:
    explicit ScopedMemoryEventSession(MemoryEventSession* session) : previous_(g_active_memory_event_session) {
        g_active_memory_event_session = session;
    }

    ~ScopedMemoryEventSession() {
        g_active_memory_event_session = previous_;
    }

    ScopedMemoryEventSession(const ScopedMemoryEventSession&) = delete;
    ScopedMemoryEventSession& operator=(const ScopedMemoryEventSession&) = delete;

   private:
    MemoryEventSession* previous_ = nullptr;
};

inline MemoryEventSession* GetActiveMemoryEventSession() {
    return g_active_memory_event_session;
}

template <typename ValueType>
inline void RecordCapacityGrowth(MemoryEventSession* session, const char* label, size_t old_capacity,
                                 size_t new_capacity) {
    if (session == nullptr || new_capacity <= old_capacity) {
        return;
    }
    session->RecordGrowth(label, (new_capacity - old_capacity) * sizeof(ValueType));
}

template <typename Vec>
inline void TrackVectorReserve(Vec& vec, size_t target_capacity, const char* label) {
    MemoryEventSession* session = GetActiveMemoryEventSession();
    const size_t old_capacity = session == nullptr ? 0 : vec.capacity();
    vec.reserve(target_capacity);
    RecordCapacityGrowth<typename Vec::value_type>(session, label, old_capacity, vec.capacity());
}

template <typename Vec>
inline void TrackVectorResize(Vec& vec, size_t target_size, const char* label) {
    MemoryEventSession* session = GetActiveMemoryEventSession();
    const size_t old_capacity = session == nullptr ? 0 : vec.capacity();
    vec.resize(target_size);
    RecordCapacityGrowth<typename Vec::value_type>(session, label, old_capacity, vec.capacity());
}

template <typename Vec, typename Value>
inline void TrackVectorPushBack(Vec& vec, Value&& value, MemoryEventSession* session, const char* label) {
    const size_t old_capacity = session == nullptr ? 0 : vec.capacity();
    vec.push_back(std::forward<Value>(value));
    RecordCapacityGrowth<typename Vec::value_type>(session, label, old_capacity, vec.capacity());
}

template <typename Vec, typename InputIt>
inline void TrackVectorInsertEnd(Vec& vec, InputIt first, InputIt last, const char* label) {
    MemoryEventSession* session = GetActiveMemoryEventSession();
    const size_t old_capacity = session == nullptr ? 0 : vec.capacity();
    vec.insert(vec.end(), first, last);
    RecordCapacityGrowth<typename Vec::value_type>(session, label, old_capacity, vec.capacity());
}

template <typename T, typename Container, typename Compare>
class InspectablePriorityQueue : public std::priority_queue<T, Container, Compare> {
   public:
    using Base = std::priority_queue<T, Container, Compare>;
    using Base::Base;

    size_t capacity() const {
        return this->c.capacity();
    }
};

template <typename PriorityQueue, typename Value>
inline void TrackPriorityQueuePush(PriorityQueue& pq, Value&& value, MemoryEventSession* session, const char* label) {
    const size_t old_capacity = session == nullptr ? 0 : pq.capacity();
    pq.push(std::forward<Value>(value));
    RecordCapacityGrowth<typename PriorityQueue::value_type>(session, label, old_capacity, pq.capacity());
}

class MemoryEventLogCollector {
   public:
    explicit MemoryEventLogCollector(bool enabled) : enabled_(enabled) {}

    void Reserve(size_t expected_line_count) {
        if (!enabled_) {
            return;
        }
        lines_.reserve(expected_line_count);
    }

    void AppendSession(const MemoryEventSession* session) {
        if (!enabled_ || session == nullptr) {
            return;
        }
        session->AppendLogLines(lines_);
    }

    bool Flush(const std::filesystem::path& path) const {
        if (!enabled_) {
            return true;
        }

        std::ofstream out(path, std::ios::out | std::ios::trunc);
        if (!out.is_open()) {
            return false;
        }

        for (const auto& line : lines_) {
            out << line << '\n';
        }
        return out.good();
    }

   private:
    bool enabled_ = false;
    std::vector<std::string> lines_;
};
