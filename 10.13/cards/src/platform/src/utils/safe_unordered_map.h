/*
 * Stub for NpuRetrieval `src/utils/safe_unordered_map.h`.
 *
 * A thread-safe unordered_map. `engine/` uses it in gm_memory_manager.h with:
 *   bool Find(const K& key, V& out);   // true + fills out when present
 *   void Insert(const K& key, const V& value);
 * Backed by a shared_mutex (concurrent reads, exclusive writes).
 */
#pragma once

#include <shared_mutex>
#include <unordered_map>

namespace NpuRetrieval {

template <typename K, typename V>
class SafeUnorderedMap {
   public:
    bool Find(const K& key, V& out) const {
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        auto it = m_map.find(key);
        if (it == m_map.end()) {
            return false;
        }
        out = it->second;
        return true;
    }

    void Insert(const K& key, const V& value) {
        std::unique_lock<std::shared_mutex> lock(m_mutex);
        m_map[key] = value;
    }

    bool Erase(const K& key) {
        std::unique_lock<std::shared_mutex> lock(m_mutex);
        return m_map.erase(key) > 0;
    }

    size_t Size() const {
        std::shared_lock<std::shared_mutex> lock(m_mutex);
        return m_map.size();
    }

   private:
    mutable std::shared_mutex m_mutex;
    std::unordered_map<K, V> m_map;
};

}  // namespace NpuRetrieval
