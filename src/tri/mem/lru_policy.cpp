/**
 * @file lru_policy.cpp
 * @brief LRU eviction policy implementation
 * @author Yongze
 * @date 2025-08-14
 */

#include "tri/mem/lru_policy.hpp"

#include <algorithm>

namespace tri {
namespace mem {

void LRUPolicy::touch(const BlockKey& key) { move_to_front(key); }

void LRUPolicy::pin(const BlockKey& key) { pinned_blocks_.insert(key); }

void LRUPolicy::unpin(const BlockKey& key) { pinned_blocks_.erase(key); }

std::optional<BlockKey> LRUPolicy::victim() {
    // Search from the back (least recently used) for an unpinned block
    for (auto it = lru_list_.rbegin(); it != lru_list_.rend(); ++it) {
        if (pinned_blocks_.find(*it) == pinned_blocks_.end()) {
            return *it;
        }
    }
    return std::nullopt;
}

void LRUPolicy::reset() {
    lru_list_.clear();
    key_to_iter_.clear();
    pinned_blocks_.clear();
}

void LRUPolicy::on_evict(const BlockKey& key) {
    auto it = key_to_iter_.find(key);
    if (it != key_to_iter_.end()) {
        lru_list_.erase(it->second);
        key_to_iter_.erase(it);
    }
    pinned_blocks_.erase(key);
}

void LRUPolicy::on_load(const BlockKey& key) {
    // Add new block to the front (most recently used)
    if (key_to_iter_.find(key) == key_to_iter_.end()) {
        lru_list_.push_front(key);
        key_to_iter_[key] = lru_list_.begin();
    } else {
        move_to_front(key);
    }
}

void LRUPolicy::move_to_front(const BlockKey& key) {
    auto it = key_to_iter_.find(key);
    if (it != key_to_iter_.end()) {
        // Move existing entry to front
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second);
        it->second = lru_list_.begin();
    } else {
        // Add new entry to front
        lru_list_.push_front(key);
        key_to_iter_[key] = lru_list_.begin();
    }
}

}  // namespace mem
}  // namespace tri