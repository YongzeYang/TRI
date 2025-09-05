/**
 * @file block_manager.cpp
 * @brief Block memory manager implementation
 * @author Yongze
 * @date 2025-08-14
 */

#include "tri/mem/block_manager.hpp"

#include <stdexcept>

#include "tri/mem/access_count_policy.hpp"
#include "tri/mem/lru_policy.hpp"

namespace tri {
namespace mem {

BlockManager::BlockManager(std::size_t max_blocks, std::unique_ptr<EvictionPolicy> policy,
                           LoadCallback load_cb, UnloadCallback unload_cb)
    : max_blocks_(max_blocks),
      policy_(std::move(policy)),
      load_callback_(std::move(load_cb)),
      unload_callback_(std::move(unload_cb)) {
    if (!policy_) {
        throw std::invalid_argument("EvictionPolicy cannot be null");
    }
    if (!load_callback_) {
        throw std::invalid_argument("Load callback cannot be null");
    }
    if (!unload_callback_) {
        throw std::invalid_argument("Unload callback cannot be null");
    }
    if (max_blocks_ == 0) {
        throw std::invalid_argument("max_blocks must be greater than 0");
    }
}

void BlockManager::on_access(const BlockKey& key) {
    std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
    if (thread_safe_) lock.lock();

    // Check if block is already loaded
    if (loaded_blocks_.find(key) != loaded_blocks_.end()) {
        // Cache hit
        policy_->touch(key);
        stats_.cache_hits++;
    } else {
        // Cache miss - need to load
        stats_.cache_misses++;

        // Check if we need to evict before loading
        while (loaded_blocks_.size() >= max_blocks_) {
            auto victim = policy_->victim();
            if (!victim) {
                // No evictable blocks (all pinned?)
                throw std::runtime_error("Cannot evict any blocks - all blocks may be pinned");
            }

            if (!evict_internal(*victim)) {
                throw std::runtime_error("Failed to evict block");
            }
        }

        // Load the new block
        if (!load_internal(key)) {
            throw std::runtime_error("Failed to load block");
        }
    }
}

void BlockManager::pin(const BlockKey& key) {
    std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
    if (thread_safe_) lock.lock();

    policy_->pin(key);
}

void BlockManager::unpin(const BlockKey& key) {
    std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
    if (thread_safe_) lock.lock();

    policy_->unpin(key);
}

bool BlockManager::maybe_evict() {
    std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
    if (thread_safe_) lock.lock();

    if (loaded_blocks_.size() <= max_blocks_) {
        return false;
    }

    auto victim = policy_->victim();
    if (!victim) {
        return false;
    }

    return evict_internal(*victim);
}

bool BlockManager::evict(const BlockKey& key) {
    std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
    if (thread_safe_) lock.lock();

    return evict_internal(key);
}

bool BlockManager::evict_internal(const BlockKey& key) {
    if (loaded_blocks_.find(key) == loaded_blocks_.end()) {
        return false;  // Not loaded
    }

    if (unload_callback_(key)) {
        loaded_blocks_.erase(key);
        policy_->on_evict(key);
        stats_.total_evictions++;
        return true;
    }

    return false;
}

bool BlockManager::load(const BlockKey& key) {
    std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
    if (thread_safe_) lock.lock();

    return load_internal(key);
}

bool BlockManager::load_internal(const BlockKey& key) {
    if (loaded_blocks_.find(key) != loaded_blocks_.end()) {
        return true;  // Already loaded
    }

    if (load_callback_(key)) {
        loaded_blocks_.insert(key);
        policy_->on_load(key);
        stats_.total_loads++;
        return true;
    }

    return false;
}

bool BlockManager::is_loaded(const BlockKey& key) const {
    std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
    if (thread_safe_) lock.lock();

    return loaded_blocks_.find(key) != loaded_blocks_.end();
}

bool BlockManager::is_pinned(const BlockKey& key) const {
    // This would need to be added to EvictionPolicy interface
    // For now, we can't query this directly
    return false;
}

std::size_t BlockManager::loaded_count() const {
    std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
    if (thread_safe_) lock.lock();

    return loaded_blocks_.size();
}

BlockManager::Stats BlockManager::get_stats() const {
    std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
    if (thread_safe_) lock.lock();

    return stats_;
}

void BlockManager::reset_stats() {
    std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
    if (thread_safe_) lock.lock();

    stats_ = Stats{};
}

void BlockManager::clear() {
    std::unique_lock<std::mutex> lock(mutex_, std::defer_lock);
    if (thread_safe_) lock.lock();

    // Unload all blocks
    for (const auto& key : loaded_blocks_) {
        unload_callback_(key);
    }

    loaded_blocks_.clear();
    policy_->reset();
    stats_ = Stats{};
}

// Factory functions
std::unique_ptr<BlockManager> make_lru_manager(std::size_t max_blocks,
                                               BlockManager::LoadCallback load_cb,
                                               BlockManager::UnloadCallback unload_cb) {
    return std::make_unique<BlockManager>(max_blocks, std::make_unique<LRUPolicy>(),
                                          std::move(load_cb), std::move(unload_cb));
}

std::unique_ptr<BlockManager> make_access_count_manager(std::size_t max_blocks,
                                                        BlockManager::LoadCallback load_cb,
                                                        BlockManager::UnloadCallback unload_cb,
                                                        const AccessCountPolicyConfig& config) {
    return std::make_unique<BlockManager>(max_blocks, std::make_unique<AccessCountPolicy>(config),
                                          std::move(load_cb), std::move(unload_cb));
}

}  // namespace mem
}  // namespace tri