/**
 * @file block_manager.hpp
 * @brief Block memory manager for BlockedTriMatrix
 * @author Yongze
 * @date 2025-08-14
 */

#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_set>

#include "block_key.hpp"
#include "eviction_policy.hpp"

namespace tri {
namespace mem {

/**
 * @brief Configuration for AccessCountPolicy
 */
struct AccessCountPolicyConfig {
    bool use_aging = false;       ///< Apply aging to access counts
    double aging_factor = 0.9;    ///< Factor for aging (< 1.0)
    bool use_prediction = false;  ///< Use predicted access counts
};

// Forward declaration
class AccessCountPolicy;

/**
 * @brief Memory manager for block-based matrices
 *
 * This class manages the loading and eviction of blocks to maintain
 * memory constraints. It uses a pluggable eviction policy to determine
 * which blocks to evict when memory limits are reached.
 */
class BlockManager {
   public:
    /**
     * @brief Callback function for loading a block
     * @param key The block to load
     * @return true if load was successful
     */
    using LoadCallback = std::function<bool(const BlockKey&)>;

    /**
     * @brief Callback function for unloading a block
     * @param key The block to unload
     * @return true if unload was successful
     */
    using UnloadCallback = std::function<bool(const BlockKey&)>;

    /**
     * @brief Constructor
     * @param max_blocks Maximum number of blocks to keep in memory
     * @param policy Eviction policy to use
     * @param load_cb Callback for loading blocks
     * @param unload_cb Callback for unloading blocks
     */
    BlockManager(std::size_t max_blocks, std::unique_ptr<EvictionPolicy> policy,
                 LoadCallback load_cb, UnloadCallback unload_cb);

    /**
     * @brief Destructor
     */
    ~BlockManager() = default;

    /**
     * @brief Record access to a block
     * @param key The block being accessed
     *
     * This method updates access statistics and triggers loading
     * if the block is not in memory. It may also trigger eviction
     * if memory limits are exceeded.
     */
    void on_access(const BlockKey& key);

    /**
     * @brief Pin a block to prevent eviction
     * @param key The block to pin
     *
     * Pinned blocks will not be evicted even if memory limits are exceeded.
     */
    void pin(const BlockKey& key);

    /**
     * @brief Unpin a block to allow eviction
     * @param key The block to unpin
     */
    void unpin(const BlockKey& key);

    /**
     * @brief Check if eviction is needed and perform if necessary
     * @return true if eviction was performed
     *
     * This method checks if the current number of loaded blocks exceeds
     * the maximum and evicts blocks according to the policy if needed.
     */
    bool maybe_evict();

    /**
     * @brief Force eviction of a specific block
     * @param key The block to evict
     * @return true if eviction was successful
     */
    bool evict(const BlockKey& key);

    /**
     * @brief Load a specific block
     * @param key The block to load
     * @return true if load was successful
     */
    bool load(const BlockKey& key);

    /**
     * @brief Check if a block is currently loaded
     * @param key The block to check
     * @return true if the block is in memory
     */
    bool is_loaded(const BlockKey& key) const;

    /**
     * @brief Check if a block is pinned
     * @param key The block to check
     * @return true if the block is pinned
     */
    bool is_pinned(const BlockKey& key) const;

    /**
     * @brief Get the current number of loaded blocks
     * @return Number of blocks in memory
     */
    std::size_t loaded_count() const;

    /**
     * @brief Get the maximum number of blocks allowed
     * @return Maximum block count
     */
    std::size_t max_blocks() const { return max_blocks_; }

    /**
     * @brief Get statistics about the manager
     */
    struct Stats {
        std::size_t total_loads = 0;      ///< Total number of loads
        std::size_t total_evictions = 0;  ///< Total number of evictions
        std::size_t cache_hits = 0;       ///< Number of cache hits
        std::size_t cache_misses = 0;     ///< Number of cache misses

        double hit_rate() const {
            auto total = cache_hits + cache_misses;
            return total > 0 ? static_cast<double>(cache_hits) / total : 0.0;
        }
    };

    /**
     * @brief Get current statistics
     * @return Statistics structure
     */
    Stats get_stats() const;

    /**
     * @brief Reset statistics counters
     */
    void reset_stats();

    /**
     * @brief Clear all blocks and reset state
     */
    void clear();

    /**
     * @brief Set whether to use thread-safe operations
     * @param thread_safe true to enable thread safety
     */
    void set_thread_safe(bool thread_safe) { thread_safe_ = thread_safe; }

   private:
    std::size_t max_blocks_;
    std::unique_ptr<EvictionPolicy> policy_;
    LoadCallback load_callback_;
    UnloadCallback unload_callback_;

    // Currently loaded blocks
    std::unordered_set<BlockKey> loaded_blocks_;

    // Statistics
    mutable Stats stats_;

    // Thread safety
    bool thread_safe_ = false;
    mutable std::mutex mutex_;

    // Helper to perform eviction without lock
    bool evict_internal(const BlockKey& key);

    // Helper to perform load without lock
    bool load_internal(const BlockKey& key);
};

/**
 * @brief Create a BlockManager with LRU policy
 */
std::unique_ptr<BlockManager> make_lru_manager(std::size_t max_blocks,
                                               BlockManager::LoadCallback load_cb,
                                               BlockManager::UnloadCallback unload_cb);

/**
 * @brief Create a BlockManager with access count policy
 */
std::unique_ptr<BlockManager> make_access_count_manager(
    std::size_t max_blocks, BlockManager::LoadCallback load_cb,
    BlockManager::UnloadCallback unload_cb,
    const AccessCountPolicyConfig& config = AccessCountPolicyConfig{});

}  // namespace mem
}  // namespace tri