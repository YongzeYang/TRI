/**
 * @file eviction_policy.hpp
 * @brief Abstract base class for block eviction policies
 * @author Yongze
 * @date 2025-08-19
 */

#pragma once

#include <memory>
#include <optional>

#include "block_key.hpp"

namespace tri {
namespace mem {

/**
 * @brief Abstract base class for block eviction policies
 *
 * This class defines the interface for different eviction strategies
 * used by BlockManager to determine which blocks to evict when memory
 * limits are reached.
 */
class EvictionPolicy {
   public:
    /**
     * @brief Virtual destructor
     */
    virtual ~EvictionPolicy() = default;

    /**
     * @brief Record access to a block
     * @param key The block identifier
     *
     * This method should update internal statistics to track block usage.
     */
    virtual void touch(const BlockKey& key) = 0;

    /**
     * @brief Pin a block to prevent eviction
     * @param key The block identifier
     *
     * Pinned blocks should never be returned by victim().
     */
    virtual void pin(const BlockKey& key) = 0;

    /**
     * @brief Unpin a block to allow eviction
     * @param key The block identifier
     *
     * After unpinning, the block becomes eligible for eviction again.
     */
    virtual void unpin(const BlockKey& key) = 0;

    /**
     * @brief Select a block for eviction
     * @return Block to evict, or nullopt if no block can be evicted
     *
     * This method should return a block that is not pinned and is
     * suitable for eviction according to the policy's strategy.
     */
    virtual std::optional<BlockKey> victim() = 0;

    /**
     * @brief Reset the policy state
     *
     * Clear all internal statistics and tracking data.
     */
    virtual void reset() = 0;

    /**
     * @brief Notify that a block has been evicted
     * @param key The block identifier
     *
     * This allows the policy to update its internal state.
     */
    virtual void on_evict(const BlockKey& key) = 0;

    /**
     * @brief Notify that a new block has been loaded
     * @param key The block identifier
     *
     * This allows the policy to track newly loaded blocks.
     */
    virtual void on_load(const BlockKey& key) = 0;

    /**
     * @brief Get policy name for debugging
     * @return Name of the eviction policy
     */
    virtual std::string name() const = 0;
};

/**
 * @brief Factory function type for creating eviction policies
 */
using EvictionPolicyFactory = std::function<std::unique_ptr<EvictionPolicy>()>;

}  // namespace mem
}  // namespace tri