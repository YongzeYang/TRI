/**
 * @file lru_policy.hpp
 * @brief Least Recently Used (LRU) eviction policy
 * @author Yongze
 * @date 2025-08-19
 */

#pragma once

#include <list>
#include <unordered_map>
#include <unordered_set>

#include "eviction_policy.hpp"

namespace tri {
namespace mem {

/**
 * @brief Least Recently Used (LRU) eviction policy
 *
 * This policy evicts the block that has been accessed least recently.
 * Pinned blocks are never evicted.
 */
class LRUPolicy : public EvictionPolicy {
   public:
    /**
     * @brief Default constructor
     */
    LRUPolicy() = default;

    /**
     * @brief Destructor
     */
    ~LRUPolicy() override = default;

    /**
     * @brief Record access to a block
     * @param key The block identifier
     *
     * Moves the block to the front of the LRU list.
     */
    void touch(const BlockKey& key) override;

    /**
     * @brief Pin a block to prevent eviction
     * @param key The block identifier
     */
    void pin(const BlockKey& key) override;

    /**
     * @brief Unpin a block to allow eviction
     * @param key The block identifier
     */
    void unpin(const BlockKey& key) override;

    /**
     * @brief Select a block for eviction
     * @return LRU block that is not pinned, or nullopt
     */
    std::optional<BlockKey> victim() override;

    /**
     * @brief Reset the policy state
     */
    void reset() override;

    /**
     * @brief Notify that a block has been evicted
     * @param key The block identifier
     */
    void on_evict(const BlockKey& key) override;

    /**
     * @brief Notify that a new block has been loaded
     * @param key The block identifier
     */
    void on_load(const BlockKey& key) override;

    /**
     * @brief Get policy name
     * @return "LRU"
     */
    std::string name() const override { return "LRU"; }

   private:
    // LRU list: front = most recently used, back = least recently used
    std::list<BlockKey> lru_list_;

    // Map from BlockKey to iterator in lru_list for O(1) access
    std::unordered_map<BlockKey, std::list<BlockKey>::iterator> key_to_iter_;

    // Set of pinned blocks
    std::unordered_set<BlockKey> pinned_blocks_;

    // Helper to move a block to the front of LRU list
    void move_to_front(const BlockKey& key);
};

}  // namespace mem
}  // namespace tri