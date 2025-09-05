/**
 * @file access_count_policy.hpp
 * @brief Access count based eviction policy
 * @author Yongze
 * @date 2025-08-14
 */

#pragma once

#include <unordered_map>
#include <unordered_set>

#include "block_manager.hpp"
#include "eviction_policy.hpp"

namespace tri {
namespace mem {

/**
 * @brief Access count based eviction policy
 *
 * This policy evicts the block with the lowest access count.
 * It can be configured to use either pure frequency (LFU) or
 * a combination with predicted access patterns.
 */
class AccessCountPolicy : public EvictionPolicy {
   public:
    /**
     * @brief Configuration for the policy
     */
    struct Config {
        bool use_aging;       ///< Apply aging to access counts
        double aging_factor;  ///< Factor for aging (< 1.0)
        bool use_prediction;  ///< Use predicted access counts

        // Default constructor
        Config() : use_aging(false), aging_factor(0.9), use_prediction(false) {}
    };

    /**
     * @brief Default constructor
     */
    AccessCountPolicy();

    /**
     * @brief Constructor with external configuration
     * @param config External policy configuration
     */
    explicit AccessCountPolicy(const AccessCountPolicyConfig& config);

    /**
     * @brief Constructor with configuration
     * @param config Policy configuration
     */
    explicit AccessCountPolicy(const Config& config);

    /**
     * @brief Destructor
     */
    ~AccessCountPolicy() override = default;

    /**
     * @brief Record access to a block
     * @param key The block identifier
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
     * @return Block with lowest access count that is not pinned
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
     * @return "AccessCount" or "AccessCount+Prediction"
     */
    std::string name() const override;

    /**
     * @brief Set predicted access counts for blocks
     * @param predictions Map from BlockKey to predicted access count
     *
     * This can be used to incorporate precomputed access patterns.
     */
    void set_predictions(const std::unordered_map<BlockKey, double>& predictions);

    /**
     * @brief Apply aging to all access counts
     *
     * Multiplies all counts by the aging factor to give preference
     * to more recent accesses.
     */
    void apply_aging();

   private:
    Config config_;

    // Actual access counts
    std::unordered_map<BlockKey, double> access_counts_;

    // Predicted access counts (optional)
    std::unordered_map<BlockKey, double> predicted_counts_;

    // Set of pinned blocks
    std::unordered_set<BlockKey> pinned_blocks_;

    // Get effective count (actual + predicted if enabled)
    double get_effective_count(const BlockKey& key) const;
};

}  // namespace mem
}  // namespace tri