/**
 * @file access_count_policy.cpp
 * @brief Access count based eviction policy implementation
 * @author Yongze
 * @date 2025-08-14
 */

#include "tri/mem/access_count_policy.hpp"

#include <algorithm>
#include <limits>

namespace tri {
namespace mem {

AccessCountPolicy::AccessCountPolicy() : config_() {}

AccessCountPolicy::AccessCountPolicy(const AccessCountPolicyConfig& ext_config) : config_() {
    // Convert external config to internal config
    config_.use_aging = ext_config.use_aging;
    config_.aging_factor = ext_config.aging_factor;
    config_.use_prediction = ext_config.use_prediction;
}

AccessCountPolicy::AccessCountPolicy(const Config& config) : config_(config) {}

void AccessCountPolicy::touch(const BlockKey& key) { access_counts_[key] += 1.0; }

void AccessCountPolicy::pin(const BlockKey& key) { pinned_blocks_.insert(key); }

void AccessCountPolicy::unpin(const BlockKey& key) { pinned_blocks_.erase(key); }

std::optional<BlockKey> AccessCountPolicy::victim() {
    std::optional<BlockKey> victim_key;
    double min_count = std::numeric_limits<double>::max();

    for (const auto& [key, count] : access_counts_) {
        // Skip pinned blocks
        if (pinned_blocks_.find(key) != pinned_blocks_.end()) {
            continue;
        }

        double effective_count = get_effective_count(key);
        if (effective_count < min_count) {
            min_count = effective_count;
            victim_key = key;
        }
    }

    return victim_key;
}

void AccessCountPolicy::reset() {
    access_counts_.clear();
    predicted_counts_.clear();
    pinned_blocks_.clear();
}

void AccessCountPolicy::on_evict(const BlockKey& key) {
    access_counts_.erase(key);
    pinned_blocks_.erase(key);
}

void AccessCountPolicy::on_load(const BlockKey& key) {
    // Initialize with zero count if not present
    if (access_counts_.find(key) == access_counts_.end()) {
        access_counts_[key] = 0.0;
    }
}

std::string AccessCountPolicy::name() const {
    if (config_.use_prediction) {
        return "AccessCount+Prediction";
    }
    return "AccessCount";
}

void AccessCountPolicy::set_predictions(const std::unordered_map<BlockKey, double>& predictions) {
    predicted_counts_ = predictions;
}

void AccessCountPolicy::apply_aging() {
    if (config_.use_aging) {
        for (auto& [key, count] : access_counts_) {
            count *= config_.aging_factor;
        }
    }
}

double AccessCountPolicy::get_effective_count(const BlockKey& key) const {
    double count = 0.0;

    auto it = access_counts_.find(key);
    if (it != access_counts_.end()) {
        count = it->second;
    }

    if (config_.use_prediction) {
        auto pred_it = predicted_counts_.find(key);
        if (pred_it != predicted_counts_.end()) {
            count += pred_it->second;
        }
    }

    return count;
}

}  // namespace mem
}  // namespace tri