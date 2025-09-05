/**
 * @file test_eviction_policy.cpp
 * @brief Unit tests for eviction policies
 * @author Yongze
 * @date 2025-08-14
 */

#include <memory>
#include <vector>

#include "framework/test_assertions.hpp"
#include "framework/test_runner.hpp"
#include "tri/mem/access_count_policy.hpp"
#include "tri/mem/lru_policy.hpp"

namespace tri::test {

// Test LRU Policy
class LRUPolicyTest : public TestCase {
   protected:
    std::unique_ptr<mem::LRUPolicy> policy_;

    void SetUp() override { policy_ = std::make_unique<mem::LRUPolicy>(); }
};

TEST_F(LRUPolicyTest, BasicLRU) {
    // Load blocks in order
    policy_->on_load(mem::BlockKey(0, 0));
    policy_->on_load(mem::BlockKey(1, 0));
    policy_->on_load(mem::BlockKey(2, 0));

    // Block 0,0 should be LRU
    auto victim = policy_->victim();
    ASSERT_TRUE(victim.has_value());
    ASSERT_EQ(mem::BlockKey(0, 0), *victim);
}

TEST_F(LRUPolicyTest, TouchUpdatesOrder) {
    policy_->on_load(mem::BlockKey(0, 0));
    policy_->on_load(mem::BlockKey(1, 0));
    policy_->on_load(mem::BlockKey(2, 0));

    // Touch block 0,0 to make it MRU
    policy_->touch(mem::BlockKey(0, 0));

    // Now block 1,0 should be LRU
    auto victim = policy_->victim();
    ASSERT_TRUE(victim.has_value());
    ASSERT_EQ(mem::BlockKey(1, 0), *victim);
}

TEST_F(LRUPolicyTest, PinnedBlocksNotEvicted) {
    policy_->on_load(mem::BlockKey(0, 0));
    policy_->on_load(mem::BlockKey(1, 0));
    policy_->on_load(mem::BlockKey(2, 0));

    // Pin the LRU block
    policy_->pin(mem::BlockKey(0, 0));

    // Should return next LRU (1,0)
    auto victim = policy_->victim();
    ASSERT_TRUE(victim.has_value());
    ASSERT_EQ(mem::BlockKey(1, 0), *victim);
}

TEST_F(LRUPolicyTest, UnpinAllowsEviction) {
    policy_->on_load(mem::BlockKey(0, 0));
    policy_->on_load(mem::BlockKey(1, 0));

    policy_->pin(mem::BlockKey(0, 0));
    policy_->pin(mem::BlockKey(1, 0));

    // All blocks pinned - no victim
    ASSERT_FALSE(policy_->victim().has_value());

    // Unpin one
    policy_->unpin(mem::BlockKey(0, 0));

    // Now we should have a victim
    auto victim = policy_->victim();
    ASSERT_TRUE(victim.has_value());
    ASSERT_EQ(mem::BlockKey(0, 0), *victim);
}

TEST_F(LRUPolicyTest, EvictRemovesFromList) {
    policy_->on_load(mem::BlockKey(0, 0));
    policy_->on_load(mem::BlockKey(1, 0));
    policy_->on_load(mem::BlockKey(2, 0));

    policy_->on_evict(mem::BlockKey(1, 0));

    // After evicting 1,0, the LRU should be 0,0
    auto victim = policy_->victim();
    ASSERT_TRUE(victim.has_value());
    ASSERT_EQ(mem::BlockKey(0, 0), *victim);
}

TEST_F(LRUPolicyTest, Reset) {
    policy_->on_load(mem::BlockKey(0, 0));
    policy_->on_load(mem::BlockKey(1, 0));
    policy_->pin(mem::BlockKey(0, 0));

    policy_->reset();

    // After reset, no victims available
    ASSERT_FALSE(policy_->victim().has_value());
}

TEST_F(LRUPolicyTest, PolicyName) { ASSERT_EQ("LRU", policy_->name()); }

// Test Access Count Policy
class AccessCountPolicyTest : public TestCase {
   protected:
    std::unique_ptr<mem::AccessCountPolicy> policy_;

    void SetUp() override { policy_ = std::make_unique<mem::AccessCountPolicy>(); }
};

TEST_F(AccessCountPolicyTest, BasicAccessCount) {
    policy_->on_load(mem::BlockKey(0, 0));
    policy_->on_load(mem::BlockKey(1, 0));
    policy_->on_load(mem::BlockKey(2, 0));

    // Touch blocks different number of times
    policy_->touch(mem::BlockKey(1, 0));
    policy_->touch(mem::BlockKey(1, 0));
    policy_->touch(mem::BlockKey(2, 0));

    // Block 0,0 has 0 touches, should be victim
    auto victim = policy_->victim();
    ASSERT_TRUE(victim.has_value());
    ASSERT_EQ(mem::BlockKey(0, 0), *victim);
}

TEST_F(AccessCountPolicyTest, TieBreaking) {
    policy_->on_load(mem::BlockKey(0, 0));
    policy_->on_load(mem::BlockKey(1, 0));
    policy_->on_load(mem::BlockKey(2, 0));

    // All have same count (0)
    auto victim = policy_->victim();
    ASSERT_TRUE(victim.has_value());
    // Any of them could be victim, just check we got one
    ASSERT_TRUE(*victim == mem::BlockKey(0, 0) || *victim == mem::BlockKey(1, 0) ||
                *victim == mem::BlockKey(2, 0));
}

TEST_F(AccessCountPolicyTest, PinnedNotEvicted) {
    policy_->on_load(mem::BlockKey(0, 0));
    policy_->on_load(mem::BlockKey(1, 0));

    // Pin the one with lowest count
    policy_->pin(mem::BlockKey(0, 0));
    policy_->touch(mem::BlockKey(1, 0));

    // Should return 1,0 even though it has higher count
    auto victim = policy_->victim();
    ASSERT_TRUE(victim.has_value());
    ASSERT_EQ(mem::BlockKey(1, 0), *victim);
}

TEST_F(AccessCountPolicyTest, WithAging) {
    mem::AccessCountPolicy::Config config;
    config.use_aging = true;
    config.aging_factor = 0.5;

    auto aging_policy = std::make_unique<mem::AccessCountPolicy>(config);

    aging_policy->on_load(mem::BlockKey(0, 0));
    aging_policy->on_load(mem::BlockKey(1, 0));

    // Touch first block many times
    for (int i = 0; i < 10; ++i) {
        aging_policy->touch(mem::BlockKey(0, 0));
    }

    // Apply aging
    aging_policy->apply_aging();

    // Touch second block fewer times
    for (int i = 0; i < 3; ++i) {
        aging_policy->touch(mem::BlockKey(1, 0));
    }

    // After aging, block 0 has count ~5, block 1 has count 3
    // Block 1 should be victim
    auto victim = aging_policy->victim();
    ASSERT_TRUE(victim.has_value());
    ASSERT_EQ(mem::BlockKey(1, 0), *victim);
}

TEST_F(AccessCountPolicyTest, WithPrediction) {
    mem::AccessCountPolicy::Config config;
    config.use_prediction = true;

    auto pred_policy = std::make_unique<mem::AccessCountPolicy>(config);

    pred_policy->on_load(mem::BlockKey(0, 0));
    pred_policy->on_load(mem::BlockKey(1, 0));

    // Set predictions
    std::unordered_map<mem::BlockKey, double> predictions;
    predictions[mem::BlockKey(0, 0)] = 10.0;  // High predicted access
    predictions[mem::BlockKey(1, 0)] = 1.0;   // Low predicted access
    pred_policy->set_predictions(predictions);

    // Even without actual accesses, block 1 should be victim due to lower prediction
    auto victim = pred_policy->victim();
    ASSERT_TRUE(victim.has_value());
    ASSERT_EQ(mem::BlockKey(1, 0), *victim);
}

TEST_F(AccessCountPolicyTest, PolicyName) {
    ASSERT_EQ("AccessCount", policy_->name());

    mem::AccessCountPolicy::Config config;
    config.use_prediction = true;
    auto pred_policy = std::make_unique<mem::AccessCountPolicy>(config);
    ASSERT_EQ("AccessCount+Prediction", pred_policy->name());
}

// Test multiple policies with same interface
TEST(EvictionPolicy, PolymorphicUsage) {
    std::vector<std::unique_ptr<mem::EvictionPolicy>> policies;
    policies.push_back(std::make_unique<mem::LRUPolicy>());
    policies.push_back(std::make_unique<mem::AccessCountPolicy>());

    for (auto& policy : policies) {
        policy->on_load(mem::BlockKey(0, 0));
        policy->on_load(mem::BlockKey(1, 0));
        policy->touch(mem::BlockKey(1, 0));

        auto victim = policy->victim();
        ASSERT_TRUE(victim.has_value());

        policy->reset();
        ASSERT_FALSE(policy->victim().has_value());
    }
}

}  // namespace tri::test