/**
 * @file test_block_manager.cpp
 * @brief Unit tests for BlockManager
 * @author Yongze
 * @date 2025-08-14
 */

#include <chrono>
#include <random>
#include <thread>
#include <unordered_set>

#include "framework/test_assertions.hpp"
#include "framework/test_runner.hpp"
#include "tri/mem/access_count_policy.hpp"
#include "tri/mem/block_manager.hpp"
#include "tri/mem/lru_policy.hpp"

namespace tri::test {

class BlockManagerTest : public TestCase {
   protected:
    std::unordered_set<mem::BlockKey> loaded_blocks_;
    int load_count_ = 0;
    int unload_count_ = 0;
    bool fail_next_load_ = false;
    bool fail_next_unload_ = false;

    mem::BlockManager::LoadCallback make_load_callback() {
        return [this](const mem::BlockKey& key) {
            if (fail_next_load_) {
                fail_next_load_ = false;
                return false;
            }
            loaded_blocks_.insert(key);
            load_count_++;
            return true;
        };
    }

    mem::BlockManager::UnloadCallback make_unload_callback() {
        return [this](const mem::BlockKey& key) {
            if (fail_next_unload_) {
                fail_next_unload_ = false;
                return false;
            }
            loaded_blocks_.erase(key);
            unload_count_++;
            return true;
        };
    }

    void SetUp() override {
        loaded_blocks_.clear();
        load_count_ = 0;
        unload_count_ = 0;
        fail_next_load_ = false;
        fail_next_unload_ = false;
    }
};

TEST_F(BlockManagerTest, Construction) {
    auto manager = mem::make_lru_manager(10, make_load_callback(), make_unload_callback());

    ASSERT_EQ(10u, manager->max_blocks());
    ASSERT_EQ(0u, manager->loaded_count());
}

TEST_F(BlockManagerTest, BasicLoadAndAccess) {
    auto manager = mem::make_lru_manager(3, make_load_callback(), make_unload_callback());

    // First access loads the block
    manager->on_access(mem::BlockKey(0, 0));
    ASSERT_EQ(1u, manager->loaded_count());
    ASSERT_EQ(1, load_count_);
    ASSERT_TRUE(manager->is_loaded(mem::BlockKey(0, 0)));

    // Second access is a cache hit
    manager->on_access(mem::BlockKey(0, 0));
    ASSERT_EQ(1u, manager->loaded_count());
    ASSERT_EQ(1, load_count_);  // No additional load

    auto stats = manager->get_stats();
    ASSERT_EQ(1u, stats.cache_misses);
    ASSERT_EQ(1u, stats.cache_hits);
}

TEST_F(BlockManagerTest, EvictionOnCapacity) {
    const std::size_t max_blocks = 3;
    auto manager = mem::make_lru_manager(max_blocks, make_load_callback(), make_unload_callback());

    // Load max_blocks
    for (std::size_t i = 0; i < max_blocks; ++i) {
        manager->on_access(mem::BlockKey(i, 0));
    }

    ASSERT_EQ(max_blocks, manager->loaded_count());
    ASSERT_EQ(0, unload_count_);

    // Load one more - should trigger eviction
    manager->on_access(mem::BlockKey(max_blocks, 0));

    ASSERT_EQ(max_blocks, manager->loaded_count());
    ASSERT_EQ(1, unload_count_);
    ASSERT_TRUE(manager->is_loaded(mem::BlockKey(max_blocks, 0)));
    ASSERT_FALSE(manager->is_loaded(mem::BlockKey(0, 0)));  // LRU was evicted
}

TEST_F(BlockManagerTest, PinPreventsEviction) {
    auto manager = mem::make_lru_manager(2, make_load_callback(), make_unload_callback());

    manager->on_access(mem::BlockKey(0, 0));
    manager->pin(mem::BlockKey(0, 0));
    manager->on_access(mem::BlockKey(1, 0));

    // Try to load third block - should evict unpinned block
    manager->on_access(mem::BlockKey(2, 0));

    ASSERT_TRUE(manager->is_loaded(mem::BlockKey(0, 0)));   // Pinned, not evicted
    ASSERT_FALSE(manager->is_loaded(mem::BlockKey(1, 0)));  // Evicted
    ASSERT_TRUE(manager->is_loaded(mem::BlockKey(2, 0)));
}

TEST_F(BlockManagerTest, UnpinAllowsEviction) {
    auto manager = mem::make_lru_manager(1, make_load_callback(), make_unload_callback());

    manager->on_access(mem::BlockKey(0, 0));
    manager->pin(mem::BlockKey(0, 0));

    // Try to load another - should fail due to pinned block
    ASSERT_THROW(manager->on_access(mem::BlockKey(1, 0)), std::runtime_error);

    // Unpin and try again
    manager->unpin(mem::BlockKey(0, 0));
    manager->on_access(mem::BlockKey(1, 0));

    ASSERT_FALSE(manager->is_loaded(mem::BlockKey(0, 0)));
    ASSERT_TRUE(manager->is_loaded(mem::BlockKey(1, 0)));
}

TEST_F(BlockManagerTest, ManualEviction) {
    auto manager = mem::make_lru_manager(5, make_load_callback(), make_unload_callback());

    manager->load(mem::BlockKey(0, 0));
    manager->load(mem::BlockKey(1, 0));

    ASSERT_TRUE(manager->is_loaded(mem::BlockKey(0, 0)));

    // Manually evict
    bool evicted = manager->evict(mem::BlockKey(0, 0));
    ASSERT_TRUE(evicted);
    ASSERT_FALSE(manager->is_loaded(mem::BlockKey(0, 0)));
    ASSERT_EQ(1u, manager->loaded_count());
}

TEST_F(BlockManagerTest, MaybeEvict) {
    auto manager = mem::make_lru_manager(2, make_load_callback(), make_unload_callback());

    manager->load(mem::BlockKey(0, 0));
    manager->load(mem::BlockKey(1, 0));

    // Should not evict - not over capacity
    ASSERT_FALSE(manager->maybe_evict());

    // Load one more manually
    loaded_blocks_.insert(mem::BlockKey(2, 0));

    // Now pretend we're over capacity (this is a bit of a hack for testing)
    // In real usage, maybe_evict is called internally
}

TEST_F(BlockManagerTest, Statistics) {
    auto manager = mem::make_lru_manager(2, make_load_callback(), make_unload_callback());

    // Generate some activity
    manager->on_access(mem::BlockKey(0, 0));  // Miss
    manager->on_access(mem::BlockKey(0, 0));  // Hit
    manager->on_access(mem::BlockKey(1, 0));  // Miss
    manager->on_access(mem::BlockKey(1, 0));  // Hit
    manager->on_access(mem::BlockKey(2, 0));  // Miss, causes eviction

    auto stats = manager->get_stats();
    ASSERT_EQ(3u, stats.cache_misses);
    ASSERT_EQ(2u, stats.cache_hits);
    ASSERT_EQ(3u, stats.total_loads);
    ASSERT_EQ(1u, stats.total_evictions);
    ASSERT_GT(stats.hit_rate(), 0.0);
    ASSERT_LT(stats.hit_rate(), 1.0);
}

TEST_F(BlockManagerTest, ResetStats) {
    auto manager = mem::make_lru_manager(2, make_load_callback(), make_unload_callback());

    manager->on_access(mem::BlockKey(0, 0));
    manager->on_access(mem::BlockKey(1, 0));

    auto stats = manager->get_stats();
    ASSERT_GT(stats.total_loads, 0u);

    manager->reset_stats();
    stats = manager->get_stats();
    ASSERT_EQ(0u, stats.total_loads);
    ASSERT_EQ(0u, stats.cache_hits);
    ASSERT_EQ(0u, stats.cache_misses);
}

TEST_F(BlockManagerTest, Clear) {
    auto manager = mem::make_lru_manager(3, make_load_callback(), make_unload_callback());

    manager->on_access(mem::BlockKey(0, 0));
    manager->on_access(mem::BlockKey(1, 0));
    manager->on_access(mem::BlockKey(2, 0));

    ASSERT_EQ(3u, manager->loaded_count());

    manager->clear();

    ASSERT_EQ(0u, manager->loaded_count());
    ASSERT_EQ(3, unload_count_);  // All blocks unloaded
}

TEST_F(BlockManagerTest, LoadFailureHandling) {
    auto manager = mem::make_lru_manager(2, make_load_callback(), make_unload_callback());

    fail_next_load_ = true;
    ASSERT_THROW(manager->on_access(mem::BlockKey(0, 0)), std::runtime_error);

    // Should not be loaded
    ASSERT_FALSE(manager->is_loaded(mem::BlockKey(0, 0)));
}

TEST_F(BlockManagerTest, AccessCountPolicy) {
    mem::AccessCountPolicyConfig config;
    auto manager = mem::make_access_count_manager(3, make_load_callback(), make_unload_callback(), config);

    // Load and access blocks with different frequencies
    manager->on_access(mem::BlockKey(0, 0));  // 1 access

    manager->on_access(mem::BlockKey(1, 0));  // 3 accesses
    manager->on_access(mem::BlockKey(1, 0));
    manager->on_access(mem::BlockKey(1, 0));

    manager->on_access(mem::BlockKey(2, 0));  // 2 accesses
    manager->on_access(mem::BlockKey(2, 0));

    // Load new block - should evict block with lowest count
    manager->on_access(mem::BlockKey(3, 0));

    ASSERT_FALSE(manager->is_loaded(mem::BlockKey(0, 0)));  // Evicted (lowest count)
    ASSERT_TRUE(manager->is_loaded(mem::BlockKey(1, 0)));
    ASSERT_TRUE(manager->is_loaded(mem::BlockKey(2, 0)));
    ASSERT_TRUE(manager->is_loaded(mem::BlockKey(3, 0)));
}

TEST_F(BlockManagerTest, ThreadSafety) {
    auto manager = mem::make_lru_manager(10, make_load_callback(), make_unload_callback());
    manager->set_thread_safe(true);

    const int num_threads = 4;
    const int accesses_per_thread = 100;
    std::vector<std::thread> threads;

    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&manager, t, accesses_per_thread]() {
            for (int i = 0; i < accesses_per_thread; ++i) {
                mem::BlockKey key(t, i % 5);
                manager->on_access(key);

                // Small delay to increase contention
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Check that manager is in consistent state
    ASSERT_LE(manager->loaded_count(), manager->max_blocks());

    auto stats = manager->get_stats();
    ASSERT_EQ(num_threads * accesses_per_thread, stats.cache_hits + stats.cache_misses);
}

TEST_F(BlockManagerTest, LargeScale) {
    const std::size_t max_blocks = 100;
    const std::size_t total_blocks = 1000;
    const int num_accesses = 10000;

    auto manager = mem::make_lru_manager(max_blocks, make_load_callback(), make_unload_callback());

    std::mt19937 rng(42);
    std::uniform_int_distribution<std::size_t> dist(0, total_blocks - 1);

    for (int i = 0; i < num_accesses; ++i) {
        mem::BlockKey key(dist(rng), 0);
        manager->on_access(key);

        // Invariant: never exceed max_blocks
        ASSERT_LE(manager->loaded_count(), max_blocks);
    }

    auto stats = manager->get_stats();
    std::cout << "\nLarge scale test results:\n";
    std::cout << "  Max blocks: " << max_blocks << "\n";
    std::cout << "  Total unique blocks: " << total_blocks << "\n";
    std::cout << "  Total accesses: " << num_accesses << "\n";
    std::cout << "  Cache hit rate: " << stats.hit_rate() * 100 << "%\n";
    std::cout << "  Total evictions: " << stats.total_evictions << "\n";
}

}  // namespace tri::test