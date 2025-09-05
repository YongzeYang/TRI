/**
 * @file test_blocked_tri_with_manager.cpp
 * @brief Unit tests for BlockedTriMatrix with memory management
 * @author Yongze
 * @date 2025-08-14
 */

#include <chrono>
#include <random>

#include "framework/test_assertions.hpp"
#include "framework/test_runner.hpp"
#include "tri/core/blocked_tri.hpp"
#include "tri/mem/block_manager.hpp"
#include "tri/mem/block_storage.hpp"

namespace tri::test {

class BlockedTriWithManagerTest : public TestCase {
   protected:
    static constexpr std::size_t DEFAULT_N = 64;
    static constexpr std::size_t DEFAULT_BLOCK_SIZE = 8;

    template <typename T>
    void fill_matrix_pattern(core::BlockedTriMatrix<T>& mat) {
        std::size_t n = mat.rows();
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                mat(i, j) = static_cast<T>(i * 1000 + j);
            }
        }
    }

    template <typename T>
    void verify_matrix_pattern(const core::BlockedTriMatrix<T>& mat) {
        std::size_t n = mat.rows();
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                T expected = static_cast<T>(i * 1000 + j);
                ASSERT_EQ(expected, mat(i, j));
            }
        }
    }
};

TEST_F(BlockedTriWithManagerTest, CreateWithoutManager) {
    // Matrix should work without a manager
    core::BlockedTriMatrix<double> mat(32, 8);

    ASSERT_EQ(nullptr, mat.get_block_manager());

    // Should still be able to access elements
    mat(10, 5) = 3.14;
    ASSERT_EQ(3.14, mat(10, 5));
}

TEST_F(BlockedTriWithManagerTest, CreateManagerLRU) {
    core::BlockedTriMatrix<double> mat(DEFAULT_N, DEFAULT_BLOCK_SIZE);

    // Create manager with limited blocks
    const std::size_t max_blocks = 5;
    ASSERT_TRUE(mat.create_block_manager(max_blocks, "LRU"));
    ASSERT_NE(nullptr, mat.get_block_manager());

    auto manager = mat.get_block_manager();
    ASSERT_EQ(max_blocks, manager->max_blocks());
}

TEST_F(BlockedTriWithManagerTest, CreateManagerAccessCount) {
    core::BlockedTriMatrix<double> mat(DEFAULT_N, DEFAULT_BLOCK_SIZE);

    const std::size_t max_blocks = 5;
    ASSERT_TRUE(mat.create_block_manager(max_blocks, "AccessCount"));
    ASSERT_NE(nullptr, mat.get_block_manager());
}

TEST_F(BlockedTriWithManagerTest, InvalidPolicyName) {
    core::BlockedTriMatrix<double> mat(DEFAULT_N, DEFAULT_BLOCK_SIZE);

    ASSERT_FALSE(mat.create_block_manager(5, "InvalidPolicy"));
    ASSERT_EQ(nullptr, mat.get_block_manager());
}

TEST_F(BlockedTriWithManagerTest, MemoryLimitRespected) {
    const std::size_t n = 32;
    const std::size_t block_size = 4;
    const std::size_t max_blocks = 3;

    core::BlockedTriMatrix<double> mat(n, block_size);
    ASSERT_TRUE(mat.create_block_manager(max_blocks, "LRU"));

    // Access many different blocks
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            mat(i, j) = static_cast<double>(i + j);
        }
    }

    // Check that we never exceeded the limit
    auto manager = mat.get_block_manager();
    ASSERT_LE(manager->loaded_count(), max_blocks);

    // Check stats
    auto stats = manager->get_stats();
    std::cout << "\nMemory limit test stats:\n";
    std::cout << "  Loaded blocks: " << manager->loaded_count() << "/" << max_blocks << "\n";
    std::cout << "  Total evictions: " << stats.total_evictions << "\n";
    std::cout << "  Cache hit rate: " << stats.hit_rate() * 100 << "%\n";
}

TEST_F(BlockedTriWithManagerTest, DataPersistence) {
    const std::size_t n = 48;
    const std::size_t block_size = 8;
    const std::size_t max_blocks = 4;

    core::BlockedTriMatrix<int> mat(n, block_size);
    ASSERT_TRUE(mat.create_block_manager(max_blocks, "LRU"));

    // Fill with pattern
    fill_matrix_pattern(mat);

    // Force many evictions by accessing all blocks randomly
    std::mt19937 rng(42);
    std::uniform_int_distribution<std::size_t> dist(0, n - 1);

    for (int iter = 0; iter < 1000; ++iter) {
        std::size_t i = dist(rng);
        std::size_t j = dist(rng);
        if (j > i) std::swap(i, j);

        // Just access the element
        volatile int val = mat(i, j);
        (void)val;
    }

    // Verify all data is still correct despite evictions
    verify_matrix_pattern(mat);
}

TEST_F(BlockedTriWithManagerTest, DataPersistenceWithMemoryStorage) {
    const std::size_t n = 48;
    const std::size_t block_size = 8;
    const std::size_t max_blocks = 4;

    core::BlockedTriMatrix<int> mat(n, block_size);

    // Set memory storage backend BEFORE creating block manager
    ASSERT_TRUE(mat.set_storage_backend(core::BlockedTriMatrix<int>::StorageType::Memory));

    ASSERT_TRUE(mat.create_block_manager(max_blocks, "LRU"));

    // Fill with pattern
    fill_matrix_pattern(mat);

    // Force many evictions by accessing all blocks randomly
    std::mt19937 rng(42);
    std::uniform_int_distribution<std::size_t> dist(0, n - 1);

    for (int iter = 0; iter < 1000; ++iter) {
        std::size_t i = dist(rng);
        std::size_t j = dist(rng);
        if (j > i) std::swap(i, j);

        // Just access the element
        volatile int val = mat(i, j);
        (void)val;
    }

    // Verify all data is still correct despite evictions
    verify_matrix_pattern(mat);
}

TEST_F(BlockedTriWithManagerTest, DataPersistenceWithDiskStorage) {
    const std::size_t n = 32;
    const std::size_t block_size = 8;
    const std::size_t max_blocks = 2;  // Very limited memory

    // Create temp directory for test
    std::string temp_dir = "./test_blocks_" + std::to_string(std::time(nullptr));
    std::filesystem::create_directory(temp_dir);

    {
        core::BlockedTriMatrix<double> mat(n, block_size);

        // Use disk storage
        ASSERT_TRUE(
            mat.set_storage_backend(core::BlockedTriMatrix<double>::StorageType::Disk, temp_dir));

        ASSERT_TRUE(mat.create_block_manager(max_blocks, "LRU"));

        // Fill with test pattern
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                mat(i, j) = static_cast<double>(i * 100 + j);
            }
        }

        // Force evictions
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                volatile double val = mat(i, j);
                (void)val;
            }
        }

        // Verify data
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                ASSERT_EQ(static_cast<double>(i * 100 + j), mat(i, j));
            }
        }
    }

    // Clean up temp directory
    std::filesystem::remove_all(temp_dir);
}

TEST_F(BlockedTriWithManagerTest, HybridStorage) {
    const std::size_t n = 64;
    const std::size_t block_size = 16;
    const std::size_t max_blocks = 3;

    core::BlockedTriMatrix<float> mat(n, block_size);

    // Use hybrid storage (memory cache + disk)
    ASSERT_TRUE(mat.set_storage_backend(core::BlockedTriMatrix<float>::StorageType::Hybrid,
                                        "./hybrid_test"));

    ASSERT_TRUE(mat.create_block_manager(max_blocks, "AccessCount"));

    // Fill and access pattern
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            mat(i, j) = static_cast<float>(std::sin(i) + std::cos(j));
        }
    }

    // Verify
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            float expected = static_cast<float>(std::sin(i) + std::cos(j));
            ASSERT_NEAR(expected, mat(i, j), 1e-6f);
        }
    }

    // Clean up
    std::filesystem::remove_all("./hybrid_test");
}

TEST_F(BlockedTriWithManagerTest, StorageStatistics) {
    const std::size_t n = 32;
    const std::size_t block_size = 8;

    core::BlockedTriMatrix<double> mat(n, block_size);
    mat.set_storage_backend(core::BlockedTriMatrix<double>::StorageType::Memory);
    mat.create_block_manager(2, "LRU");

    // Fill matrix
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            mat(i, j) = static_cast<double>(i + j);
        }
    }

    // Check storage backend statistics
    auto storage = mat.get_storage_backend();
    ASSERT_NE(nullptr, storage);

    std::cout << "\nStorage statistics:\n";
    std::cout << "  Type: " << storage->type() << "\n";
    std::cout << "  Size: " << storage->size_bytes() << " bytes\n";
}

TEST_F(BlockedTriWithManagerTest, PinningBlocks) {
    const std::size_t n = 32;
    const std::size_t block_size = 8;
    const std::size_t max_blocks = 2;

    core::BlockedTriMatrix<double> mat(n, block_size);
    ASSERT_TRUE(mat.create_block_manager(max_blocks, "LRU"));

    // Access and pin first block
    mat(0, 0) = 1.0;
    mat.pin_block(0, 0);

    // Access and pin second block
    mat(8, 0) = 2.0;
    mat.pin_block(1, 0);

    // Try to access third block - should fail since both are pinned
    ASSERT_THROW(mat(16, 0) = 3.0, std::runtime_error);

    // Unpin first block
    mat.unpin_block(0, 0);

    // Now should work
    mat(16, 0) = 3.0;
    ASSERT_EQ(3.0, mat(16, 0));
}

TEST_F(BlockedTriWithManagerTest, AccessPatternLRU) {
    const std::size_t n = 32;
    const std::size_t block_size = 8;
    const std::size_t max_blocks = 3;

    core::BlockedTriMatrix<double> mat(n, block_size);
    ASSERT_TRUE(mat.create_block_manager(max_blocks, "LRU"));

    // Access blocks in specific order
    mat(0, 0) = 1.0;   // Block (0,0)
    mat(8, 8) = 2.0;   // Block (1,1)
    mat(16, 0) = 3.0;  // Block (2,0)

    // Access fourth block - should evict (0,0) as LRU
    mat(24, 8) = 4.0;  // Block (3,1)

    // Check which blocks are loaded by trying to access them
    auto manager = mat.get_block_manager();
    auto initial_evictions = manager->get_stats().total_evictions;

    // Accessing (8,8) should be a hit (not evicted)
    volatile double val1 = mat(8, 8);
    ASSERT_EQ(initial_evictions, manager->get_stats().total_evictions);

    // Accessing (0,0) should cause a load (was evicted)
    volatile double val2 = mat(0, 0);
    ASSERT_GT(manager->get_stats().total_evictions, initial_evictions);
}

TEST_F(BlockedTriWithManagerTest, BlockBoundaryAccess) {
    const std::size_t n = 33;  // Not evenly divisible by block size
    const std::size_t block_size = 8;
    const std::size_t max_blocks = 5;

    core::BlockedTriMatrix<double> mat(n, block_size);
    ASSERT_TRUE(mat.create_block_manager(max_blocks, "LRU"));

    // Access elements at block boundaries
    mat(7, 7) = 1.0;    // Last element of block (0,0)
    mat(8, 0) = 2.0;    // First element of block (1,0)
    mat(8, 8) = 3.0;    // First element of block (1,1)
    mat(15, 15) = 4.0;  // Last element of block (1,1)
    mat(32, 32) = 5.0;  // Last element of last block

    // Verify
    ASSERT_EQ(1.0, mat(7, 7));
    ASSERT_EQ(2.0, mat(8, 0));
    ASSERT_EQ(3.0, mat(8, 8));
    ASSERT_EQ(4.0, mat(15, 15));
    ASSERT_EQ(5.0, mat(32, 32));
}

TEST_F(BlockedTriWithManagerTest, PerformanceComparison) {
    const std::size_t n = 256;
    const std::size_t block_size = 16;

    // Matrix without manager (all blocks in memory)
    core::BlockedTriMatrix<double> mat_unlimited(n, block_size);

    // Matrix with limited memory
    core::BlockedTriMatrix<double> mat_limited(n, block_size);
    mat_limited.create_block_manager(10, "LRU");  // Only 10 blocks in memory

    // Sequential access pattern
    auto sequential_test = [](core::BlockedTriMatrix<double>& mat) {
        auto start = std::chrono::high_resolution_clock::now();

        std::size_t n = mat.rows();
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                mat(i, j) = static_cast<double>(i + j);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    double time_unlimited = sequential_test(mat_unlimited);
    double time_limited = sequential_test(mat_limited);

    std::cout << "\nPerformance comparison:\n";
    std::cout << "  Unlimited memory: " << time_unlimited << " ms\n";
    std::cout << "  Limited memory (10 blocks): " << time_limited << " ms\n";
    std::cout << "  Overhead: " << (time_limited / time_unlimited - 1) * 100 << "%\n";

    if (mat_limited.get_block_manager()) {
        auto stats = mat_limited.get_block_manager()->get_stats();
        std::cout << "  Cache hit rate: " << stats.hit_rate() * 100 << "%\n";
        std::cout << "  Total evictions: " << stats.total_evictions << "\n";
    }
}

TEST_F(BlockedTriWithManagerTest, RandomAccessPattern) {
    const std::size_t n = 128;
    const std::size_t block_size = 16;
    const std::size_t max_blocks = 8;

    core::BlockedTriMatrix<double> mat(n, block_size);
    ASSERT_TRUE(mat.create_block_manager(max_blocks, "AccessCount"));

    // Fill with initial values
    fill_matrix_pattern(mat);

    // Random access pattern
    std::mt19937 rng(42);
    std::uniform_int_distribution<std::size_t> dist(0, n - 1);

    const int num_accesses = 5000;
    for (int i = 0; i < num_accesses; ++i) {
        std::size_t row = dist(rng);
        std::size_t col = dist(rng);
        if (col > row) std::swap(row, col);

        // Read and modify
        double val = mat(row, col);
        mat(row, col) = val + 1.0;
    }

    auto manager = mat.get_block_manager();
    auto stats = manager->get_stats();

    std::cout << "\nRandom access pattern results:\n";
    std::cout << "  Total accesses: " << num_accesses << "\n";
    std::cout << "  Cache hits: " << stats.cache_hits << "\n";
    std::cout << "  Cache misses: " << stats.cache_misses << "\n";
    std::cout << "  Hit rate: " << stats.hit_rate() * 100 << "%\n";
    std::cout << "  Total evictions: " << stats.total_evictions << "\n";
}

TEST_F(BlockedTriWithManagerTest, WorkingSetSize) {
    const std::size_t n = 64;
    const std::size_t block_size = 8;

    // Test with different cache sizes
    std::vector<std::size_t> cache_sizes = {1, 2, 4, 8, 16, 32};

    for (auto max_blocks : cache_sizes) {
        core::BlockedTriMatrix<double> mat(n, block_size);
        mat.create_block_manager(max_blocks, "LRU");

        // Access pattern with locality
        for (std::size_t bi = 0; bi < n / block_size; ++bi) {
            for (std::size_t bj = 0; bj <= bi; ++bj) {
                // Access all elements in block (bi, bj)
                std::size_t row_start = bi * block_size;
                std::size_t col_start = bj * block_size;
                std::size_t row_end = std::min(row_start + block_size, n);
                std::size_t col_end = std::min(col_start + block_size, n);

                for (std::size_t i = row_start; i < row_end; ++i) {
                    for (std::size_t j = col_start; j < col_end && j <= i; ++j) {
                        mat(i, j) = static_cast<double>(i + j);
                    }
                }
            }
        }

        auto stats = mat.get_block_manager()->get_stats();
        std::cout << "\nCache size " << max_blocks << " blocks: ";
        std::cout << "Hit rate = " << stats.hit_rate() * 100 << "%, ";
        std::cout << "Evictions = " << stats.total_evictions << "\n";
    }
}

TEST_F(BlockedTriWithManagerTest, ExportToDenseWithManager) {
    const std::size_t n = 32;
    const std::size_t block_size = 8;
    const std::size_t max_blocks = 3;

    core::BlockedTriMatrix<double> mat(n, block_size);
    mat.create_block_manager(max_blocks, "LRU");

    // Fill with pattern
    fill_matrix_pattern(mat);

    // Export to dense
    std::vector<double> dense = mat.to_dense_vector();

    // Verify exported data
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (j <= i) {
                double expected = static_cast<double>(i * 1000 + j);
                ASSERT_EQ(expected, dense[i * n + j]);
            } else {
                ASSERT_EQ(0.0, dense[i * n + j]);
            }
        }
    }
}

}  // namespace tri::test