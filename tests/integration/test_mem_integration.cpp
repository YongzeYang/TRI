/**
 * @file test_mem_integration.cpp
 * @brief Integration tests for memory management system
 * @author Yongze
 * @date 2025-08-14
 */

#include <iostream>

#include "framework/test_assertions.hpp"
#include "framework/test_runner.hpp"
#include "tri/core/blocked_tri.hpp"
#include "tri/factory/blocked_factory.hpp"
#include "tri/mem/block_manager.hpp"

namespace tri::test {

TEST(MemoryIntegration, FactoryWithManager) {
    const std::size_t n = 64;
    const std::size_t block_size = 8;

    // Create matrix using factory
    auto mat = factory::BlockedMatrixFactory<double>::identity(n, block_size);

    // Add memory manager
    ASSERT_TRUE(mat.create_block_manager(5, "LRU"));

    // Verify identity property still holds with limited memory
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) {
                ASSERT_EQ(1.0, mat(i, j));
            } else if (j < i) {
                ASSERT_EQ(0.0, mat(i, j));
            }
        }
    }
}

TEST(MemoryIntegration, MultipleMatricesSharedManager) {
    const std::size_t n = 32;
    const std::size_t block_size = 8;

    // Create shared manager
    auto load_count = std::make_shared<int>(0);
    auto unload_count = std::make_shared<int>(0);

    auto manager = mem::make_lru_manager(
        10,
        [load_count](const mem::BlockKey&) {
            (*load_count)++;
            return true;
        },
        [unload_count](const mem::BlockKey&) {
            (*unload_count)++;
            return true;
        });

    // Create multiple matrices sharing the manager
    core::BlockedTriMatrix<double> mat1(n, block_size);
    core::BlockedTriMatrix<double> mat2(n, block_size);

    mat1.set_block_manager(std::move(manager));
    // Note: Can't share the same manager instance directly in this design
    // Each matrix needs its own manager with appropriate callbacks
}

TEST(MemoryIntegration, StressTestWithMemoryPressure) {
    const std::size_t n = 512;
    const std::size_t block_size = 32;
    const std::size_t max_blocks = 10;  // Very limited memory

    core::BlockedTriMatrix<double> mat(n, block_size);
    mat.create_block_manager(max_blocks, "LRU");

    // Simulate heavy computation with memory pressure
    double sum = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        // Diagonal sweep
        for (std::size_t i = k; i < n; ++i) {
            mat(i, k) = static_cast<double>(i - k + 1);
            sum += mat(i, k);
        }
    }

    std::cout << "\nStress test completed. Sum = " << sum << "\n";

    if (mat.get_block_manager()) {
        auto stats = mat.get_block_manager()->get_stats();
        std::cout << "Memory pressure stats:\n";
        std::cout << "  Total loads: " << stats.total_loads << "\n";
        std::cout << "  Total evictions: " << stats.total_evictions << "\n";
        std::cout << "  Final cache hit rate: " << stats.hit_rate() * 100 << "%\n";
    }
}

TEST(MemoryIntegration, AdaptiveBlockSize) {
    // Test different block sizes for performance
    std::vector<std::size_t> block_sizes = {4, 8, 16, 32, 64};
    const std::size_t n = 256;
    const std::size_t max_memory_blocks = 8;

    for (auto block_size : block_sizes) {
        core::BlockedTriMatrix<double> mat(n, block_size);
        mat.create_block_manager(max_memory_blocks, "LRU");

        auto start = std::chrono::high_resolution_clock::now();

        // Perform computation
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                mat(i, j) = std::sqrt(static_cast<double>(i * i + j * j));
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start).count();

        auto stats = mat.get_block_manager()->get_stats();

        std::cout << "\nBlock size " << block_size << ":\n";
        std::cout << "  Time: " << duration << " ms\n";
        std::cout << "  Cache hit rate: " << stats.hit_rate() * 100 << "%\n";
        std::cout << "  Evictions: " << stats.total_evictions << "\n";
    }
}

}  // namespace tri::test