/**
 * @file test_blocked_tri.cpp
 * @brief Unit tests for BlockedTriMatrix class
 * @author Yongze
 * @date 2025-08-14
 */

#include <iostream>

#include "test_assertions.hpp"
#include "test_runner.hpp"
#include "test_utils.hpp"
#include "tri/core/blocked_tri.hpp"
#include "tri/core/dense_rm.hpp"
#include "tri/core/lower_tri_rm.hpp"

namespace tri::test {

// Test fixture for double precision
class BlockedTriMatrixTestDouble : public TestCase {
   protected:
    double tolerance_ = 1e-10;
};

// Test fixture for float precision
class BlockedTriMatrixTestFloat : public TestCase {
   protected:
    float tolerance_ = 1e-5f;
};

// Double precision tests
TEST_F(BlockedTriMatrixTestDouble, DefaultConstruction) {
    core::BlockedTriMatrix<double> m;
    ASSERT_EQ(0u, m.rows());
    ASSERT_EQ(0u, m.cols());
    ASSERT_TRUE(m.empty());
    ASSERT_EQ(0u, m.total_blocks());
    ASSERT_EQ(0u, m.allocated_blocks());
}

TEST_F(BlockedTriMatrixTestDouble, SizeConstructionDefaultBlockSize) {
    const std::size_t n = 100;
    core::BlockedTriMatrix<double> m(n);

    ASSERT_EQ(n, m.rows());
    ASSERT_EQ(n, m.cols());
    ASSERT_EQ(tri::config::DEFAULT_BLOCK_SIZE, m.block_size());
    ASSERT_FALSE(m.empty());
    ASSERT_TRUE(m.is_square());
}

TEST_F(BlockedTriMatrixTestDouble, SizeConstructionCustomBlockSize) {
    const std::size_t n = 50;
    const std::size_t block_size = 16;
    core::BlockedTriMatrix<double> m(n, block_size);

    ASSERT_EQ(n, m.rows());
    ASSERT_EQ(n, m.cols());
    ASSERT_EQ(block_size, m.block_size());

    // Calculate expected number of blocks (ceiling division)
    std::size_t expected_blocks = (n + block_size - 1) / block_size;
    ASSERT_EQ(expected_blocks, m.num_block_rows());
    ASSERT_EQ(expected_blocks, m.num_block_cols());
}

TEST_F(BlockedTriMatrixTestDouble, ValueConstruction) {
    const std::size_t n = 32;
    const std::size_t block_size = 8;
    const double value = 3.14;

    core::BlockedTriMatrix<double> m(n, block_size, value);

    ASSERT_EQ(n, m.rows());
    ASSERT_EQ(block_size, m.block_size());

    // Verify all elements in lower triangle
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            ASSERT_NEAR(value, m(i, j), tolerance_);
        }
    }
}

TEST_F(BlockedTriMatrixTestDouble, BlockIndexComputation) {
    core::BlockedTriMatrix<double> m(16, 4);

    // Test block index computation
    auto idx1 = m.compute_block_index(7, 3);
    ASSERT_EQ(1u, idx1.block_row);   // 7/4 = 1
    ASSERT_EQ(0u, idx1.block_col);   // 3/4 = 0
    ASSERT_EQ(3u, idx1.row_offset);  // 7%4 = 3
    ASSERT_EQ(3u, idx1.col_offset);  // 3%4 = 3
    ASSERT_TRUE(idx1.is_valid());

    // Test invalid block index for upper triangle
    auto invalid_idx = m.compute_block_index(3, 7);
    ASSERT_FALSE(invalid_idx.is_valid());
}

TEST_F(BlockedTriMatrixTestDouble, Fill) {
    core::BlockedTriMatrix<double> m(12, 4);
    const double fill_value = 2.718;

    m.fill(fill_value);

    for (std::size_t i = 0; i < 12; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            ASSERT_NEAR(fill_value, m(i, j), tolerance_);
        }
    }
}

TEST_F(BlockedTriMatrixTestDouble, SetDiagonal) {
    core::BlockedTriMatrix<double> m(16, 4);
    m.fill(0.0);

    m.set_diagonal(1.0);

    for (std::size_t i = 0; i < 16; ++i) {
        ASSERT_EQ(1.0, m(i, i));
        if (i > 0) {
            ASSERT_EQ(0.0, m(i, 0));  // Non-diagonal element
        }
    }
}

// Float precision tests
TEST_F(BlockedTriMatrixTestFloat, ElementAccess) {
    const std::size_t n = 20;
    const std::size_t block_size = 5;
    core::BlockedTriMatrix<float> m(n, block_size);

    // Set elements in lower triangle
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            m.set(i, j, static_cast<float>(i * 100 + j));
        }
    }

    // Verify lower triangle
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            ASSERT_EQ(static_cast<float>(i * 100 + j), m(i, j));
        }
    }

    // Test upper triangle returns 0
    const auto& const_m = m;  // 创建const引用
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            ASSERT_EQ(0.0f, const_m(i, j));
        }
    }
}

TEST_F(BlockedTriMatrixTestFloat, AccessTracking) {
    core::BlockedTriMatrix<float> m(16, 4);

    // Initial access count should be 0
    ASSERT_EQ(0u, m.get_block_access_count(0, 0));

    // Access some elements
    m(0, 0);
    m(1, 0);
    m(1, 1);

    // Block (0,0) should have been accessed 3 times
    ASSERT_EQ(3u, m.get_block_access_count(0, 0));

    // Reset access counts
    m.reset_access_counts();
    ASSERT_EQ(0u, m.get_block_access_count(0, 0));
}

// Non-fixture tests
TEST(BlockedTriMatrix, InvalidSetThrows) {
    core::BlockedTriMatrix<double> m(12, 4);

    // Setting upper triangle should throw
    ASSERT_THROW(m.set(3, 5, 10.0), std::logic_error);
    ASSERT_THROW(m.set(0, 10, 10.0), std::logic_error);
    ASSERT_THROW(m(2, 8) = 10.0, std::logic_error);
}

TEST(BlockedTriMatrix, StaticFactories) {
    // Identity
    auto id = core::BlockedTriMatrix<double>::identity(12, 4);
    ASSERT_EQ(12u, id.rows());
    ASSERT_EQ(4u, id.block_size());

    for (std::size_t i = 0; i < 12; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            if (i == j) {
                ASSERT_EQ(1.0, id(i, j));
            } else {
                ASSERT_EQ(0.0, id(i, j));
            }
        }
    }

    // Zeros
    auto z = core::BlockedTriMatrix<double>::zeros(15, 5);
    ASSERT_EQ(15u, z.rows());
    ASSERT_EQ(5u, z.block_size());

    for (std::size_t i = 0; i < 15; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            ASSERT_EQ(0.0, z(i, j));
        }
    }
}

TEST(BlockedTriMatrix, ExportToDense) {
    core::BlockedTriMatrix<double> m(10, 3);

    // Fill with pattern
    for (std::size_t i = 0; i < 10; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            m(i, j) = static_cast<double>(i * 10 + j);
        }
    }

    // Export to dense array
    std::vector<double> dense(10 * 10);
    m.export_to_dense(dense.data(), 10);

    // Verify exported data
    for (std::size_t i = 0; i < 10; ++i) {
        for (std::size_t j = 0; j < 10; ++j) {
            if (j <= i) {
                ASSERT_EQ(static_cast<double>(i * 10 + j), dense[i * 10 + j]);
            } else {
                ASSERT_EQ(0.0, dense[i * 10 + j]);
            }
        }
    }
}

}  // namespace tri::test