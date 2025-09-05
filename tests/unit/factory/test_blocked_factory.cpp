/**
 * @file test_blocked_factory.cpp
 * @brief Unit tests for BlockedMatrixFactory
 * @author Yongze
 * @date 2025-08-14
 */

#include <iostream>

#include "framework/test_assertions.hpp"
#include "framework/test_runner.hpp"
#include "framework/test_utils.hpp"
#include "tri/core/dense_rm.hpp"
#include "tri/core/lower_tri_rm.hpp"
#include "tri/factory/blocked_factory.hpp"

namespace tri::test {

// Test fixtures - NO TEMPLATES IN CLASS NAME!
class BlockedFactoryTestDouble : public TestCase {
   protected:
    double tolerance_ = 1e-10;
};

class BlockedFactoryTestFloat : public TestCase {
   protected:
    float tolerance_ = 1e-5f;
};

class BlockedFactoryTestInt : public TestCase {
   protected:
    int tolerance_ = 0;
};

// Double precision tests
TEST_F(BlockedFactoryTestDouble, Identity) {
    const std::size_t n = 16;
    const std::size_t block_size = 4;
    auto m = factory::BlockedMatrixFactory<double>::identity(n, block_size);

    ASSERT_EQ(n, m.rows());
    ASSERT_EQ(block_size, m.block_size());

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            if (i == j) {
                ASSERT_EQ(1.0, m(i, j));
            } else {
                ASSERT_EQ(0.0, m(i, j));
            }
        }
    }
}

TEST_F(BlockedFactoryTestDouble, Zeros) {
    auto m = factory::BlockedMatrixFactory<double>::zeros(20u, 5u);

    ASSERT_EQ(20u, m.rows());
    ASSERT_EQ(5u, m.block_size());

    for (std::size_t i = 0; i < 20; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            ASSERT_EQ(0.0, m(i, j));
        }
    }
}

TEST_F(BlockedFactoryTestDouble, Ones) {
    auto m = factory::BlockedMatrixFactory<double>::ones(15u, 3u);

    ASSERT_EQ(15u, m.rows());
    ASSERT_EQ(3u, m.block_size());

    for (std::size_t i = 0; i < 15; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            ASSERT_EQ(1.0, m(i, j));
        }
    }
}

TEST_F(BlockedFactoryTestDouble, Constant) {
    const double value = 3.14159;
    auto m = factory::BlockedMatrixFactory<double>::constant(24u, value, 6u);

    ASSERT_EQ(24u, m.rows());
    ASSERT_EQ(6u, m.block_size());

    for (std::size_t i = 0; i < 24; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            ASSERT_NEAR(value, m(i, j), tolerance_);
        }
    }
}

TEST_F(BlockedFactoryTestDouble, Random) {
    const double min_val = -1.0;
    const double max_val = 1.0;
    auto m = factory::BlockedMatrixFactory<double>::random(32u, min_val, max_val, 8u);

    ASSERT_EQ(32u, m.rows());
    ASSERT_EQ(8u, m.block_size());

    // Check all values in lower triangle are in range
    for (std::size_t i = 0; i < 32; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double val = m(i, j);
            ASSERT_GE(val, min_val);
            ASSERT_LE(val, max_val);
        }
    }
}

TEST_F(BlockedFactoryTestDouble, FromDense) {
    // Create a dense matrix
    core::DenseRM<double> dense(16, 16);
    for (std::size_t i = 0; i < 16; ++i) {
        for (std::size_t j = 0; j < 16; ++j) {
            dense(i, j) = static_cast<double>(i + j);
        }
    }

    auto m = factory::BlockedMatrixFactory<double>::from_dense(dense, 4u);

    ASSERT_EQ(16u, m.rows());
    ASSERT_EQ(4u, m.block_size());

    // Check lower triangular part matches
    for (std::size_t i = 0; i < 16; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            ASSERT_EQ(dense(i, j), m(i, j));
        }
    }
}

TEST_F(BlockedFactoryTestDouble, FromTriangular) {
    // Create a triangular matrix
    core::LowerTriangularRM<double> tri(12u);
    for (std::size_t i = 0; i < 12; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            tri(i, j) = static_cast<double>(i * j);
        }
    }

    auto m = factory::BlockedMatrixFactory<double>::from_triangular(tri, 3u);

    ASSERT_EQ(12u, m.rows());
    ASSERT_EQ(3u, m.block_size());

    // Check values match
    for (std::size_t i = 0; i < 12; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            ASSERT_EQ(tri(i, j), m(i, j));
        }
    }
}

// Float precision tests
TEST_F(BlockedFactoryTestFloat, IdentityFloat) {
    auto m = factory::BlockedMatrixFactory<float>::identity(12u, 3u);

    ASSERT_EQ(12u, m.rows());
    ASSERT_EQ(3u, m.block_size());

    for (std::size_t i = 0; i < 12; ++i) {
        ASSERT_EQ(1.0f, m(i, i));
    }
}

// Integer tests
TEST_F(BlockedFactoryTestInt, ConstantInteger) {
    const int value = 42;
    auto m = factory::BlockedMatrixFactory<int>::constant(8u, value, 2u);

    ASSERT_EQ(8u, m.rows());
    ASSERT_EQ(2u, m.block_size());

    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            ASSERT_EQ(value, m(i, j));
        }
    }
}

}  // namespace tri::test