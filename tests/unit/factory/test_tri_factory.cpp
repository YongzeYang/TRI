/**
 * @file test_tri_factory.cpp
 * @brief Unit tests for TriangularMatrixFactory
 * @author Yongze
 * @date 2025-08-14
 */

#include <iostream>

#include "framework/test_assertions.hpp"
#include "framework/test_runner.hpp"
#include "framework/test_utils.hpp"
#include "tri/core/dense_rm.hpp"
#include "tri/factory/tri_factory.hpp"

namespace tri::test {

// Test fixtures - NO TEMPLATES IN CLASS NAME!
class TriangularFactoryTestDouble : public TestCase {
   protected:
    double tolerance_ = 1e-10;
};

class TriangularFactoryTestFloat : public TestCase {
   protected:
    float tolerance_ = 1e-5f;
};

// Double precision tests
TEST_F(TriangularFactoryTestDouble, LowerIdentity) {
    auto m = factory::TriangularMatrixFactory<double>::lower_identity(5u);

    ASSERT_EQ(5u, m.dimension());

    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            if (i == j) {
                ASSERT_EQ(1.0, m(i, j));
            } else {
                ASSERT_EQ(0.0, m(i, j));
            }
        }
    }
}

TEST_F(TriangularFactoryTestDouble, LowerZeros) {
    auto m = factory::TriangularMatrixFactory<double>::lower_zeros(4u);

    ASSERT_EQ(4u, m.dimension());

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            ASSERT_EQ(0.0, m(i, j));
        }
    }
}

TEST_F(TriangularFactoryTestDouble, LowerOnes) {
    auto m = factory::TriangularMatrixFactory<double>::lower_ones(3u);

    ASSERT_EQ(3u, m.dimension());

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            ASSERT_EQ(1.0, m(i, j));
        }
    }
}

TEST_F(TriangularFactoryTestDouble, LowerConstant) {
    const double value = 2.718;
    auto m = factory::TriangularMatrixFactory<double>::lower_constant(5u, value);

    ASSERT_EQ(5u, m.dimension());

    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            ASSERT_NEAR(value, m(i, j), tolerance_);
        }
    }
}

TEST_F(TriangularFactoryTestDouble, LowerRandom) {
    const double min_val = -10.0;
    const double max_val = 10.0;
    auto m = factory::TriangularMatrixFactory<double>::lower_random(8u, min_val, max_val);

    ASSERT_EQ(8u, m.dimension());

    // Check all values in lower triangle are in range
    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double val = m(i, j);
            ASSERT_GE(val, min_val);
            ASSERT_LE(val, max_val);
        }
    }

    // Check upper triangle is zero
    const auto& const_m = m;  // 创建const引用
    for (std::size_t i = 0; i < 8; ++i) {
        for (std::size_t j = i + 1; j < 8; ++j) {
            ASSERT_EQ(0.0, const_m(i, j));
        }
    }
}

// Float precision tests
TEST_F(TriangularFactoryTestFloat, LowerIdentityFloat) {
    auto m = factory::TriangularMatrixFactory<float>::lower_identity(3u);

    ASSERT_EQ(3u, m.dimension());

    ASSERT_EQ(1.0f, m(0, 0));
    ASSERT_EQ(1.0f, m(1, 1));
    ASSERT_EQ(1.0f, m(2, 2));
    ASSERT_EQ(0.0f, m(1, 0));
    ASSERT_EQ(0.0f, m(2, 0));
    ASSERT_EQ(0.0f, m(2, 1));
}

TEST_F(TriangularFactoryTestFloat, LowerConstantFloat) {
    const float value = 3.14f;
    auto m = factory::TriangularMatrixFactory<float>::lower_constant(4u, value);

    ASSERT_EQ(4u, m.dimension());

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            ASSERT_NEAR(value, m(i, j), tolerance_);
        }
    }
}

// Tests without fixtures
TEST(TriangularFactory, CreateFromDenseManually) {
    // Create a dense matrix with known values
    core::DenseRM<double> dense(4, 4);
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            dense(i, j) = static_cast<double>(i * 4 + j + 1);
        }
    }

    // Create triangular matrix from dense using constructor
    core::LowerTriangularRM<double> m(dense);

    ASSERT_EQ(4u, m.dimension());

    // Check lower triangular part matches
    ASSERT_EQ(1.0, m(0, 0));
    ASSERT_EQ(5.0, m(1, 0));
    ASSERT_EQ(6.0, m(1, 1));
    ASSERT_EQ(9.0, m(2, 0));
    ASSERT_EQ(10.0, m(2, 1));
    ASSERT_EQ(11.0, m(2, 2));

    // Check upper triangle is zero
    const auto& const_m = m;  // 创建const引用
    ASSERT_EQ(0.0, const_m(0, 1));
    ASSERT_EQ(0.0, const_m(0, 2));
    ASSERT_EQ(0.0, const_m(1, 2));
}

TEST(TriangularFactory, CreateFromPackedManually) {
    std::vector<double> packed{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    // Create triangular matrix from packed data using constructor
    core::LowerTriangularRM<double> m(3u, packed);

    ASSERT_EQ(3u, m.dimension());

    ASSERT_EQ(1.0, m(0, 0));
    ASSERT_EQ(2.0, m(1, 0));
    ASSERT_EQ(3.0, m(1, 1));
    ASSERT_EQ(4.0, m(2, 0));
    ASSERT_EQ(5.0, m(2, 1));
    ASSERT_EQ(6.0, m(2, 2));
}

}  // namespace tri::test