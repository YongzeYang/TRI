/**
 * @file test_dense_factory.cpp
 * @brief Unit tests for DenseMatrixFactory
 * @author Yongze
 * @date 2025-08-14
 */

#include <iostream>

#include "framework/test_assertions.hpp"
#include "framework/test_runner.hpp"
#include "framework/test_utils.hpp"
#include "tri/factory/dense_factory.hpp"

namespace tri::test {

// Test fixture for DenseMatrixFactory - NO TEMPLATES IN CLASS NAME!
class DenseFactoryTestDouble : public TestCase {
   protected:
    double tolerance_ = 1e-10;
};

class DenseFactoryTestFloat : public TestCase {
   protected:
    float tolerance_ = 1e-5f;
};

class DenseFactoryTestInt : public TestCase {
   protected:
    int tolerance_ = 0;
};

// Double precision tests
TEST_F(DenseFactoryTestDouble, Identity) {
    auto m = factory::DenseMatrixFactory<double>::identity(5);

    ASSERT_EQ(5u, m.rows());
    ASSERT_EQ(5u, m.cols());
    ValidateIdentityMatrix(m, tolerance_);
}

TEST_F(DenseFactoryTestDouble, ZerosRectangular) {
    auto m = factory::DenseMatrixFactory<double>::zeros(4, 6);

    ASSERT_EQ(4u, m.rows());
    ASSERT_EQ(6u, m.cols());
    ValidateZeroMatrix(m, tolerance_);
}

TEST_F(DenseFactoryTestDouble, ZerosSquare) {
    auto m = factory::DenseMatrixFactory<double>::zeros(5);

    ASSERT_EQ(5u, m.rows());
    ASSERT_EQ(5u, m.cols());
    ValidateZeroMatrix(m, tolerance_);
}

TEST_F(DenseFactoryTestDouble, OnesRectangular) {
    auto m = factory::DenseMatrixFactory<double>::ones(3, 4);

    ASSERT_EQ(3u, m.rows());
    ASSERT_EQ(4u, m.cols());

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            ASSERT_EQ(1.0, m(i, j));
        }
    }
}

TEST_F(DenseFactoryTestDouble, OnesSquare) {
    auto m = factory::DenseMatrixFactory<double>::ones(4);

    ASSERT_EQ(4u, m.rows());
    ASSERT_EQ(4u, m.cols());

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            ASSERT_EQ(1.0, m(i, j));
        }
    }
}

TEST_F(DenseFactoryTestDouble, ConstantRectangular) {
    const double value = 3.14;
    auto m = factory::DenseMatrixFactory<double>::constant(3, 5, value);

    ASSERT_EQ(3u, m.rows());
    ASSERT_EQ(5u, m.cols());

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            ASSERT_NEAR(value, m(i, j), tolerance_);
        }
    }
}

TEST_F(DenseFactoryTestDouble, Diagonal) {
    std::vector<double> diag_values{1.0, 2.0, 3.0, 4.0};
    auto m = factory::DenseMatrixFactory<double>::diagonal(diag_values);

    ASSERT_EQ(4u, m.rows());
    ASSERT_EQ(4u, m.cols());

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            if (i == j) {
                ASSERT_EQ(diag_values[i], m(i, j));
            } else {
                ASSERT_EQ(0.0, m(i, j));
            }
        }
    }
}

TEST_F(DenseFactoryTestDouble, RandomWithBounds) {
    const double min_val = -5.0;
    const double max_val = 5.0;
    auto m = factory::DenseMatrixFactory<double>::random(10, 8, min_val, max_val);

    ASSERT_EQ(10u, m.rows());
    ASSERT_EQ(8u, m.cols());

    // Check all values are in range
    for (std::size_t i = 0; i < 10; ++i) {
        for (std::size_t j = 0; j < 8; ++j) {
            double val = m(i, j);
            ASSERT_GE(val, min_val);
            ASSERT_LE(val, max_val);
        }
    }
}

TEST_F(DenseFactoryTestDouble, RandomDefaultBounds) {
    // Test with default bounds [0, 1]
    auto m = factory::DenseMatrixFactory<double>::random(5, 5);

    ASSERT_EQ(5u, m.rows());
    ASSERT_EQ(5u, m.cols());

    // Check all values are in range [0, 1]
    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            double val = m(i, j);
            ASSERT_GE(val, 0.0);
            ASSERT_LE(val, 1.0);
        }
    }
}

// Float precision tests
TEST_F(DenseFactoryTestFloat, IdentityFloat) {
    auto m = factory::DenseMatrixFactory<float>::identity(3);

    ASSERT_EQ(3u, m.rows());
    ASSERT_EQ(3u, m.cols());
    ValidateIdentityMatrix(m, tolerance_);
}

TEST_F(DenseFactoryTestFloat, DiagonalFloat) {
    std::vector<float> diag_values{1.5f, 2.5f, 3.5f};
    auto m = factory::DenseMatrixFactory<float>::diagonal(diag_values);

    ASSERT_EQ(3u, m.rows());
    ASSERT_EQ(3u, m.cols());

    ASSERT_EQ(1.5f, m(0, 0));
    ASSERT_EQ(2.5f, m(1, 1));
    ASSERT_EQ(3.5f, m(2, 2));
    ASSERT_EQ(0.0f, m(0, 1));
}

TEST_F(DenseFactoryTestFloat, OnesFloat) {
    auto m = factory::DenseMatrixFactory<float>::ones(2, 3);

    ASSERT_EQ(2u, m.rows());
    ASSERT_EQ(3u, m.cols());

    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            ASSERT_EQ(1.0f, m(i, j));
        }
    }
}

TEST_F(DenseFactoryTestFloat, ZerosFloat) {
    auto m = factory::DenseMatrixFactory<float>::zeros(4, 4);

    ASSERT_EQ(4u, m.rows());
    ASSERT_EQ(4u, m.cols());
    ValidateZeroMatrix(m, tolerance_);
}

// Integer tests
TEST_F(DenseFactoryTestInt, ConstantInt) {
    const int value = 42;
    auto m = factory::DenseMatrixFactory<int>::constant(3, 3, value);

    ASSERT_EQ(3u, m.rows());
    ASSERT_EQ(3u, m.cols());

    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            ASSERT_EQ(value, m(i, j));
        }
    }
}

TEST_F(DenseFactoryTestInt, OnesInt) {
    auto m = factory::DenseMatrixFactory<int>::ones(4, 2);

    ASSERT_EQ(4u, m.rows());
    ASSERT_EQ(2u, m.cols());

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            ASSERT_EQ(1, m(i, j));
        }
    }
}

// Tests without fixtures (using TEST macro)
TEST(DenseFactory, IdentityData) {
    // Create identity matrix and verify
    auto m = factory::DenseMatrixFactory<double>::identity(3);

    ASSERT_EQ(3u, m.rows());
    ASSERT_EQ(3u, m.cols());

    // Check diagonal elements
    ASSERT_EQ(1.0, m(0, 0));
    ASSERT_EQ(1.0, m(1, 1));
    ASSERT_EQ(1.0, m(2, 2));

    // Check off-diagonal elements
    ASSERT_EQ(0.0, m(0, 1));
    ASSERT_EQ(0.0, m(0, 2));
    ASSERT_EQ(0.0, m(1, 0));
    ASSERT_EQ(0.0, m(1, 2));
    ASSERT_EQ(0.0, m(2, 0));
    ASSERT_EQ(0.0, m(2, 1));
}

TEST(DenseFactory, DiagonalWithVector) {
    std::vector<float> diag{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto m = factory::DenseMatrixFactory<float>::diagonal(diag);

    ASSERT_EQ(5u, m.rows());
    ASSERT_EQ(5u, m.cols());

    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j < 5; ++j) {
            if (i == j) {
                ASSERT_EQ(diag[i], m(i, j));
            } else {
                ASSERT_EQ(0.0f, m(i, j));
            }
        }
    }
}

TEST(DenseFactory, RandomIntegerRange) {
    const int min_val = -100;
    const int max_val = 100;
    auto m = factory::DenseMatrixFactory<int>::random(10, 10, min_val, max_val);

    ASSERT_EQ(10u, m.rows());
    ASSERT_EQ(10u, m.cols());

    // Check all values are in range
    for (std::size_t i = 0; i < 10; ++i) {
        for (std::size_t j = 0; j < 10; ++j) {
            int val = m(i, j);
            ASSERT_GE(val, min_val);
            ASSERT_LE(val, max_val);
        }
    }
}

// Test that different factory methods produce different matrix types
TEST(DenseFactory, DifferentFactoryMethods) {
    const std::size_t n = 4;

    auto identity = factory::DenseMatrixFactory<double>::identity(n);
    auto zeros = factory::DenseMatrixFactory<double>::zeros(n);
    auto ones = factory::DenseMatrixFactory<double>::ones(n);

    // All should have same dimensions
    ASSERT_EQ(n, identity.rows());
    ASSERT_EQ(n, zeros.rows());
    ASSERT_EQ(n, ones.rows());

    // But different values
    ASSERT_EQ(1.0, identity(0, 0));
    ASSERT_EQ(0.0, identity(0, 1));

    ASSERT_EQ(0.0, zeros(0, 0));
    ASSERT_EQ(0.0, zeros(1, 1));

    ASSERT_EQ(1.0, ones(0, 0));
    ASSERT_EQ(1.0, ones(1, 1));
}

}  // namespace tri::test