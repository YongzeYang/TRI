/**
 * @file test_lower_tri_rm.cpp
 * @brief Unit tests for LowerTriangularRM matrix class
 * @author Yongze
 * @date 2025-08-14
 */

#include <iostream>

#include "test_assertions.hpp"
#include "test_runner.hpp"
#include "test_utils.hpp"
#include "tri/core/dense_rm.hpp"
#include "tri/core/lower_tri_rm.hpp"

namespace tri::test {

// Test fixture for double precision
class LowerTriangularRMTestDouble : public TestCase {
   protected:
    double tolerance_ = 1e-10;
};

// Test fixture for float precision
class LowerTriangularRMTestFloat : public TestCase {
   protected:
    float tolerance_ = 1e-5f;
};

// Double precision tests
TEST_F(LowerTriangularRMTestDouble, DefaultConstruction) {
    core::LowerTriangularRM<double> m;
    ASSERT_EQ(0u, m.rows());
    ASSERT_EQ(0u, m.cols());
    ASSERT_EQ(0u, m.dimension());
    ASSERT_TRUE(m.empty());
    ASSERT_EQ(0u, m.size());
    ASSERT_EQ(0u, m.packed_size());
}

TEST_F(LowerTriangularRMTestDouble, SizeConstruction) {
    const std::size_t n = 5;
    core::LowerTriangularRM<double> m(n);

    ASSERT_EQ(n, m.rows());
    ASSERT_EQ(n, m.cols());
    ASSERT_EQ(n, m.dimension());
    ASSERT_EQ(n * n, m.size());
    ASSERT_EQ(n * (n + 1) / 2, m.packed_size());
    ASSERT_FALSE(m.empty());
    ASSERT_TRUE(m.is_square());
}

TEST_F(LowerTriangularRMTestDouble, ValueConstruction) {
    const std::size_t n = 4;
    const double value = 2.5;
    core::LowerTriangularRM<double> m(n, value);

    ValidateSquareMatrix(m, n);

    // Check lower triangular part
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            ASSERT_NEAR(value, m(i, j), tolerance_);
        }
    }

    // Check upper triangular part is zero
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const auto& const_m = m;  // 创建const引用
            ASSERT_NEAR(0.0, const_m(i, j), tolerance_);
        }
    }
}

TEST_F(LowerTriangularRMTestDouble, ConstructionFromDense) {
    core::DenseRM<double> dense(3, 3);
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            dense(i, j) = static_cast<double>(i * 3 + j + 1);
        }
    }

    core::LowerTriangularRM<double> m(dense);

    ASSERT_EQ(3u, m.dimension());

    // Check that lower triangular part is copied
    ASSERT_EQ(1.0, m(0, 0));
    ASSERT_EQ(4.0, m(1, 0));
    ASSERT_EQ(5.0, m(1, 1));
    ASSERT_EQ(7.0, m(2, 0));
    ASSERT_EQ(8.0, m(2, 1));
    ASSERT_EQ(9.0, m(2, 2));

    // Upper part should be zero
    const auto& const_m = m;  // 创建const引用
    ASSERT_EQ(0.0, const_m(0, 1));
    ASSERT_EQ(0.0, const_m(0, 2));
    ASSERT_EQ(0.0, const_m(1, 2));
}

TEST_F(LowerTriangularRMTestDouble, PackedDataConstruction) {
    std::vector<double> packed{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};  // 3x3 lower triangular
    core::LowerTriangularRM<double> m(3, packed);

    ASSERT_EQ(3u, m.dimension());
    ASSERT_EQ(1.0, m(0, 0));
    ASSERT_EQ(2.0, m(1, 0));
    ASSERT_EQ(3.0, m(1, 1));
    ASSERT_EQ(4.0, m(2, 0));
    ASSERT_EQ(5.0, m(2, 1));
    ASSERT_EQ(6.0, m(2, 2));
}

TEST_F(LowerTriangularRMTestDouble, Fill) {
    core::LowerTriangularRM<double> m(4);
    const double fill_value = 3.14;

    m.fill(fill_value);

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            ASSERT_EQ(fill_value, m(i, j));
        }
    }
}

TEST_F(LowerTriangularRMTestDouble, SetDiagonal) {
    core::LowerTriangularRM<double> m(4, 2.0);
    const double diag_value = 1.0;

    m.set_diagonal(diag_value);

    for (std::size_t i = 0; i < 4; ++i) {
        ASSERT_EQ(diag_value, m(i, i));
        if (i > 0) {
            ASSERT_EQ(2.0, m(i, 0));  // Non-diagonal element unchanged
        }
    }
}

TEST_F(LowerTriangularRMTestDouble, Resize) {
    core::LowerTriangularRM<double> m(3, 1.0);

    m.resize(5);
    ASSERT_EQ(5u, m.dimension());
    ASSERT_EQ(15u, m.packed_size());

    // After resize, elements should be zero-initialized
    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            ASSERT_EQ(0.0, m(i, j));
        }
    }
}

TEST_F(LowerTriangularRMTestDouble, Clear) {
    core::LowerTriangularRM<double> m(4);
    ASSERT_FALSE(m.empty());

    m.clear();
    ASSERT_TRUE(m.empty());
    ASSERT_EQ(0u, m.dimension());
    ASSERT_EQ(0u, m.packed_size());
}

// Float precision tests
TEST_F(LowerTriangularRMTestFloat, ElementAccess) {
    const std::size_t n = 4;
    core::LowerTriangularRM<float> m(n);

    // Set elements in lower triangle
    float value = 1.0f;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            m.set(i, j, value++);
        }
    }

    // Verify elements
    value = 1.0f;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            ASSERT_EQ(value++, m(i, j));
        }
    }
}

TEST_F(LowerTriangularRMTestFloat, CopyConstruction) {
    core::LowerTriangularRM<float> m1(3);
    m1(0, 0) = 1.0f;
    m1(1, 0) = 2.0f;
    m1(1, 1) = 3.0f;
    m1(2, 0) = 4.0f;
    m1(2, 1) = 5.0f;
    m1(2, 2) = 6.0f;

    core::LowerTriangularRM<float> m2(m1);

    ASSERT_EQ(m1.dimension(), m2.dimension());
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            ASSERT_EQ(m1(i, j), m2(i, j));
        }
    }

    // Ensure deep copy
    m2(0, 0) = 999.0f;
    ASSERT_NE(m1(0, 0), m2(0, 0));
}

// Non-fixture tests
TEST(LowerTriangularRM, UpperTriangleAccess) {
    const core::LowerTriangularRM<double> m(3, 1.0);

    // Upper triangle should always return 0
    ASSERT_EQ(0.0, m(0, 1));
    ASSERT_EQ(0.0, m(0, 2));
    ASSERT_EQ(0.0, m(1, 2));
}

TEST(LowerTriangularRM, InvalidSetThrows) {
    core::LowerTriangularRM<double> m(4);

    // Setting upper triangle should throw
    ASSERT_THROW(m.set(0, 1, 10.0), std::logic_error);
    ASSERT_THROW(m.set(1, 3, 10.0), std::logic_error);
    ASSERT_THROW(m(0, 2) = 10.0, std::logic_error);
}

TEST(LowerTriangularRM, StaticPackedSize) {
    ASSERT_EQ(1u, core::LowerTriangularRM<double>::packed_size(1));
    ASSERT_EQ(3u, core::LowerTriangularRM<double>::packed_size(2));
    ASSERT_EQ(6u, core::LowerTriangularRM<double>::packed_size(3));
    ASSERT_EQ(10u, core::LowerTriangularRM<double>::packed_size(4));
    ASSERT_EQ(15u, core::LowerTriangularRM<double>::packed_size(5));
    ASSERT_EQ(55u, core::LowerTriangularRM<double>::packed_size(10));
}

TEST(LowerTriangularRM, StaticFactories) {
    // Identity
    auto id = core::LowerTriangularRM<double>::identity(5);
    ValidateSquareMatrix(id, 5u);

    for (std::size_t i = 0; i < 5; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            if (i == j) {
                ASSERT_EQ(1.0, id(i, j));
            } else {
                ASSERT_EQ(0.0, id(i, j));
            }
        }
    }

    // Zeros
    auto z = core::LowerTriangularRM<double>::zeros(4);
    ValidateSquareMatrix(z, 4u);

    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            ASSERT_EQ(0.0, z(i, j));
        }
    }
}

}  // namespace tri::test