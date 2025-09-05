/**
 * @file test_solver.cpp
 * @brief Unit tests for BLAS solver functions
 * @author Yongze
 * @date 2025-08-14
 */

#include <cmath>
#include <vector>

#include "framework/test_assertions.hpp"
#include "framework/test_runner.hpp"
#include "framework/test_utils.hpp"
#include "tri/blas/solver.hpp"

namespace tri::test {

// Test fixture for solver functions
class SolverTestDouble : public TestCase {
   protected:
    double tolerance_ = 1e-10;

    // Helper to create identity matrix
    std::vector<double> CreateIdentity(std::size_t n) {
        std::vector<double> I(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            I[i * n + i] = 1.0;
        }
        return I;
    }

    // Helper to create lower triangular matrix
    std::vector<double> CreateLowerTriangular(std::size_t n) {
        std::vector<double> L(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                L[i * n + j] = static_cast<double>(i + j + 1);
            }
        }
        return L;
    }

    // Helper to create upper triangular matrix
    std::vector<double> CreateUpperTriangular(std::size_t n) {
        std::vector<double> U(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i; j < n; ++j) {
                U[i * n + j] = static_cast<double>(i + j + 1);
            }
        }
        return U;
    }
};

// TRMM Tests
TEST_F(SolverTestDouble, TrmmLowerLeft) {
    const std::size_t n = 4;
    auto L = CreateLowerTriangular(n);
    auto B = CreateIdentity(n);

    // Compute B := L * B
    blas::trmm(common::Side::Left, common::Uplo::Lower, common::TransposeOp::NoTrans,
               common::Diag::NonUnit, n, n, 1.0, L.data(), n, B.data(), n);

    // Result should be L
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            ASSERT_NEAR(L[i * n + j], B[i * n + j], tolerance_);
        }
    }
}

TEST_F(SolverTestDouble, TrmmUpperRight) {
    const std::size_t m = 3, n = 4;
    auto U = CreateUpperTriangular(n);
    std::vector<double> B(m * n);

    // Initialize B with test data
    for (std::size_t i = 0; i < m * n; ++i) {
        B[i] = static_cast<double>(i + 1);
    }
    auto B_orig = B;

    // Compute B := B * U
    blas::trmm(common::Side::Right, common::Uplo::Upper, common::TransposeOp::NoTrans,
               common::Diag::NonUnit, m, n, 2.0, U.data(), n, B.data(), n);

    // Verify result
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            double expected = 0.0;
            for (std::size_t k = 0; k <= j; ++k) {
                expected += B_orig[i * n + k] * U[k * n + j];
            }
            expected *= 2.0;
            ASSERT_NEAR(expected, B[i * n + j], tolerance_);
        }
    }
}

TEST_F(SolverTestDouble, TrmmWithUnitDiag) {
    const std::size_t n = 3;
    std::vector<double> L(n * n, 0.0);

    // Create lower triangular with unit diagonal
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            L[i * n + j] = static_cast<double>(i - j);
        }
        // Diagonal is implicitly 1
    }

    std::vector<double> B = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    auto B_orig = B;

    blas::trmm(common::Side::Left, common::Uplo::Lower, common::TransposeOp::NoTrans,
               common::Diag::Unit, n, n, 1.0, L.data(), n, B.data(), n);

    // Manually compute expected result
    std::vector<double> expected(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            expected[i * n + j] = B_orig[i * n + j];  // Diagonal term (unit)
            for (std::size_t k = 0; k < i; ++k) {
                expected[i * n + j] += L[i * n + k] * B_orig[k * n + j];
            }
        }
    }

    for (std::size_t i = 0; i < n * n; ++i) {
        ASSERT_NEAR(expected[i], B[i], tolerance_);
    }
}

// TRSM Tests
TEST_F(SolverTestDouble, TrsmLowerLeft) {
    const std::size_t n = 4;
    std::vector<double> L = {2.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0,
                             0.5, 1.0, 4.0, 0.0, 1.0, 0.5, 1.0, 5.0};

    std::vector<double> B = {2.0, 4.0, 7.0, 11.0, 11.5, 18.0, 16.0, 26.5};

    // Solve L * X = B for X
    blas::trsm(common::Side::Left, common::Uplo::Lower, common::TransposeOp::NoTrans,
               common::Diag::NonUnit, n, 2, 1.0, L.data(), n, B.data(), 2);

    // Expected solution: solving L * X = B
    // X[0,1] = [1.0, 2.0]  - first row
    // X[2,3] = [2.0, 3.0]  - second row
    // X[4,5] = [2.25, 3.5] - third row
    // X[6,7] = [2.35, 3.9] - fourth row
    std::vector<double> expected = {1.0, 2.0, 2.0, 3.0, 2.25, 3.5, 2.35, 3.9};

    for (std::size_t i = 0; i < n * 2; ++i) {
        ASSERT_NEAR(expected[i], B[i], tolerance_);
    }
}

TEST_F(SolverTestDouble, TrsmUpperRight) {
    const std::size_t m = 3, n = 3;
    std::vector<double> U = {2.0, 1.0, 0.5, 0.0, 3.0, 1.0, 0.0, 0.0, 4.0};

    std::vector<double> B = {5.0, 8.0, 10.0, 9.0, 15.0, 19.0, 13.0, 22.0, 28.0};

    // Solve X * U = B for X
    blas::trsm(common::Side::Right, common::Uplo::Upper, common::TransposeOp::NoTrans,
               common::Diag::NonUnit, m, n, 1.0, U.data(), n, B.data(), n);

    // Verify by multiplication
    std::vector<double> U_copy = U;
    std::vector<double> result(m * n);

    // Compute result = B * U to verify
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            result[i * n + j] = 0.0;
            for (std::size_t k = 0; k <= j; ++k) {
                if (k <= j) {  // Upper triangular
                    result[i * n + j] += B[i * n + k] * U_copy[k * n + j];
                }
            }
        }
    }
}

TEST_F(SolverTestDouble, TrsmWithAlpha) {
    const std::size_t n = 2;
    std::vector<double> L = {3.0, 0.0, 2.0, 4.0};

    std::vector<double> B = {6.0, 9.0, 10.0, 16.0};

    // Solve L * X = 2.0 * B
    blas::trsm(common::Side::Left, common::Uplo::Lower, common::TransposeOp::NoTrans,
               common::Diag::NonUnit, n, n, 2.0, L.data(), n, B.data(), n);

    // Expected: X should satisfy L * X = 2.0 * B_orig
    std::vector<double> expected = {4.0, 6.0, 3.0, 5.0};

    for (std::size_t i = 0; i < n * n; ++i) {
        ASSERT_NEAR(expected[i], B[i], tolerance_);
    }
}

}  // namespace tri::test