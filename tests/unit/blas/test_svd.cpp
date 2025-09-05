/**
 * @file test_svd.cpp
 * @brief Unit tests for SVD functions
 * @author Yongze
 * @date 2025-08-14
 */

#include <algorithm>
#include <cmath>
#include <vector>

#include "framework/test_assertions.hpp"
#include "framework/test_runner.hpp"
#include "framework/test_utils.hpp"
#include "tri/blas/svd.hpp"

namespace tri::test {

class SVDTestDouble : public TestCase {
   protected:
    double tolerance_ = 1e-8;

    // Helper to verify SVD decomposition
    void VerifySVD(const std::vector<double>& A_orig, const std::vector<double>& U,
                   const std::vector<double>& S, const std::vector<double>& VT, std::size_t m,
                   std::size_t n) {
        std::size_t min_mn = std::min(m, n);

        // Reconstruct A from U*S*V^T
        std::vector<double> A_reconstructed(m * n, 0.0);

        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                for (std::size_t k = 0; k < min_mn; ++k) {
                    A_reconstructed[i * n + j] += U[i * m + k] * S[k] * VT[k * n + j];
                }
            }
        }

        // Compare with original
        for (std::size_t i = 0; i < m * n; ++i) {
            ASSERT_NEAR(A_orig[i], A_reconstructed[i], tolerance_);
        }
    }

    // Helper to verify pseudoinverse properties
    void VerifyPseudoinverse(const std::vector<double>& A, const std::vector<double>& Ainv,
                             std::size_t m, std::size_t n) {
        // Verify A * A+ * A = A
        std::vector<double> temp1(m * m);  // A * A+ is m x m
        std::vector<double> temp2(m * n);  // (A * A+) * A is m x n

        // temp1 = A * A+ (m x n) * (n x m) = (m x m)
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < m; ++j) {
                temp1[i * m + j] = 0.0;
                for (std::size_t k = 0; k < n; ++k) {
                    temp1[i * m + j] += A[i * n + k] * Ainv[k * m + j];
                }
            }
        }

        // temp2 = (A * A+) * A  (m x m) * (m x n) = (m x n)
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                temp2[i * n + j] = 0.0;
                for (std::size_t k = 0; k < m; ++k) {
                    temp2[i * n + j] += temp1[i * m + k] * A[k * n + j];
                }
            }
        }

        // Check temp2 == A
        for (std::size_t i = 0; i < m * n; ++i) {
            ASSERT_NEAR(A[i], temp2[i], tolerance_);
        }
    }
};

TEST_F(SVDTestDouble, BasicSVD_2x2) {
    const std::size_t m = 2, n = 2;
    std::vector<double> A = {3.0, 2.0, 2.0, 3.0};
    auto A_orig = A;

    std::vector<double> S(2);
    std::vector<double> U(4);
    std::vector<double> VT(4);

    int result = blas::gesvd(m, n, A.data(), n, S.data(), U.data(), m, VT.data(), n);

    ASSERT_EQ(0, result);

    // Check singular values are non-negative and sorted
    ASSERT_GE(S[0], 0.0);
    ASSERT_GE(S[1], 0.0);
    ASSERT_GE(S[0], S[1]);

    // For this symmetric matrix, singular values should be 5 and 1
    ASSERT_NEAR(5.0, S[0], tolerance_);
    ASSERT_NEAR(1.0, S[1], tolerance_);

    VerifySVD(A_orig, U, S, VT, m, n);
}

TEST_F(SVDTestDouble, RectangularSVD_3x2) {
    const std::size_t m = 3, n = 2;
    std::vector<double> A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    auto A_orig = A;

    std::vector<double> S(2);
    std::vector<double> U(9);
    std::vector<double> VT(4);

    int result = blas::gesvd(m, n, A.data(), n, S.data(), U.data(), m, VT.data(), n);

    ASSERT_EQ(0, result);

    // Check singular values
    ASSERT_GE(S[0], S[1]);
    ASSERT_GE(S[1], 0.0);

    VerifySVD(A_orig, U, S, VT, m, n);
}

TEST_F(SVDTestDouble, RankDeficientMatrix) {
    const std::size_t m = 3, n = 3;
    std::vector<double> A = {1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0};  // Rank 1 matrix
    auto A_orig = A;

    std::vector<double> S(3);
    std::vector<double> U(9);
    std::vector<double> VT(9);

    int result = blas::gesvd(m, n, A.data(), n, S.data(), U.data(), m, VT.data(), n);

    ASSERT_EQ(0, result);

    // Should have one large singular value and two near-zero
    ASSERT_GT(S[0], 1.0);
    ASSERT_LT(S[1], tolerance_);
    ASSERT_LT(S[2], tolerance_);
}

TEST_F(SVDTestDouble, PseudoinverseSquare) {
    const std::size_t n = 3;
    std::vector<double> A = {1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0};
    auto A_orig = A;

    std::vector<double> Ainv(n * n);

    int result = blas::pinv(n, n, A.data(), n, Ainv.data(), n);

    ASSERT_EQ(0, result);

    VerifyPseudoinverse(A_orig, Ainv, n, n);
}

TEST_F(SVDTestDouble, PseudoinverseRectangular) {
    const std::size_t m = 4, n = 2;
    std::vector<double> A = {1.0, 0.0,  // Row-major: A is 4x2
                             0.0, 1.0, 1.0, 1.0, 0.0, 1.0};
    auto A_orig = A;

    std::vector<double> Ainv(n * m);

    int result = blas::pinv(m, n, A.data(), n, Ainv.data(), m);

    ASSERT_EQ(0, result);

    VerifyPseudoinverse(A_orig, Ainv, m, n);
}

TEST_F(SVDTestDouble, PseudoinverseWithTolerance) {
    const std::size_t n = 2;
    std::vector<double> A = {1.0, 1e-15,  // Near-singular matrix
                             1e-15, 1.0};
    auto A_orig = A;

    std::vector<double> Ainv(n * n);

    // Use explicit tolerance
    int result = blas::pinv(n, n, A.data(), n, Ainv.data(), n, 1e-10);

    ASSERT_EQ(0, result);

    // With tolerance, small singular values should be treated as zero
    // Result should be close to identity
    ASSERT_NEAR(1.0, Ainv[0], tolerance_);
    ASSERT_NEAR(0.0, Ainv[1], tolerance_);
    ASSERT_NEAR(0.0, Ainv[2], tolerance_);
    ASSERT_NEAR(1.0, Ainv[3], tolerance_);
}

TEST_F(SVDTestDouble, IdentityMatrixSVD) {
    const std::size_t n = 3;
    std::vector<double> I(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        I[i * n + i] = 1.0;
    }
    auto I_orig = I;

    std::vector<double> S(n);
    std::vector<double> U(n * n);
    std::vector<double> VT(n * n);

    int result = blas::gesvd(n, n, I.data(), n, S.data(), U.data(), n, VT.data(), n);

    ASSERT_EQ(0, result);

    // All singular values should be 1
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(1.0, S[i], tolerance_);
    }

    VerifySVD(I_orig, U, S, VT, n, n);
}

}  // namespace tri::test