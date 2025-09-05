/**
 * @file test_gemm.cpp
 * @brief Unit tests for GEMM (General Matrix Multiplication)
 * @author Yongze
 * @date 2025-08-14
 */

#include <cmath>
#include <vector>

#include "framework/test_assertions.hpp"
#include "framework/test_runner.hpp"
#include "framework/test_utils.hpp"
#include "tri/blas/gemm.hpp"

namespace tri::test {

class GemmTestDouble : public TestCase {
   protected:
    double tolerance_ = 1e-10;

    // Helper to create test matrices
    std::vector<double> CreateTestMatrix(std::size_t rows, std::size_t cols,
                                         double start_val = 1.0) {
        std::vector<double> mat(rows * cols);
        for (std::size_t i = 0; i < rows * cols; ++i) {
            mat[i] = start_val + static_cast<double>(i);
        }
        return mat;
    }

    // Manual matrix multiplication for verification
    void ManualGemm(const double* A, const double* B, double* C, std::size_t m, std::size_t n,
                    std::size_t k, std::size_t lda, std::size_t ldb, std::size_t ldc, double alpha,
                    double beta, bool transA = false, bool transB = false) {
        // First scale C by beta
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                C[i * ldc + j] *= beta;
            }
        }

        // Then add alpha * A * B
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                double sum = 0.0;
                for (std::size_t l = 0; l < k; ++l) {
                    double a_val = transA ? A[l * lda + i] : A[i * lda + l];
                    double b_val = transB ? B[j * ldb + l] : B[l * ldb + j];
                    sum += a_val * b_val;
                }
                C[i * ldc + j] += alpha * sum;
            }
        }
    }
};

TEST_F(GemmTestDouble, BasicMultiplication) {
    const std::size_t m = 3, n = 4, k = 2;

    std::vector<double> A = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};  // 3x2

    std::vector<double> B = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0};  // 2x4

    std::vector<double> C(m * n, 0.0);  // 3x4
    std::vector<double> C_expected(m * n, 0.0);

    // Compute C = A * B
    blas::gemm(common::TransposeOp::NoTrans, common::TransposeOp::NoTrans, m, n, k, 1.0, A.data(),
               k, B.data(), n, 0.0, C.data(), n);

    // Compute expected result manually
    ManualGemm(A.data(), B.data(), C_expected.data(), m, n, k, k, n, n, 1.0, 0.0);

    // Verify
    for (std::size_t i = 0; i < m * n; ++i) {
        ASSERT_NEAR(C_expected[i], C[i], tolerance_);
    }
}

TEST_F(GemmTestDouble, TransposeA) {
    const std::size_t m = 2, n = 3, k = 4;

    std::vector<double> A = {1.0, 2.0,  // 4x2 (will be transposed to 2x4)
                             3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    std::vector<double> B = {1.0, 0.0, 1.0,  // 4x3
                             0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0};

    std::vector<double> C(m * n, 0.0);

    // C = A^T * B
    blas::gemm(common::TransposeOp::Trans, common::TransposeOp::NoTrans, m, n, k, 1.0, A.data(), m,
               B.data(), n, 0.0, C.data(), n);

    // Expected result
    std::vector<double> expected(m * n);
    ManualGemm(A.data(), B.data(), expected.data(), m, n, k, m, n, n, 1.0, 0.0, true, false);

    for (std::size_t i = 0; i < m * n; ++i) {
        ASSERT_NEAR(expected[i], C[i], tolerance_);
    }
}

TEST_F(GemmTestDouble, TransposeB) {
    const std::size_t m = 3, n = 2, k = 3;

    std::vector<double> A = {1.0, 2.0, 3.0,  // 3x3
                             4.0, 5.0, 6.0, 7.0, 8.0, 9.0};

    std::vector<double> B = {1.0, 0.0, 1.0,  // 2x3 (will be transposed to 3x2)
                             0.0, 1.0, 0.0};

    std::vector<double> C(m * n, 0.0);

    // C = A * B^T
    blas::gemm(common::TransposeOp::NoTrans, common::TransposeOp::Trans, m, n, k, 1.0, A.data(), k,
               B.data(), k, 0.0, C.data(), n);

    std::vector<double> expected(m * n);
    ManualGemm(A.data(), B.data(), expected.data(), m, n, k, k, k, n, 1.0, 0.0, false, true);

    for (std::size_t i = 0; i < m * n; ++i) {
        ASSERT_NEAR(expected[i], C[i], tolerance_);
    }
}

TEST_F(GemmTestDouble, BothTransposed) {
    const std::size_t m = 2, n = 2, k = 3;

    std::vector<double> A = {1.0, 2.0,  // 3x2 (transposed to 2x3)
                             3.0, 4.0, 5.0, 6.0};

    std::vector<double> B = {7.0,  8.0,  9.0,  // 2x3 (transposed to 3x2)
                             10.0, 11.0, 12.0};

    std::vector<double> C(m * n, 0.0);

    // C = A^T * B^T
    // A is 3x2, so lda = 2 (number of columns in original A)
    // B is 2x3, so ldb = 3 (number of columns in original B)
    blas::gemm(common::TransposeOp::Trans, common::TransposeOp::Trans, m, n, k, 1.0, A.data(), 2,
               B.data(), 3, 0.0, C.data(), n);

    std::vector<double> expected(m * n);
    ManualGemm(A.data(), B.data(), expected.data(), m, n, k, 2, 3, n, 1.0, 0.0, true, true);

    for (std::size_t i = 0; i < m * n; ++i) {
        ASSERT_NEAR(expected[i], C[i], tolerance_);
    }
}

TEST_F(GemmTestDouble, AlphaBetaScaling) {
    const std::size_t m = 2, n = 2, k = 2;

    std::vector<double> A = {1.0, 2.0, 3.0, 4.0};

    std::vector<double> B = {5.0, 6.0, 7.0, 8.0};

    std::vector<double> C = {1.0, 1.0, 1.0, 1.0};

    double alpha = 2.0;
    double beta = 3.0;

    // C = alpha * A * B + beta * C
    blas::gemm(common::TransposeOp::NoTrans, common::TransposeOp::NoTrans, m, n, k, alpha, A.data(),
               k, B.data(), n, beta, C.data(), n);

    // Expected: 2*[[19,22],[43,50]] + 3*[[1,1],[1,1]] = [[41,47],[89,103]]
    std::vector<double> expected = {41.0, 47.0, 89.0, 103.0};

    for (std::size_t i = 0; i < m * n; ++i) {
        ASSERT_NEAR(expected[i], C[i], tolerance_);
    }
}

TEST_F(GemmTestDouble, SquareMatrices) {
    const std::size_t n = 4;

    auto A = CreateTestMatrix(n, n, 1.0);
    auto B = CreateTestMatrix(n, n, 0.5);
    std::vector<double> C(n * n, 0.0);

    blas::gemm(common::TransposeOp::NoTrans, common::TransposeOp::NoTrans, n, n, n, 1.0, A.data(),
               n, B.data(), n, 0.0, C.data(), n);

    // Verify result is computed
    bool all_zero = true;
    for (const auto& val : C) {
        if (std::abs(val) > tolerance_) {
            all_zero = false;
            break;
        }
    }
    ASSERT_FALSE(all_zero);
}

TEST_F(GemmTestDouble, LargeMatrices) {
    const std::size_t m = 100, n = 80, k = 60;

    auto A = CreateTestMatrix(m, k);
    auto B = CreateTestMatrix(k, n);
    std::vector<double> C(m * n, 0.0);

    // Just test that it runs without error
    ASSERT_NO_THROW(blas::gemm(common::TransposeOp::NoTrans, common::TransposeOp::NoTrans, m, n, k,
                               1.0, A.data(), k, B.data(), n, 0.0, C.data(), n));
}

TEST_F(GemmTestDouble, ZeroAlpha) {
    const std::size_t n = 3;

    auto A = CreateTestMatrix(n, n);
    auto B = CreateTestMatrix(n, n);
    std::vector<double> C = CreateTestMatrix(n, n, 10.0);
    auto C_orig = C;

    // C = 0 * A * B + 1 * C = C
    blas::gemm(common::TransposeOp::NoTrans, common::TransposeOp::NoTrans, n, n, n, 0.0, A.data(),
               n, B.data(), n, 1.0, C.data(), n);

    // C should be unchanged
    for (std::size_t i = 0; i < n * n; ++i) {
        ASSERT_NEAR(C_orig[i], C[i], tolerance_);
    }
}

TEST_F(GemmTestDouble, ZeroBeta) {
    const std::size_t n = 3;

    auto A = CreateTestMatrix(n, n);
    auto B = CreateTestMatrix(n, n);
    std::vector<double> C = CreateTestMatrix(n, n, 100.0);  // Initial values should be ignored

    // C = 1 * A * B + 0 * C = A * B
    blas::gemm(common::TransposeOp::NoTrans, common::TransposeOp::NoTrans, n, n, n, 1.0, A.data(),
               n, B.data(), n, 0.0, C.data(), n);

    // Compute expected
    std::vector<double> expected(n * n, 0.0);
    ManualGemm(A.data(), B.data(), expected.data(), n, n, n, n, n, n, 1.0, 0.0);

    for (std::size_t i = 0; i < n * n; ++i) {
        ASSERT_NEAR(expected[i], C[i], tolerance_);
    }
}

// Float precision tests
class GemmTestFloat : public TestCase {
   protected:
    float tolerance_ = 1e-5f;
};

TEST_F(GemmTestFloat, BasicFloat) {
    const std::size_t m = 2, n = 3, k = 2;

    std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f};

    std::vector<float> B = {5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};

    std::vector<float> C(m * n, 0.0f);

    blas::gemm(common::TransposeOp::NoTrans, common::TransposeOp::NoTrans, m, n, k, 1.0f, A.data(),
               k, B.data(), n, 0.0f, C.data(), n);

    // Expected: [[21,24,27],[47,54,61]]
    std::vector<float> expected = {21.0f, 24.0f, 27.0f, 47.0f, 54.0f, 61.0f};

    for (std::size_t i = 0; i < m * n; ++i) {
        ASSERT_NEAR(expected[i], C[i], tolerance_);
    }
}

}  // namespace tri::test