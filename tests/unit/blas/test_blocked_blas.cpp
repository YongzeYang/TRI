/**
 * @file test_blocked_blas.cpp
 * @brief Unit tests for BLAS operations with BlockedTriMatrix
 * @author Yongze
 * @date 2025-08-14
 */

#include <cmath>
#include <vector>

#include "framework/test_assertions.hpp"
#include "framework/test_runner.hpp"
#include "framework/test_utils.hpp"
#include "tri/blas/gemm.hpp"
#include "tri/blas/solver.hpp"
#include "tri/core/blocked_tri.hpp"
#include "tri/core/dense_rm.hpp"

namespace tri::test {

class BlockedBLASTestDouble : public TestCase {
   protected:
    double tolerance_ = 1e-10;
    static constexpr std::size_t DEFAULT_BLOCK_SIZE = 4;

    // Helper to convert BlockedTriMatrix to dense for BLAS operations
    std::vector<double> BlockedToDense(const core::BlockedTriMatrix<double>& blocked) {
        std::size_t n = blocked.rows();
        std::vector<double> dense(n * n, 0.0);

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                dense[i * n + j] = blocked(i, j);
            }
        }

        return dense;
    }

    // Helper to create a test blocked matrix
    core::BlockedTriMatrix<double> CreateTestBlocked(std::size_t n, std::size_t block_size) {
        core::BlockedTriMatrix<double> mat(n, block_size);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                mat(i, j) = static_cast<double>(i * n + j + 1);
            }
        }
        return mat;
    }
};

TEST_F(BlockedBLASTestDouble, BlockedToGemm) {
    const std::size_t n = 16;
    const std::size_t block_size = 4;

    // Create blocked triangular matrix
    auto blocked = CreateTestBlocked(n, block_size);

    // Convert to dense
    auto L = BlockedToDense(blocked);

    // Create another matrix for multiplication
    std::vector<double> B(n * n);
    for (std::size_t i = 0; i < n * n; ++i) {
        B[i] = static_cast<double>(i) * 0.1;
    }

    std::vector<double> C(n * n, 0.0);

    // Perform GEMM: C = L * B
    blas::gemm(common::TransposeOp::NoTrans, common::TransposeOp::NoTrans, n, n, n, 1.0, L.data(),
               n, B.data(), n, 0.0, C.data(), n);

    // Verify result is computed
    bool has_nonzero = false;
    for (const auto& val : C) {
        if (std::abs(val) > tolerance_) {
            has_nonzero = true;
            break;
        }
    }
    ASSERT_TRUE(has_nonzero);
}

TEST_F(BlockedBLASTestDouble, BlockedTriangularSolve) {
    const std::size_t n = 12;
    const std::size_t block_size = 3;

    // Create blocked lower triangular matrix with non-zero diagonal
    core::BlockedTriMatrix<double> blocked(n, block_size);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            if (i == j) {
                blocked(i, j) = static_cast<double>(i + 1);  // Diagonal
            } else {
                blocked(i, j) = static_cast<double>(i - j) * 0.1;  // Lower triangle
            }
        }
    }

    // Convert to dense
    auto L = BlockedToDense(blocked);

    // Create right-hand side
    std::vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i) {
        b[i] = static_cast<double>(i + 1);
    }

    // Solve L * x = b using TRSM
    blas::trsm(common::Side::Left, common::Uplo::Lower, common::TransposeOp::NoTrans,
               common::Diag::NonUnit, n, 1, 1.0, L.data(), n, b.data(), 1);

    // Verify solution by computing L * x
    std::vector<double> result(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            result[i] += L[i * n + j] * b[j];
        }
    }

    // Should get back original b
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(static_cast<double>(i + 1), result[i], tolerance_ * 100);
    }
}

TEST_F(BlockedBLASTestDouble, BlockedMultiplication) {
    const std::size_t n = 8;
    const std::size_t block_size = 2;

    // Create two blocked matrices
    core::BlockedTriMatrix<double> L1(n, block_size);
    core::BlockedTriMatrix<double> L2(n, block_size);

    // Initialize with test data
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            L1(i, j) = (i == j) ? 2.0 : 1.0;
            L2(i, j) = (i == j) ? 3.0 : 0.5;
        }
    }

    // Convert to dense
    auto D1 = BlockedToDense(L1);
    auto D2 = BlockedToDense(L2);

    // Multiply using TRMM
    blas::trmm(common::Side::Left, common::Uplo::Lower, common::TransposeOp::NoTrans,
               common::Diag::NonUnit, n, n, 1.0, D1.data(), n, D2.data(), n);

    // D2 now contains L1 * L2
    // Verify diagonal elements are products of original diagonals
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(6.0, D2[i * n + i], tolerance_);  // 2.0 * 3.0 = 6.0
    }
}

TEST_F(BlockedBLASTestDouble, LargeBlockedMatrix) {
    const std::size_t n = 128;
    const std::size_t block_size = 16;

    // Create large blocked matrix
    core::BlockedTriMatrix<double> blocked(n, block_size);

    // Initialize as identity
    for (std::size_t i = 0; i < n; ++i) {
        blocked(i, i) = 1.0;
    }

    // Convert to dense
    auto I = BlockedToDense(blocked);

    // Create vector
    std::vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<double>(i);
    }
    auto x_orig = x;

    // Multiply I * x using GEMM (treating x as nx1 matrix)
    std::vector<double> result(n, 0.0);
    blas::gemm(common::TransposeOp::NoTrans, common::TransposeOp::NoTrans, n, 1, n, 1.0, I.data(),
               n, x.data(), 1, 0.0, result.data(), 1);

    // Result should equal x
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(x_orig[i], result[i], tolerance_);
    }
}

TEST_F(BlockedBLASTestDouble, MixedBlockSizes) {
    // Test with different block sizes
    std::vector<std::size_t> block_sizes = {2, 4, 8, 16, 32};
    const std::size_t n = 32;

    for (auto block_size : block_sizes) {
        // Create blocked matrix
        core::BlockedTriMatrix<double> blocked(n, block_size);

        // Fill with test pattern
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                blocked(i, j) = std::sin(static_cast<double>(i + j));
            }
        }

        // Convert to dense
        auto dense = BlockedToDense(blocked);

        // Verify conversion preserves values
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                ASSERT_NEAR(blocked(i, j), dense[i * n + j], tolerance_);
            }
        }
    }
}

TEST_F(BlockedBLASTestDouble, BlockBoundaryAccess) {
    const std::size_t n = 17;  // Not divisible by block size
    const std::size_t block_size = 4;

    core::BlockedTriMatrix<double> blocked(n, block_size);

    // Test boundary blocks
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            blocked(i, j) = static_cast<double>(i * 100 + j);
        }
    }

    // Convert and verify
    auto dense = BlockedToDense(blocked);

    // Test that boundary elements are correct
    ASSERT_NEAR(blocked(n - 1, n - 1), dense[(n - 1) * n + (n - 1)], tolerance_);
    ASSERT_NEAR(blocked(n - 1, 0), dense[(n - 1) * n], tolerance_);
}

// Performance comparison test
TEST_F(BlockedBLASTestDouble, BlockedVsDensePerformance) {
    const std::size_t n = 256;
    const std::size_t block_size = 32;

    // Create blocked matrix
    auto blocked = CreateTestBlocked(n, block_size);

    // Convert to dense
    auto dense = BlockedToDense(blocked);

    // Create test vector
    std::vector<double> x(n, 1.0);
    std::vector<double> y1(n, 0.0);
    std::vector<double> y2(n, 0.0);

    // Time dense multiplication
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10; ++iter) {
        blas::gemm(common::TransposeOp::NoTrans, common::TransposeOp::NoTrans, n, 1, n, 1.0,
                   dense.data(), n, x.data(), 1, 0.0, y1.data(), 1);
    }
    auto dense_time = std::chrono::high_resolution_clock::now() - start;

    // Time blocked access (simulated)
    start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < 10; ++iter) {
        for (std::size_t i = 0; i < n; ++i) {
            y2[i] = 0.0;
            for (std::size_t j = 0; j <= i; ++j) {
                y2[i] += blocked(i, j) * x[j];
            }
        }
    }
    auto blocked_time = std::chrono::high_resolution_clock::now() - start;

    // Results should be similar
    for (std::size_t i = 0; i < n; ++i) {
        ASSERT_NEAR(y1[i], y2[i], tolerance_ * 1000);  // Relax tolerance for accumulated errors
    }

    // Report times (informational)
    auto dense_ms = std::chrono::duration<double, std::milli>(dense_time).count();
    auto blocked_ms = std::chrono::duration<double, std::milli>(blocked_time).count();
    std::cout << "\n  Dense BLAS time: " << dense_ms / 10 << " ms/iter\n";
    std::cout << "  Blocked access time: " << blocked_ms / 10 << " ms/iter\n";
}

}  // namespace tri::test