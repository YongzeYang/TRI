#pragma once

/**
 * @file matrix_ops.hpp
 * @brief Basic matrix operations for triangular and dense matrices
 * @author Yongze
 * @date 2025-08-09
 */

#include "tri/core/lower_triangular_rm.hpp"
#include "tri/core/dense_rm.hpp"
#include "tri/blas/gemm.hpp"
#include "tri/blas/solver.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>

#ifdef TRI_USE_BLAS
#include <cblas.h>
#endif

namespace tri {
namespace linalg {

/**
 * @brief Compute Frobenius norm of lower triangular matrix
 * 
 * @tparam T Data type
 * @param L Lower triangular matrix
 * @return Frobenius norm (sqrt of sum of squares)
 */
template<typename T>
[[nodiscard]] inline T frobenius_norm(
    const tri::core::LowerTriangularRM<T>& L) noexcept
{
    const T* data = L.data();
    const std::size_t size = L.packed_size();
    
#ifdef TRI_USE_BLAS
    if constexpr (std::is_same_v<T, float>) {
        return cblas_snrm2(static_cast<int>(size), data, 1);
    } else if constexpr (std::is_same_v<T, double>) {
        return cblas_dnrm2(static_cast<int>(size), data, 1);
    } else {
        T sum = T{0};
        for (std::size_t i = 0; i < size; ++i) {
            sum += data[i] * data[i];
        }
        return std::sqrt(sum);
    }
#else
    T sum = T{0};
    for (std::size_t i = 0; i < size; ++i) {
        sum += data[i] * data[i];
    }
    return std::sqrt(sum);
#endif
}

/**
 * @brief Compute Frobenius norm of dense matrix
 * 
 * @tparam T Data type
 * @param A Dense matrix
 * @return Frobenius norm
 */
template<typename T>
[[nodiscard]] inline T frobenius_norm(
    const tri::core::DenseRM<T>& A) noexcept
{
    const T* data = A.data();
    const std::size_t size = A.size();
    
#ifdef TRI_USE_BLAS
    if constexpr (std::is_same_v<T, float>) {
        return cblas_snrm2(static_cast<int>(size), data, 1);
    } else if constexpr (std::is_same_v<T, double>) {
        return cblas_dnrm2(static_cast<int>(size), data, 1);
    } else {
        T sum = T{0};
        for (std::size_t i = 0; i < size; ++i) {
            sum += data[i] * data[i];
        }
        return std::sqrt(sum);
    }
#else
    T sum = T{0};
    for (std::size_t i = 0; i < size; ++i) {
        sum += data[i] * data[i];
    }
    return std::sqrt(sum);
#endif
}

/**
 * @brief Compute trace of lower triangular matrix
 * 
 * @tparam T Data type
 * @param L Lower triangular matrix
 * @return Sum of diagonal elements
 */
template<typename T>
[[nodiscard]] inline T trace(
    const tri::core::LowerTriangularRM<T>& L) noexcept
{
    T sum = T{0};
    const std::size_t n = L.rows();
    
    for (std::size_t i = 0; i < n; ++i) {
        sum += L(i, i);
    }
    
    return sum;
}

/**
 * @brief Compute trace of dense matrix
 * 
 * @tparam T Data type
 * @param A Dense matrix
 * @return Sum of diagonal elements
 */
template<typename T>
[[nodiscard]] inline T trace(
    const tri::core::DenseRM<T>& A)
{
    if (!A.is_square()) {
        throw std::invalid_argument("Trace requires square matrix");
    }
    
    T sum = T{0};
    const std::size_t n = A.rows();
    
    for (std::size_t i = 0; i < n; ++i) {
        sum += A(i, i);
    }
    
    return sum;
}

/**
 * @brief Matrix-vector multiplication y = L*x for lower triangular
 * 
 * @tparam T Data type
 * @param L Lower triangular matrix
 * @param x Input vector
 * @return Result vector y = L*x
 */
template<typename T>
[[nodiscard]] inline std::vector<T> matvec_multiply(
    const tri::core::LowerTriangularRM<T>& L,
    const std::vector<T>& x)
{
    if (L.cols() != x.size()) {
        throw std::invalid_argument("Dimension mismatch in matrix-vector multiplication");
    }

    const std::size_t n = L.rows();
    std::vector<T> y(n, T{0});

#ifdef TRI_USE_BLAS
    // Use BLAS tpmv for packed triangular matrix-vector product
    y = x;  // tpmv overwrites the vector
    if constexpr (std::is_same_v<T, float>) {
        cblas_stpmv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                    static_cast<int>(n), L.data(), y.data(), 1);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dtpmv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                    static_cast<int>(n), L.data(), y.data(), 1);
    } else {
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                y[i] += L(i, j) * x[j];
            }
        }
    }
#else
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            y[i] += L(i, j) * x[j];
        }
    }
#endif

    return y;
}

/**
 * @brief Matrix-vector multiplication y = A*x for dense matrix
 * 
 * @tparam T Data type
 * @param A Dense matrix (m x n)
 * @param x Vector (n x 1)
 * @return Result vector y (m x 1)
 */
template<typename T>
[[nodiscard]] inline std::vector<T> matvec_multiply(
    const tri::core::DenseRM<T>& A,
    const std::vector<T>& x)
{
    if (A.cols() != x.size()) {
        throw std::invalid_argument("Dimension mismatch in matrix-vector multiplication");
    }
    
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();
    std::vector<T> y(m);
    
#ifdef TRI_USE_BLAS
    if constexpr (std::is_same_v<T, float>) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    static_cast<int>(m), static_cast<int>(n),
                    1.0f, A.data(), static_cast<int>(A.ld()),
                    x.data(), 1,
                    0.0f, y.data(), 1);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dgemv(CblasRowMajor, CblasNoTrans,
                    static_cast<int>(m), static_cast<int>(n),
                    1.0, A.data(), static_cast<int>(A.ld()),
                    x.data(), 1,
                    0.0, y.data(), 1);
    } else {
        for (std::size_t i = 0; i < m; ++i) {
            T sum = T{0};
            for (std::size_t j = 0; j < n; ++j) {
                sum += A(i, j) * x[j];
            }
            y[i] = sum;
        }
    }
#else
    for (std::size_t i = 0; i < m; ++i) {
        T sum = T{0};
        for (std::size_t j = 0; j < n; ++j) {
            sum += A(i, j) * x[j];
        }
        y[i] = sum;
    }
#endif
    
    return y;
}

/**
 * @brief Lower triangular matrix multiplication C = L1 * L2
 * 
 * @tparam T Data type
 * @param L1 First lower triangular matrix
 * @param L2 Second lower triangular matrix
 * @return Product matrix (lower triangular)
 */
template<typename T>
[[nodiscard]] inline tri::core::LowerTriangularRM<T> multiply_lower(
    const tri::core::LowerTriangularRM<T>& L1,
    const tri::core::LowerTriangularRM<T>& L2)
{
    if (L1.cols() != L2.rows()) {
        throw std::invalid_argument("Dimension mismatch in lower triangular multiplication");
    }

    const std::size_t n = L1.rows();
    tri::core::LowerTriangularRM<T> result(n);

#ifdef TRI_USE_BLAS
    // Convert to dense, multiply, then extract lower triangular
    tri::core::DenseRM<T> D1(n, n, T{0});
    tri::core::DenseRM<T> D2(n, n, T{0});
    
    // Fill dense matrices from triangular
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            D1(i, j) = L1(i, j);
            D2(i, j) = L2(i, j);
        }
    }
    
    // Use TRMM for triangular multiplication
    tri::blas::trmm(tri::blas::Side::Right, tri::blas::Uplo::Lower,
                    tri::blas::TransposeOp::NoTrans, tri::blas::Diag::NonUnit,
                    n, n, T{1}, D2.data(), D2.ld(), D1.data(), D1.ld());
    
    // Extract result
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            result(i, j) = D1(i, j);
        }
    }
#else
    // Manual multiplication
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            T sum = T{0};
            const std::size_t k_max = std::min(i, j) + 1;
            for (std::size_t k = 0; k < k_max; ++k) {
                sum += L1(i, k) * L2(k, j);
            }
            result(i, j) = sum;
        }
    }
#endif

    return result;
}

/**
 * @brief Dense matrix multiplication C = A * B
 * 
 * @tparam T Data type
 * @param A First matrix (m x k)
 * @param B Second matrix (k x n)
 * @return Product matrix C (m x n)
 */
template<typename T>
[[nodiscard]] inline tri::core::DenseRM<T> multiply(
    const tri::core::DenseRM<T>& A,
    const tri::core::DenseRM<T>& B)
{
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Matrix dimension mismatch in multiplication");
    }

    const std::size_t m = A.rows();
    const std::size_t n = B.cols();
    const std::size_t k = A.cols();
    
    tri::core::DenseRM<T> C(m, n);
    
    tri::blas::gemm(tri::blas::TransposeOp::NoTrans, tri::blas::TransposeOp::NoTrans,
                    m, n, k, T{1}, A.data(), A.ld(), B.data(), B.ld(),
                    T{0}, C.data(), C.ld());
    
    return C;
}

/**
 * @brief Transpose of dense matrix
 * 
 * @tparam T Data type
 * @param A Input matrix (m x n)
 * @return Transposed matrix (n x m)
 */
template<typename T>
[[nodiscard]] inline tri::core::DenseRM<T> transpose(
    const tri::core::DenseRM<T>& A)
{
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();
    
    tri::core::DenseRM<T> AT(n, m);
    
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            AT(j, i) = A(i, j);
        }
    }
    
    return AT;
}

/**
 * @brief Matrix multiplication C = A * B for dense matrices
 * 
 * @tparam T Data type
 * @param A Left matrix (m×k)
 * @param B Right matrix (k×n)
 * @return Result matrix C (m×n)
 * @throw std::invalid_argument if dimensions don't match
 */
template<typename T>
[[nodiscard]] inline tri::core::DenseRM<T> matrix_multiply(
    const tri::core::DenseRM<T>& A,
    const tri::core::DenseRM<T>& B)
{
    if (A.cols() != B.rows()) {
        throw std::invalid_argument("Matrix dimensions don't match for multiplication");
    }
    
    const std::size_t m = A.rows();
    const std::size_t k = A.cols();
    const std::size_t n = B.cols();
    
    tri::core::DenseRM<T> C(m, n, T{0});
    
#ifdef TRI_USE_BLAS
    if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                   static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                   1.0f,
                   A.data(), static_cast<int>(A.ld()),
                   B.data(), static_cast<int>(B.ld()),
                   0.0f,
                   C.data(), static_cast<int>(C.ld()));
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                   static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                   1.0,
                   A.data(), static_cast<int>(A.ld()),
                   B.data(), static_cast<int>(B.ld()),
                   0.0,
                   C.data(), static_cast<int>(C.ld()));
    } else {
        // Fallback for non-BLAS types
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                for (std::size_t l = 0; l < k; ++l) {
                    C(i, j) += A(i, l) * B(l, j);
                }
            }
        }
    }
#else
    // Standard matrix multiplication
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            for (std::size_t l = 0; l < k; ++l) {
                C(i, j) += A(i, l) * B(l, j);
            }
        }
    }
#endif
    
    return C;
}

} // namespace linalg
} // namespace tri