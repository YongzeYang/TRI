#pragma once

/**
 * @file dense_ops.hpp
 * @brief Dense matrix operations
 * @author Yongze
 * @date 2025-08-09
 */

#include "tri/core/dense_rm.hpp"
#include "tri/blas/gemm.hpp"
#include "tri/blas/solver.hpp"
#include "tri/blas/svd.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace tri {
namespace linalg {

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
 * @brief Dense matrix transpose
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
 * @brief Dense matrix inverse
 * 
 * @tparam T Data type
 * @param A Square matrix to invert
 * @return Inverse matrix
 * @throw std::runtime_error if matrix is singular
 */
template<typename T>
[[nodiscard]] inline tri::core::DenseRM<T> inverse(
    const tri::core::DenseRM<T>& A)
{
    if (!A.is_square()) {
        throw std::invalid_argument("Matrix must be square for inversion");
    }
    
    const std::size_t n = A.rows();
    tri::core::DenseRM<T> Ainv = A;  // Copy A
    
    int info = tri::blas::getri(n, Ainv.data(), Ainv.ld());
    
    if (info > 0) {
        throw std::runtime_error("Matrix is singular at diagonal " + std::to_string(info));
    } else if (info < 0) {
        throw std::runtime_error("Invalid argument to getri: " + std::to_string(-info));
    }
    
    return Ainv;
}

/**
 * @brief Moore-Penrose pseudoinverse
 * 
 * @tparam T Data type
 * @param A Input matrix (m x n)
 * @param tolerance Relative tolerance for singular values
 * @return Pseudoinverse (n x m)
 */
template<typename T>
[[nodiscard]] inline tri::core::DenseRM<T> pinv(
    const tri::core::DenseRM<T>& A,
    T tolerance = -1)
{
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();
    
    tri::core::DenseRM<T> Acopy = A;  // SVD will destroy input
    tri::core::DenseRM<T> Ainv(n, m);
    
    int info = tri::blas::pinv(m, n, Acopy.data(), Acopy.ld(),
                               Ainv.data(), Ainv.ld(), tolerance);
    
    if (info != 0) {
        throw std::runtime_error("Pseudoinverse computation failed with error " + std::to_string(info));
    }
    
    return Ainv;
}

/**
 * @brief Frobenius norm of dense matrix
 * 
 * @tparam T Data type
 * @param A Input matrix
 * @return Frobenius norm
 */
template<typename T>
[[nodiscard]] inline T frobenius_norm(const tri::core::DenseRM<T>& A) noexcept
{
    T sum = T{0};
    const T* data = A.data();
    const std::size_t size = A.size();
    
    for (std::size_t i = 0; i < size; ++i) {
        sum += data[i] * data[i];
    }
    
    return std::sqrt(sum);
}

/**
 * @brief Trace of dense matrix
 * 
 * @tparam T Data type
 * @param A Square matrix
 * @return Sum of diagonal elements
 */
template<typename T>
[[nodiscard]] inline T trace(const tri::core::DenseRM<T>& A)
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
 * @brief Matrix-vector multiplication y = A*x
 * 
 * @tparam T Data type
 * @param A Matrix (m x n)
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
    tri::blas::gemm(tri::blas::TransposeOp::NoTrans, tri::blas::TransposeOp::NoTrans,
                    m, 1, n, T{1}, A.data(), A.ld(), x.data(), 1,
                    T{0}, y.data(), 1);
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

} // namespace linalg
} // namespace tri