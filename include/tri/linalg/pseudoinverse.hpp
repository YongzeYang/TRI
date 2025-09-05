#pragma once

/**
 * @file pseudoinverse.hpp
 * @brief Moore-Penrose pseudoinverse implementation
 * @author Yongze
 * @date 2025-08-10
 */

#include "tri/core/dense_rm.hpp"
#include "tri/core/matrix_base.hpp"
#include "tri/blas/svd.hpp"
#include "tri/linalg/matrix_ops.hpp"
#include "tri/linalg/dense_ops.hpp"
#include "tri/linalg/inverse.hpp"
#include <stdexcept>
#include <limits>
#include <cmath>
#include <vector>
#include <algorithm>

#ifdef TRI_USE_BLAS
#include <lapacke.h>
#endif

namespace tri {
namespace linalg {

/**
 * @brief Compute Moore-Penrose pseudoinverse using SVD
 * 
 * For matrix A = U * Σ * V^T, the pseudoinverse is A^+ = V * Σ^+ * U^T
 * where Σ^+ is the pseudoinverse of the diagonal matrix Σ.
 * 
 * @tparam T Data type (must be floating point)
 * @param A Input matrix (m×n)
 * @param tolerance Tolerance for singular value cutoff (default: machine epsilon * max(m,n) * max_singular_value)
 * @return Moore-Penrose pseudoinverse A^+ (n×m)
 * @throw std::invalid_argument if tolerance is negative
 */
template<typename T>
[[nodiscard]] inline tri::core::DenseRM<T> pseudoinverse(
    const tri::core::DenseRM<T>& A,
    T tolerance = T{-1})
{
    static_assert(std::is_floating_point_v<T>, "Pseudoinverse requires floating point type");
    
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();
    const std::size_t min_mn = std::min(m, n);
    
    if (m == 0 || n == 0) {
        return tri::core::DenseRM<T>(n, m, T{0});
    }
    
    if (tolerance < T{0}) {
        tolerance = std::numeric_limits<T>::epsilon() * static_cast<T>(std::max(m, n));
    } else if (tolerance < T{0}) {
        throw std::invalid_argument("Tolerance must be non-negative");
    }

#ifdef TRI_USE_BLAS
    // Use LAPACK SVD for optimal performance
    return pseudoinverse_svd_lapack(A, tolerance);
#else
    // Fallback implementation using normal equations for overdetermined systems
    if (m >= n) {
        // A^+ = (A^T * A)^(-1) * A^T for overdetermined systems
        return pseudoinverse_normal_equations(A, tolerance);
    } else {
        // A^+ = A^T * (A * A^T)^(-1) for underdetermined systems
        return pseudoinverse_underdetermined(A, tolerance);
    }
#endif
}

#ifdef TRI_USE_BLAS
/**
 * @brief Compute pseudoinverse using LAPACK SVD
 * 
 * @tparam T Data type
 * @param A Input matrix
 * @param tolerance Singular value tolerance
 * @return Pseudoinverse matrix
 */
template<typename T>
[[nodiscard]] inline tri::core::DenseRM<T> pseudoinverse_svd_lapack(
    const tri::core::DenseRM<T>& A,
    T tolerance)
{
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();
    const std::size_t min_mn = std::min(m, n);
    
    // Copy A for SVD (LAPACK modifies input)
    tri::core::DenseRM<T> A_copy = A;
    
    // Allocate arrays for SVD
    std::vector<T> s(min_mn);  // Singular values
    tri::core::DenseRM<T> U(m, m);  // Left singular vectors
    tri::core::DenseRM<T> VT(n, n); // Right singular vectors (transposed)
    
    // Compute SVD: A = U * diag(s) * VT
    int info = 0;
    if constexpr (std::is_same_v<T, float>) {
        info = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'A', 'A',
                             static_cast<int>(m), static_cast<int>(n),
                             A_copy.data(), static_cast<int>(A_copy.ld()),
                             s.data(),
                             U.data(), static_cast<int>(U.ld()),
                             VT.data(), static_cast<int>(VT.ld()),
                             nullptr); // superb not used when computing all vectors
    } else if constexpr (std::is_same_v<T, double>) {
        info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'A', 'A',
                             static_cast<int>(m), static_cast<int>(n),
                             A_copy.data(), static_cast<int>(A_copy.ld()),
                             s.data(),
                             U.data(), static_cast<int>(U.ld()),
                             VT.data(), static_cast<int>(VT.ld()),
                             nullptr);
    }
    
    if (info != 0) {
        throw std::runtime_error("SVD computation failed with error code: " + std::to_string(info));
    }
    
    // Determine effective rank based on tolerance
    T max_s = *std::max_element(s.begin(), s.end());
    T effective_tolerance = tolerance * max_s;
    
    std::size_t rank = 0;
    for (std::size_t i = 0; i < min_mn; ++i) {
        if (s[i] > effective_tolerance) {
            ++rank;
        }
    }
    
    // Compute pseudoinverse: A^+ = V * Σ^+ * U^T
    tri::core::DenseRM<T> result(n, m, T{0});
    
    for (std::size_t i = 0; i < rank; ++i) {
        T inv_s = T{1} / s[i];
        
        // result += inv_s * V(:,i) * U(:,i)^T
        for (std::size_t row = 0; row < n; ++row) {
            for (std::size_t col = 0; col < m; ++col) {
                result(row, col) += inv_s * VT(i, row) * U(col, i);
            }
        }
    }
    
    return result;
}
#endif

/**
 * @brief Compute pseudoinverse using normal equations (overdetermined case)
 * 
 * @tparam T Data type
 * @param A Input matrix (m×n with m >= n)
 * @param tolerance Tolerance for conditioning
 * @return Pseudoinverse matrix
 */
template<typename T>
[[nodiscard]] inline tri::core::DenseRM<T> pseudoinverse_normal_equations(
    const tri::core::DenseRM<T>& A,
    T tolerance)
{
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();
    
    // Compute A^T
    tri::core::DenseRM<T> AT = transpose(A);
    
    // Compute A^T * A
    tri::core::DenseRM<T> ATA = matrix_multiply(AT, A);
    
    try {
        // Try to compute (A^T * A)^(-1)
        inverse_inplace(ATA);
        
        // Return (A^T * A)^(-1) * A^T
        return matrix_multiply(ATA, AT);
    } catch (const std::runtime_error&) {
        // Matrix is singular, fall back to regularized solution
        return pseudoinverse_regularized(A, tolerance);
    }
}

/**
 * @brief Compute pseudoinverse for underdetermined systems
 * 
 * @tparam T Data type
 * @param A Input matrix (m×n with m < n)
 * @param tolerance Tolerance for conditioning
 * @return Pseudoinverse matrix
 */
template<typename T>
[[nodiscard]] inline tri::core::DenseRM<T> pseudoinverse_underdetermined(
    const tri::core::DenseRM<T>& A,
    T tolerance)
{
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();
    
    // Compute A^T
    tri::core::DenseRM<T> AT = transpose(A);
    
    // Compute A * A^T
    tri::core::DenseRM<T> AAT = matrix_multiply(A, AT);
    
    try {
        // Try to compute (A * A^T)^(-1)
        inverse_inplace(AAT);
        
        // Return A^T * (A * A^T)^(-1)
        return matrix_multiply(AT, AAT);
    } catch (const std::runtime_error&) {
        // Matrix is singular, fall back to regularized solution
        return pseudoinverse_regularized(A, tolerance);
    }
}

/**
 * @brief Compute regularized pseudoinverse (Tikhonov regularization)
 * 
 * @tparam T Data type
 * @param A Input matrix
 * @param tolerance Regularization parameter
 * @return Regularized pseudoinverse
 */
template<typename T>
[[nodiscard]] inline tri::core::DenseRM<T> pseudoinverse_regularized(
    const tri::core::DenseRM<T>& A,
    T tolerance)
{
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();
    
    tri::core::DenseRM<T> AT = transpose(A);
    
    if (m >= n) {
        // (A^T * A + λI)^(-1) * A^T
        tri::core::DenseRM<T> ATA = matrix_multiply(AT, A);
        
        // Add regularization to diagonal
        for (std::size_t i = 0; i < n; ++i) {
            ATA(i, i) += tolerance;
        }
        
        inverse_inplace(ATA);
        return matrix_multiply(ATA, AT);
    } else {
        // A^T * (A * A^T + λI)^(-1)
        tri::core::DenseRM<T> AAT = matrix_multiply(A, AT);
        
        // Add regularization to diagonal
        for (std::size_t i = 0; i < m; ++i) {
            AAT(i, i) += tolerance;
        }
        
        inverse_inplace(AAT);
        return matrix_multiply(AT, AAT);
    }
}

/**
 * @brief Compute pseudoinverse rank
 * 
 * @tparam T Data type
 * @param A Input matrix
 * @param tolerance Tolerance for rank determination
 * @return Effective rank of matrix
 */
template<typename T>
[[nodiscard]] inline std::size_t pseudoinverse_rank(
    const tri::core::DenseRM<T>& A,
    T tolerance = std::numeric_limits<T>::epsilon() * 100)
{
#ifdef TRI_USE_BLAS
    const std::size_t m = A.rows();
    const std::size_t n = A.cols();
    const std::size_t min_mn = std::min(m, n);
    
    // Copy A for SVD
    tri::core::DenseRM<T> A_copy = A;
    std::vector<T> s(min_mn);
    
    // Compute singular values only
    int info = 0;
    if constexpr (std::is_same_v<T, float>) {
        info = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, 'N', 'N',
                             static_cast<int>(m), static_cast<int>(n),
                             A_copy.data(), static_cast<int>(A_copy.ld()),
                             s.data(), nullptr, 1, nullptr, 1, nullptr);
    } else if constexpr (std::is_same_v<T, double>) {
        info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, 'N', 'N',
                             static_cast<int>(m), static_cast<int>(n),
                             A_copy.data(), static_cast<int>(A_copy.ld()),
                             s.data(), nullptr, 1, nullptr, 1, nullptr);
    }
    
    if (info != 0) {
        throw std::runtime_error("SVD computation failed");
    }
    
    T max_s = *std::max_element(s.begin(), s.end());
    T effective_tolerance = tolerance * max_s;
    
    std::size_t rank = 0;
    for (const T& sv : s) {
        if (sv > effective_tolerance) {
            ++rank;
        }
    }
    
    return rank;
#else
    // Fallback: estimate rank using condition number
    if (!A.is_square()) {
        return std::min(A.rows(), A.cols()); // Conservative estimate
    }
    
    try {
        T cond = condition_number_dense(A);
        return (cond < T{1} / tolerance) ? A.rows() : A.rows() - 1;
    } catch (...) {
        return 0;
    }
#endif
}

/**
 * @brief Check if pseudoinverse computation is well-conditioned
 * 
 * @tparam T Data type
 * @param A Input matrix
 * @param tolerance Conditioning tolerance
 * @return True if pseudoinverse computation should be stable
 */
template<typename T>
[[nodiscard]] inline bool is_pseudoinverse_stable(
    const tri::core::DenseRM<T>& A,
    T tolerance = std::numeric_limits<T>::epsilon() * 1000) noexcept
{
    try {
        std::size_t rank = pseudoinverse_rank(A, tolerance);
        std::size_t min_dim = std::min(A.rows(), A.cols());
        
        // Consider stable if rank is close to minimum dimension
        return static_cast<double>(rank) / static_cast<double>(min_dim) > 0.8;
    } catch (...) {
        return false;
    }
}

} // namespace linalg
} // namespace tri
