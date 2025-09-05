#pragma once

/**
 * @file inverse.hpp
 * @brief Matrix inversion for triangular and dense matrices
 * @author Yongze
 * @date 2025-08-09
 */

#include "tri/core/lower_triangular_rm.hpp"
#include "tri/core/dense_rm.hpp"
#include "tri/core/matrix_base.hpp"
#include "tri/blas/solver.hpp"
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
 * @brief Compute inverse of lower triangular matrix
 * 
 * @tparam T Data type
 * @param L Lower triangular matrix to invert
 * @return Inverse matrix L^(-1)
 * @throw std::runtime_error if matrix is singular
 */
template<typename T>
[[nodiscard]] inline tri::core::LowerTriangularRM<T> inverse_lower(
    const tri::core::LowerTriangularRM<T>& L)
{
    const std::size_t n = L.rows();
    tri::core::LowerTriangularRM<T> inv(n);

    // Check for singularity
    constexpr T epsilon = std::numeric_limits<T>::epsilon() * 100;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::abs(L(i, i)) < epsilon) {
            throw std::runtime_error("Matrix is singular: cannot compute inverse");
        }
    }

#ifdef TRI_USE_BLAS
    // Copy L to inv
    std::copy(L.data(), L.data() + L.packed_size(), inv.data());
    
    int info = 0;
    if constexpr (std::is_same_v<T, float>) {
        info = LAPACKE_stptri(LAPACK_ROW_MAJOR, 'L', 'N',
                               static_cast<int>(n), inv.data());
    } else if constexpr (std::is_same_v<T, double>) {
        info = LAPACKE_dtptri(LAPACK_ROW_MAJOR, 'L', 'N',
                               static_cast<int>(n), inv.data());
    } else {
        throw std::runtime_error("Unsupported type for triangular inverse");
    }
    
    if (info > 0) {
        throw std::runtime_error("Matrix is singular at element " + std::to_string(info));
    } else if (info < 0) {
        throw std::runtime_error("Invalid argument to tptri: " + std::to_string(-info));
    }
#else
    // Manual triangular inversion
    // Initialize diagonal
    for (std::size_t i = 0; i < n; ++i) {
        inv(i, i) = T{1} / L(i, i);
    }
    
    // Compute off-diagonal elements column by column
    for (std::size_t j = 0; j < n; ++j) {
        for (std::size_t i = j + 1; i < n; ++i) {
            T sum = T{0};
            for (std::size_t k = j; k < i; ++k) {
                sum += L(i, k) * inv(k, j);
            }
            inv(i, j) = -sum / L(i, i);
        }
    }
#endif

    return inv;
}

/**
 * @brief Compute inverse of dense matrix
 * 
 * @tparam T Data type
 * @param A Dense matrix to invert
 * @return Inverse matrix A^(-1)
 * @throw std::runtime_error if matrix is singular
 */
template<typename T>
[[nodiscard]] inline tri::core::DenseRM<T> inverse_dense(
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
 * @brief Check if lower triangular matrix is well-conditioned
 * 
 * @tparam T Data type
 * @param L Lower triangular matrix
 * @return Estimated condition number
 */
template<typename T>
[[nodiscard]] inline T condition_number(
    const tri::core::LowerTriangularRM<T>& L) noexcept
{
    const std::size_t n = L.rows();
    if (n == 0) return T{1};
    
    T max_diag = std::abs(L(0, 0));
    T min_diag = std::abs(L(0, 0));
    
    for (std::size_t i = 1; i < n; ++i) {
        T abs_diag = std::abs(L(i, i));
        max_diag = std::max(max_diag, abs_diag);
        min_diag = std::min(min_diag, abs_diag);
    }
    
    if (min_diag == T{0}) {
        return std::numeric_limits<T>::infinity();
    }
    
    return max_diag / min_diag;
}

/**
 * @brief Estimate condition number of dense matrix
 * 
 * @tparam T Data type
 * @param A Dense matrix
 * @return Estimated condition number (using 1-norm)
 */
template<typename T>
[[nodiscard]] inline T condition_number(
    const tri::core::DenseRM<T>& A)
{
    if (!A.is_square()) {
        throw std::invalid_argument("Condition number requires square matrix");
    }
    
#ifdef TRI_USE_BLAS
    const std::size_t n = A.rows();
    tri::core::DenseRM<T> Acopy = A;
    std::vector<int> ipiv(n);
    
    // Compute LU factorization
    int info = 0;
    if constexpr (std::is_same_v<T, float>) {
        info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR,
                              static_cast<int>(n), static_cast<int>(n),
                              Acopy.data(), static_cast<int>(Acopy.ld()),
                              ipiv.data());
    } else if constexpr (std::is_same_v<T, double>) {
        info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR,
                              static_cast<int>(n), static_cast<int>(n),
                              Acopy.data(), static_cast<int>(Acopy.ld()),
                              ipiv.data());
    }
    
    if (info != 0) {
        return std::numeric_limits<T>::infinity();
    }
    
    // Estimate condition number
    T rcond = 0;
    T anorm = matrix_1norm(A);  // Need to implement this
    
    if constexpr (std::is_same_v<T, float>) {
        LAPACKE_sgecon(LAPACK_ROW_MAJOR, '1',
                       static_cast<int>(n),
                       Acopy.data(), static_cast<int>(Acopy.ld()),
                       anorm, &rcond);
    } else if constexpr (std::is_same_v<T, double>) {
        LAPACKE_dgecon(LAPACK_ROW_MAJOR, '1',
                       static_cast<int>(n),
                       Acopy.data(), static_cast<int>(Acopy.ld()),
                       anorm, &rcond);
    }
    
    return (rcond > 0) ? T{1} / rcond : std::numeric_limits<T>::infinity();
#else
    // Simple estimate using diagonal dominance
    const std::size_t n = A.rows();
    T max_ratio = T{0};
    
    for (std::size_t i = 0; i < n; ++i) {
        T diag = std::abs(A(i, i));
        T sum = T{0};
        for (std::size_t j = 0; j < n; ++j) {
            if (i != j) {
                sum += std::abs(A(i, j));
            }
        }
        if (diag > T{0}) {
            max_ratio = std::max(max_ratio, (diag + sum) / diag);
        }
    }
    
    return max_ratio;
#endif
}

/**
 * @brief Compute 1-norm of matrix (maximum absolute column sum)
 * 
 * @tparam T Data type
 * @param A Input matrix
 * @return 1-norm
 */
template<typename T>
[[nodiscard]] inline T matrix_1norm(const tri::core::DenseRM<T>& A) noexcept
{
    T max_sum = T{0};
    
    for (std::size_t j = 0; j < A.cols(); ++j) {
        T col_sum = T{0};
        for (std::size_t i = 0; i < A.rows(); ++i) {
            col_sum += std::abs(A(i, j));
        }
        max_sum = std::max(max_sum, col_sum);
    }
    
    return max_sum;
}

/**
 * @brief Generic matrix inverse with automatic method selection
 * 
 * @tparam T Data type
 * @param A Input matrix (must be square)
 * @return Inverse matrix A^(-1)
 * @throw std::invalid_argument if matrix is not square
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
    
    // For small matrices, use direct methods
    if (n == 1) {
        constexpr T epsilon = std::numeric_limits<T>::epsilon() * 100;
        if (std::abs(A(0, 0)) < epsilon) {
            throw std::runtime_error("Matrix is singular: cannot compute inverse");
        }
        return tri::core::DenseRM<T>(1, 1, T{1} / A(0, 0));
    }
    
    if (n == 2) {
        T det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
        constexpr T epsilon = std::numeric_limits<T>::epsilon() * 100;
        
        if (std::abs(det) < epsilon) {
            throw std::runtime_error("Matrix is singular: cannot compute inverse");
        }
        
        tri::core::DenseRM<T> inv(2, 2);
        T inv_det = T{1} / det;
        inv(0, 0) = A(1, 1) * inv_det;
        inv(0, 1) = -A(0, 1) * inv_det;
        inv(1, 0) = -A(1, 0) * inv_det;
        inv(1, 1) = A(0, 0) * inv_det;
        return inv;
    }
    
    // For larger matrices, use LU decomposition with BLAS/LAPACK
    return inverse_dense(A);
}

/**
 * @brief In-place matrix inversion
 * 
 * @tparam T Data type
 * @param A Matrix to invert (modified in place)
 * @throw std::invalid_argument if matrix is not square
 * @throw std::runtime_error if matrix is singular
 */
template<typename T>
inline void inverse_inplace(tri::core::DenseRM<T>& A)
{
    if (!A.is_square()) {
        throw std::invalid_argument("Matrix must be square for inversion");
    }
    
    const std::size_t n = A.rows();
    
#ifdef TRI_USE_BLAS
    std::vector<int> ipiv(n);
    
    // LU factorization
    int info = 0;
    if constexpr (std::is_same_v<T, float>) {
        info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, static_cast<int>(n), static_cast<int>(n),
                             A.data(), static_cast<int>(A.ld()), ipiv.data());
    } else if constexpr (std::is_same_v<T, double>) {
        info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, static_cast<int>(n), static_cast<int>(n),
                             A.data(), static_cast<int>(A.ld()), ipiv.data());
    }
    
    if (info > 0) {
        throw std::runtime_error("Matrix is singular at diagonal " + std::to_string(info));
    } else if (info < 0) {
        throw std::runtime_error("Invalid argument to getrf: " + std::to_string(-info));
    }
    
    // Compute inverse
    if constexpr (std::is_same_v<T, float>) {
        info = LAPACKE_sgetri(LAPACK_ROW_MAJOR, static_cast<int>(n),
                             A.data(), static_cast<int>(A.ld()), ipiv.data());
    } else if constexpr (std::is_same_v<T, double>) {
        info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, static_cast<int>(n),
                             A.data(), static_cast<int>(A.ld()), ipiv.data());
    }
    
    if (info > 0) {
        throw std::runtime_error("Matrix is singular at diagonal " + std::to_string(info));
    } else if (info < 0) {
        throw std::runtime_error("Invalid argument to getri: " + std::to_string(-info));
    }
#else
    // Fallback: Gauss-Jordan elimination
    gauss_jordan_inplace(A);
#endif
}

/**
 * @brief Gauss-Jordan elimination for matrix inversion (fallback implementation)
 * 
 * @tparam T Data type
 * @param A Matrix to invert (modified in place)
 * @throw std::runtime_error if matrix is singular
 */
template<typename T>
inline void gauss_jordan_inplace(tri::core::DenseRM<T>& A)
{
    const std::size_t n = A.rows();
    constexpr T epsilon = std::numeric_limits<T>::epsilon() * 100;
    
    // Create augmented matrix [A | I]
    tri::core::DenseRM<T> augmented(n, 2 * n, T{0});
    
    // Copy A to left part and identity to right part
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            augmented(i, j) = A(i, j);
        }
        augmented(i, n + i) = T{1};
    }
    
    // Forward elimination
    for (std::size_t k = 0; k < n; ++k) {
        // Find pivot
        std::size_t pivot_row = k;
        T max_val = std::abs(augmented(k, k));
        
        for (std::size_t i = k + 1; i < n; ++i) {
            T abs_val = std::abs(augmented(i, k));
            if (abs_val > max_val) {
                max_val = abs_val;
                pivot_row = i;
            }
        }
        
        if (max_val < epsilon) {
            throw std::runtime_error("Matrix is singular: cannot compute inverse");
        }
        
        // Swap rows if needed
        if (pivot_row != k) {
            for (std::size_t j = 0; j < 2 * n; ++j) {
                std::swap(augmented(k, j), augmented(pivot_row, j));
            }
        }
        
        // Scale pivot row
        T pivot = augmented(k, k);
        for (std::size_t j = 0; j < 2 * n; ++j) {
            augmented(k, j) /= pivot;
        }
        
        // Eliminate column
        for (std::size_t i = 0; i < n; ++i) {
            if (i != k) {
                T factor = augmented(i, k);
                for (std::size_t j = 0; j < 2 * n; ++j) {
                    augmented(i, j) -= factor * augmented(k, j);
                }
            }
        }
    }
    
    // Copy result back to A
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            A(i, j) = augmented(i, n + j);
        }
    }
}

/**
 * @brief Check if matrix is invertible
 * 
 * @tparam T Data type
 * @param A Input matrix
 * @param tolerance Tolerance for singularity check
 * @return True if matrix is likely invertible
 */
template<typename T>
[[nodiscard]] inline bool is_invertible(
    const tri::core::DenseRM<T>& A, 
    T tolerance = std::numeric_limits<T>::epsilon() * 100) noexcept
{
    if (!A.is_square()) {
        return false;
    }
    
    try {
        T cond = condition_number_dense(A);
        return cond < (T{1} / tolerance);
    } catch (...) {
        return false;
    }
}


} // namespace linalg
} // namespace tri