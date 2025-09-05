#pragma once

/**
 * @file cholesky.hpp
 * @brief Cholesky decomposition for symmetric positive definite matrices
 * @author Yongze
 * @date 2025-08-09
 */

#include "tri/core/lower_triangular_rm.hpp"
#include "tri/core/dense_rm.hpp"
#include <stdexcept>
#include <cmath>
#include <limits>

#ifdef TRI_USE_BLAS
#include <lapacke.h>
#endif

namespace tri {
namespace linalg {

/**
 * @brief Cholesky decomposition A = L*L^T
 * 
 * @tparam T Data type
 * @param A Symmetric positive definite matrix
 * @return Lower triangular matrix L
 */
template<typename T>
[[nodiscard]] inline tri::core::LowerTriangularRM<T> cholesky_decomposition(
    const tri::core::DenseRM<T>& A)
{
    if (!A.is_square()) {
        throw std::invalid_argument("Cholesky decomposition requires square matrix");
    }

    const std::size_t n = A.rows();
    tri::core::LowerTriangularRM<T> L(n);

#ifdef TRI_USE_BLAS
    std::vector<T> temp(n * n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            temp[i * n + j] = A(i, j);
        }
    }

    int info = 0;
    if constexpr (std::is_same_v<T, float>) {
        info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'L', 
                              static_cast<int>(n), 
                              temp.data(), 
                              static_cast<int>(n));
    } else if constexpr (std::is_same_v<T, double>) {
        info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', 
                              static_cast<int>(n), 
                              temp.data(), 
                              static_cast<int>(n));
    }
    
    if (info > 0) {
        throw std::runtime_error("Matrix is not positive definite at minor " + std::to_string(info));
    } else if (info < 0) {
        throw std::runtime_error("Invalid argument to potrf: " + std::to_string(-info));
    }

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            L(i, j) = temp[i * n + j];
        }
    }
#else
    constexpr T epsilon = std::numeric_limits<T>::epsilon() * 100;

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            T sum = T{0};

            if (i == j) {
                for (std::size_t k = 0; k < j; ++k) {
                    sum += L(j, k) * L(j, k);
                }
                
                T val = A(j, j) - sum;
                
                if (val <= epsilon) {
                    throw std::runtime_error("Matrix is not positive definite at diagonal " + std::to_string(j));
                }
                
                L(j, j) = std::sqrt(val);
            } else {
                for (std::size_t k = 0; k < j; ++k) {
                    sum += L(i, k) * L(j, k);
                }
                
                L(i, j) = (A(i, j) - sum) / L(j, j);
            }
        }
    }
#endif

    return L;
}

/**
 * @brief In-place Cholesky decomposition
 * 
 * @tparam T Data type
 * @param L Lower triangular matrix to decompose in-place
 */
template<typename T>
inline void cholesky_decomposition_inplace(
    tri::core::LowerTriangularRM<T>& L)
{
    const std::size_t n = L.rows();
    
#ifdef TRI_USE_BLAS
    int info = 0;
    if constexpr (std::is_same_v<T, float>) {
        info = LAPACKE_spptrf(LAPACK_ROW_MAJOR, 'L', 
                               static_cast<int>(n), 
                               L.data());
    } else if constexpr (std::is_same_v<T, double>) {
        info = LAPACKE_dpptrf(LAPACK_ROW_MAJOR, 'L', 
                               static_cast<int>(n), 
                               L.data());
    }
    
    if (info > 0) {
        throw std::runtime_error("Matrix is not positive definite at minor " + std::to_string(info));
    } else if (info < 0) {
        throw std::runtime_error("Invalid argument to pptrf: " + std::to_string(-info));
    }
#else
    constexpr T epsilon = std::numeric_limits<T>::epsilon() * 100;

    for (std::size_t i = 0; i < n; ++i) {
        T sum = T{0};
        for (std::size_t k = 0; k < i; ++k) {
            sum += L(i, k) * L(i, k);
        }
        
        T diag = L(i, i) - sum;
        if (diag <= epsilon) {
            throw std::runtime_error("Matrix is not positive definite at diagonal " + std::to_string(i));
        }
        L(i, i) = std::sqrt(diag);
        
        for (std::size_t j = i + 1; j < n; ++j) {
            sum = T{0};
            for (std::size_t k = 0; k < i; ++k) {
                sum += L(j, k) * L(i, k);
            }
            L(j, i) = (L(j, i) - sum) / L(i, i);
        }
    }
#endif
}

} // namespace linalg
} // namespace tri