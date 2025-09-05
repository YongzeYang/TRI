#pragma once

/**
 * @file vector_ops.hpp
 * @brief Vector operations
 * @author Yongze
 * @date 2025-08-09
 */

#include <vector>
#include <numeric>
#include <cmath>
#include <stdexcept>
#include <algorithm>

#ifdef TRI_USE_BLAS
#include <cblas.h>
#endif

namespace tri {
namespace linalg {

/**
 * @brief Compute dot product of two vectors
 * 
 * @tparam T Data type
 * @param x First vector
 * @param y Second vector
 * @return Dot product x^T * y
 */
template<typename T>
[[nodiscard]] inline T dot_product(
    const std::vector<T>& x,
    const std::vector<T>& y)
{
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector dimension mismatch in dot product");
    }

#ifdef TRI_USE_BLAS
    if constexpr (std::is_same_v<T, float>) {
        return cblas_sdot(static_cast<int>(x.size()), x.data(), 1, y.data(), 1);
    } else if constexpr (std::is_same_v<T, double>) {
        return cblas_ddot(static_cast<int>(x.size()), x.data(), 1, y.data(), 1);
    } else {
        return std::inner_product(x.begin(), x.end(), y.begin(), T{0});
    }
#else
    return std::inner_product(x.begin(), x.end(), y.begin(), T{0});
#endif
}

/**
 * @brief Compute L2 norm of vector
 * 
 * @tparam T Data type
 * @param x Input vector
 * @return L2 norm (Euclidean length)
 */
template<typename T>
[[nodiscard]] inline T vector_norm(const std::vector<T>& x) noexcept
{
#ifdef TRI_USE_BLAS
    if constexpr (std::is_same_v<T, float>) {
        return cblas_snrm2(static_cast<int>(x.size()), x.data(), 1);
    } else if constexpr (std::is_same_v<T, double>) {
        return cblas_dnrm2(static_cast<int>(x.size()), x.data(), 1);
    } else {
        T sum = T{0};
        for (const T& val : x) {
            sum += val * val;
        }
        return std::sqrt(sum);
    }
#else
    T sum = T{0};
    for (const T& val : x) {
        sum += val * val;
    }
    return std::sqrt(sum);
#endif
}

/**
 * @brief Vector addition y = x + y
 * 
 * @tparam T Data type
 * @param x First vector
 * @param y Second vector (modified in place)
 */
template<typename T>
inline void axpy(T alpha, const std::vector<T>& x, std::vector<T>& y)
{
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector dimension mismatch");
    }

#ifdef TRI_USE_BLAS
    if constexpr (std::is_same_v<T, float>) {
        cblas_saxpy(static_cast<int>(x.size()), alpha, x.data(), 1, y.data(), 1);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_daxpy(static_cast<int>(x.size()), alpha, x.data(), 1, y.data(), 1);
    } else {
        for (std::size_t i = 0; i < x.size(); ++i) {
            y[i] += alpha * x[i];
        }
    }
#else
    for (std::size_t i = 0; i < x.size(); ++i) {
        y[i] += alpha * x[i];
    }
#endif
}

/**
 * @brief Scale vector by scalar
 * 
 * @tparam T Data type
 * @param alpha Scalar multiplier
 * @param x Vector to scale (modified in place)
 */
template<typename T>
inline void scal(T alpha, std::vector<T>& x)
{
#ifdef TRI_USE_BLAS
    if constexpr (std::is_same_v<T, float>) {
        cblas_sscal(static_cast<int>(x.size()), alpha, x.data(), 1);
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dscal(static_cast<int>(x.size()), alpha, x.data(), 1);
    } else {
        for (T& val : x) {
            val *= alpha;
        }
    }
#else
    for (T& val : x) {
        val *= alpha;
    }
#endif
}

/**
 * @brief Normalize vector to unit length
 * 
 * @tparam T Data type
 * @param x Vector to normalize (modified in place)
 * @return Original norm before normalization
 */
template<typename T>
inline T normalize(std::vector<T>& x)
{
    T norm = vector_norm(x);
    if (norm > std::numeric_limits<T>::epsilon()) {
        scal(T{1} / norm, x);
    }
    return norm;
}

/**
 * @brief Element-wise vector multiplication
 * 
 * @tparam T Data type
 * @param x First vector
 * @param y Second vector
 * @return Element-wise product
 */
template<typename T>
[[nodiscard]] inline std::vector<T> hadamard_product(
    const std::vector<T>& x,
    const std::vector<T>& y)
{
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector dimension mismatch");
    }
    
    std::vector<T> result(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        result[i] = x[i] * y[i];
    }
    
    return result;
}

} // namespace linalg
} // namespace tri