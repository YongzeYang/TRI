#pragma once

/**
 * @file backward_substitution.hpp
 * @brief Backward substitution solver for upper triangular systems
 * @author Yongze
 * @date 2025-08-09
 */

#include "tri/core/lower_triangular_rm.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <limits>

namespace tri {
namespace linalg {

/**
 * @brief Backward substitution for L^T x = b
 * 
 * @tparam T Data type
 * @param L Lower triangular matrix (whose transpose is used)
 * @param b Right-hand side vector
 * @return Solution vector x such that L^T x = b
 */
template<typename T>
[[nodiscard]] inline std::vector<T> backward_substitution(
    const tri::core::LowerTriangularRM<T>& L,
    const std::vector<T>& b)
{
    if (L.rows() != b.size()) {
        throw std::invalid_argument("Dimension mismatch in backward substitution");
    }

    const std::size_t n = L.rows();
    std::vector<T> x = b;

    constexpr T epsilon = std::numeric_limits<T>::epsilon() * 100;

    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i) {
        const std::size_t idx = static_cast<std::size_t>(i);
        
        if (std::abs(L(idx, idx)) < epsilon) {
            throw std::runtime_error("Matrix is singular at diagonal " + std::to_string(idx));
        }

        T sum = T{0};
        for (std::size_t j = idx + 1; j < n; ++j) {
            sum += L(j, idx) * x[j];  // L(j, idx) for transpose
        }
        
        x[idx] = (x[idx] - sum) / L(idx, idx);
    }

    return x;
}

} // namespace linalg
} // namespace tri