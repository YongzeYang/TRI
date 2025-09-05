#pragma once

/**
 * @file forward_substitution.hpp
 * @brief Forward substitution solver for lower triangular systems
 * @author Yongze
 * @date 2025-08-09
 */

#include "tri/core/lower_triangular_rm.hpp"
#include "tri/blas/solver.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <limits>

namespace tri {
namespace linalg {

/**
 * @brief Forward substitution solver for Lx = b
 * 
 * @tparam T Data type
 * @param L Lower triangular matrix (n x n)
 * @param b Right-hand side vector (n x 1)
 * @return Solution vector x such that Lx = b
 */
template<typename T>
[[nodiscard]] inline std::vector<T> forward_substitution(
    const tri::core::LowerTriangularRM<T>& L,
    const std::vector<T>& b)
{
    if (L.rows() != b.size()) {
        throw std::invalid_argument("Dimension mismatch in forward substitution");
    }

    const std::size_t n = L.rows();
    if (n == 0) return {};

    constexpr T epsilon = std::numeric_limits<T>::epsilon() * 100;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::abs(L(i, i)) < epsilon) {
            throw std::runtime_error("Matrix is singular at diagonal " + std::to_string(i));
        }
    }

    std::vector<T> x = b;

#ifdef TRI_USE_BLAS
    // Use BLAS trsm for packed triangular solve
    tri::blas::trsm(tri::blas::Side::Left, tri::blas::Uplo::Lower,
                    tri::blas::TransposeOp::NoTrans, tri::blas::Diag::NonUnit,
                    n, 1, T{1}, L.data(), n, x.data(), 1);
#else
    for (std::size_t i = 0; i < n; ++i) {
        T sum = T{0};
        for (std::size_t j = 0; j < i; ++j) {
            sum += L(i, j) * x[j];
        }
        x[i] = (x[i] - sum) / L(i, i);
    }
#endif

    return x;
}

} // namespace linalg
} // namespace tri