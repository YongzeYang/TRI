#pragma once

/**
 * @file tri_factory.hpp
 * @brief Factory functions for creating triangular matrices
 * @author Yongze
 * @date 2025-08-13
 */

#include "tri/core/lower_tri_rm.hpp"
#include "tri/common/types.hpp"
#include <vector>
#include <random>

namespace tri {
namespace factory {

/**
 * @brief Lower triangular matrix factory class
 * @tparam T Element type (must be numeric)
 */
template<typename T>
class TriangularMatrixFactory {
    static_assert(tri::common::is_numeric_v<T>, "Matrix element type must be numeric");

public:
    using value_type = T;
    using size_type = std::size_t;
    using LowerTriangularMatrix = tri::core::LowerTriangularRM<T>;

    // Basic creation methods
    [[nodiscard]] static LowerTriangularMatrix lower_identity(size_type n);
    [[nodiscard]] static LowerTriangularMatrix lower_zeros(size_type n);
    [[nodiscard]] static LowerTriangularMatrix lower_ones(size_type n);
    [[nodiscard]] static LowerTriangularMatrix lower_constant(size_type n, const T& value);
    [[nodiscard]] static LowerTriangularMatrix lower_random(size_type n,
                                                           T min_val = T{0}, T max_val = T{1});

private:
    TriangularMatrixFactory() = delete;
    ~TriangularMatrixFactory() = delete;
    TriangularMatrixFactory(const TriangularMatrixFactory&) = delete;
    TriangularMatrixFactory& operator=(const TriangularMatrixFactory&) = delete;
};

// Convenience type aliases
using FloatTriFactory = TriangularMatrixFactory<float>;
using DoubleTriFactory = TriangularMatrixFactory<double>;

// Convenience functions for common cases
namespace triangular {

[[nodiscard]] inline tri::core::LowerTriangularRM<float> lower_identity_f(std::size_t n);
[[nodiscard]] inline tri::core::LowerTriangularRM<double> lower_identity_d(std::size_t n);
[[nodiscard]] inline tri::core::LowerTriangularRM<float> lower_zeros_f(std::size_t n);
[[nodiscard]] inline tri::core::LowerTriangularRM<double> lower_zeros_d(std::size_t n);

} // namespace triangular

} // namespace factory
} // namespace tri