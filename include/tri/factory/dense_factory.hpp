#pragma once

/**
 * @file dense_factory.hpp
 * @brief Factory functions for creating dense matrices
 * @author Yongze
 * @date 2025-08-13
 */

#include <random>
#include <vector>

#include "tri/common/types.hpp"
#include "tri/core/dense_rm.hpp"

namespace tri {
namespace factory {

/**
 * @brief Dense matrix factory class
 * @tparam T Element type (must be numeric)
 */
template <typename T> 
class DenseMatrixFactory {
    static_assert(tri::common::is_numeric_v<T>, "Matrix element type must be numeric");

   public:
    using value_type = T;
    using size_type = std::size_t;
    using DenseMatrix = tri::core::DenseRM<T>;

    // Basic creation methods
    [[nodiscard]] static DenseMatrix identity(size_type n);
    [[nodiscard]] static DenseMatrix zeros(size_type rows, size_type cols);
    [[nodiscard]] static DenseMatrix zeros(size_type n);
    [[nodiscard]] static DenseMatrix ones(size_type rows, size_type cols);
    [[nodiscard]] static DenseMatrix ones(size_type n);
    [[nodiscard]] static DenseMatrix constant(size_type rows, size_type cols, const T& value);
    [[nodiscard]] static DenseMatrix diagonal(const std::vector<T>& diagonal_elements);
    [[nodiscard]] static DenseMatrix random(size_type rows, size_type cols, T min_val = T{0},
                                            T max_val = T{1});

   private:
    DenseMatrixFactory() = delete;
    ~DenseMatrixFactory() = delete;
    DenseMatrixFactory(const DenseMatrixFactory&) = delete;
    DenseMatrixFactory& operator=(const DenseMatrixFactory&) = delete;
};

// Convenience type aliases
using FloatDenseFactory = DenseMatrixFactory<float>;
using DoubleDenseFactory = DenseMatrixFactory<double>;

// Convenience functions for common cases
namespace dense {

[[nodiscard]] inline tri::core::DenseRM<float> identity_f(std::size_t n);
[[nodiscard]] inline tri::core::DenseRM<double> identity_d(std::size_t n);
[[nodiscard]] inline tri::core::DenseRM<float> zeros_f(std::size_t rows, std::size_t cols);
[[nodiscard]] inline tri::core::DenseRM<double> zeros_d(std::size_t rows, std::size_t cols);
[[nodiscard]] inline tri::core::DenseRM<float> ones_f(std::size_t rows, std::size_t cols);
[[nodiscard]] inline tri::core::DenseRM<double> ones_d(std::size_t rows, std::size_t cols);

}  // namespace dense

}  // namespace factory
}  // namespace tri