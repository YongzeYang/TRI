#pragma once

/**
 * @file blocked_factory.hpp
 * @brief Factory functions for creating blocked triangular matrices
 * @author Yongze
 * @date 2025-08-13
 */
#include "tri/core/dense_rm.hpp"
#include "tri/core/lower_tri_rm.hpp"
#include "tri/core/blocked_tri.hpp"
#include "tri/common/types.hpp"
#include <vector>
#include <random>

namespace tri {
namespace factory {

/**
 * @brief Blocked triangular matrix factory class
 * @tparam T Element type (must be numeric)
 */
template<typename T>
class BlockedMatrixFactory {
    static_assert(tri::common::is_numeric_v<T>, "Matrix element type must be numeric");

public:
    using value_type = T;
    using size_type = std::size_t;
    using BlockedMatrix = tri::core::BlockedTriMatrix<T>;

    // Basic creation methods
    [[nodiscard]] static BlockedMatrix identity(size_type n, 
                                               size_type block_size = tri::config::DEFAULT_BLOCK_SIZE);
    [[nodiscard]] static BlockedMatrix zeros(size_type n,
                                            size_type block_size = tri::config::DEFAULT_BLOCK_SIZE);
    [[nodiscard]] static BlockedMatrix ones(size_type n,
                                           size_type block_size = tri::config::DEFAULT_BLOCK_SIZE);
    [[nodiscard]] static BlockedMatrix constant(size_type n, const T& value,
                                               size_type block_size = tri::config::DEFAULT_BLOCK_SIZE);
    [[nodiscard]] static BlockedMatrix random(size_type n,
                                             T min_val = T{0}, T max_val = T{1},
                                             size_type block_size = tri::config::DEFAULT_BLOCK_SIZE);
    
    // Create from dense matrix
    [[nodiscard]] static BlockedMatrix from_dense(const tri::core::DenseRM<T>& dense,
                                                 size_type block_size = tri::config::DEFAULT_BLOCK_SIZE);
    
    // Create from packed triangular matrix
    [[nodiscard]] static BlockedMatrix from_triangular(const tri::core::LowerTriangularRM<T>& tri,
                                                      size_type block_size = tri::config::DEFAULT_BLOCK_SIZE);

private:
    BlockedMatrixFactory() = delete;
    ~BlockedMatrixFactory() = delete;
    BlockedMatrixFactory(const BlockedMatrixFactory&) = delete;
    BlockedMatrixFactory& operator=(const BlockedMatrixFactory&) = delete;
};

// Convenience type aliases
using FloatBlockedFactory = BlockedMatrixFactory<float>;
using DoubleBlockedFactory = BlockedMatrixFactory<double>;

// Convenience functions for common cases
namespace blocked {

[[nodiscard]] inline tri::core::BlockedTriMatrix<float> identity_f(std::size_t n,
                                                                  std::size_t block_size = tri::config::DEFAULT_BLOCK_SIZE);
[[nodiscard]] inline tri::core::BlockedTriMatrix<double> identity_d(std::size_t n,
                                                                   std::size_t block_size = tri::config::DEFAULT_BLOCK_SIZE);
[[nodiscard]] inline tri::core::BlockedTriMatrix<float> zeros_f(std::size_t n,
                                                               std::size_t block_size = tri::config::DEFAULT_BLOCK_SIZE);
[[nodiscard]] inline tri::core::BlockedTriMatrix<double> zeros_d(std::size_t n,
                                                                std::size_t block_size = tri::config::DEFAULT_BLOCK_SIZE);

} // namespace blocked

} // namespace factory
} // namespace tri