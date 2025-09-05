/**
 * @file blocked_factory.cpp
 * @brief Factory functions implementation for blocked triangular matrices
 * @author Yongze
 * @date 2025-08-13
 */

#include "tri/factory/blocked_factory.hpp"

#include <stdexcept>

#include "tri/core/dense_rm.hpp"
#include "tri/core/lower_tri_rm.hpp"

namespace tri {
namespace factory {

template <typename T>
typename BlockedMatrixFactory<T>::BlockedMatrix BlockedMatrixFactory<T>::identity(
    size_type n, size_type block_size) {
    if (n == 0) {
        throw std::invalid_argument("Matrix dimension must be positive");
    }
    return BlockedMatrix::identity(n, block_size);
}

template <typename T>
typename BlockedMatrixFactory<T>::BlockedMatrix BlockedMatrixFactory<T>::zeros(
    size_type n, size_type block_size) {
    if (n == 0) {
        throw std::invalid_argument("Matrix dimension must be positive");
    }
    return BlockedMatrix::zeros(n, block_size);
}

template <typename T>
typename BlockedMatrixFactory<T>::BlockedMatrix BlockedMatrixFactory<T>::ones(
    size_type n, size_type block_size) {
    if (n == 0) {
        throw std::invalid_argument("Matrix dimension must be positive");
    }
    return BlockedMatrix(n, block_size, T{1});
}

template <typename T>
typename BlockedMatrixFactory<T>::BlockedMatrix BlockedMatrixFactory<T>::constant(
    size_type n, const T& value, size_type block_size) {
    if (n == 0) {
        throw std::invalid_argument("Matrix dimension must be positive");
    }
    return BlockedMatrix(n, block_size, value);
}

template <typename T>
typename BlockedMatrixFactory<T>::BlockedMatrix BlockedMatrixFactory<T>::random(
    size_type n, T min_val, T max_val, size_type block_size) {
    if (n == 0) {
        throw std::invalid_argument("Matrix dimension must be positive");
    }
    if (min_val >= max_val) {
        throw std::invalid_argument("min_val must be less than max_val");
    }

    BlockedMatrix result(n, block_size);
    std::random_device rd;
    std::mt19937 gen(rd());

    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(min_val, max_val);
        for (size_type i = 0; i < n; ++i) {
            for (size_type j = 0; j <= i; ++j) {
                result.set(i, j, dist(gen));
            }
        }
    } else {
        std::uniform_int_distribution<T> dist(min_val, max_val);
        for (size_type i = 0; i < n; ++i) {
            for (size_type j = 0; j <= i; ++j) {
                result.set(i, j, dist(gen));
            }
        }
    }

    return result;
}

template <typename T>
typename BlockedMatrixFactory<T>::BlockedMatrix BlockedMatrixFactory<T>::from_dense(
    const tri::core::DenseRM<T>& dense, size_type block_size) {
    if (!dense.is_square()) {
        throw std::invalid_argument("Dense matrix must be square");
    }
    return BlockedMatrix(dense, block_size);
}

template <typename T>
typename BlockedMatrixFactory<T>::BlockedMatrix BlockedMatrixFactory<T>::from_triangular(
    const tri::core::LowerTriangularRM<T>& tri, size_type block_size) {
    return BlockedMatrix(tri, block_size);
}

// Explicit instantiations
template class BlockedMatrixFactory<float>;
template class BlockedMatrixFactory<double>;
template class BlockedMatrixFactory<int>;
template class BlockedMatrixFactory<long>;

namespace blocked {

tri::core::BlockedTriMatrix<float> identity_f(std::size_t n, std::size_t block_size) {
    return FloatBlockedFactory::identity(n, block_size);
}

tri::core::BlockedTriMatrix<double> identity_d(std::size_t n, std::size_t block_size) {
    return DoubleBlockedFactory::identity(n, block_size);
}

tri::core::BlockedTriMatrix<float> zeros_f(std::size_t n, std::size_t block_size) {
    return FloatBlockedFactory::zeros(n, block_size);
}

tri::core::BlockedTriMatrix<double> zeros_d(std::size_t n, std::size_t block_size) {
    return DoubleBlockedFactory::zeros(n, block_size);
}

}  // namespace blocked

}  // namespace factory
}  // namespace tri