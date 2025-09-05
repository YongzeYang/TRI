/**
 * @file dense_factory.cpp
 * @brief Factory functions implementation for dense matrices
 * @author Yongze
 * @date 2025-08-13
 */

#include "tri/factory/dense_factory.hpp"
#include <stdexcept>
#include <cmath>

namespace tri {
namespace factory {

template<typename T>
typename DenseMatrixFactory<T>::DenseMatrix 
DenseMatrixFactory<T>::identity(size_type n) {
    if (n == 0) {
        throw std::invalid_argument("Matrix dimension must be positive");
    }
    
    DenseMatrix result(n, n, T{0});
    for (size_type i = 0; i < n; ++i) {
        result(i, i) = T{1};
    }
    return result;
}

template<typename T>
typename DenseMatrixFactory<T>::DenseMatrix 
DenseMatrixFactory<T>::zeros(size_type rows, size_type cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    return DenseMatrix(rows, cols, T{0});
}

template<typename T>
typename DenseMatrixFactory<T>::DenseMatrix 
DenseMatrixFactory<T>::zeros(size_type n) {
    return zeros(n, n);
}

template<typename T>
typename DenseMatrixFactory<T>::DenseMatrix 
DenseMatrixFactory<T>::ones(size_type rows, size_type cols) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    return DenseMatrix(rows, cols, T{1});
}

template<typename T>
typename DenseMatrixFactory<T>::DenseMatrix 
DenseMatrixFactory<T>::ones(size_type n) {
    return ones(n, n);
}

template<typename T>
typename DenseMatrixFactory<T>::DenseMatrix 
DenseMatrixFactory<T>::constant(size_type rows, size_type cols, const T& value) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    return DenseMatrix(rows, cols, value);
}

template<typename T>
typename DenseMatrixFactory<T>::DenseMatrix 
DenseMatrixFactory<T>::diagonal(const std::vector<T>& diagonal_elements) {
    if (diagonal_elements.empty()) {
        throw std::invalid_argument("Diagonal vector cannot be empty");
    }
    
    const size_type n = diagonal_elements.size();
    DenseMatrix result(n, n, T{0});
    for (size_type i = 0; i < n; ++i) {
        result(i, i) = diagonal_elements[i];
    }
    return result;
}

template<typename T>
typename DenseMatrixFactory<T>::DenseMatrix 
DenseMatrixFactory<T>::random(size_type rows, size_type cols, T min_val, T max_val) {
    if (rows == 0 || cols == 0) {
        throw std::invalid_argument("Matrix dimensions must be positive");
    }
    if (min_val >= max_val) {
        throw std::invalid_argument("min_val must be less than max_val");
    }
    
    DenseMatrix result(rows, cols);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(min_val, max_val);
        for (size_type i = 0; i < rows; ++i) {
            for (size_type j = 0; j < cols; ++j) {
                result(i, j) = dist(gen);
            }
        }
    } else {
        std::uniform_int_distribution<T> dist(min_val, max_val);
        for (size_type i = 0; i < rows; ++i) {
            for (size_type j = 0; j < cols; ++j) {
                result(i, j) = dist(gen);
            }
        }
    }
    return result;
}

// Explicit instantiations
template class DenseMatrixFactory<float>;
template class DenseMatrixFactory<double>;
template class DenseMatrixFactory<int>;
template class DenseMatrixFactory<long>;

namespace dense {

tri::core::DenseRM<float> identity_f(std::size_t n) {
    return FloatDenseFactory::identity(n);
}

tri::core::DenseRM<double> identity_d(std::size_t n) {
    return DoubleDenseFactory::identity(n);
}

tri::core::DenseRM<float> zeros_f(std::size_t rows, std::size_t cols) {
    return FloatDenseFactory::zeros(rows, cols);
}

tri::core::DenseRM<double> zeros_d(std::size_t rows, std::size_t cols) {
    return DoubleDenseFactory::zeros(rows, cols);
}

tri::core::DenseRM<float> ones_f(std::size_t rows, std::size_t cols) {
    return FloatDenseFactory::ones(rows, cols);
}

tri::core::DenseRM<double> ones_d(std::size_t rows, std::size_t cols) {
    return DoubleDenseFactory::ones(rows, cols);
}

} // namespace dense

} // namespace factory
} // namespace tri