/**
 * @file tri_factory.cpp
 * @brief Factory functions implementation for triangular matrices
 * @author Yongze
 * @date 2025-08-13
 */

#include "tri/factory/tri_factory.hpp"
#include <stdexcept>

namespace tri {
namespace factory {

template<typename T>
typename TriangularMatrixFactory<T>::LowerTriangularMatrix 
TriangularMatrixFactory<T>::lower_identity(size_type n) {
    if (n == 0) {
        throw std::invalid_argument("Matrix dimension must be positive");
    }
    
    LowerTriangularMatrix result(n, T{0});
    for (size_type i = 0; i < n; ++i) {
        result(i, i) = T{1};
    }
    return result;
}

template<typename T>
typename TriangularMatrixFactory<T>::LowerTriangularMatrix 
TriangularMatrixFactory<T>::lower_zeros(size_type n) {
    if (n == 0) {
        throw std::invalid_argument("Matrix dimension must be positive");
    }
    return LowerTriangularMatrix(n, T{0});
}

template<typename T>
typename TriangularMatrixFactory<T>::LowerTriangularMatrix 
TriangularMatrixFactory<T>::lower_ones(size_type n) {
    if (n == 0) {
        throw std::invalid_argument("Matrix dimension must be positive");
    }
    return LowerTriangularMatrix(n, T{1});
}

template<typename T>
typename TriangularMatrixFactory<T>::LowerTriangularMatrix 
TriangularMatrixFactory<T>::lower_constant(size_type n, const T& value) {
    if (n == 0) {
        throw std::invalid_argument("Matrix dimension must be positive");
    }
    return LowerTriangularMatrix(n, value);
}

template<typename T>
typename TriangularMatrixFactory<T>::LowerTriangularMatrix 
TriangularMatrixFactory<T>::lower_random(size_type n, T min_val, T max_val) {
    if (n == 0) {
        throw std::invalid_argument("Matrix dimension must be positive");
    }
    if (min_val >= max_val) {
        throw std::invalid_argument("min_val must be less than max_val");
    }
    
    LowerTriangularMatrix result(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    
    if constexpr (std::is_floating_point_v<T>) {
        std::uniform_real_distribution<T> dist(min_val, max_val);
        for (size_type i = 0; i < n; ++i) {
            for (size_type j = 0; j <= i; ++j) {
                result(i, j) = dist(gen);
            }
        }
    } else {
        std::uniform_int_distribution<T> dist(min_val, max_val);
        for (size_type i = 0; i < n; ++i) {
            for (size_type j = 0; j <= i; ++j) {
                result(i, j) = dist(gen);
            }
        }
    }
    return result;
}

// Explicit instantiations
template class TriangularMatrixFactory<float>;
template class TriangularMatrixFactory<double>;
template class TriangularMatrixFactory<int>;
template class TriangularMatrixFactory<long>;

namespace triangular {

tri::core::LowerTriangularRM<float> lower_identity_f(std::size_t n) {
    return FloatTriFactory::lower_identity(n);
}

tri::core::LowerTriangularRM<double> lower_identity_d(std::size_t n) {
    return DoubleTriFactory::lower_identity(n);
}

tri::core::LowerTriangularRM<float> lower_zeros_f(std::size_t n) {
    return FloatTriFactory::lower_zeros(n);
}

tri::core::LowerTriangularRM<double> lower_zeros_d(std::size_t n) {
    return DoubleTriFactory::lower_zeros(n);
}

} // namespace triangular

} // namespace factory
} // namespace tri