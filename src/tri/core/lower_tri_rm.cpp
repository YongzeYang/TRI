/**
 * @file lower_tri_rm.cpp
 * @brief Row-major packed lower triangular matrix implementation
 * @author Yongze
 * @date 2025-08-13
 */

#include "tri/core/lower_tri_rm.hpp"

#include <algorithm>
#include <stdexcept>

#include "tri/core/dense_rm.hpp"

namespace tri {
namespace core {

// Matrix constructor implementation
template <typename T>
template <typename MatrixType, typename>
LowerTriangularRM<T>::LowerTriangularRM(const MatrixType& dense) {
    if (dense.rows() != dense.cols()) {
        throw std::invalid_argument("Matrix must be square");
    }

    n_ = dense.rows();
    data_.resize(packed_size(n_));

    for (size_type i = 0; i < n_; ++i) {
        for (size_type j = 0; j <= i; ++j) {
            (*this)(i, j) = static_cast<T>(dense(i, j));
        }
    }
}

template <typename T>
LowerTriangularRM<T>::LowerTriangularRM(size_type n, std::vector<T> packed_data)
    : n_(n), data_(std::move(packed_data)) {
    if (data_.size() != packed_size(n_)) {
        throw std::invalid_argument("Packed data size mismatch");
    }
}

// Non-inline member functions

template <typename T>
void LowerTriangularRM<T>::set(size_type i, size_type j, const T& value) {
    if (i >= n_ || j >= n_) {
        throw std::out_of_range("Index out of range");
    }

    if (j > i) {
        throw std::logic_error("Cannot set upper triangular elements");
    }

    data_[get_packed_index(i, j)] = value;
}

template <typename T>
void LowerTriangularRM<T>::clear() noexcept {
    n_ = 0;
    data_.clear();
}

template <typename T>
void LowerTriangularRM<T>::resize(size_type new_n) {
    n_ = new_n;
    data_.resize(packed_size(new_n));
    std::fill(data_.begin(), data_.end(), T{0});
}

template <typename T>
void LowerTriangularRM<T>::fill(const T& value) noexcept {
    std::fill(data_.begin(), data_.end(), value);
}

template <typename T>
void LowerTriangularRM<T>::set_diagonal(const T& value) noexcept {
    for (size_type i = 0; i < n_; ++i) {
        data_[get_packed_index(i, i)] = value;
    }
}

template <typename T>
void LowerTriangularRM<T>::swap(LowerTriangularRM& other) noexcept {
    std::swap(n_, other.n_);
    data_.swap(other.data_);
}

template <typename T>
LowerTriangularRM<T> LowerTriangularRM<T>::identity(size_type n) {
    LowerTriangularRM result(static_cast<int>(n));  // Explicitly use int constructor
    result.set_diagonal(T{1});
    return result;
}

template <typename T>
LowerTriangularRM<T> LowerTriangularRM<T>::zeros(size_type n) {
    return LowerTriangularRM(static_cast<int>(n), T{0});  // Explicitly use int constructor
}

template <typename T>
void swap(LowerTriangularRM<T>& lhs, LowerTriangularRM<T>& rhs) noexcept {
    lhs.swap(rhs);
}

// Explicit instantiations
template class LowerTriangularRM<float>;
template class LowerTriangularRM<double>;
template class LowerTriangularRM<int>;
template class LowerTriangularRM<long>;

template void swap(LowerTriangularRM<float>&, LowerTriangularRM<float>&) noexcept;
template void swap(LowerTriangularRM<double>&, LowerTriangularRM<double>&) noexcept;
template void swap(LowerTriangularRM<int>&, LowerTriangularRM<int>&) noexcept;
template void swap(LowerTriangularRM<long>&, LowerTriangularRM<long>&) noexcept;

// Explicit instantiation of template constructors
template LowerTriangularRM<float>::LowerTriangularRM(const DenseRM<float>&);
template LowerTriangularRM<double>::LowerTriangularRM(const DenseRM<double>&);
template LowerTriangularRM<int>::LowerTriangularRM(const DenseRM<int>&);
template LowerTriangularRM<long>::LowerTriangularRM(const DenseRM<long>&);

// Explicit instantiation for integral type constructors
template LowerTriangularRM<float>::LowerTriangularRM<int>(int);
template LowerTriangularRM<float>::LowerTriangularRM<int>(int, const float&);
template LowerTriangularRM<double>::LowerTriangularRM<int>(int);
template LowerTriangularRM<double>::LowerTriangularRM<int>(int, const double&);
template LowerTriangularRM<float>::LowerTriangularRM<unsigned int>(unsigned int);
template LowerTriangularRM<float>::LowerTriangularRM<unsigned int>(unsigned int, const float&);
template LowerTriangularRM<double>::LowerTriangularRM<unsigned int>(unsigned int);
template LowerTriangularRM<double>::LowerTriangularRM<unsigned int>(unsigned int, const double&);

}  // namespace core
}  // namespace tri