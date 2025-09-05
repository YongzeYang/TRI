/**
 * @file dense_rm.cpp
 * @brief Row-major dense matrix implementation
 * @author Yongze
 * @date 2025-08-13
 */

#include "tri/core/dense_rm.hpp"

#include <algorithm>
#include <stdexcept>

namespace tri {
namespace core {

template <typename T>
DenseRM<T>::DenseRM(size_type rows, size_type cols)
    : rows_(rows), cols_(cols), data_(rows * cols) {}

template <typename T>
DenseRM<T>::DenseRM(size_type rows, size_type cols, const T& value)
    : rows_(rows), cols_(cols), data_(rows * cols, value) {}

template <typename T>
DenseRM<T>::DenseRM(size_type rows, size_type cols, std::vector<T> data)
    : rows_(rows), cols_(cols), data_(std::move(data)) {
    if (data_.size() != rows_ * cols_) {
        throw std::invalid_argument("Data size mismatch with matrix dimensions");
    }
}

template <typename T>
DenseRM<T>::DenseRM(std::initializer_list<std::initializer_list<T>> init) {
    rows_ = init.size();
    cols_ = rows_ > 0 ? init.begin()->size() : 0;

    data_.reserve(rows_ * cols_);

    for (const auto& row : init) {
        if (row.size() != cols_) {
            throw std::invalid_argument("All rows must have the same size");
        }
        data_.insert(data_.end(), row.begin(), row.end());
    }
}

#ifdef __cpp_lib_span
template <typename T>
DenseRM<T>::DenseRM(size_type rows, size_type cols, std::span<const T> data)
    : rows_(rows), cols_(cols), data_(data.begin(), data.end()) {
    if (data.size() != rows_ * cols_) {
        throw std::invalid_argument("Span size mismatch with matrix dimensions");
    }
}
#endif

// Non-inline member functions

template <typename T>
void DenseRM<T>::resize(size_type new_rows, size_type new_cols) {
    rows_ = new_rows;
    cols_ = new_cols;
    data_.clear();                            // 清除所有现有数据
    data_.resize(new_rows * new_cols, T{0});  // 重新分配并零初始化
}

template <typename T>
void DenseRM<T>::clear() noexcept {
    rows_ = 0;
    cols_ = 0;
    data_.clear();
}

template <typename T>
void DenseRM<T>::fill(const T& value) noexcept {
    std::fill(data_.begin(), data_.end(), value);
}

template <typename T>
void DenseRM<T>::set_diagonal(const T& value) noexcept {
    const size_type min_dim = std::min(rows_, cols_);
    for (size_type i = 0; i < min_dim; ++i) {
        (*this)(i, i) = value;
    }
}

template <typename T>
void DenseRM<T>::swap(DenseRM& other) noexcept {
    std::swap(rows_, other.rows_);
    std::swap(cols_, other.cols_);
    data_.swap(other.data_);
}

template <typename T>
DenseRM<T> DenseRM<T>::zeros(size_type rows, size_type cols) {
    return DenseRM(rows, cols, T{0});
}

template <typename T>
DenseRM<T> DenseRM<T>::identity(size_type rows, size_type cols) {
    DenseRM result(rows, cols, T{0});
    result.set_diagonal(T{1});
    return result;
}

template <typename T>
DenseRM<T> DenseRM<T>::eye(size_type n) {
    return identity(n, n);
}

template <typename T>
void swap(DenseRM<T>& lhs, DenseRM<T>& rhs) noexcept {
    lhs.swap(rhs);
}

// Explicit instantiations
template class DenseRM<float>;
template class DenseRM<double>;
template class DenseRM<int>;
template class DenseRM<long>;

template void swap(DenseRM<float>&, DenseRM<float>&) noexcept;
template void swap(DenseRM<double>&, DenseRM<double>&) noexcept;
template void swap(DenseRM<int>&, DenseRM<int>&) noexcept;
template void swap(DenseRM<long>&, DenseRM<long>&) noexcept;

}  // namespace core
}  // namespace tri