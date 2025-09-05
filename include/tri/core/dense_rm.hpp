#pragma once

/**
 * @file dense_rm.hpp
 * @brief Row-major dense matrix declaration
 * @author Yongze
 * @date 2025-08-13
 */

#include "tri/core/matrix_base.hpp"
#include "tri/common/macros.hpp"
#include <vector>
#include <initializer_list>
#include <stdexcept>

#ifdef __cpp_lib_span
#include <span>
#endif

namespace tri {
namespace core {

/**
 * @brief Row-major dense matrix template class
 * @tparam T Element type (default: float)
 */
template<typename T = float>
class TRI_API DenseRM : public MatrixBase<T> {
public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;

    // Constructors
    DenseRM() noexcept = default;
    DenseRM(size_type rows, size_type cols);
    DenseRM(size_type rows, size_type cols, const T& value);
    DenseRM(size_type rows, size_type cols, std::vector<T> data);
    DenseRM(std::initializer_list<std::initializer_list<T>> init);
    
#ifdef __cpp_lib_span
    DenseRM(size_type rows, size_type cols, std::span<const T> data);
#endif

    // Rule of Five
    ~DenseRM() = default;
    DenseRM(const DenseRM&) = default;
    DenseRM(DenseRM&&) noexcept = default;
    DenseRM& operator=(const DenseRM&) = default;
    DenseRM& operator=(DenseRM&&) noexcept = default;

    // Element access
    [[nodiscard]] TRI_FORCE_INLINE reference operator()(size_type i, size_type j) override;
    [[nodiscard]] TRI_FORCE_INLINE const_reference operator()(size_type i, size_type j) const override;

    // Row access
    [[nodiscard]] TRI_FORCE_INLINE pointer row(size_type i) noexcept;
    [[nodiscard]] TRI_FORCE_INLINE const_pointer row(size_type i) const noexcept;

    // Data access
    [[nodiscard]] TRI_FORCE_INLINE pointer data() noexcept override;
    [[nodiscard]] TRI_FORCE_INLINE const_pointer data() const noexcept override;

    // Matrix properties
    [[nodiscard]] TRI_FORCE_INLINE size_type rows() const noexcept override;
    [[nodiscard]] TRI_FORCE_INLINE size_type cols() const noexcept override;
    [[nodiscard]] TRI_FORCE_INLINE size_type size() const noexcept override;
    [[nodiscard]] TRI_FORCE_INLINE bool empty() const noexcept override;
    [[nodiscard]] TRI_FORCE_INLINE bool is_square() const noexcept override;
    [[nodiscard]] TRI_FORCE_INLINE size_type ld() const noexcept;

    // Modifiers
    void resize(size_type new_rows, size_type new_cols);
    void clear() noexcept;
    void fill(const T& value) noexcept;
    void set_diagonal(const T& value) noexcept;
    void swap(DenseRM& other) noexcept;

    // Static factory methods
    [[nodiscard]] static DenseRM zeros(size_type rows, size_type cols);
    [[nodiscard]] static DenseRM identity(size_type rows, size_type cols);
    [[nodiscard]] static DenseRM eye(size_type n);

private:
    size_type rows_ = 0;
    size_type cols_ = 0;
    std::vector<T> data_;
};

// Non-member swap
template<typename T>
void swap(DenseRM<T>& lhs, DenseRM<T>& rhs) noexcept;

// Inline implementations (must be in header for TRI_FORCE_INLINE)
template<typename T>
TRI_FORCE_INLINE typename DenseRM<T>::reference 
DenseRM<T>::operator()(size_type i, size_type j) {
#ifdef TRI_DEBUG
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
#endif
    return data_[i * cols_ + j];
}

template<typename T>
TRI_FORCE_INLINE typename DenseRM<T>::const_reference 
DenseRM<T>::operator()(size_type i, size_type j) const {
#ifdef TRI_DEBUG
    if (i >= rows_ || j >= cols_) {
        throw std::out_of_range("Matrix index out of range");
    }
#endif
    return data_[i * cols_ + j];
}

template<typename T>
TRI_FORCE_INLINE typename DenseRM<T>::pointer 
DenseRM<T>::row(size_type i) noexcept {
    return data_.data() + i * cols_;
}

template<typename T>
TRI_FORCE_INLINE typename DenseRM<T>::const_pointer 
DenseRM<T>::row(size_type i) const noexcept {
    return data_.data() + i * cols_;
}

template<typename T>
TRI_FORCE_INLINE typename DenseRM<T>::pointer 
DenseRM<T>::data() noexcept {
    return data_.data();
}

template<typename T>
TRI_FORCE_INLINE typename DenseRM<T>::const_pointer 
DenseRM<T>::data() const noexcept {
    return data_.data();
}

template<typename T>
TRI_FORCE_INLINE typename DenseRM<T>::size_type 
DenseRM<T>::rows() const noexcept {
    return rows_;
}

template<typename T>
TRI_FORCE_INLINE typename DenseRM<T>::size_type 
DenseRM<T>::cols() const noexcept {
    return cols_;
}

template<typename T>
TRI_FORCE_INLINE typename DenseRM<T>::size_type 
DenseRM<T>::size() const noexcept {
    return rows_ * cols_;
}

template<typename T>
TRI_FORCE_INLINE bool 
DenseRM<T>::empty() const noexcept {
    return data_.empty();
}

template<typename T>
TRI_FORCE_INLINE bool 
DenseRM<T>::is_square() const noexcept {
    return rows_ == cols_;
}

template<typename T>
TRI_FORCE_INLINE typename DenseRM<T>::size_type 
DenseRM<T>::ld() const noexcept {
    return cols_;
}

// Extern template instantiations for common types
extern template class DenseRM<float>;
extern template class DenseRM<double>;

} // namespace core
} // namespace tri