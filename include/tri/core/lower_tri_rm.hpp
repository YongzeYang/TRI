#pragma once

/**
 * @file lower_tri_rm.hpp
 * @brief Row-major packed lower triangular matrix declaration
 * @author Yongze
 * @date 2025-08-13
 */

#include <stdexcept>
#include <type_traits>
#include <vector>

#include "tri/common/macros.hpp"
#include "tri/core/matrix_base.hpp"

namespace tri {
namespace core {

/**
 * @brief Lower triangular matrix with packed row-major storage
 * @tparam T Element type (default: float)
 */
template <typename T = float>
class TRI_API LowerTriangularRM : public MatrixBase<T> {
   public:
    using value_type = T;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;

    // Constructors
    LowerTriangularRM() = default;

    // Single-argument constructor for integral types
    template <typename IntType, typename = std::enable_if_t<std::is_integral_v<IntType> &&
                                                            !std::is_same_v<IntType, bool>>>
    explicit LowerTriangularRM(IntType n);

    // Two-argument constructor for integral type + value
    template <typename IntType, typename = std::enable_if_t<std::is_integral_v<IntType> &&
                                                            !std::is_same_v<IntType, bool>>>
    LowerTriangularRM(IntType n, const T& value);

    // Constructor from matrix (must have rows() and cols() methods)
    template <typename MatrixType, typename = std::enable_if_t<!std::is_integral_v<MatrixType>>>
    explicit LowerTriangularRM(const MatrixType& dense);

    LowerTriangularRM(size_type n, std::vector<T> packed_data);

    // Rule of Five
    ~LowerTriangularRM() = default;
    LowerTriangularRM(const LowerTriangularRM&) = default;
    LowerTriangularRM(LowerTriangularRM&&) noexcept = default;
    LowerTriangularRM& operator=(const LowerTriangularRM&) = default;
    LowerTriangularRM& operator=(LowerTriangularRM&&) noexcept = default;

    // Element access
    [[nodiscard]] TRI_FORCE_INLINE reference operator()(size_type i, size_type j) override;
    [[nodiscard]] TRI_FORCE_INLINE const_reference operator()(size_type i,
                                                              size_type j) const override;
    [[nodiscard]] TRI_FORCE_INLINE T at(size_type i, size_type j) const noexcept;
    void set(size_type i, size_type j, const T& value);

    // Matrix properties
    [[nodiscard]] TRI_FORCE_INLINE size_type rows() const noexcept override;
    [[nodiscard]] TRI_FORCE_INLINE size_type cols() const noexcept override;
    [[nodiscard]] TRI_FORCE_INLINE size_type dimension() const noexcept;
    [[nodiscard]] TRI_FORCE_INLINE size_type size() const noexcept override;
    [[nodiscard]] TRI_FORCE_INLINE bool empty() const noexcept override;
    [[nodiscard]] TRI_FORCE_INLINE bool is_square() const noexcept override;
    [[nodiscard]] TRI_FORCE_INLINE size_type packed_size() const noexcept;
    [[nodiscard]] static TRI_FORCE_INLINE size_type packed_size(size_type n) noexcept;

    // Data access
    [[nodiscard]] TRI_FORCE_INLINE pointer data() noexcept override;
    [[nodiscard]] TRI_FORCE_INLINE const_pointer data() const noexcept override;

    // Modifiers
    void clear() noexcept;
    void resize(size_type new_n);
    void fill(const T& value) noexcept;
    void set_diagonal(const T& value) noexcept;
    void swap(LowerTriangularRM& other) noexcept;

    // Static factory methods
    [[nodiscard]] static LowerTriangularRM identity(size_type n);
    [[nodiscard]] static LowerTriangularRM zeros(size_type n);

   private:
    [[nodiscard]] TRI_FORCE_INLINE size_type get_packed_index(size_type i,
                                                              size_type j) const noexcept;

    size_type n_ = 0;
    std::vector<T> data_;
};

// Template constructor implementations
template <typename T>
template <typename IntType, typename>
LowerTriangularRM<T>::LowerTriangularRM(IntType n)
    : n_(static_cast<size_type>(n)), data_(packed_size(static_cast<size_type>(n)), T{0}) {
    if (n < 0) {
        throw std::invalid_argument("Matrix dimension must be non-negative");
    }
}

template <typename T>
template <typename IntType, typename>
LowerTriangularRM<T>::LowerTriangularRM(IntType n, const T& value)
    : n_(static_cast<size_type>(n)), data_(packed_size(static_cast<size_type>(n)), value) {
    if (n < 0) {
        throw std::invalid_argument("Matrix dimension must be non-negative");
    }
}

// Non-member swap
template <typename T>
void swap(LowerTriangularRM<T>& lhs, LowerTriangularRM<T>& rhs) noexcept;

// Inline implementations (must be in header for TRI_FORCE_INLINE)
template <typename T>
TRI_FORCE_INLINE typename LowerTriangularRM<T>::reference LowerTriangularRM<T>::operator()(
    size_type i, size_type j) {
    if (i >= n_ || j >= n_) {
        throw std::out_of_range("Index out of range");
    }

    if (j > i) {
        throw std::logic_error("Cannot modify upper triangular elements");
    }

    return data_[get_packed_index(i, j)];
}

template <typename T>
TRI_FORCE_INLINE typename LowerTriangularRM<T>::const_reference LowerTriangularRM<T>::operator()(
    size_type i, size_type j) const {
    if (i >= n_ || j >= n_) {
        throw std::out_of_range("Index out of range");
    }

    if (j > i) {
        static const T zero = T{0};
        return zero;
    }

    return data_[get_packed_index(i, j)];
}

template <typename T>
TRI_FORCE_INLINE typename LowerTriangularRM<T>::size_type LowerTriangularRM<T>::rows()
    const noexcept {
    return n_;
}

template <typename T>
TRI_FORCE_INLINE typename LowerTriangularRM<T>::size_type LowerTriangularRM<T>::cols()
    const noexcept {
    return n_;
}

template <typename T>
TRI_FORCE_INLINE typename LowerTriangularRM<T>::size_type LowerTriangularRM<T>::dimension()
    const noexcept {
    return n_;
}

template <typename T>
TRI_FORCE_INLINE typename LowerTriangularRM<T>::size_type LowerTriangularRM<T>::size()
    const noexcept {
    return n_ * n_;
}

template <typename T>
TRI_FORCE_INLINE bool LowerTriangularRM<T>::empty() const noexcept {
    return n_ == 0;
}

template <typename T>
TRI_FORCE_INLINE bool LowerTriangularRM<T>::is_square() const noexcept {
    return true;
}

template <typename T>
TRI_FORCE_INLINE typename LowerTriangularRM<T>::size_type LowerTriangularRM<T>::packed_size()
    const noexcept {
    return packed_size(n_);
}

template <typename T>
TRI_FORCE_INLINE typename LowerTriangularRM<T>::size_type LowerTriangularRM<T>::get_packed_index(
    size_type i, size_type j) const noexcept {
    return i * (i + 1) / 2 + j;
}

template <typename T>
TRI_FORCE_INLINE T LowerTriangularRM<T>::at(size_type i, size_type j) const noexcept {
    if (j > i) return T{0};
    return data_[get_packed_index(i, j)];
}

template <typename T>
TRI_FORCE_INLINE typename LowerTriangularRM<T>::size_type LowerTriangularRM<T>::packed_size(
    size_type n) noexcept {
    return n * (n + 1) / 2;
}

template <typename T>
TRI_FORCE_INLINE typename LowerTriangularRM<T>::pointer LowerTriangularRM<T>::data() noexcept {
    return data_.data();
}

template <typename T>
TRI_FORCE_INLINE typename LowerTriangularRM<T>::const_pointer LowerTriangularRM<T>::data()
    const noexcept {
    return data_.data();
}

// Extern template instantiations for common types
extern template class LowerTriangularRM<float>;
extern template class LowerTriangularRM<double>;

}  // namespace core
}  // namespace tri