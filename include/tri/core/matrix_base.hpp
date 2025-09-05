#pragma once

/**
 * @file matrix_base.hpp
 * @brief Base interfaces and concepts for matrix types
 * @author Yongze
 * @date 2025-08-13
 */

#include "tri/common/types.hpp"
#include <type_traits>

#if __cplusplus >= 202002L
#include <concepts>
#endif

namespace tri {
namespace core {

/**
 * @brief Abstract base interface for all matrix types
 * @tparam T Element type
 */
template<typename T>
class MatrixBase {
public:
    using value_type = T;
    using size_type = std::size_t;
    using reference = T&;
    using const_reference = const T&;
    using pointer = T*;
    using const_pointer = const T*;

    virtual ~MatrixBase() = default;

    [[nodiscard]] virtual size_type rows() const noexcept = 0;
    [[nodiscard]] virtual size_type cols() const noexcept = 0;
    [[nodiscard]] virtual size_type size() const noexcept = 0;
    
    [[nodiscard]] virtual bool is_square() const noexcept {
        return rows() == cols();
    }
    
    [[nodiscard]] virtual bool empty() const noexcept {
        return size() == 0;
    }

    [[nodiscard]] virtual const_reference operator()(size_type i, size_type j) const = 0;
    virtual reference operator()(size_type i, size_type j) = 0;

    [[nodiscard]] virtual const_pointer data() const noexcept = 0;
    virtual pointer data() noexcept = 0;
};

#if __cplusplus >= 202002L
template<typename T>
concept MatrixLike = requires(T t, const T ct, std::size_t i, std::size_t j) {
    typename T::value_type;
    typename T::size_type;
    
    { ct.rows() } -> std::convertible_to<std::size_t>;
    { ct.cols() } -> std::convertible_to<std::size_t>;
    { ct.size() } -> std::convertible_to<std::size_t>;
    { ct.is_square() } -> std::convertible_to<bool>;
    { ct.empty() } -> std::convertible_to<bool>;
    
    { ct(i, j) } -> std::convertible_to<const typename T::value_type&>;
    { t(i, j) } -> std::convertible_to<typename T::value_type&>;
    
    { ct.data() } -> std::convertible_to<const typename T::value_type*>;
    { t.data() } -> std::convertible_to<typename T::value_type*>;
};
#endif

template<typename T, typename = void>
struct is_matrix_like : std::false_type {};

template<typename T>
struct is_matrix_like<T, std::void_t<
    typename T::value_type,
    typename T::size_type,
    decltype(std::declval<const T&>().rows()),
    decltype(std::declval<const T&>().cols()),
    decltype(std::declval<const T&>().size()),
    decltype(std::declval<const T&>().is_square()),
    decltype(std::declval<const T&>().empty()),
    decltype(std::declval<const T&>()(std::size_t{}, std::size_t{})),
    decltype(std::declval<T&>()(std::size_t{}, std::size_t{})),
    decltype(std::declval<const T&>().data()),
    decltype(std::declval<T&>().data())
>> : std::true_type {};

template<typename T>
inline constexpr bool is_matrix_like_v = is_matrix_like<T>::value;

template<typename T>
using enable_if_matrix_like_t = std::enable_if_t<is_matrix_like_v<T>>;

} // namespace core
} // namespace tri