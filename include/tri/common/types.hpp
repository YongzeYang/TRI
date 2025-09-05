#pragma once

/**
 * @file types.hpp
 * @brief Common types and type traits for the tri library
 * @author Yongze
 * @date 2025-08-13
 */

#include <cstddef>
#include <type_traits>

namespace tri {
namespace common {

// Basic type aliases
using size_type = std::size_t;
using difference_type = std::ptrdiff_t;

// Type traits for numeric types
template<typename T>
struct is_numeric : std::integral_constant<bool,
    std::is_arithmetic_v<T> && !std::is_same_v<T, bool>> {};

template<typename T>
inline constexpr bool is_numeric_v = is_numeric<T>::value;

template<typename T>
struct is_floating_point : std::is_floating_point<T> {};

template<typename T>
inline constexpr bool is_floating_point_v = is_floating_point<T>::value;

// BLAS operation enums
enum class TransposeOp {
    NoTrans = 0,
    Trans = 1,
    ConjTrans = 2
};

enum class Side {
    Left = 0,
    Right = 1
};

enum class Uplo {
    Upper = 0,
    Lower = 1
};

enum class Diag {
    NonUnit = 0,
    Unit = 1
};

} // namespace common
} // namespace tri