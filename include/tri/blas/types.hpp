#pragma once

/**
 * @file types.hpp
 * @brief Common types and enums for BLAS operations
 * @author Yongze
 * @date 2025-08-09
 */

namespace tri {
namespace blas {

/**
 * @brief Transpose operation flag
 */
enum class TransposeOp {
    NoTrans = 0,
    Trans = 1,
    ConjTrans = 2
};

/**
 * @brief Matrix side flag
 */
enum class Side {
    Left = 0,
    Right = 1
};

/**
 * @brief Triangle type
 */
enum class Uplo {
    Upper = 0,
    Lower = 1
};

/**
 * @brief Diagonal type
 */
enum class Diag {
    NonUnit = 0,
    Unit = 1
};

} // namespace blas
} // namespace tri