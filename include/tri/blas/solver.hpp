#pragma once

/**
 * @file solver.hpp
 * @brief Linear system solver LAPACK wrappers
 * @author Yongze
 * @date 2025-08-13
 */

#include "tri/common/types.hpp"
#include <cstddef>

namespace tri {
namespace blas {

/**
 * @brief Triangular matrix multiplication
 */
template<typename T>
void trmm(tri::common::Side side, tri::common::Uplo uplo, 
          tri::common::TransposeOp transA, tri::common::Diag diag,
          std::size_t m, std::size_t n,
          T alpha, const T* A, std::size_t lda,
          T* B, std::size_t ldb);

/**
 * @brief Triangular solve
 */
template<typename T>
void trsm(tri::common::Side side, tri::common::Uplo uplo,
          tri::common::TransposeOp transA, tri::common::Diag diag,
          std::size_t m, std::size_t n,
          T alpha, const T* A, std::size_t lda,
          T* B, std::size_t ldb);

// Explicit instantiations
extern template void trmm<float>(tri::common::Side, tri::common::Uplo,
                                 tri::common::TransposeOp, tri::common::Diag,
                                 std::size_t, std::size_t,
                                 float, const float*, std::size_t,
                                 float*, std::size_t);

extern template void trmm<double>(tri::common::Side, tri::common::Uplo,
                                  tri::common::TransposeOp, tri::common::Diag,
                                  std::size_t, std::size_t,
                                  double, const double*, std::size_t,
                                  double*, std::size_t);

extern template void trsm<float>(tri::common::Side, tri::common::Uplo,
                                 tri::common::TransposeOp, tri::common::Diag,
                                 std::size_t, std::size_t,
                                 float, const float*, std::size_t,
                                 float*, std::size_t);

extern template void trsm<double>(tri::common::Side, tri::common::Uplo,
                                  tri::common::TransposeOp, tri::common::Diag,
                                  std::size_t, std::size_t,
                                  double, const double*, std::size_t,
                                  double*, std::size_t);

} // namespace blas
} // namespace tri