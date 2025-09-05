#pragma once

/**
 * @file gemm.hpp
 * @brief General matrix multiplication BLAS wrapper
 * @author Yongze
 * @date 2025-08-13
 */

#include "tri/common/types.hpp"
#include <cstddef>

namespace tri {
namespace blas {

/**
 * @brief General matrix multiplication C = alpha*op(A)*op(B) + beta*C
 */
template<typename T>
void gemm(tri::common::TransposeOp transA, tri::common::TransposeOp transB,
          std::size_t m, std::size_t n, std::size_t k,
          T alpha, const T* A, std::size_t lda,
          const T* B, std::size_t ldb,
          T beta, T* C, std::size_t ldc);

// Explicit instantiations
extern template void gemm<float>(tri::common::TransposeOp, tri::common::TransposeOp,
                                 std::size_t, std::size_t, std::size_t,
                                 float, const float*, std::size_t,
                                 const float*, std::size_t,
                                 float, float*, std::size_t);

extern template void gemm<double>(tri::common::TransposeOp, tri::common::TransposeOp,
                                  std::size_t, std::size_t, std::size_t,
                                  double, const double*, std::size_t,
                                  const double*, std::size_t,
                                  double, double*, std::size_t);

} // namespace blas
} // namespace tri