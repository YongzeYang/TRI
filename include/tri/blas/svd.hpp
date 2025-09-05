#pragma once

/**
 * @file svd.hpp
 * @brief Singular value decomposition wrapper
 * @author Yongze
 * @date 2025-08-13
 */

#include <cstddef>

namespace tri {
namespace blas {

/**
 * @brief Compute singular value decomposition A = U*S*V^T
 */
template<typename T>
int gesvd(std::size_t m, std::size_t n,
          T* A, std::size_t lda,
          T* S, T* U, std::size_t ldu,
          T* VT, std::size_t ldvt);

/**
 * @brief Compute Moore-Penrose pseudoinverse using SVD
 */
template<typename T>
int pinv(std::size_t m, std::size_t n,
         T* A, std::size_t lda,
         T* Ainv, std::size_t ldainv,
         T tolerance = -1);

// Explicit instantiations
extern template int gesvd<float>(std::size_t, std::size_t,
                                 float*, std::size_t,
                                 float*, float*, std::size_t,
                                 float*, std::size_t);

extern template int gesvd<double>(std::size_t, std::size_t,
                                  double*, std::size_t,
                                  double*, double*, std::size_t,
                                  double*, std::size_t);

extern template int pinv<float>(std::size_t, std::size_t,
                                float*, std::size_t,
                                float*, std::size_t,
                                float);

extern template int pinv<double>(std::size_t, std::size_t,
                                 double*, std::size_t,
                                 double*, std::size_t,
                                 double);

} // namespace blas
} // namespace tri