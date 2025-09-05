/**
 * @file gemm.cpp
 * @brief General matrix multiplication BLAS wrapper implementation
 * @author Yongze
 * @date 2025-08-13
 */

#include "tri/blas/gemm.hpp"
#include <stdexcept>

#ifdef TRI_USE_BLAS
#include <cblas.h>
#endif

namespace tri {
namespace blas {

template<typename T>
void gemm(tri::common::TransposeOp transA, tri::common::TransposeOp transB,
          std::size_t m, std::size_t n, std::size_t k,
          T alpha, const T* A, std::size_t lda,
          const T* B, std::size_t ldb,
          T beta, T* C, std::size_t ldc) {
    
#ifdef TRI_USE_BLAS
    CBLAS_TRANSPOSE cblas_transA = (transA == tri::common::TransposeOp::Trans) ? CblasTrans : CblasNoTrans;
    CBLAS_TRANSPOSE cblas_transB = (transB == tri::common::TransposeOp::Trans) ? CblasTrans : CblasNoTrans;
    
    if constexpr (std::is_same_v<T, float>) {
        cblas_sgemm(CblasRowMajor, cblas_transA, cblas_transB,
                    static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                    alpha, A, static_cast<int>(lda),
                    B, static_cast<int>(ldb),
                    beta, C, static_cast<int>(ldc));
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dgemm(CblasRowMajor, cblas_transA, cblas_transB,
                    static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                    alpha, A, static_cast<int>(lda),
                    B, static_cast<int>(ldb),
                    beta, C, static_cast<int>(ldc));
    } else {
        throw std::runtime_error("Unsupported type for GEMM");
    }
#else
    // Fallback naive implementation
    if (beta == T{0}) {
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                C[i * ldc + j] = T{0};
            }
        }
    } else if (beta != T{1}) {
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                C[i * ldc + j] *= beta;
            }
        }
    }
    
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            T sum = T{0};
            for (std::size_t l = 0; l < k; ++l) {
                T a_val = (transA == tri::common::TransposeOp::NoTrans) ? 
                    A[i * lda + l] : A[l * lda + i];
                T b_val = (transB == tri::common::TransposeOp::NoTrans) ? 
                    B[l * ldb + j] : B[j * ldb + l];
                sum += a_val * b_val;
            }
            C[i * ldc + j] += alpha * sum;
        }
    }
#endif
}

// Explicit instantiations
template void gemm<float>(tri::common::TransposeOp, tri::common::TransposeOp,
                         std::size_t, std::size_t, std::size_t,
                         float, const float*, std::size_t,
                         const float*, std::size_t,
                         float, float*, std::size_t);

template void gemm<double>(tri::common::TransposeOp, tri::common::TransposeOp,
                          std::size_t, std::size_t, std::size_t,
                          double, const double*, std::size_t,
                          const double*, std::size_t,
                          double, double*, std::size_t);

} // namespace blas
} // namespace tri