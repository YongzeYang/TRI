/**
 * @file solver.cpp
 * @brief Linear system solver LAPACK wrapper implementation
 * @author Yongze
 * @date 2025-08-13
 */

#include "tri/blas/solver.hpp"

#include <cstring>
#include <stdexcept>

#ifdef TRI_USE_BLAS
#include <cblas.h>
#endif

namespace tri {
namespace blas {

template <typename T>
void trmm(tri::common::Side side, tri::common::Uplo uplo, tri::common::TransposeOp transA,
          tri::common::Diag diag, std::size_t m, std::size_t n, T alpha, const T* A,
          std::size_t lda, T* B, std::size_t ldb) {
#ifdef TRI_USE_BLAS
    CBLAS_SIDE cblas_side = (side == tri::common::Side::Left) ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == tri::common::Uplo::Upper) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_transA =
        (transA == tri::common::TransposeOp::Trans) ? CblasTrans : CblasNoTrans;
    CBLAS_DIAG cblas_diag = (diag == tri::common::Diag::Unit) ? CblasUnit : CblasNonUnit;

    if constexpr (std::is_same_v<T, float>) {
        cblas_strmm(CblasRowMajor, cblas_side, cblas_uplo, cblas_transA, cblas_diag,
                    static_cast<int>(m), static_cast<int>(n), alpha, A, static_cast<int>(lda), B,
                    static_cast<int>(ldb));
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dtrmm(CblasRowMajor, cblas_side, cblas_uplo, cblas_transA, cblas_diag,
                    static_cast<int>(m), static_cast<int>(n), alpha, A, static_cast<int>(lda), B,
                    static_cast<int>(ldb));
    } else {
        throw std::runtime_error("Unsupported type for TRMM");
    }
#else
    // Fallback implementation for triangular matrix multiplication
    // B := alpha * op(A) * B (side == Left)
    // B := alpha * B * op(A) (side == Right)

    // Create temporary storage for result
    std::vector<T> temp(m * n);

    if (side == tri::common::Side::Left) {
        // B := alpha * op(A) * B
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                T sum = T{0};

                if (uplo == tri::common::Uplo::Lower) {
                    // Lower triangular
                    if (transA == tri::common::TransposeOp::NoTrans) {
                        for (std::size_t k = 0; k <= i; ++k) {
                            T a_val =
                                (k == i && diag == tri::common::Diag::Unit) ? T{1} : A[i * lda + k];
                            sum += a_val * B[k * ldb + j];
                        }
                    } else {
                        for (std::size_t k = i; k < m; ++k) {
                            T a_val =
                                (k == i && diag == tri::common::Diag::Unit) ? T{1} : A[k * lda + i];
                            sum += a_val * B[k * ldb + j];
                        }
                    }
                } else {
                    // Upper triangular
                    if (transA == tri::common::TransposeOp::NoTrans) {
                        for (std::size_t k = i; k < m; ++k) {
                            T a_val =
                                (k == i && diag == tri::common::Diag::Unit) ? T{1} : A[i * lda + k];
                            sum += a_val * B[k * ldb + j];
                        }
                    } else {
                        for (std::size_t k = 0; k <= i; ++k) {
                            T a_val =
                                (k == i && diag == tri::common::Diag::Unit) ? T{1} : A[k * lda + i];
                            sum += a_val * B[k * ldb + j];
                        }
                    }
                }
                temp[i * n + j] = alpha * sum;
            }
        }
    } else {
        // B := alpha * B * op(A)
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                T sum = T{0};

                if (uplo == tri::common::Uplo::Lower) {
                    if (transA == tri::common::TransposeOp::NoTrans) {
                        for (std::size_t k = 0; k <= j; ++k) {
                            T a_val =
                                (k == j && diag == tri::common::Diag::Unit) ? T{1} : A[j * lda + k];
                            sum += B[i * ldb + k] * a_val;
                        }
                    } else {
                        for (std::size_t k = j; k < n; ++k) {
                            T a_val =
                                (k == j && diag == tri::common::Diag::Unit) ? T{1} : A[k * lda + j];
                            sum += B[i * ldb + k] * a_val;
                        }
                    }
                } else {
                    if (transA == tri::common::TransposeOp::NoTrans) {
                        for (std::size_t k = j; k < n; ++k) {
                            T a_val =
                                (k == j && diag == tri::common::Diag::Unit) ? T{1} : A[j * lda + k];
                            sum += B[i * ldb + k] * a_val;
                        }
                    } else {
                        for (std::size_t k = 0; k <= j; ++k) {
                            T a_val =
                                (k == j && diag == tri::common::Diag::Unit) ? T{1} : A[k * lda + j];
                            sum += B[i * ldb + k] * a_val;
                        }
                    }
                }
                temp[i * n + j] = alpha * sum;
            }
        }
    }

    // Copy result back to B
    std::memcpy(B, temp.data(), m * n * sizeof(T));
#endif
}

template <typename T>
void trsm(tri::common::Side side, tri::common::Uplo uplo, tri::common::TransposeOp transA,
          tri::common::Diag diag, std::size_t m, std::size_t n, T alpha, const T* A,
          std::size_t lda, T* B, std::size_t ldb) {
#ifdef TRI_USE_BLAS
    CBLAS_SIDE cblas_side = (side == tri::common::Side::Left) ? CblasLeft : CblasRight;
    CBLAS_UPLO cblas_uplo = (uplo == tri::common::Uplo::Upper) ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_transA =
        (transA == tri::common::TransposeOp::Trans) ? CblasTrans : CblasNoTrans;
    CBLAS_DIAG cblas_diag = (diag == tri::common::Diag::Unit) ? CblasUnit : CblasNonUnit;

    if constexpr (std::is_same_v<T, float>) {
        cblas_strsm(CblasRowMajor, cblas_side, cblas_uplo, cblas_transA, cblas_diag,
                    static_cast<int>(m), static_cast<int>(n), alpha, A, static_cast<int>(lda), B,
                    static_cast<int>(ldb));
    } else if constexpr (std::is_same_v<T, double>) {
        cblas_dtrsm(CblasRowMajor, cblas_side, cblas_uplo, cblas_transA, cblas_diag,
                    static_cast<int>(m), static_cast<int>(n), alpha, A, static_cast<int>(lda), B,
                    static_cast<int>(ldb));
    } else {
        throw std::runtime_error("Unsupported type for TRSM");
    }
#else
    // Fallback implementation for triangular solve
    // Solve op(A) * X = alpha * B (side == Left)
    // Solve X * op(A) = alpha * B (side == Right)

    // First scale B by alpha
    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            B[i * ldb + j] *= alpha;
        }
    }

    if (side == tri::common::Side::Left) {
        // Solve op(A) * X = B for X, overwriting B with X
        if (uplo == tri::common::Uplo::Lower) {
            if (transA == tri::common::TransposeOp::NoTrans) {
                // Forward substitution for lower triangular system
                for (std::size_t i = 0; i < m; ++i) {
                    for (std::size_t j = 0; j < n; ++j) {
                        T sum = B[i * ldb + j];
                        for (std::size_t k = 0; k < i; ++k) {
                            sum -= A[i * lda + k] * B[k * ldb + j];
                        }
                        if (diag != tri::common::Diag::Unit) {
                            sum /= A[i * lda + i];
                        }
                        B[i * ldb + j] = sum;
                    }
                }
            } else {
                // Backward substitution for transposed lower (= upper) triangular
                for (std::ptrdiff_t i = m - 1; i >= 0; --i) {
                    for (std::size_t j = 0; j < n; ++j) {
                        T sum = B[i * ldb + j];
                        for (std::size_t k = i + 1; k < m; ++k) {
                            sum -= A[k * lda + i] * B[k * ldb + j];
                        }
                        if (diag != tri::common::Diag::Unit) {
                            sum /= A[i * lda + i];
                        }
                        B[i * ldb + j] = sum;
                    }
                }
            }
        } else {
            // Upper triangular
            if (transA == tri::common::TransposeOp::NoTrans) {
                // Backward substitution for upper triangular system
                for (std::ptrdiff_t i = m - 1; i >= 0; --i) {
                    for (std::size_t j = 0; j < n; ++j) {
                        T sum = B[i * ldb + j];
                        for (std::size_t k = i + 1; k < m; ++k) {
                            sum -= A[i * lda + k] * B[k * ldb + j];
                        }
                        if (diag != tri::common::Diag::Unit) {
                            sum /= A[i * lda + i];
                        }
                        B[i * ldb + j] = sum;
                    }
                }
            } else {
                // Forward substitution for transposed upper (= lower) triangular
                for (std::size_t i = 0; i < m; ++i) {
                    for (std::size_t j = 0; j < n; ++j) {
                        T sum = B[i * ldb + j];
                        for (std::size_t k = 0; k < i; ++k) {
                            sum -= A[k * lda + i] * B[k * ldb + j];
                        }
                        if (diag != tri::common::Diag::Unit) {
                            sum /= A[i * lda + i];
                        }
                        B[i * ldb + j] = sum;
                    }
                }
            }
        }
    } else {
        // Solve X * op(A) = B for X, overwriting B with X
        if (uplo == tri::common::Uplo::Lower) {
            if (transA == tri::common::TransposeOp::NoTrans) {
                // Solve from right to left for lower triangular
                for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (diag != tri::common::Diag::Unit) {
                        T divisor = A[j * lda + j];
                        for (std::size_t i = 0; i < m; ++i) {
                            B[i * ldb + j] /= divisor;
                        }
                    }
                    for (std::ptrdiff_t k = j - 1; k >= 0; --k) {
                        T factor = A[j * lda + k];
                        for (std::size_t i = 0; i < m; ++i) {
                            B[i * ldb + k] -= factor * B[i * ldb + j];
                        }
                    }
                }
            } else {
                // Solve from left to right for transposed lower
                for (std::size_t j = 0; j < n; ++j) {
                    if (diag != tri::common::Diag::Unit) {
                        T divisor = A[j * lda + j];
                        for (std::size_t i = 0; i < m; ++i) {
                            B[i * ldb + j] /= divisor;
                        }
                    }
                    for (std::size_t k = j + 1; k < n; ++k) {
                        T factor = A[k * lda + j];
                        for (std::size_t i = 0; i < m; ++i) {
                            B[i * ldb + k] -= factor * B[i * ldb + j];
                        }
                    }
                }
            }
        } else {
            // Upper triangular
            if (transA == tri::common::TransposeOp::NoTrans) {
                // Solve from left to right for upper triangular
                for (std::size_t j = 0; j < n; ++j) {
                    if (diag != tri::common::Diag::Unit) {
                        T divisor = A[j * lda + j];
                        for (std::size_t i = 0; i < m; ++i) {
                            B[i * ldb + j] /= divisor;
                        }
                    }
                    for (std::size_t k = j + 1; k < n; ++k) {
                        T factor = A[j * lda + k];
                        for (std::size_t i = 0; i < m; ++i) {
                            B[i * ldb + k] -= factor * B[i * ldb + j];
                        }
                    }
                }
            } else {
                // Solve from right to left for transposed upper
                for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (diag != tri::common::Diag::Unit) {
                        T divisor = A[j * lda + j];
                        for (std::size_t i = 0; i < m; ++i) {
                            B[i * ldb + j] /= divisor;
                        }
                    }
                    for (std::ptrdiff_t k = j - 1; k >= 0; --k) {
                        T factor = A[k * lda + j];
                        for (std::size_t i = 0; i < m; ++i) {
                            B[i * ldb + k] -= factor * B[i * ldb + j];
                        }
                    }
                }
            }
        }
    }
#endif
}

// Explicit instantiations
template void trmm<float>(tri::common::Side, tri::common::Uplo, tri::common::TransposeOp,
                          tri::common::Diag, std::size_t, std::size_t, float, const float*,
                          std::size_t, float*, std::size_t);

template void trmm<double>(tri::common::Side, tri::common::Uplo, tri::common::TransposeOp,
                           tri::common::Diag, std::size_t, std::size_t, double, const double*,
                           std::size_t, double*, std::size_t);

template void trsm<float>(tri::common::Side, tri::common::Uplo, tri::common::TransposeOp,
                          tri::common::Diag, std::size_t, std::size_t, float, const float*,
                          std::size_t, float*, std::size_t);

template void trsm<double>(tri::common::Side, tri::common::Uplo, tri::common::TransposeOp,
                           tri::common::Diag, std::size_t, std::size_t, double, const double*,
                           std::size_t, double*, std::size_t);

}  // namespace blas
}  // namespace tri