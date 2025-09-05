#pragma once

/**
 * @file factory.hpp
 * @brief Matrix factory functions for creating common matrix types
 * @author Yongze
 * @date 2025-08-10
 */

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

#include "dense_rm.hpp"
#include "lower_tri_rm.hpp"
#include "matrix_base.hpp"

#if __cplusplus >= 202002L
#include <concepts>
#endif

namespace tri {
namespace core {

/**
 * @brief Factory class for creating matrices with common patterns
 * @tparam T Element type (must be numeric)
 */
template <typename T>
class MatrixFactory {
    static_assert(std::is_arithmetic_v<T>, "Matrix element type must be arithmetic");

   public:
    using value_type = T;
    using size_type = std::size_t;
    using DenseMatrix = DenseRM<T>;
    using LowerTriangularMatrix = LowerTriangularRM<T>;

    /**
     * @brief Create dense identity matrix
     * @param n Matrix dimension
     * @return n×n identity matrix
     * @throw std::invalid_argument if n == 0
     */
    [[nodiscard]] static DenseMatrix identity(size_type n) {
        if (n == 0) {
            throw std::invalid_argument("Matrix dimension must be positive");
        }

        DenseMatrix result(n, n, T{0});
        for (size_type i = 0; i < n; ++i) {
            result(i, i) = T{1};
        }
        return result;
    }

    /**
     * @brief Create dense zero matrix
     * @param rows Number of rows
     * @param cols Number of columns
     * @return rows×cols zero matrix
     * @throw std::invalid_argument if rows or cols is 0
     */
    [[nodiscard]] static DenseMatrix zeros(size_type rows, size_type cols) {
        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("Matrix dimensions must be positive");
        }
        return DenseMatrix(rows, cols, T{0});
    }

    /**
     * @brief Create dense zero matrix (square)
     * @param n Matrix dimension
     * @return n×n zero matrix
     * @throw std::invalid_argument if n == 0
     */
    [[nodiscard]] static DenseMatrix zeros(size_type n) { return zeros(n, n); }

    /**
     * @brief Create dense matrix filled with ones
     * @param rows Number of rows
     * @param cols Number of columns
     * @return rows×cols matrix filled with ones
     * @throw std::invalid_argument if rows or cols is 0
     */
    [[nodiscard]] static DenseMatrix ones(size_type rows, size_type cols) {
        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("Matrix dimensions must be positive");
        }
        return DenseMatrix(rows, cols, T{1});
    }

    /**
     * @brief Create dense matrix filled with ones (square)
     * @param n Matrix dimension
     * @return n×n matrix filled with ones
     * @throw std::invalid_argument if n == 0
     */
    [[nodiscard]] static DenseMatrix ones(size_type n) { return ones(n, n); }

    /**
     * @brief Create dense matrix filled with specific value
     * @param rows Number of rows
     * @param cols Number of columns
     * @param value Fill value
     * @return rows×cols matrix filled with value
     * @throw std::invalid_argument if rows or cols is 0
     */
    [[nodiscard]] static DenseMatrix constant(size_type rows, size_type cols, const T& value) {
        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("Matrix dimensions must be positive");
        }
        return DenseMatrix(rows, cols, value);
    }

    /**
     * @brief Create diagonal matrix from vector
     * @param diagonal_elements Vector of diagonal elements
     * @return Diagonal matrix with given elements
     * @throw std::invalid_argument if vector is empty
     */
    [[nodiscard]] static DenseMatrix diagonal(const std::vector<T>& diagonal_elements) {
        if (diagonal_elements.empty()) {
            throw std::invalid_argument("Diagonal vector cannot be empty");
        }

        const size_type n = diagonal_elements.size();
        DenseMatrix result(n, n, T{0});
        for (size_type i = 0; i < n; ++i) {
            result(i, i) = diagonal_elements[i];
        }
        return result;
    }

    /**
     * @brief Create random matrix with uniform distribution
     * @param rows Number of rows
     * @param cols Number of columns
     * @param min_val Minimum value (default: 0)
     * @param max_val Maximum value (default: 1)
     * @return rows×cols random matrix
     * @throw std::invalid_argument if dimensions invalid or min_val >= max_val
     */
    [[nodiscard]] static DenseMatrix random(size_type rows, size_type cols, T min_val = T{0},
                                            T max_val = T{1}) {
        if (rows == 0 || cols == 0) {
            throw std::invalid_argument("Matrix dimensions must be positive");
        }
        if (min_val >= max_val) {
            throw std::invalid_argument("min_val must be less than max_val");
        }

        DenseMatrix result(rows, cols);
        std::random_device rd;
        std::mt19937 gen(rd());

        if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution<T> dist(min_val, max_val);
            for (size_type i = 0; i < rows; ++i) {
                for (size_type j = 0; j < cols; ++j) {
                    result(i, j) = dist(gen);
                }
            }
        } else {
            std::uniform_int_distribution<T> dist(min_val, max_val);
            for (size_type i = 0; i < rows; ++i) {
                for (size_type j = 0; j < cols; ++j) {
                    result(i, j) = dist(gen);
                }
            }
        }
        return result;
    }

    // Lower triangular matrix factories

    /**
     * @brief Create lower triangular identity matrix
     * @param n Matrix dimension
     * @return n×n lower triangular identity matrix
     * @throw std::invalid_argument if n == 0
     */
    [[nodiscard]] static LowerTriangularMatrix lower_identity(size_type n) {
        if (n == 0) {
            throw std::invalid_argument("Matrix dimension must be positive");
        }

        LowerTriangularMatrix result(n, T{0});
        for (size_type i = 0; i < n; ++i) {
            result(i, i) = T{1};
        }
        return result;
    }

    /**
     * @brief Create lower triangular zero matrix
     * @param n Matrix dimension
     * @return n×n lower triangular zero matrix
     * @throw std::invalid_argument if n == 0
     */
    [[nodiscard]] static LowerTriangularMatrix lower_zeros(size_type n) {
        if (n == 0) {
            throw std::invalid_argument("Matrix dimension must be positive");
        }
        return LowerTriangularMatrix(n, T{0});
    }

    /**
     * @brief Create lower triangular matrix filled with ones
     * @param n Matrix dimension
     * @return n×n lower triangular matrix filled with ones
     * @throw std::invalid_argument if n == 0
     */
    [[nodiscard]] static LowerTriangularMatrix lower_ones(size_type n) {
        if (n == 0) {
            throw std::invalid_argument("Matrix dimension must be positive");
        }
        return LowerTriangularMatrix(n, T{1});
    }

    /**
     * @brief Create lower triangular matrix with specific value
     * @param n Matrix dimension
     * @param value Fill value
     * @return n×n lower triangular matrix filled with value
     * @throw std::invalid_argument if n == 0
     */
    [[nodiscard]] static LowerTriangularMatrix lower_constant(size_type n, const T& value) {
        if (n == 0) {
            throw std::invalid_argument("Matrix dimension must be positive");
        }
        return LowerTriangularMatrix(n, value);
    }

    /**
     * @brief Create random lower triangular matrix
     * @param n Matrix dimension
     * @param min_val Minimum value (default: 0)
     * @param max_val Maximum value (default: 1)
     * @return n×n random lower triangular matrix
     * @throw std::invalid_argument if n == 0 or min_val >= max_val
     */
    [[nodiscard]] static LowerTriangularMatrix lower_random(size_type n, T min_val = T{0},
                                                            T max_val = T{1}) {
        if (n == 0) {
            throw std::invalid_argument("Matrix dimension must be positive");
        }
        if (min_val >= max_val) {
            throw std::invalid_argument("min_val must be less than max_val");
        }

        LowerTriangularMatrix result(n);
        std::random_device rd;
        std::mt19937 gen(rd());

        if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution<T> dist(min_val, max_val);
            for (size_type i = 0; i < n; ++i) {
                for (size_type j = 0; j <= i; ++j) {
                    result(i, j) = dist(gen);
                }
            }
        } else {
            std::uniform_int_distribution<T> dist(min_val, max_val);
            for (size_type i = 0; i < n; ++i) {
                for (size_type j = 0; j <= i; ++j) {
                    result(i, j) = dist(gen);
                }
            }
        }
        return result;
    }

   private:
    // Factory class should not be instantiated
    MatrixFactory() = delete;
    ~MatrixFactory() = delete;
    MatrixFactory(const MatrixFactory&) = delete;
    MatrixFactory& operator=(const MatrixFactory&) = delete;
};

// Convenience type aliases
using FloatFactory = MatrixFactory<float>;
using DoubleFactory = MatrixFactory<double>;

// Convenience functions for common cases
namespace factory {

/**
 * @brief Create identity matrix (float)
 * @param n Matrix dimension
 * @return n×n identity matrix
 */
[[nodiscard]] inline DenseRM<float> identity_f(std::size_t n) { return FloatFactory::identity(n); }

/**
 * @brief Create identity matrix (double)
 * @param n Matrix dimension
 * @return n×n identity matrix
 */
[[nodiscard]] inline DenseRM<double> identity_d(std::size_t n) {
    return DoubleFactory::identity(n);
}

/**
 * @brief Create zero matrix (float)
 * @param rows Number of rows
 * @param cols Number of columns
 * @return rows×cols zero matrix
 */
[[nodiscard]] inline DenseRM<float> zeros_f(std::size_t rows, std::size_t cols) {
    return FloatFactory::zeros(rows, cols);
}

/**
 * @brief Create zero matrix (double)
 * @param rows Number of rows
 * @param cols Number of columns
 * @return rows×cols zero matrix
 */
[[nodiscard]] inline DenseRM<double> zeros_d(std::size_t rows, std::size_t cols) {
    return DoubleFactory::zeros(rows, cols);
}

}  // namespace factory

}  // namespace core
}  // namespace tri