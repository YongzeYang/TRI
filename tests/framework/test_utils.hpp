/**
 * @file test_utils.hpp
 * @brief Testing utilities and helper functions
 * @author Yongze
 * @date 2025-08-14
 */

#pragma once

#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

namespace tri::test {

// Math utilities
template<typename T>
class MathUtils {
public:
    // Generate random value in range
    static T Random(T min_val, T max_val) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        
        if constexpr (std::is_floating_point_v<T>) {
            std::uniform_real_distribution<T> dist(min_val, max_val);
            return dist(gen);
        } else {
            std::uniform_int_distribution<T> dist(min_val, max_val);
            return dist(gen);
        }
    }
    
    // Generate random vector
    static std::vector<T> RandomVector(std::size_t size, T min_val, T max_val) {
        std::vector<T> vec(size);
        for (auto& val : vec) {
            val = Random(min_val, max_val);
        }
        return vec;
    }
    
    // Check if value is near zero
    static bool IsNearZero(T value, T tolerance = std::numeric_limits<T>::epsilon() * 100) {
        return std::abs(value) <= tolerance;
    }
};

// Performance measurement utilities
class PerformanceTimer {
private:
    std::chrono::high_resolution_clock::time_point start_;
    std::string name_;
    bool stopped_;
    
public:
    explicit PerformanceTimer(const std::string& name = "Operation")
        : name_(name), stopped_(false) {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    ~PerformanceTimer() {
        if (!stopped_) {
            Stop();
        }
    }
    
    double Elapsed() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }
    
    void Stop() {
        if (!stopped_) {
            stopped_ = true;
            double time_ms = Elapsed();
            std::cout << name_ << " took " << std::fixed << std::setprecision(3) 
                     << time_ms << " ms\n";
        }
    }
};

// Test data generators
template<typename T>
class TestDataGenerator {
public:
    // Generate identity matrix data
    static std::vector<T> IdentityData(std::size_t n) {
        std::vector<T> data(n * n, T(0));
        for (std::size_t i = 0; i < n; ++i) {
            data[i * n + i] = T(1);
        }
        return data;
    }
    
    // Generate diagonal matrix data
    static std::vector<T> DiagonalData(std::size_t n, const std::vector<T>& diag_values) {
        std::vector<T> data(n * n, T(0));
        std::size_t min_size = std::min(n, diag_values.size());
        for (std::size_t i = 0; i < min_size; ++i) {
            data[i * n + i] = diag_values[i];
        }
        return data;
    }
    
    // Generate lower triangular data (packed format)
    static std::vector<T> LowerTriangularPackedData(std::size_t n, T min_val = T(0), T max_val = T(10)) {
        std::size_t packed_size = n * (n + 1) / 2;
        return MathUtils<T>::RandomVector(packed_size, min_val, max_val);
    }
    
    // Generate test pattern (useful for debugging)
    static std::vector<T> TestPattern(std::size_t rows, std::size_t cols) {
        std::vector<T> data(rows * cols);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                data[i * cols + j] = static_cast<T>(i * cols + j);
            }
        }
        return data;
    }
};

// Matrix validation helpers (non-member functions to avoid template issues)
template<typename MatrixType>
void ValidateMatrixDimensions(const MatrixType& m, std::size_t expected_rows, std::size_t expected_cols) {
    ASSERT_EQ(expected_rows, m.rows());
    ASSERT_EQ(expected_cols, m.cols());
    ASSERT_EQ(expected_rows * expected_cols, m.size());
}

template<typename MatrixType>
void ValidateSquareMatrix(const MatrixType& m, std::size_t expected_dim) {
    ASSERT_TRUE(m.is_square());
    ASSERT_EQ(expected_dim, m.rows());
    ASSERT_EQ(expected_dim, m.cols());
}

template<typename MatrixType, typename T>
void ValidateIdentityMatrix(const MatrixType& m, T tolerance = std::numeric_limits<T>::epsilon() * 100) {
    ASSERT_TRUE(m.is_square());
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) {
            if (i == j) {
                ASSERT_NEAR(T(1), m(i, j), tolerance);
            } else {
                ASSERT_NEAR(T(0), m(i, j), tolerance);
            }
        }
    }
}

template<typename MatrixType, typename T>
void ValidateZeroMatrix(const MatrixType& m, T tolerance = std::numeric_limits<T>::epsilon() * 100) {
    for (std::size_t i = 0; i < m.rows(); ++i) {
        for (std::size_t j = 0; j < m.cols(); ++j) {
            ASSERT_NEAR(T(0), m(i, j), tolerance);
        }
    }
}

template<typename MatrixType1, typename MatrixType2, typename T>
void ValidateMatricesEqual(const MatrixType1& m1, const MatrixType2& m2, T tolerance = std::numeric_limits<T>::epsilon() * 100) {
    ASSERT_EQ(m1.rows(), m2.rows());
    ASSERT_EQ(m1.cols(), m2.cols());
    for (std::size_t i = 0; i < m1.rows(); ++i) {
        for (std::size_t j = 0; j < m1.cols(); ++j) {
            ASSERT_NEAR(m1(i, j), m2(i, j), tolerance);
        }
    }
}

} // namespace tri::test