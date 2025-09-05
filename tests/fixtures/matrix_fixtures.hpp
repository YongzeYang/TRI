/**
 * @file matrix_fixtures.hpp
 * @brief Test fixtures for matrix testing
 * @author Yongze
 * @date 2025-08-14
 */

#pragma once

#include "test_runner.hpp"
#include "test_assertions.hpp"
#include "test_utils.hpp"
#include "tri/core/dense_rm.hpp"
#include "tri/core/lower_tri_rm.hpp"
#include "tri/core/blocked_tri.hpp"
#include <vector>
#include <memory>

namespace tri::test {

// Base fixture for all matrix tests
template<typename T>
class MatrixTestFixture : public TestCase {
protected:
    // Common test sizes
    static constexpr std::size_t TINY_SIZE = 2;
    static constexpr std::size_t SMALL_SIZE = 4;
    static constexpr std::size_t MEDIUM_SIZE = 32;
    static constexpr std::size_t LARGE_SIZE = 128;
    static constexpr std::size_t HUGE_SIZE = 1024;
    
    // Common block sizes for blocked matrices
    static constexpr std::size_t SMALL_BLOCK = 4;
    static constexpr std::size_t MEDIUM_BLOCK = 16;
    static constexpr std::size_t LARGE_BLOCK = 64;
    
    // Tolerance for floating point comparisons
    T tolerance_;
    
    // Random number generator
    std::mt19937 rng_;
    
    // Test data generators
    TestDataGenerator<T> data_gen_;
    
    void SetUp() override {
        // Set tolerance based on type
        if constexpr (std::is_same_v<T, float>) {
            tolerance_ = 1e-5f;
        } else if constexpr (std::is_same_v<T, double>) {
            tolerance_ = 1e-10;
        } else {
            tolerance_ = T(0);
        }
        
        // Initialize random number generator with fixed seed for reproducibility
        rng_.seed(42);
    }
    
    void TearDown() override {
        // Cleanup if needed
    }
    
    // Factory methods for creating test matrices
    auto CreateDenseMatrix(std::size_t rows, std::size_t cols, T value = T(0)) {
        return core::DenseRM<T>(rows, cols, value);
    }
    
    auto CreateRandomDenseMatrix(std::size_t rows, std::size_t cols, T min_val = T(-10), T max_val = T(10)) {
        core::DenseRM<T> m(rows, cols);
        for (std::size_t i = 0; i < rows; ++i) {
            for (std::size_t j = 0; j < cols; ++j) {
                m(i, j) = MathUtils<T>::Random(min_val, max_val);
            }
        }
        return m;
    }
    
    auto CreateIdentityMatrix(std::size_t n) {
        return core::DenseRM<T>::identity(n, n);
    }
    
    auto CreateTriangularMatrix(std::size_t n, T value = T(0)) {
        return core::LowerTriangularRM<T>(n, value);
    }
    
    auto CreateRandomTriangularMatrix(std::size_t n, T min_val = T(-10), T max_val = T(10)) {
        core::LowerTriangularRM<T> m(n);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                m(i, j) = MathUtils<T>::Random(min_val, max_val);
            }
        }
        return m;
    }
    
    auto CreateBlockedMatrix(std::size_t n, std::size_t block_size, T value = T(0)) {
        return core::BlockedTriMatrix<T>(n, block_size, value);
    }
    
    auto CreateRandomBlockedMatrix(std::size_t n, std::size_t block_size, T min_val = T(-10), T max_val = T(10)) {
        core::BlockedTriMatrix<T> m(n, block_size);
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                m(i, j) = MathUtils<T>::Random(min_val, max_val);
            }
        }
        return m;
    }
    
    // Validation helpers
    void ValidateMatrixDimensions(const auto& m, std::size_t expected_rows, std::size_t expected_cols) {
        ASSERT_EQ(expected_rows, m.rows());
        ASSERT_EQ(expected_cols, m.cols());
        ASSERT_EQ(expected_rows * expected_cols, m.size());
    }
    
    void ValidateSquareMatrix(const auto& m, std::size_t expected_dim) {
        ASSERT_TRUE(m.is_square());
        ASSERT_EQ(expected_dim, m.rows());
        ASSERT_EQ(expected_dim, m.cols());
    }
    
    void ValidateIdentityMatrix(const auto& m) {
        ASSERT_TRUE(m.is_square());
        for (std::size_t i = 0; i < m.rows(); ++i) {
            for (std::size_t j = 0; j < m.cols(); ++j) {
                if (i == j) {
                    ASSERT_NEAR(T(1), m(i, j), tolerance_);
                } else {
                    ASSERT_NEAR(T(0), m(i, j), tolerance_);
                }
            }
        }
    }
    
    void ValidateZeroMatrix(const auto& m) {
        for (std::size_t i = 0; i < m.rows(); ++i) {
            for (std::size_t j = 0; j < m.cols(); ++j) {
                ASSERT_NEAR(T(0), m(i, j), tolerance_);
            }
        }
    }
    
    void ValidateLowerTriangular(const auto& m) {
        ASSERT_TRUE(m.is_square());
        for (std::size_t i = 0; i < m.rows(); ++i) {
            for (std::size_t j = i + 1; j < m.cols(); ++j) {
                ASSERT_NEAR(T(0), m(i, j), tolerance_);
            }
        }
    }
    
    void ValidateUpperTriangular(const auto& m) {
        ASSERT_TRUE(m.is_square());
        for (std::size_t i = 1; i < m.rows(); ++i) {
            for (std::size_t j = 0; j < i; ++j) {
                ASSERT_NEAR(T(0), m(i, j), tolerance_);
            }
        }
    }
    
    void ValidateMatricesEqual(const auto& m1, const auto& m2) {
        ASSERT_EQ(m1.rows(), m2.rows());
        ASSERT_EQ(m1.cols(), m2.cols());
        for (std::size_t i = 0; i < m1.rows(); ++i) {
            for (std::size_t j = 0; j < m1.cols(); ++j) {
                ASSERT_NEAR(m1(i, j), m2(i, j), tolerance_);
            }
        }
    }
};

// Specialized fixture for dense matrix tests
template<typename T>
class DenseMatrixFixture : public MatrixTestFixture<T> {
protected:
    core::DenseRM<T> small_matrix_;
    core::DenseRM<T> medium_matrix_;
    core::DenseRM<T> identity_matrix_;
    
    void SetUp() override {
        MatrixTestFixture<T>::SetUp();
        
        // Create test matrices
        small_matrix_ = this->CreateDenseMatrix(this->SMALL_SIZE, this->SMALL_SIZE, T(1));
        medium_matrix_ = this->CreateRandomDenseMatrix(this->MEDIUM_SIZE, this->MEDIUM_SIZE);
        identity_matrix_ = this->CreateIdentityMatrix(this->SMALL_SIZE);
    }
};

// Specialized fixture for triangular matrix tests
template<typename T>
class TriangularMatrixFixture : public MatrixTestFixture<T> {
protected:
    core::LowerTriangularRM<T> small_tri_;
    core::LowerTriangularRM<T> medium_tri_;
    core::LowerTriangularRM<T> identity_tri_;
    
    void SetUp() override {
        MatrixTestFixture<T>::SetUp();
        
        // Create test matrices
        small_tri_ = this->CreateTriangularMatrix(this->SMALL_SIZE, T(2));
        medium_tri_ = this->CreateRandomTriangularMatrix(this->MEDIUM_SIZE);
        identity_tri_ = core::LowerTriangularRM<T>::identity(this->SMALL_SIZE);
    }
};

// Specialized fixture for blocked matrix tests
template<typename T>
class BlockedMatrixFixture : public MatrixTestFixture<T> {
protected:
    core::BlockedTriMatrix<T> small_blocked_;
    core::BlockedTriMatrix<T> medium_blocked_;
    core::BlockedTriMatrix<T> large_blocked_;
    
    void SetUp() override {
        MatrixTestFixture<T>::SetUp();
        
        // Create test matrices with different block sizes
        small_blocked_ = this->CreateBlockedMatrix(this->SMALL_SIZE, this->SMALL_BLOCK, T(3));
        medium_blocked_ = this->CreateRandomBlockedMatrix(this->MEDIUM_SIZE, this->SMALL_BLOCK);
        large_blocked_ = this->CreateBlockedMatrix(this->LARGE_SIZE, this->MEDIUM_BLOCK, T(0));
    }
};

// Fixture for factory tests
template<typename T>
class FactoryTestFixture : public MatrixTestFixture<T> {
protected:
    void SetUp() override {
        MatrixTestFixture<T>::SetUp();
    }
    
    // Validate factory-created matrix
    template<typename MatrixType>
    void ValidateFactoryMatrix(const MatrixType& m, std::size_t expected_rows, 
                              std::size_t expected_cols, const std::string& description) {
        ASSERT_EQ(expected_rows, m.rows()) << "Failed for: " << description;
        ASSERT_EQ(expected_cols, m.cols()) << "Failed for: " << description;
        ASSERT_FALSE(m.empty()) << "Failed for: " << description;
    }
};

// Parametrized test fixture for size variations
template<typename T>
class ParametrizedSizeFixture : public MatrixTestFixture<T> {
public:
    struct SizeParam {
        std::size_t size;
        std::string description;
    };
    
protected:
    std::vector<SizeParam> GetTestSizes() const {
        return {
            {1, "single element"},
            {2, "tiny (2x2)"},
            {4, "small (4x4)"},
            {8, "small-medium (8x8)"},
            {16, "medium (16x16)"},
            {32, "medium-large (32x32)"},
            {64, "large (64x64)"}
        };
    }
    
    std::vector<SizeParam> GetBlockSizes() const {
        return {
            {2, "tiny blocks"},
            {4, "small blocks"},
            {8, "medium blocks"},
            {16, "large blocks"}
        };
    }
};

// Performance testing fixture
template<typename T>
class PerformanceTestFixture : public MatrixTestFixture<T> {
protected:
    // Measure operation time
    template<typename Func>
    double MeasureTime(Func&& operation, int iterations = 1) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            operation();
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start);
        return duration.count() / iterations;
    }
    
    // Compare performance of two operations
    template<typename Func1, typename Func2>
    void ComparePerformance(const std::string& name1, Func1&& op1,
                           const std::string& name2, Func2&& op2,
                           int iterations = 100) {
        double time1 = MeasureTime(op1, iterations);
        double time2 = MeasureTime(op2, iterations);
        
        std::cout << "\nPerformance Comparison:\n";
        std::cout << "  " << name1 << ": " << std::fixed << std::setprecision(3) 
                 << time1 << " ms\n";
        std::cout << "  " << name2 << ": " << std::fixed << std::setprecision(3) 
                 << time2 << " ms\n";
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
                 << (time1 / time2) << "x\n";
    }
};

} // namespace tri::test