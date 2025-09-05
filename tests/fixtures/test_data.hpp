/**
 * @file test_data.hpp
 * @brief Predefined test data for matrix testing
 * @author Yongze
 * @date 2025-08-14
 */

#pragma once

#include <vector>
#include <array>
#include <cmath>

namespace tri::test {

// Predefined test matrices
template<typename T>
class TestMatrices {
public:
    // 2x2 matrices
    static std::vector<T> Get2x2Identity() {
        return {T(1), T(0), 
                T(0), T(1)};
    }
    
    static std::vector<T> Get2x2Singular() {
        return {T(1), T(2), 
                T(2), T(4)};
    }
    
    static std::vector<T> Get2x2Invertible() {
        return {T(4), T(7), 
                T(2), T(6)};
    }
    
    // 3x3 matrices
    static std::vector<T> Get3x3Identity() {
        return {T(1), T(0), T(0),
                T(0), T(1), T(0),
                T(0), T(0), T(1)};
    }
    
    static std::vector<T> Get3x3Symmetric() {
        return {T(2), T(1), T(0),
                T(1), T(2), T(1),
                T(0), T(1), T(2)};
    }
    
    static std::vector<T> Get3x3LowerTriangular() {
        return {T(1), T(0), T(0),
                T(2), T(3), T(0),
                T(4), T(5), T(6)};
    }
    
    static std::vector<T> Get3x3UpperTriangular() {
        return {T(1), T(2), T(3),
                T(0), T(4), T(5),
                T(0), T(0), T(6)};
    }
    
    static std::vector<T> Get3x3PositiveDefinite() {
        return {T(4), T(1), T(1),
                T(1), T(4), T(1),
                T(1), T(1), T(4)};
    }
    
    // 4x4 matrices
    static std::vector<T> Get4x4Identity() {
        return {T(1), T(0), T(0), T(0),
                T(0), T(1), T(0), T(0),
                T(0), T(0), T(1), T(0),
                T(0), T(0), T(0), T(1)};
    }
    
    static std::vector<T> Get4x4Hilbert() {
        return {T(1),    T(1)/T(2), T(1)/T(3), T(1)/T(4),
                T(1)/T(2), T(1)/T(3), T(1)/T(4), T(1)/T(5),
                T(1)/T(3), T(1)/T(4), T(1)/T(5), T(1)/T(6),
                T(1)/T(4), T(1)/T(5), T(1)/T(6), T(1)/T(7)};
    }
    
    static std::vector<T> Get4x4Tridiagonal() {
        return {T(2), T(-1), T(0), T(0),
                T(-1), T(2), T(-1), T(0),
                T(0), T(-1), T(2), T(-1),
                T(0), T(0), T(-1), T(2)};
    }
    
    // Special matrices
    static std::vector<T> GetPascalTriangle(std::size_t n) {
        std::vector<T> matrix(n * n, T(0));
        
        for (std::size_t i = 0; i < n; ++i) {
            matrix[i * n] = T(1);  // First column
            matrix[i] = T(1);      // First row
        }
        
        for (std::size_t i = 1; i < n; ++i) {
            for (std::size_t j = 1; j < n; ++j) {
                matrix[i * n + j] = matrix[(i-1) * n + j] + matrix[i * n + (j-1)];
            }
        }
        
        return matrix;
    }
    
    static std::vector<T> GetVandermonde(const std::vector<T>& x) {
        std::size_t n = x.size();
        std::vector<T> matrix(n * n);
        
        for (std::size_t i = 0; i < n; ++i) {
            T xi = x[i];
            T power = T(1);
            for (std::size_t j = 0; j < n; ++j) {
                matrix[i * n + j] = power;
                power *= xi;
            }
        }
        
        return matrix;
    }
    
    // Packed storage for triangular matrices
    static std::vector<T> GetPackedLowerTriangular3x3() {
        // Elements: [1, 2, 3, 4, 5, 6] for:
        // [1, 0, 0]
        // [2, 3, 0]
        // [4, 5, 6]
        return {T(1), T(2), T(3), T(4), T(5), T(6)};
    }
    
    static std::vector<T> GetPackedIdentity(std::size_t n) {
        std::size_t packed_size = n * (n + 1) / 2;
        std::vector<T> packed(packed_size, T(0));
        
        // Set diagonal elements to 1
        std::size_t idx = 0;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j <= i; ++j) {
                if (i == j) {
                    packed[idx] = T(1);
                }
                idx++;
            }
        }
        
        return packed;
    }
};

// Expected results for common operations
template<typename T>
class ExpectedResults {
public:
    // 2x2 matrix operations
    static std::vector<T> Get2x2InvertibleInverse() {
        // Inverse of [[4, 7], [2, 6]]
        T det = T(4*6 - 7*2);  // = 10
        return {T(6)/det, T(-7)/det,
                T(-2)/det, T(4)/det};
    }
    
    static T Get2x2InvertibleDeterminant() {
        return T(10);  // det([[4, 7], [2, 6]])
    }
    
    // 3x3 matrix operations
    static std::vector<T> Get3x3SymmetricEigenvalues() {
        // Eigenvalues of [[2, 1, 0], [1, 2, 1], [0, 1, 2]]
        // Computed analytically: 2, 2-sqrt(2), 2+sqrt(2)
        T sqrt2 = std::sqrt(T(2));
        return {T(2) - sqrt2, T(2), T(2) + sqrt2};
    }
    
    static T Get3x3LowerTriangularDeterminant() {
        return T(1 * 3 * 6);  // Product of diagonal elements
    }
    
    // Cholesky decomposition results
    static std::vector<T> Get3x3PositiveDefiniteCholesky() {
        // Cholesky of [[4, 1, 1], [1, 4, 1], [1, 1, 4]]
        return {T(2), T(0), T(0),
                T(0.5), T(1.936492), T(0),  // sqrt(15/4) ≈ 1.936492
                T(0.5), T(0.387298), T(1.932184)};  // Values computed analytically
    }
};

// Test case scenarios
template<typename T>
class TestScenarios {
public:
    struct MatrixPair {
        std::vector<T> A;
        std::vector<T> B;
        std::size_t rows_A;
        std::size_t cols_A;
        std::size_t rows_B;
        std::size_t cols_B;
    };
    
    struct MatrixOperation {
        std::vector<T> input;
        std::vector<T> expected_output;
        std::size_t rows;
        std::size_t cols;
        std::string operation_name;
    };
    
    // Matrix multiplication test cases
    static MatrixPair GetMultiplicationCase1() {
        return {
            {T(1), T(2), T(3), T(4)},  // 2x2
            {T(5), T(6), T(7), T(8)},  // 2x2
            2, 2, 2, 2
        };
    }
    
    static std::vector<T> GetMultiplicationCase1Result() {
        // [1, 2] * [5, 6] = [19, 22]
        // [3, 4]   [7, 8]   [43, 50]
        return {T(19), T(22), T(43), T(50)};
    }
    
    // System solving test cases
    static MatrixOperation GetLinearSystemCase1() {
        return {
            {T(2), T(1), T(5),   // Ax = b where A = [[2, 1], [1, 2]], b = [5, 4]
             T(1), T(2), T(4)},
            {T(2), T(1)},        // Solution x = [2, 1]
            2, 3,
            "2x2 linear system"
        };
    }
    
    // Edge cases
    static std::vector<MatrixOperation> GetEdgeCases() {
        return {
            // Empty matrix
            {{}, {}, 0, 0, "empty matrix"},
            
            // Single element
            {{T(42)}, {T(42)}, 1, 1, "single element"},
            
            // Zero matrix
            {{T(0), T(0), T(0), T(0)}, {T(0), T(0), T(0), T(0)}, 2, 2, "zero matrix"},
            
            // Large values
            {{T(1e10), T(1e10), T(1e10), T(1e10)}, 
             {T(1e10), T(1e10), T(1e10), T(1e10)}, 2, 2, "large values"},
            
            // Small values
            {{T(1e-10), T(1e-10), T(1e-10), T(1e-10)}, 
             {T(1e-10), T(1e-10), T(1e-10), T(1e-10)}, 2, 2, "small values"}
        };
    }
};

} // namespace tri::test