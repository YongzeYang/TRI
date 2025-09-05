/**
 * @file test_dense_rm.cpp
 * @brief Unit tests for DenseRM matrix class
 * @author Yongze
 * @date 2025-08-14
 */

#include "test_runner.hpp"
#include "test_assertions.hpp"
#include "test_utils.hpp"
#include "tri/core/dense_rm.hpp"
#include <iostream>

namespace tri::test {

// Test fixture for double precision tests
class DenseRMTestDouble : public TestCase {
protected:
    double tolerance_ = 1e-10;
    
    void SetUp() override {
        // Any setup code here
    }
    
    void TearDown() override {
        // Any cleanup code here
    }
};

// Test fixture for float precision tests
class DenseRMTestFloat : public TestCase {
protected:
    float tolerance_ = 1e-5f;
};

// Tests using double precision fixture
TEST_F(DenseRMTestDouble, DefaultConstruction) {
    core::DenseRM<double> m;
    ASSERT_EQ(0u, m.rows());
    ASSERT_EQ(0u, m.cols());
    ASSERT_TRUE(m.empty());
    ASSERT_EQ(0u, m.size());
}

TEST_F(DenseRMTestDouble, SizeConstruction) {
    core::DenseRM<double> m(3, 4);
    ASSERT_EQ(3u, m.rows());
    ASSERT_EQ(4u, m.cols());
    ASSERT_EQ(12u, m.size());
    ASSERT_FALSE(m.empty());
    ASSERT_FALSE(m.is_square());
}

TEST_F(DenseRMTestDouble, ValueConstruction) {
    const double value = 3.14;
    core::DenseRM<double> m(4, 4, value);
    
    ValidateSquareMatrix(m, 4u);
    
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            ASSERT_NEAR(value, m(i, j), tolerance_);
        }
    }
}

TEST_F(DenseRMTestDouble, RowAccess) {
    core::DenseRM<double> m(3, 3);
    
    // Fill matrix
    for (std::size_t i = 0; i < 3; ++i) {
        double* row = m.row(i);
        for (std::size_t j = 0; j < 3; ++j) {
            row[j] = static_cast<double>(i * 3 + j);
        }
    }
    
    // Verify through normal access
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            ASSERT_EQ(static_cast<double>(i * 3 + j), m(i, j));
        }
    }
}

TEST_F(DenseRMTestDouble, DataPointerAccess) {
    core::DenseRM<double> m(2, 3);
    
    double* data = m.data();
    ASSERT_NOT_NULL(data);
    
    // Fill through data pointer
    for (std::size_t i = 0; i < 6; ++i) {
        data[i] = static_cast<double>(i);
    }
    
    // Verify through element access
    ASSERT_EQ(0.0, m(0, 0));
    ASSERT_EQ(1.0, m(0, 1));
    ASSERT_EQ(2.0, m(0, 2));
    ASSERT_EQ(3.0, m(1, 0));
    ASSERT_EQ(4.0, m(1, 1));
    ASSERT_EQ(5.0, m(1, 2));
}

TEST_F(DenseRMTestDouble, Fill) {
    core::DenseRM<double> m(3, 3);
    const double fill_value = 7.5;
    
    m.fill(fill_value);
    
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            ASSERT_EQ(fill_value, m(i, j));
        }
    }
}

TEST_F(DenseRMTestDouble, SetDiagonal) {
    core::DenseRM<double> m(4, 4, 0.0);
    const double diag_value = 3.0;
    
    m.set_diagonal(diag_value);
    
    for (std::size_t i = 0; i < 4; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            if (i == j) {
                ASSERT_EQ(diag_value, m(i, j));
            } else {
                ASSERT_EQ(0.0, m(i, j));
            }
        }
    }
}

TEST_F(DenseRMTestDouble, Resize) {
    core::DenseRM<double> m(3, 3, 1.0);
    
    m.resize(2, 4);
    ASSERT_EQ(2u, m.rows());
    ASSERT_EQ(4u, m.cols());
    
    // After resize, elements should be zero-initialized
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            ASSERT_EQ(0.0, m(i, j));
        }
    }
}

TEST_F(DenseRMTestDouble, Clear) {
    core::DenseRM<double> m(5, 5);
    ASSERT_FALSE(m.empty());
    
    m.clear();
    ASSERT_TRUE(m.empty());
    ASSERT_EQ(0u, m.rows());
    ASSERT_EQ(0u, m.cols());
    ASSERT_EQ(0u, m.size());
}

TEST_F(DenseRMTestDouble, CopyConstruction) {
    core::DenseRM<double> m1(2, 3);
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            m1(i, j) = static_cast<double>(i * 3 + j);
        }
    }
    
    core::DenseRM<double> m2(m1);
    
    ValidateMatricesEqual(m1, m2, tolerance_);
    
    // Ensure deep copy
    m2(0, 0) = 999.0;
    ASSERT_NE(m1(0, 0), m2(0, 0));
}

TEST_F(DenseRMTestDouble, MoveConstruction) {
    core::DenseRM<double> m1(4, 4, 3.14);
    auto original_data = m1.data();
    
    core::DenseRM<double> m2(std::move(m1));
    
    ASSERT_EQ(4u, m2.rows());
    ASSERT_EQ(4u, m2.cols());
    ASSERT_EQ(3.14, m2(0, 0));
    ASSERT_EQ(original_data, m2.data());  // Should have taken ownership
}

TEST_F(DenseRMTestDouble, Swap) {
    core::DenseRM<double> m1(2, 3, 1.0);
    core::DenseRM<double> m2(4, 5, 2.0);
    
    auto rows1 = m1.rows();
    auto cols1 = m1.cols();
    auto val1 = m1(0, 0);
    
    auto rows2 = m2.rows();
    auto cols2 = m2.cols();
    auto val2 = m2(0, 0);
    
    m1.swap(m2);
    
    ASSERT_EQ(rows2, m1.rows());
    ASSERT_EQ(cols2, m1.cols());
    ASSERT_EQ(val2, m1(0, 0));
    
    ASSERT_EQ(rows1, m2.rows());
    ASSERT_EQ(cols1, m2.cols());
    ASSERT_EQ(val1, m2(0, 0));
}

// Tests using float precision fixture
TEST_F(DenseRMTestFloat, ElementAccess) {
    core::DenseRM<float> m(3, 4);
    
    // Set elements
    float value = 0.0f;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            m(i, j) = value++;
        }
    }
    
    // Verify elements
    value = 0.0f;
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 4; ++j) {
            ASSERT_EQ(value++, m(i, j));
        }
    }
}

TEST_F(DenseRMTestFloat, CopyAssignment) {
    core::DenseRM<float> m1(3, 2, 5.0f);
    core::DenseRM<float> m2;
    
    m2 = m1;
    
    ValidateMatricesEqual(m1, m2, tolerance_);
    
    // Ensure deep copy
    m2(0, 0) = 999.0f;
    ASSERT_NE(m1(0, 0), m2(0, 0));
}

// Non-fixture tests using TEST macro
TEST(DenseRM, InitializerListConstruction) {
    core::DenseRM<double> m{
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };
    
    ASSERT_EQ(2u, m.rows());
    ASSERT_EQ(3u, m.cols());
    ASSERT_EQ(1.0, m(0, 0));
    ASSERT_EQ(2.0, m(0, 1));
    ASSERT_EQ(3.0, m(0, 2));
    ASSERT_EQ(4.0, m(1, 0));
    ASSERT_EQ(5.0, m(1, 1));
    ASSERT_EQ(6.0, m(1, 2));
}

TEST(DenseRM, VectorDataConstruction) {
    std::vector<double> data = TestDataGenerator<double>::IdentityData(3);
    core::DenseRM<double> m(3, 3, data);
    
    ValidateIdentityMatrix(m, 1e-10);
}

TEST(DenseRM, StaticFactories) {
    // Test zeros
    auto z = core::DenseRM<double>::zeros(3, 4);
    ASSERT_EQ(3u, z.rows());
    ASSERT_EQ(4u, z.cols());
    ValidateZeroMatrix(z, 1e-10);
    
    // Test identity
    auto id = core::DenseRM<double>::identity(4, 4);
    ValidateSquareMatrix(id, 4u);
    ValidateIdentityMatrix(id, 1e-10);
    
    // Test eye
    auto eye = core::DenseRM<double>::eye(3);
    ValidateSquareMatrix(eye, 3u);
    ValidateIdentityMatrix(eye, 1e-10);
}

TEST(DenseRM, MatrixProperties) {
    core::DenseRM<double> m1(4, 4);
    ASSERT_TRUE(m1.is_square());
    ASSERT_EQ(4u, m1.ld());
    
    core::DenseRM<double> m2(3, 5);
    ASSERT_FALSE(m2.is_square());
    ASSERT_EQ(5u, m2.ld());
    
    core::DenseRM<double> m3;
    ASSERT_TRUE(m3.empty());
    ASSERT_TRUE(m3.is_square());  // 0x0 is considered square
}

// Performance test
class DenseRMPerformanceTest : public TestCase {
public:
    std::string GetName() const override { return "Performance"; }
    std::string GetSuite() const override { return "DenseRM"; }
    
    void Run() override {
        TestConstructionPerformance();
        TestAccessPerformance();
        TestFillPerformance();
    }
    
private:
    void TestConstructionPerformance() {
        const std::size_t size = 1000;
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            core::DenseRM<double> m(size, size);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "  Construction (1000x1000): " << duration / 100 << " ms\n";
    }
    
    void TestAccessPerformance() {
        const std::size_t size = 500;
        core::DenseRM<double> m(size, size);
        
        auto start = std::chrono::high_resolution_clock::now();
        double sum = 0;
        for (int iter = 0; iter < 10; ++iter) {
            for (std::size_t i = 0; i < size; ++i) {
                for (std::size_t j = 0; j < size; ++j) {
                    sum += m(i, j);
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "  Element access (500x500): " << duration / 10 << " ms\n";
        
        // Use sum to prevent optimization
        volatile double dummy = sum;
        (void)dummy;
    }
    
    void TestFillPerformance() {
        const std::size_t size = 1000;
        core::DenseRM<double> m(size, size);
        
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i) {
            m.fill(3.14);
        }
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "  Fill operation (1000x1000): " << duration / 100 << " ms\n";
    }
};

// Register performance test
static TestRegistrar<DenseRMPerformanceTest> perf_test_reg("DenseRM", "Performance");

} // namespace tri::test