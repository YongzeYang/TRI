/**
 * @file test_assertions.hpp
 * @brief Test assertion macros and utilities
 * @author Yongze
 * @date 2025-08-14
 */

#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cmath>
#include <type_traits>

namespace tri::test {

// Custom exception for test failures
class TestFailure : public std::runtime_error {
public:
    TestFailure(const std::string& message, const std::string& file, int line)
        : std::runtime_error(format_message(message, file, line)) {}
        
private:
    static std::string format_message(const std::string& msg, const std::string& file, int line) {
        std::ostringstream oss;
        oss << file << ":" << line << " - " << msg;
        return oss.str();
    }
};

// Helper function for floating point comparison
template<typename T>
inline bool almost_equal(T a, T b, T tolerance = std::numeric_limits<T>::epsilon() * 100) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::abs(a - b) <= tolerance;
    } else {
        return a == b;
    }
}

// Assertion macros
#define ASSERT_TRUE(condition) \
    do { \
        if (!(condition)) { \
            std::ostringstream oss; \
            oss << "Assertion failed: " #condition " is false"; \
            throw tri::test::TestFailure(oss.str(), __FILE__, __LINE__); \
        } \
    } while(0)

#define ASSERT_FALSE(condition) \
    do { \
        if (condition) { \
            std::ostringstream oss; \
            oss << "Assertion failed: " #condition " is true"; \
            throw tri::test::TestFailure(oss.str(), __FILE__, __LINE__); \
        } \
    } while(0)

#define ASSERT_EQ(expected, actual) \
    do { \
        if (!((expected) == (actual))) { \
            std::ostringstream oss; \
            oss << "Assertion failed: expected " << (expected) << " but got " << (actual); \
            throw tri::test::TestFailure(oss.str(), __FILE__, __LINE__); \
        } \
    } while(0)

#define ASSERT_NE(expected, actual) \
    do { \
        if ((expected) == (actual)) { \
            std::ostringstream oss; \
            oss << "Assertion failed: expected != " << (expected) << " but got " << (actual); \
            throw tri::test::TestFailure(oss.str(), __FILE__, __LINE__); \
        } \
    } while(0)

#define ASSERT_LT(val1, val2) \
    do { \
        if (!((val1) < (val2))) { \
            std::ostringstream oss; \
            oss << "Assertion failed: expected " << (val1) << " < " << (val2); \
            throw tri::test::TestFailure(oss.str(), __FILE__, __LINE__); \
        } \
    } while(0)

#define ASSERT_LE(val1, val2) \
    do { \
        if (!((val1) <= (val2))) { \
            std::ostringstream oss; \
            oss << "Assertion failed: expected " << (val1) << " <= " << (val2); \
            throw tri::test::TestFailure(oss.str(), __FILE__, __LINE__); \
        } \
    } while(0)

#define ASSERT_GT(val1, val2) \
    do { \
        if (!((val1) > (val2))) { \
            std::ostringstream oss; \
            oss << "Assertion failed: expected " << (val1) << " > " << (val2); \
            throw tri::test::TestFailure(oss.str(), __FILE__, __LINE__); \
        } \
    } while(0)

#define ASSERT_GE(val1, val2) \
    do { \
        if (!((val1) >= (val2))) { \
            std::ostringstream oss; \
            oss << "Assertion failed: expected " << (val1) << " >= " << (val2); \
            throw tri::test::TestFailure(oss.str(), __FILE__, __LINE__); \
        } \
    } while(0)

#define ASSERT_NEAR(expected, actual, tolerance) \
    do { \
        auto diff = std::abs((expected) - (actual)); \
        if (diff > (tolerance)) { \
            std::ostringstream oss; \
            oss << "Assertion failed: expected " << (expected) << " ± " << (tolerance); \
            oss << " but got " << (actual) << " (diff: " << diff << ")"; \
            throw tri::test::TestFailure(oss.str(), __FILE__, __LINE__); \
        } \
    } while(0)

#define ASSERT_THROW(expression, exception_type) \
    do { \
        bool caught_expected = false; \
        try { \
            expression; \
        } catch (const exception_type&) { \
            caught_expected = true; \
        } catch (...) { \
            throw tri::test::TestFailure("Caught unexpected exception type", __FILE__, __LINE__); \
        } \
        if (!caught_expected) { \
            throw tri::test::TestFailure("Expected exception " #exception_type " was not thrown", __FILE__, __LINE__); \
        } \
    } while(0)

#define ASSERT_NO_THROW(expression) \
    do { \
        try { \
            expression; \
        } catch (const std::exception& e) { \
            std::ostringstream oss; \
            oss << "Unexpected exception: " << e.what(); \
            throw tri::test::TestFailure(oss.str(), __FILE__, __LINE__); \
        } catch (...) { \
            throw tri::test::TestFailure("Unexpected unknown exception", __FILE__, __LINE__); \
        } \
    } while(0)

#define ASSERT_NOT_NULL(pointer) \
    do { \
        if ((pointer) == nullptr) { \
            throw tri::test::TestFailure("Assertion failed: pointer is null", __FILE__, __LINE__); \
        } \
    } while(0)

#define ASSERT_NULL(pointer) \
    do { \
        if ((pointer) != nullptr) { \
            throw tri::test::TestFailure("Assertion failed: pointer is not null", __FILE__, __LINE__); \
        } \
    } while(0)

} // namespace tri::test