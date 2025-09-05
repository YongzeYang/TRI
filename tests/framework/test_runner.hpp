/**
 * @file test_runner.hpp
 * @brief Test runner and test case base classes
 * @author Yongze
 * @date 2025-08-14
 */

#pragma once

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <map>
#include <algorithm>

namespace tri::test {

// Terminal color codes
struct Colors {
    static constexpr const char* RESET = "\033[0m";
    static constexpr const char* RED = "\033[31m";
    static constexpr const char* GREEN = "\033[32m";
    static constexpr const char* YELLOW = "\033[33m";
    static constexpr const char* BLUE = "\033[34m";
    static constexpr const char* MAGENTA = "\033[35m";
    static constexpr const char* CYAN = "\033[36m";
    static constexpr const char* BOLD = "\033[1m";
};

// Test result structure
struct TestResult {
    std::string suite_name;
    std::string test_name;
    bool passed;
    std::string failure_message;
    double execution_time_ms;
};

// Base class for all test cases
class TestCase {
public:
    TestCase() = default;
    virtual ~TestCase() = default;
    
    // Setup and teardown hooks
    virtual void SetUp() {}
    virtual void TearDown() {}
    
    // Main test execution
    virtual void Run() = 0;
    
    // Test metadata
    virtual std::string GetName() const = 0;
    virtual std::string GetSuite() const = 0;
    
    // Full test name
    std::string GetFullName() const {
        return GetSuite() + "." + GetName();
    }
};

// Test registry entry
struct TestRegistryEntry {
    std::function<std::unique_ptr<TestCase>()> factory;
    std::string suite_name;
    std::string test_name;
};

// Main test runner class
class TestRunner {
private:
    TestRunner() = default;
    
    std::vector<TestRegistryEntry> registry_;
    std::vector<TestResult> results_;
    
    // Configuration
    bool verbose_ = false;
    bool stop_on_failure_ = false;
    bool show_timing_ = true;
    bool use_colors_ = true;
    std::string filter_pattern_;
    
    // Statistics
    int total_tests_ = 0;
    int passed_tests_ = 0;
    int failed_tests_ = 0;
    double total_time_ms_ = 0.0;

public:
    // Singleton instance
    static TestRunner& Instance() {
        static TestRunner instance;
        return instance;
    }
    
    // Delete copy and move
    TestRunner(const TestRunner&) = delete;
    TestRunner& operator=(const TestRunner&) = delete;
    TestRunner(TestRunner&&) = delete;
    TestRunner& operator=(TestRunner&&) = delete;
    
    // Test registration
    void RegisterTest(const std::string& suite_name,
                     const std::string& test_name,
                     std::function<std::unique_ptr<TestCase>()> factory) {
        registry_.push_back({factory, suite_name, test_name});
    }
    
    // Configuration methods
    void SetVerbose(bool verbose) { verbose_ = verbose; }
    void SetStopOnFailure(bool stop) { stop_on_failure_ = stop; }
    void SetShowTiming(bool show) { show_timing_ = show; }
    void SetUseColors(bool use) { use_colors_ = use; }
    void SetFilter(const std::string& pattern) { filter_pattern_ = pattern; }
    
    // Run all tests
    void RunAll() {
        PrintHeader();
        
        // Group tests by suite
        std::map<std::string, std::vector<TestRegistryEntry>> suites;
        for (const auto& entry : registry_) {
            if (ShouldRunTest(entry.suite_name, entry.test_name)) {
                suites[entry.suite_name].push_back(entry);
            }
        }
        
        // Run tests suite by suite
        for (const auto& [suite_name, tests] : suites) {
            RunSuite(suite_name, tests);
            if (stop_on_failure_ && failed_tests_ > 0) {
                break;
            }
        }
        
        PrintSummary();
    }
    
    // Check if all tests passed
    bool AllPassed() const { return failed_tests_ == 0; }

private:
    // Check if test should run based on filter
    bool ShouldRunTest(const std::string& suite_name, const std::string& test_name) {
        if (filter_pattern_.empty()) {
            return true;
        }
        
        std::string full_name = suite_name + "." + test_name;
        return full_name.find(filter_pattern_) != std::string::npos;
    }
    
    // Run a suite of tests
    void RunSuite(const std::string& suite_name, const std::vector<TestRegistryEntry>& tests) {
        PrintSuiteHeader(suite_name);
        
        for (const auto& entry : tests) {
            RunSingleTest(entry);
            
            if (stop_on_failure_ && failed_tests_ > 0) {
                break;
            }
        }
    }
    
    // Run a single test
    void RunSingleTest(const TestRegistryEntry& entry) {
        total_tests_++;
        
        TestResult result;
        result.suite_name = entry.suite_name;
        result.test_name = entry.test_name;
        
        if (verbose_) {
            std::cout << "[ RUN      ] " << entry.suite_name << "." << entry.test_name << "\n";
        } else {
            std::cout << "  " << entry.test_name << " ... ";
            std::cout.flush();
        }
        
        auto start = std::chrono::high_resolution_clock::now();
        
        try {
            auto test = entry.factory();
            test->SetUp();
            test->Run();
            test->TearDown();
            
            auto end = std::chrono::high_resolution_clock::now();
            result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            total_time_ms_ += result.execution_time_ms;
            
            result.passed = true;
            passed_tests_++;
            PrintTestSuccess(entry, result);
            
        } catch (const std::exception& e) {
            auto end = std::chrono::high_resolution_clock::now();
            result.execution_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            total_time_ms_ += result.execution_time_ms;
            
            result.passed = false;
            result.failure_message = e.what();
            failed_tests_++;
            
            PrintTestFailure(entry, result);
        }
        
        results_.push_back(result);
    }
    
    // Print methods
    void PrintHeader() {
        std::cout << "\n";
        std::cout << Color(Colors::CYAN) << "========================================\n";
        std::cout << "    Tri Matrix Library Test Suite\n";
        std::cout << "========================================" << Color(Colors::RESET) << "\n\n";
    }
    
    void PrintSuiteHeader(const std::string& suite_name) {
        std::cout << "\n" << Color(Colors::BLUE) << Color(Colors::BOLD);
        std::cout << "[----------] " << suite_name << Color(Colors::RESET) << "\n";
    }
    
    void PrintTestSuccess(const TestRegistryEntry& entry, const TestResult& result) {
        if (verbose_) {
            std::cout << Color(Colors::GREEN) << "[       OK ] " << Color(Colors::RESET);
            std::cout << entry.suite_name << "." << entry.test_name;
            if (show_timing_) {
                std::cout << " (" << std::fixed << std::setprecision(2) << result.execution_time_ms << " ms)";
            }
            std::cout << "\n";
        } else {
            std::cout << Color(Colors::GREEN) << "PASS" << Color(Colors::RESET);
            if (show_timing_) {
                std::cout << " (" << std::fixed << std::setprecision(2) << result.execution_time_ms << " ms)";
            }
            std::cout << "\n";
        }
    }
    
    void PrintTestFailure(const TestRegistryEntry& entry, const TestResult& result) {
        if (verbose_) {
            std::cout << Color(Colors::RED) << "[  FAILED  ] " << Color(Colors::RESET);
            std::cout << entry.suite_name << "." << entry.test_name;
            if (show_timing_) {
                std::cout << " (" << std::fixed << std::setprecision(2) << result.execution_time_ms << " ms)";
            }
            std::cout << "\n";
            
            if (!result.failure_message.empty()) {
                std::cout << Color(Colors::RED) << "  Error: " << result.failure_message << Color(Colors::RESET) << "\n";
            }
        } else {
            std::cout << Color(Colors::RED) << "FAIL" << Color(Colors::RESET) << "\n";
            if (!result.failure_message.empty()) {
                std::cout << "    Error: " << result.failure_message << "\n";
            }
        }
    }
    
    void PrintSummary() {
        std::cout << "\n" << Color(Colors::CYAN) << Color(Colors::BOLD);
        std::cout << "[==========] Test Summary" << Color(Colors::RESET) << "\n";
        
        std::cout << "[==========] " << total_tests_ << " test(s) ran";
        if (show_timing_) {
            std::cout << " (" << std::fixed << std::setprecision(2) << total_time_ms_ << " ms total)";
        }
        std::cout << "\n";
        
        std::cout << Color(Colors::GREEN) << "[  PASSED  ] " << passed_tests_ << " test(s)" << Color(Colors::RESET) << "\n";
        
        if (failed_tests_ > 0) {
            std::cout << Color(Colors::RED) << "[  FAILED  ] " << failed_tests_ << " test(s)" << Color(Colors::RESET) << "\n";
            
            std::cout << "\nFailed tests:\n";
            for (const auto& result : results_) {
                if (!result.passed) {
                    std::cout << "  " << Color(Colors::RED) << "✗" << Color(Colors::RESET);
                    std::cout << " " << result.suite_name << "." << result.test_name << "\n";
                }
            }
        }
        
        std::cout << "\n";
    }
    
    // Helper to apply color if enabled
    std::string Color(const char* color) const {
        return use_colors_ ? color : "";
    }
};

// Test registration helper
template<typename TestClass>
class TestRegistrar {
public:
    TestRegistrar(const std::string& suite_name, const std::string& test_name) {
        TestRunner::Instance().RegisterTest(
            suite_name,
            test_name,
            []() { return std::make_unique<TestClass>(); }
        );
    }
};

// Macros for test definition and registration
#define TEST_CLASS_NAME(suite_name, test_name) suite_name##_##test_name##_Test

#define TEST(suite_name, test_name) \
    class TEST_CLASS_NAME(suite_name, test_name) : public ::tri::test::TestCase { \
    public: \
        std::string GetName() const override { return #test_name; } \
        std::string GetSuite() const override { return #suite_name; } \
        void Run() override; \
    }; \
    static ::tri::test::TestRegistrar<TEST_CLASS_NAME(suite_name, test_name)> \
        suite_name##_##test_name##_Test_registrar(#suite_name, #test_name); \
    void TEST_CLASS_NAME(suite_name, test_name)::Run()

#define TEST_F(fixture_class, test_name) \
    class fixture_class##_##test_name##_Test : public fixture_class { \
    public: \
        std::string GetName() const override { return #test_name; } \
        std::string GetSuite() const override { return #fixture_class; } \
        void Run() override; \
    }; \
    static ::tri::test::TestRegistrar<fixture_class##_##test_name##_Test> \
        fixture_class##_##test_name##_Test##_registrar(#fixture_class, #test_name); \
    void fixture_class##_##test_name##_Test::Run()

} // namespace tri::test