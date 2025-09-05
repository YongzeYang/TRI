/**
 * @file main.cpp
 * @brief Main test runner for the Tri Matrix Library test suite
 * @author Yongze
 * @date 2025-08-14
 */

#include "framework/test_runner.hpp"
#include <iostream>
#include <string>
#include <vector>

using namespace tri::test;

// Print usage information
void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "\nOptions:\n";
    std::cout << "  --help, -h           Show this help message\n";
    std::cout << "  --verbose, -v        Show detailed test output\n";
    std::cout << "  --filter <pattern>   Run only tests matching pattern\n";
    std::cout << "  --no-color           Disable colored output\n";
    std::cout << "  --stop-on-failure    Stop execution on first failure\n";
    std::cout << "  --no-timing          Don't show timing information\n";
    std::cout << "\nExamples:\n";
    std::cout << "  " << program_name << " --verbose\n";
    std::cout << "  " << program_name << " --filter Dense\n";
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    TestRunner& runner = TestRunner::Instance();
    
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "--verbose" || arg == "-v") {
            runner.SetVerbose(true);
        }
        else if (arg == "--no-color") {
            runner.SetUseColors(false);
        }
        else if (arg == "--no-timing") {
            runner.SetShowTiming(false);
        }
        else if (arg == "--stop-on-failure") {
            runner.SetStopOnFailure(true);
        }
        else if (arg == "--filter" && i + 1 < argc) {
            runner.SetFilter(argv[++i]);
        }
        else {
            std::cerr << "Unknown argument: " << arg << "\n";
            std::cerr << "Use --help for usage information\n";
            return 1;
        }
    }
    
    // Run tests
    runner.RunAll();
    
    // Return exit code based on test results
    return runner.AllPassed() ? 0 : 1;
}