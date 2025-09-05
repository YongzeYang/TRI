#pragma once

/**
 * @file config.hpp
 * @brief Global configuration and version information
 * @author Yongze
 * @date 2025-08-13
 */

#include <cstddef>

namespace tri {
namespace config {

// Version information
constexpr int VERSION_MAJOR = 1;
constexpr int VERSION_MINOR = 0;
constexpr int VERSION_PATCH = 0;

// Default block size for blocked matrices (future use)
constexpr std::size_t DEFAULT_BLOCK_SIZE = 64;

// Memory alignment
constexpr std::size_t MEMORY_ALIGNMENT = 64;  // Cache line size

} // namespace config
} // namespace tri