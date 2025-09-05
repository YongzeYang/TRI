#pragma once

/**
 * @file macros.hpp
 * @brief Common macros and compile-time configurations
 * @author Yongze
 * @date 2025-08-13
 */

// Debug mode macro
#ifdef DEBUG
    #define TRI_DEBUG 1
#endif

// BLAS/LAPACK availability
#ifdef USE_BLAS
    #define TRI_USE_BLAS 1
#endif

// Inline force macro
#ifdef _MSC_VER
    #define TRI_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
    #define TRI_FORCE_INLINE __attribute__((always_inline)) inline
#else
    #define TRI_FORCE_INLINE inline
#endif

// Export/Import macros for shared library
#ifdef _WIN32
    #ifdef TRI_EXPORTS
        #define TRI_API __declspec(dllexport)
    #else
        #define TRI_API __declspec(dllimport)
    #endif
#else
    #define TRI_API
#endif