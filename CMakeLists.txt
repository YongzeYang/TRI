cmake_minimum_required(VERSION 3.14)
project(LowerTriangularMatrix 
    VERSION 1.0.0
    DESCRIPTION "TRI: a Tiled, performance-Resilient Infrastructure for large-scale matrix algebra"
    LANGUAGES CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compilation flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math -funroll-loops")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG -march=native")

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Options
option(TRI_USE_BLAS "Use BLAS/LAPACK for optimized operations" ON)
option(TRI_USE_OPENMP "Enable OpenMP parallelization" ON)
option(TRI_BUILD_TESTS "Build unit tests" ON)
option(TRI_BUILD_EXAMPLES "Build example programs" ON)
option(TRI_BUILD_BENCHMARKS "Build performance benchmarks" OFF)

# Find dependencies
set(TRI_EXTERNAL_LIBS "")

# Find BLAS/LAPACK
if(TRI_USE_BLAS)
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(OPENBLAS openblas QUIET)
    endif()
    
    if(OPENBLAS_FOUND)
        set(BLAS_LIBRARIES ${OPENBLAS_LIBRARIES})
        set(BLAS_INCLUDE_DIRS ${OPENBLAS_INCLUDE_DIRS})
        set(BLAS_FOUND TRUE)
        message(STATUS "Found OpenBLAS: ${OPENBLAS_LIBRARIES}")
        add_compile_definitions(TRI_USE_BLAS)
    else()
        find_package(BLAS QUIET)
        if(BLAS_FOUND)
            message(STATUS "Found BLAS: ${BLAS_LIBRARIES}")
            add_compile_definitions(TRI_USE_BLAS)
        endif()
    endif()
    
    find_library(LAPACKE_LIB NAMES lapacke)
    if(LAPACKE_LIB)
        set(LAPACK_LIBRARIES ${LAPACKE_LIB})
        set(LAPACK_FOUND TRUE)
        message(STATUS "Found LAPACKE: ${LAPACKE_LIB}")
    else()
        find_package(LAPACK QUIET)
        if(LAPACK_FOUND)
            message(STATUS "Found LAPACK: ${LAPACK_LIBRARIES}")
        endif()
    endif()
    
    if(BLAS_FOUND)
        list(APPEND TRI_EXTERNAL_LIBS ${BLAS_LIBRARIES})
    endif()
    if(LAPACK_FOUND)
        list(APPEND TRI_EXTERNAL_LIBS ${LAPACK_LIBRARIES})
    endif()
endif()

# Find OpenMP
if(TRI_USE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        add_compile_definitions(TRI_USE_OPENMP)
        list(APPEND TRI_EXTERNAL_LIBS OpenMP::OpenMP_CXX)
        message(STATUS "Found OpenMP: ${OpenMP_CXX_VERSION}")
    endif()
endif()

# Header-only library target
add_library(tri INTERFACE)
target_include_directories(tri INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(tri INTERFACE ${TRI_EXTERNAL_LIBS})
target_compile_features(tri INTERFACE cxx_std_17)

# Collect source files
set(CORE_SOURCES
    src/tri/core/dense_rm.cpp
    src/tri/core/lower_tri_rm.cpp
    src/tri/core/blocked_tri.cpp
)

set(FACTORY_SOURCES
    src/tri/factory/dense_factory.cpp
    src/tri/factory/tri_factory.cpp
    src/tri/factory/blocked_factory.cpp
)

set(BLAS_SOURCES
    src/tri/blas/gemm.cpp
    src/tri/blas/solver.cpp
    src/tri/blas/svd.cpp
)

set(MEM_SOURCES
    src/tri/mem/block_manager.cpp
    src/tri/mem/lru_policy.cpp
    src/tri/mem/access_count_policy.cpp
)

# Combine all sources
set(TRI_SOURCES ${CORE_SOURCES} ${FACTORY_SOURCES} ${MEM_SOURCES})
if(TRI_USE_BLAS)
    list(APPEND TRI_SOURCES ${BLAS_SOURCES})
endif()

# Create static library
if(TRI_SOURCES)
    add_library(tri_static STATIC ${TRI_SOURCES})
    target_include_directories(tri_static PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    )
    target_link_libraries(tri_static PUBLIC ${TRI_EXTERNAL_LIBS})
    set_target_properties(tri_static PROPERTIES OUTPUT_NAME tri)
    
    # Add compile definitions
    if(TRI_USE_BLAS AND BLAS_FOUND)
        target_compile_definitions(tri_static PUBLIC TRI_USE_BLAS)
    endif()
endif()

# Tests
if(TRI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Examples
if(TRI_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Benchmarks
if(TRI_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Installation
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Install headers
install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install targets
install(TARGETS tri
    EXPORT TRITargets
)

if(TARGET tri_static)
    install(TARGETS tri_static
        EXPORT TRITargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
endif()

# Export targets
install(EXPORT TRITargets
    FILE TRITargets.cmake
    NAMESPACE TRI::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TRI
)

# Create config file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/TRIConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/TRIConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TRI
)

# Create version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/TRIConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Install config files
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/TRIConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/TRIConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TRI
)

# Uninstall target
if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY
    )
    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake
    )
endif()

# Configuration summary
message(STATUS "")
message(STATUS "LowerTriangularMatrix Configuration Summary")
message(STATUS "==========================================")
message(STATUS "Version:          ${PROJECT_VERSION}")
message(STATUS "Build type:       ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler:     ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ standard:     ${CMAKE_CXX_STANDARD}")
message(STATUS "Install prefix:   ${CMAKE_INSTALL_PREFIX}")
message(STATUS "")
message(STATUS "Options:")
message(STATUS "  BLAS/LAPACK:    ${TRI_USE_BLAS} (found: ${BLAS_FOUND}/${LAPACK_FOUND})")
message(STATUS "  OpenMP:         ${TRI_USE_OPENMP} (found: ${OpenMP_CXX_FOUND})")
message(STATUS "  Build tests:    ${TRI_BUILD_TESTS}")
message(STATUS "  Build examples: ${TRI_BUILD_EXAMPLES}")
message(STATUS "  Build benchmarks: ${TRI_BUILD_BENCHMARKS}")
message(STATUS "==========================================")
message(STATUS "")