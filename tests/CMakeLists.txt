# Test configuration
enable_testing()

# Collect all test source files
set(TEST_SOURCES
    main.cpp
    # Core tests
    unit/core/test_dense_rm.cpp
    unit/core/test_lower_tri_rm.cpp
    unit/core/test_blocked_tri.cpp
    unit/core/test_blocked_tri_with_manager.cpp
    # Factory tests
    unit/factory/test_dense_factory.cpp
    unit/factory/test_tri_factory.cpp
    unit/factory/test_blocked_factory.cpp
    # Memory management tests
    unit/mem/test_block_key.cpp
    unit/mem/test_eviction_policy.cpp
    unit/mem/test_block_manager.cpp
    # Integration tests
    integration/test_mem_integration.cpp
    integration/test_matrix_operations.cpp
    integration/test_comprehensive_api.cpp
)

# Create main test executable
add_executable(tri_tests ${TEST_SOURCES})

# Include directories
target_include_directories(tri_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/framework  # For test framework
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures   # For test fixtures
    ${CMAKE_CURRENT_SOURCE_DIR}            # For other test includes
    ${CMAKE_SOURCE_DIR}/include             # For tri library headers
)

# Check if tri_static exists, otherwise use header-only tri
if(TARGET tri_static)
    target_link_libraries(tri_tests PRIVATE tri_static)
    message(STATUS "Linking tests with tri_static library")
else()
    target_link_libraries(tri_tests PRIVATE tri)
    message(STATUS "Linking tests with tri interface library")
endif()

# Link with external libraries if needed
if(TRI_EXTERNAL_LIBS)
    target_link_libraries(tri_tests PRIVATE ${TRI_EXTERNAL_LIBS})
endif()

# Set C++ standard
target_compile_features(tri_tests PRIVATE cxx_std_17)

# Add compiler warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(tri_tests PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        -Wno-unused-variable
    )
elseif(MSVC)
    target_compile_options(tri_tests PRIVATE /W4)
endif()

# Copy test executable to convenient location
set_target_properties(tri_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Register tests with CTest
add_test(NAME TriMatrixTests 
    COMMAND tri_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

add_test(NAME TriMatrixTests_Verbose 
    COMMAND tri_tests --verbose
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Add custom test targets
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS tri_tests
    COMMENT "Running all tests"
)

add_custom_target(test_quick
    COMMAND tri_tests --filter "DefaultConstruction"
    DEPENDS tri_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    COMMENT "Running quick construction tests"
)

add_custom_target(test_performance
    COMMAND tri_tests --filter "Performance"
    DEPENDS tri_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    COMMENT "Running performance tests"
)

# Print test configuration
message(STATUS "Tests configured:")
message(STATUS "  Test executable: tri_tests")
message(STATUS "  Test framework: Built-in lightweight framework")
message(STATUS "  Output directory: ${CMAKE_BINARY_DIR}/bin")
message(STATUS "  Test files:")
foreach(source ${TEST_SOURCES})
    message(STATUS "    - ${source}")
endforeach()