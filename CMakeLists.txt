# CMakeLists.txt for High-Performance Logo Analysis C++ Module
# Optimized for MacBook Pro 2024 (Apple Silicon M3 Pro/Max)

cmake_minimum_required(VERSION 3.16)
project(fourier_math_cpp LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect system and set optimizations
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
        message(STATUS "🚀 Detected Apple Silicon M3 - enabling advanced optimizations")
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mcpu=apple-m3 -mtune=apple-m3 -ffast-math -funroll-loops -fvectorize -fslp-vectorize")
        set(CMAKE_EXE_LINKER_FLAGS "-framework Accelerate")
    else()
        message(STATUS "⚡ Detected Intel Mac - enabling AVX2/FMA optimizations")
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -mavx2 -mfma -ffast-math -funroll-loops")
        set(CMAKE_EXE_LINKER_FLAGS "-framework Accelerate")
    endif()
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(STATUS "🐧 Detected Linux - enabling native optimizations")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native -mtune=native -ffast-math -funroll-loops")
endif()

# Set build type to Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find required packages
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Try to find pybind11
find_package(pybind11 QUIET)

if(NOT pybind11_FOUND)
    message(STATUS "pybind11 not found, attempting to use pip-installed version")
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    
    if(pybind11_DIR)
        find_package(pybind11 REQUIRED PATHS ${pybind11_DIR})
        message(STATUS "✅ Found pybind11 via pip installation")
    else()
        message(FATAL_ERROR "❌ pybind11 not found. Install with: pip install pybind11")
    endif()
endif()

# Create the Python module
pybind11_add_module(fourier_math_cpp
    python_bindings.cpp
    fourier_math.cpp
)

# Set properties for the module
set_target_properties(fourier_math_cpp PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN ON
)

# Add compiler-specific definitions
target_compile_definitions(fourier_math_cpp PRIVATE
    VERSION_INFO="${PROJECT_VERSION}"
    PYBIND11_DETAILED_ERROR_MESSAGES
)

# Enable parallel compilation
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_libraries(fourier_math_cpp PRIVATE "-framework Accelerate")
endif()

# Add threading support
find_package(Threads REQUIRED)
target_link_libraries(fourier_math_cpp PRIVATE Threads::Threads)

# Add OpenMP support if available (for parallel algorithms)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(fourier_math_cpp PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "✅ OpenMP found - enabling parallel algorithms")
else()
    message(STATUS "⚠️ OpenMP not found - some optimizations disabled")
endif()

# Print build configuration
message(STATUS "")
message(STATUS "🔧 BUILD CONFIGURATION:")
message(STATUS "   System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "   Compiler: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "   Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "   CXX Flags: ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "   Python: ${Python_VERSION}")
message(STATUS "   pybind11: ${pybind11_VERSION}")
message(STATUS "")

# Custom target for installation instructions
add_custom_target(install_instructions
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "📦 BUILD COMPLETE!"
    COMMAND ${CMAKE_COMMAND} -E echo "To install the module:"
    COMMAND ${CMAKE_COMMAND} -E echo "  cp fourier_math_cpp*.so /path/to/your/python/site-packages/"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Or use in place:"
    COMMAND ${CMAKE_COMMAND} -E echo "  export PYTHONPATH=$PYTHONPATH:${CMAKE_CURRENT_BINARY_DIR}"
    COMMAND ${CMAKE_COMMAND} -E echo ""
)

# Performance test target (optional)
add_custom_target(performance_test
    COMMAND ${Python_EXECUTABLE} -c "
import sys
import numpy as np
sys.path.insert(0, '${CMAKE_CURRENT_BINARY_DIR}')
try:
    import fourier_math_cpp
    print('✅ Module imported successfully')
    
    # Quick performance test
    test_images = [np.random.rand(128, 128) for _ in range(10)]
    result = fourier_math_cpp.benchmark_analysis(test_images, 5)
    print(f'⚡ Performance: {result[\"images_per_second\"]:.1f} images/second')
except Exception as e:
    print(f'❌ Test failed: {e}')
"
    DEPENDS fourier_math_cpp
)

# Development target for rebuilding
add_custom_target(rebuild
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target clean
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
)
