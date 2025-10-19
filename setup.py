#!/usr/bin/env python3
"""
Setup script for building the high-performance C++ Fourier mathematics module
Optimized for MacBook Pro 2024 (Apple Silicon M3 Pro/Max)
"""

import os
import sys
import platform
from setuptools import setup, Extension
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    from pybind11 import get_cmake_dir
    import pybind11
    PYBIND11_AVAILABLE = True
except ImportError:
    # Fallback for older pybind11 versions
    from setuptools import Extension as Pybind11Extension
    from setuptools.command.build_ext import build_ext
    import pybind11
    PYBIND11_AVAILABLE = False

# Detect system and optimize compiler flags
def get_compiler_flags():
    """Get optimized compiler flags for the current system"""
    
    flags = [
        '-std=c++17',
        '-O3',
        '-DNDEBUG',
        '-ffast-math',
        '-funroll-loops'
    ]
    
    # Apple Silicon specific optimizations
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        flags.extend([
            '-mcpu=apple-m3',  # Target M3 processor
            '-mtune=apple-m3',
            '-fvectorize',
            '-fslp-vectorize',
            '-ffp-contract=fast'
        ])
        print(" Detected Apple Silicon M3 - enabling advanced optimizations")
    
    # Intel Mac optimizations
    elif platform.system() == 'Darwin' and platform.machine() == 'x86_64':
        flags.extend([
            '-march=native',
            '-mtune=native',
            '-mavx2',
            '-mfma'
        ])
        print(" Detected Intel Mac - enabling AVX2/FMA optimizations")
    
    # Linux optimizations
    elif platform.system() == 'Linux':
        flags.extend([
            '-march=native',
            '-mtune=native'
        ])
        print(" Detected Linux - enabling native optimizations")
    
    return flags

def get_link_flags():
    """Get optimized linking flags"""
    
    flags = []
    
    if platform.system() == 'Darwin':
        # macOS specific linking
        flags.extend([
            '-framework', 'Accelerate',  # Use Apple's optimized BLAS/LAPACK
            '-Wl,-dead_strip'  # Remove unused code
        ])
    
    return flags

# Define the C++ extension
if PYBIND11_AVAILABLE:
    ext_modules = [
        Pybind11Extension(
            "fourier_math_cpp",
            sources=[
                "python_bindings.cpp",
                "fourier_math.cpp"
            ],
            include_dirs=[
                # Path to pybind11 headers
                pybind11.get_include(),
                # Current directory for our headers
                "."
            ],
            cxx_std=17,
            define_macros=[
                ('VERSION_INFO', '"dev"'),
                ('PYBIND11_DETAILED_ERROR_MESSAGES', None)
            ],
            extra_compile_args=get_compiler_flags(),
            extra_link_args=get_link_flags(),
        ),
    ]
else:
    # Fallback extension definition
    ext_modules = [
        Extension(
            "fourier_math_cpp",
            sources=[
                "python_bindings.cpp",
                "fourier_math.cpp"
            ],
            include_dirs=[
                pybind11.get_include(),
                "."
            ],
            language='c++',
            extra_compile_args=get_compiler_flags() + ['-std=c++17'],
            extra_link_args=get_link_flags(),
        ),
    ]

class BuildExt(build_ext):
    """Custom build extension class with enhanced error reporting"""
    
    def build_extensions(self):
        # Check for required dependencies
        self.check_dependencies()
        
        # Custom compiler detection and optimization
        compiler_type = self.compiler.compiler_type
        print(f" Using compiler: {compiler_type}")
        
        # Add compiler-specific flags
        if compiler_type == 'unix':
            # GCC/Clang specific optimizations
            for ext in self.extensions:
                ext.extra_compile_args.extend([
                    '-Wall',
                    '-Wextra',
                    '-Wno-unused-parameter'
                ])
        
        # Build with progress reporting
        print(" Building C++ extension with performance optimizations...")
        super().build_extensions()
        print(" C++ extension built successfully!")
    
    def check_dependencies(self):
        """Check for required build dependencies"""
        
        # Check for C++17 support
        try:
            self.compiler.compile(['fourier_math.cpp'], extra_preargs=['-std=c++17'])
            print(" C++17 support confirmed")
        except:
            print(" C++17 support required but not available")
            sys.exit(1)
        
        # Check for pybind11
        try:
            import pybind11
            print(f" pybind11 {pybind11.__version__} found")
        except ImportError:
            print(" pybind11 required but not installed")
            print("   Install with: pip install pybind11")
            sys.exit(1)

# Read long description
def get_long_description():
    """Read the README for long description"""
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return """
        High-Performance C++ Fourier Mathematics for Logo Analysis
        
        This module provides optimized C++ implementations of Fourier transforms,
        feature extraction, and similarity analysis specifically designed for
        logo image analysis on MacBook Pro 2024 systems.
        
        Key features:
        - Hardware-optimized FFT implementations
        - Multi-threaded batch processing
        - Apple Silicon M3 Pro/Max optimizations
        - Python integration via pybind11
        """

setup(
    name="fourier_math_cpp",
    version="1.0.0",
    author="Logo Analysis Pipeline",
    author_email="",
    description="High-Performance C++ Fourier Mathematics for Logo Analysis",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExt},
    python_requires=">=3.8",
    install_requires=[
        "pybind11>=2.10.0",
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-benchmark",
            "black",
            "isort",
        ],
        "full": [
            "matplotlib>=3.3.0",
            "seaborn>=0.11.0",
            "pandas>=1.3.0",
            "pillow>=8.0.0",
            "opencv-python>=4.5.0",
            "aiohttp>=3.8.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
    ],
    zip_safe=False,
    include_package_data=True,
)

# Build instructions for users
if __name__ == "__main__":
    print("\n" + "="*60)
    print(" HIGH-PERFORMANCE C++ FOURIER MATHEMATICS MODULE")
    print("="*60)
    print(f"System: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")
    
    if len(sys.argv) == 1:
        print("\nBuild instructions:")
        print("  Development build: python setup.py build_ext --inplace")
        print("  Install build:     pip install .")
        print("  Clean build:       python setup.py clean --all")
        print("\nOptimization levels:")
        print("  🟢 Full optimization (-O3, vectorization)")
        print("   Apple Silicon specific tuning (M3 Pro/Max)")
        print("   Accelerate framework integration (macOS)")
    
    print("\n" + "="*60)
