#!/usr/bin/env python3
"""
Automated Build Script for Logo Analysis C++ Module
Handles installation of dependencies and building on MacBook Pro 2024
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

class BuildManager:
    """Manages the complete build process for the C++ module"""
    
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.build_dir = self.project_dir / "build"
        self.system = platform.system()
        self.machine = platform.machine()
        
        print(f"🚀 Logo Analysis C++ Module Build Manager")
        print(f"System: {self.system} {self.machine}")
        print(f"Python: {sys.version.split()[0]}")
        print(f"Project: {self.project_dir}")
    
    def check_system_requirements(self):
        """Check if system has required tools"""
        print("\n🔍 Checking system requirements...")
        
        requirements = {
            'python3': 'Python 3.8+',
            'pip': 'pip package manager', 
            'cmake': 'CMake build system',
            'make': 'Make build tool'
        }
        
        if self.system == 'Darwin':
            requirements['clang++'] = 'Clang C++ compiler (Xcode)'
        else:
            requirements['g++'] = 'GCC C++ compiler'
        
        missing = []
        for cmd, desc in requirements.items():
            if not shutil.which(cmd):
                missing.append(f"❌ {cmd} ({desc})")
            else:
                print(f"✅ {cmd} found")
        
        if missing:
            print(f"\n❌ Missing requirements:")
            for item in missing:
                print(f"  {item}")
            
            if self.system == 'Darwin':
                print("\n💡 To install missing tools on macOS:")
                print("  - Install Xcode Command Line Tools: xcode-select --install")
                print("  - Install Homebrew: /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
                print("  - Install CMake: brew install cmake")
            
            return False
        
        print("✅ All system requirements met")
        return True
    
    def install_python_dependencies(self):
        """Install required Python packages"""
        print("\n📦 Installing Python dependencies...")
        
        dependencies = [
            'pybind11>=2.10.0',
            'numpy>=1.19.0',
            'setuptools>=45.0.0',
            'wheel'
        ]
        
        try:
            for dep in dependencies:
                print(f"Installing {dep}...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                             check=True, capture_output=True)
            
            print("✅ Python dependencies installed")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def build_with_cmake(self):
        """Build using CMake (recommended)"""
        print("\n🔨 Building with CMake...")
        
        try:
            # Create build directory
            self.build_dir.mkdir(exist_ok=True)
            
            # Configure
            print("Configuring...")
            subprocess.run([
                'cmake', 
                '-S', str(self.project_dir),
                '-B', str(self.build_dir),
                '-DCMAKE_BUILD_TYPE=Release'
            ], check=True, cwd=self.project_dir)
            
            # Build
            print("Building...")
            subprocess.run([
                'cmake', 
                '--build', str(self.build_dir),
                '--config', 'Release',
                '-j'  # Use all available cores
            ], check=True)
            
            print("✅ CMake build successful")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ CMake build failed: {e}")
            return False
    
    def build_with_setuppy(self):
        """Build using setup.py (fallback)"""
        print("\n🔨 Building with setup.py...")
        
        try:
            subprocess.run([
                sys.executable, 'setup.py', 
                'build_ext', '--inplace'
            ], check=True, cwd=self.project_dir)
            
            print("✅ setup.py build successful")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ setup.py build failed: {e}")
            return False
    
    def test_module(self):
        """Test the built module"""
        print("\n🧪 Testing built module...")
        
        # Add build directory to Python path
        build_path = str(self.build_dir)
        if build_path not in sys.path:
            sys.path.insert(0, build_path)
        
        # Also check current directory for setup.py builds
        current_path = str(self.project_dir)
        if current_path not in sys.path:
            sys.path.insert(0, current_path)
        
        try:
            # Try importing the module
            import fourier_math_cpp
            print("✅ Module import successful")
            
            # Basic functionality test
            import numpy as np
            
            # Create test images
            test_images = [np.random.rand(128, 128).astype(np.float64) for _ in range(5)]
            
            # Test single image analysis
            pipeline = fourier_math_cpp.LogoAnalysisPipeline(128)
            features = pipeline.analyze_single_logo(test_images[0])
            
            if features.is_valid:
                print("✅ Single image analysis working")
            else:
                print("⚠️ Single image analysis returned invalid features")
            
            # Test batch analysis
            batch_features = pipeline.analyze_logo_batch(test_images)
            valid_count = sum(1 for f in batch_features if f.is_valid)
            print(f"✅ Batch analysis: {valid_count}/{len(test_images)} valid")
            
            # Performance benchmark
            print("Running performance benchmark...")
            benchmark = fourier_math_cpp.benchmark_analysis(test_images, 3)
            print(f"⚡ Performance: {benchmark['images_per_second']:.1f} images/second")
            
            return True
            
        except ImportError as e:
            print(f"❌ Module import failed: {e}")
            print("   Check that the module was built correctly")
            return False
        except Exception as e:
            print(f"❌ Module test failed: {e}")
            return False
    
    def clean_build(self):
        """Clean build artifacts"""
        print("\n🧹 Cleaning build artifacts...")
        
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
            print("✅ Build directory cleaned")
        
        # Clean setup.py artifacts
        for pattern in ['*.so', '*.pyd', 'build/', 'dist/', '*.egg-info/']:
            for path in self.project_dir.glob(pattern):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
        
        print("✅ All build artifacts cleaned")
    
    def install_module(self):
        """Install the module to site-packages"""
        print("\n📦 Installing module...")
        
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '.'
            ], check=True, cwd=self.project_dir)
            
            print("✅ Module installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Installation failed: {e}")
            return False
    
    def run_full_build(self, clean_first=False, install=False):
        """Run the complete build process"""
        print("\n" + "="*60)
        print("🚀 STARTING FULL BUILD PROCESS")
        print("="*60)
        
        if clean_first:
            self.clean_build()
        
        # Step 1: Check system requirements
        if not self.check_system_requirements():
            return False
        
        # Step 2: Install Python dependencies
        if not self.install_python_dependencies():
            return False
        
        # Step 3: Build (try CMake first, fallback to setup.py)
        build_success = self.build_with_cmake()
        if not build_success:
            print("\n⚠️ CMake build failed, trying setup.py...")
            build_success = self.build_with_setuppy()
        
        if not build_success:
            print("\n❌ All build methods failed")
            return False
        
        # Step 4: Test the module
        if not self.test_module():
            return False
        
        # Step 5: Install if requested
        if install:
            if not self.install_module():
                return False
        
        print("\n" + "="*60)
        print("🎉 BUILD PROCESS COMPLETE!")
        print("="*60)
        print(f"Module location: {self.build_dir}")
        print("\nNext steps:")
        print("1. Test integration: python test_integration.py")
        print("2. Run full pipeline: python python_scraping_class.py")
        
        if not install:
            print("3. Install module: python build_module.py --install")
        
        return True


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Logo Analysis C++ Module")
    parser.add_argument('--clean', action='store_true', help='Clean before building')
    parser.add_argument('--install', action='store_true', help='Install after building')
    parser.add_argument('--test-only', action='store_true', help='Only run tests')
    parser.add_argument('--clean-only', action='store_true', help='Only clean artifacts')
    
    args = parser.parse_args()
    
    builder = BuildManager()
    
    if args.clean_only:
        builder.clean_build()
        return
    
    if args.test_only:
        success = builder.test_module()
        sys.exit(0 if success else 1)
    
    success = builder.run_full_build(
        clean_first=args.clean,
        install=args.install
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
