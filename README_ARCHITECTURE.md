# Logo Analysis Pipeline: Python + C++ Hybrid Architecture

##  Overview

This is a high-performance logo analysis pipeline redesigned as a **two-part architecture** that separates concerns for optimal performance on MacBook Pro 2024 systems:

- **Python Engine**: Handles scraping, I/O operations, data management, and visualization
- **C++ Engine**: Handles computationally intensive Fourier mathematics and similarity analysis

##  Architecture Design

```

                    PYTHON SCRAPING & VISUALIZATION              
                                                                 
       
   LogoScrapingEngine    LogoVisualization    LogoAnalysis      
                      Engine              Pipeline         
   • API Extraction   • Performance       • Orchestration  
   • Caching            Charts            • Integration    
   • Preprocessing    • Similarity        • Results        
   • Data Mgmt          Heatmaps            Management     
       

                                   
                          Python Bindings (pybind11)
                                   

                    C++ FOURIER MATHEMATICS                      
                                                                 
       
   FourierAnalyzer    SimilarityComputer   LogoClusterer    
                                                          
   • FFT/IFFT         • Cosine Sim       • Union-Find     
   • DCT/Mellin       • Hamming Dist     • Threshold      
   • Feature          • Matrix Ops       • Hierarchical   
     Extraction       • Batch Proc       • Adaptive       
       

```

##  File Structure

```
logo_matcher/
  PYTHON LAYER
    python_scraping_class.py      # Main Python engine
    test_integration.py           # Integration tests
    logo_apis_config.json         # API configuration

  C++ LAYER  
    fourier_math.hpp              # C++ header definitions
    fourier_math.cpp              # C++ implementation
    python_bindings.cpp           # pybind11 interface

  BUILD SYSTEM
    setup.py                      # Python setup script
    CMakeLists.txt               # CMake build config
    build_module.py              # Automated build script

  ORIGINAL FILES (preserved)
     complete_pipeline.py
     lightning_pipeline.py
     similarity_pipeline.py
     visualization_pipeline.py
```

##  Quick Setup

### Option 1: Automated Build (Recommended)

```bash
# Build and test everything automatically
python build_module.py

# Clean build (removes all artifacts first)  
python build_module.py --clean

# Install to site-packages
python build_module.py --install
```

### Option 2: Manual Build

```bash
# Install dependencies
pip install pybind11 numpy setuptools wheel

# Build C++ module with CMake
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# Or build with setup.py
python setup.py build_ext --inplace
```

### Option 3: Python-Only Mode

If C++ build fails, the system automatically falls back to Python implementations:

```python
from python_scraping_class import LogoAnalysisPipeline

# This works with or without C++ module
pipeline = LogoAnalysisPipeline()
results = await pipeline.run_complete_analysis(websites)
```

##  Usage Examples

### Basic Logo Analysis

```python
import asyncio
from python_scraping_class import LogoAnalysisPipeline

async def analyze_logos():
    websites = [
        'google.com',
        'microsoft.com', 
        'apple.com',
        'github.com'
    ]
    
    pipeline = LogoAnalysisPipeline()
    
    results = await pipeline.run_complete_analysis(
        websites=websites,
        create_visualizations=True,
        similarity_threshold=0.45
    )
    
    print(f"Success rate: {results['performance_metrics']['success_rate']:.1%}")
    print(f"Clusters found: {results['performance_metrics']['clusters_found']}")

# Run the analysis
asyncio.run(analyze_logos())
```

### Advanced C++ Usage (when available)

```python
import fourier_math_cpp
import numpy as np

# Create C++ pipeline for maximum performance
cpp_pipeline = fourier_math_cpp.LogoAnalysisPipeline(128)

# Analyze batch of images
images = [np.random.rand(128, 128) for _ in range(100)]
results = cpp_pipeline.compute_comprehensive_analysis(images, 0.45)

print(f"Processing time: {results['processing_time_ms']:.2f}ms")
print(f"Similarity matrix shape: {results['similarity_matrix'].shape}")
```

##  Performance Characteristics

### MacBook Pro 2024 M3 Pro/Max Optimizations

- **Apple Silicon Native**: C++ compiled with `-mcpu=apple-m3` optimizations
- **Accelerate Framework**: Uses Apple's optimized BLAS/LAPACK
- **Multi-threading**: Parallel processing across all CPU cores
- **SIMD Vectorization**: Leverages ARM Neon instructions
- **Memory Optimization**: Cache-friendly algorithms and memory pools

### Performance Targets

| Configuration | Processing Rate | Notes |
|---------------|-----------------|-------|
| **C++ + Python** | ~150-200 logos/sec | Full optimization |
| **Python Only** | ~50-80 logos/sec | Fallback mode |
| **5K Websites** | ~35-45 minutes | Complete pipeline |

##  Testing

```bash
# Run integration tests
python test_integration.py

# Test only C++ components
python build_module.py --test-only

# Performance benchmark
python -c "
import fourier_math_cpp
import numpy as np
images = [np.random.rand(128, 128) for _ in range(50)]
result = fourier_math_cpp.benchmark_analysis(images, 10)
print(f'Performance: {result[\"images_per_second\"]:.1f} images/sec')
"
```

##  Configuration

### API Configuration (logo_apis_config.json)

```json
{
  "logo_apis": [
    {
      "name": "Clearbit",
      "url": "https://logo.clearbit.com/{domain}",
      "timeout": 3,
      "tier": 1
    },
    {
      "name": "Google Favicon",
      "url": "https://www.google.com/s2/favicons",
      "params": {"domain": "{domain}", "sz": "128"},
      "timeout": 2,
      "tier": 1
    }
  ]
}
```

### Performance Tuning

```python
# Adjust concurrency for your system
scraper = LogoScrapingEngine()
scraper.max_concurrent = 200  # MacBook Pro M3 Max
scraper.batch_size = 1000     # Large batches for efficiency

# C++ optimization levels
pipeline = fourier_math_cpp.LogoAnalysisPipeline(
    image_size=128  # Balance between quality and speed
)
```

##  Architecture Comparison

| Aspect | Original Pipeline | New Hybrid Architecture |
|--------|-------------------|------------------------|
| **Language** | Pure Python | Python + C++ |
| **Performance** | ~50-80 logos/sec | ~150-200 logos/sec |
| **Fourier Math** | scipy/numpy | Optimized C++ |
| **Memory Usage** | High (Python objects) | Optimized (C++ pools) |
| **Scalability** | Limited by GIL | Multi-threaded C++ |
| **Maintainability** | Single language | Clear separation |
| **Fallback** | N/A | Automatic Python fallback |

##  Technical Details

### C++ Optimizations

- **Cache-Friendly FFT**: Cooley-Tukey algorithm with pre-computed twiddle factors
- **SIMD Operations**: Vectorized similarity computations
- **Memory Pools**: Reduced allocation overhead
- **Parallel Algorithms**: std::execution::par_unseq for batch processing

### Python Integration

- **Zero-Copy**: NumPy arrays shared with C++ via pybind11
- **Async I/O**: Non-blocking logo extraction with aiohttp
- **Smart Caching**: SHA-256 based disk caching with expiration
- **Progress Tracking**: Real-time performance monitoring

##  Troubleshooting

### C++ Build Issues

```bash
# Check system requirements
python build_module.py --clean

# Install Xcode Command Line Tools (macOS)
xcode-select --install

# Install CMake via Homebrew
brew install cmake

# Force rebuild
rm -rf build/ && python build_module.py --clean
```

### Import Errors

```python
# Add build directory to Python path
import sys
sys.path.insert(0, '/path/to/logo_matcher/build')

# Or set environment variable
export PYTHONPATH=$PYTHONPATH:/path/to/logo_matcher/build
```

### Performance Issues

```python
# Check if C++ module is being used
from python_scraping_class import LogoAnalysisPipeline
pipeline = LogoAnalysisPipeline()
print(f"C++ available: {pipeline.cpp_available}")

# Monitor resource usage
import psutil
print(f"CPU cores: {psutil.cpu_count()}")
print(f"Memory: {psutil.virtual_memory().total // (1024**3)} GB")
```

##  Future Enhancements

1. **GPU Acceleration**: CUDA/Metal compute shaders for Fourier transforms
2. **Distributed Processing**: Multi-machine clustering support
3. **Advanced ML**: Deep learning similarity models
4. **Real-time Processing**: Streaming logo analysis
5. **Cloud Integration**: AWS/Azure batch processing

##  License

This project maintains the same license as the original logo analysis pipeline.

##  Acknowledgments

- Original logo analysis pipeline developers
- pybind11 team for excellent Python-C++ integration
- Apple for optimized Accelerate framework
- Contributors to FFT algorithms and image processing techniques
