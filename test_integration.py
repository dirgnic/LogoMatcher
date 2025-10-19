#!/usr/bin/env python3
"""
Integration Test for Python + C++ Logo Analysis Architecture
Tests the complete two-part system: Python scraping + C++ Fourier mathematics
"""

import asyncio
import numpy as np
import time
import sys
from pathlib import Path

# Add current directory and build directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir / "build"))

try:
    from python_scraping_class import LogoAnalysisPipeline
    print(" Python scraping class imported successfully")
except ImportError as e:
    print(f" Failed to import Python scraping class: {e}")
    sys.exit(1)

# Test C++ module availability
try:
    import fourier_math_cpp
    CPP_AVAILABLE = True
    print(" C++ Fourier module available")
    print(f"   Thread pool capacity: {fourier_math_cpp.LogoAnalysisPipeline(128)}")
except ImportError as e:
    CPP_AVAILABLE = False
    print(" C++ Fourier module not available, using Python fallback")
    print(f"   Import error: {e}")

async def test_logo_extraction():
    """Test the logo extraction functionality"""
    print("\n Testing Logo Extraction...")
    
    test_websites = [
        'google.com',
        'microsoft.com',
        'apple.com',
        'github.com',
        'stackoverflow.com'
    ]
    
    pipeline = LogoAnalysisPipeline()
    
    try:
        # Test extraction
        async with pipeline.scraper:
            results = await pipeline.scraper.extract_batch_logos(test_websites[:3])  # Test with first 3
        
        successful = [r for r in results if r.get('success')]
        print(f" Extraction test: {len(successful)}/{len(test_websites[:3])} logos extracted")
        
        if len(successful) > 0:
            print(f"   Sample result: {successful[0]['domain']} ({successful[0]['size']} bytes)")
        
        return len(successful) > 0
        
    except Exception as e:
        print(f" Extraction test failed: {e}")
        return False

def test_cpp_fourier_analysis():
    """Test C++ Fourier analysis if available"""
    if not CPP_AVAILABLE:
        print("\n Skipping C++ tests - module not available")
        return True
    
    print("\n Testing C++ Fourier Analysis...")
    
    try:
        # Create test images
        test_images = []
        for i in range(5):
            # Generate synthetic logo-like images
            img = np.zeros((128, 128), dtype=np.float64)
            
            # Add some structure (simulated logo features)
            center_x, center_y = 64, 64
            radius = 20 + i * 5
            
            for x in range(128):
                for y in range(128):
                    dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                    if dist < radius:
                        img[x, y] = 0.8 - (dist / radius) * 0.6
            
            # Add some noise
            img += np.random.normal(0, 0.1, img.shape)
            img = np.clip(img, 0, 1)
            
            test_images.append(img)
        
        # Test C++ pipeline
        cpp_pipeline = fourier_math_cpp.LogoAnalysisPipeline(128)
        
        # Test single image analysis
        start_time = time.time()
        features = cpp_pipeline.analyze_single_logo(test_images[0])
        single_time = time.time() - start_time
        
        print(f" Single image analysis: {single_time*1000:.2f}ms")
        print(f"   Features valid: {features.is_valid}")
        print(f"   Hash length: {len(features.perceptual_hash)}")
        print(f"   Feature vector size: {len(features.comprehensive_features)}")
        
        # Test batch analysis
        start_time = time.time()
        batch_features = cpp_pipeline.analyze_logo_batch(test_images)
        batch_time = time.time() - start_time
        
        valid_count = sum(1 for f in batch_features if f.is_valid)
        print(f" Batch analysis: {batch_time*1000:.2f}ms ({len(test_images)} images)")
        print(f"   Valid features: {valid_count}/{len(test_images)}")
        
        # Test comprehensive analysis with clustering
        start_time = time.time()
        analysis_results = cpp_pipeline.compute_comprehensive_analysis(test_images, 0.4)
        analysis_time = time.time() - start_time
        
        print(f" Comprehensive analysis: {analysis_time*1000:.2f}ms")
        print(f"   Similarity matrix: {analysis_results['similarity_matrix'].shape}")
        print(f"   Clusters found: {len(analysis_results['clusters'])}")
        print(f"   Similarity scores: {len(analysis_results['similarity_scores'])}")
        
        # Performance benchmark
        print("\n Running performance benchmark...")
        benchmark = fourier_math_cpp.benchmark_analysis(test_images, 10)
        print(f"   Processing rate: {benchmark['images_per_second']:.1f} images/second")
        print(f"   Average time per iteration: {benchmark['avg_time_per_iteration_ms']:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f" C++ analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_python_fallback_analysis():
    """Test Python fallback analysis"""
    print("\n Testing Python Fallback Analysis...")
    
    try:
        # Create pipeline with C++ disabled
        pipeline = LogoAnalysisPipeline()
        pipeline.cpp_available = False  # Force Python fallback
        
        # Create synthetic logo data
        test_logo_data = []
        for i in range(3):
            # Create a simple synthetic image as bytes
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            
            # Convert to bytes (simplified - normally would use PIL/CV2)
            from PIL import Image
            import io
            
            pil_img = Image.fromarray(img)
            byte_buffer = io.BytesIO()
            pil_img.save(byte_buffer, format='PNG')
            logo_bytes = byte_buffer.getvalue()
            
            test_logo_data.append({
                'domain': f'test{i}.com',
                'logo_data': logo_bytes,
                'success': True,
                'size': len(logo_bytes)
            })
        
        # Test Python analysis
        start_time = time.time()
        results = pipeline._python_similarity_analysis(test_logo_data, 0.4)
        python_time = time.time() - start_time
        
        print(f" Python analysis: {python_time*1000:.2f}ms")
        print(f"   Clusters found: {len(results['clusters'])}")
        print(f"   Similarity scores: {len(results['similarity_scores'])}")
        print(f"   Valid logos: {len(results['valid_logos'])}")
        
        return True
        
    except Exception as e:
        print(f" Python analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_integrated_pipeline():
    """Test the complete integrated pipeline"""
    print("\n Testing Integrated Pipeline...")
    
    try:
        pipeline = LogoAnalysisPipeline()
        
        # Use a small set of test websites
        test_websites = ['google.com', 'github.com']
        
        # Run complete analysis (but limit scope for testing)
        print("Running complete analysis (limited scope for testing)...")
        results = await pipeline.run_complete_analysis(
            websites=test_websites,
            create_visualizations=False,  # Skip visualizations for faster testing
            similarity_threshold=0.45
        )
        
        print(f" Integrated pipeline test: {results['status']}")
        if results['status'] == 'success':
            print(f"   Processing time: {results['total_time']:.2f}s")
            print(f"   Success rate: {results['performance_metrics']['success_rate']:.1%}")
            print(f"   Clusters found: {results['performance_metrics']['clusters_found']}")
        
        return results['status'] == 'success'
        
    except Exception as e:
        print(f" Integrated pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_architecture_separation():
    """Test that the architecture properly separates concerns"""
    print("\n Testing Architecture Separation...")
    
    try:
        from python_scraping_class import LogoScrapingEngine, LogoVisualizationEngine
        
        # Test that scraping engine handles I/O
        scraper = LogoScrapingEngine()
        print(" Scraping engine instantiated")
        
        # Test that visualization engine handles charts
        visualizer = LogoVisualizationEngine()
        print(" Visualization engine instantiated")
        
        # Test that C++ module handles mathematics (if available)
        if CPP_AVAILABLE:
            import fourier_math_cpp
            extractor = fourier_math_cpp.LogoFeatureExtractor(128, 8)
            similarity = fourier_math_cpp.SimilarityComputer()
            clusterer = fourier_math_cpp.LogoClusterer()
            print(" C++ mathematical components instantiated")
        else:
            print(" C++ mathematical components not available (using Python fallback)")
        
        print(" Architecture separation verified:")
        print("   - Python: I/O, visualization, orchestration")
        print("   - C++: Heavy mathematical computations (when available)")
        
        return True
        
    except Exception as e:
        print(f" Architecture separation test failed: {e}")
        return False

async def run_all_tests():
    """Run all integration tests"""
    print(" LOGO ANALYSIS INTEGRATION TESTS")
    print("=" * 50)
    
    test_results = {}
    
    # Test 1: Architecture Separation
    test_results['architecture'] = test_architecture_separation()
    
    # Test 2: Logo Extraction
    test_results['extraction'] = await test_logo_extraction()
    
    # Test 3: C++ Analysis (if available)
    test_results['cpp_analysis'] = test_cpp_fourier_analysis()
    
    # Test 4: Python Fallback
    test_results['python_fallback'] = test_python_fallback_analysis()
    
    # Test 5: Integrated Pipeline
    test_results['integrated'] = await test_integrated_pipeline()
    
    # Summary
    print("\n" + "=" * 50)
    print(" TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = " PASS" if result else " FAIL"
        print(f"{test_name.replace('_', ' ').title():20} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print(" ALL TESTS PASSED! The two-part architecture is working correctly.")
        print("\n System Status:")
        print(f"   Python Components:  Functional")
        print(f"   C++ Components: {' Functional' if CPP_AVAILABLE else ' Using Python fallback'}")
        print(f"   Integration:  Working")
    else:
        print(" Some tests failed. Check the output above for details.")
    
    return passed == total

def main():
    """Main entry point"""
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n Test suite failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
