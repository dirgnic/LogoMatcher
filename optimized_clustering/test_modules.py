"""
Test script for optimized clustering modules

Run this to verify all modules are working correctly.
"""

import os
import sys


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from optimized_clustering.brand_intelligence import BrandIntelligence
        from optimized_clustering.visual_analyzer import OptimizedVisualAnalyzer
        from optimized_clustering.hashing import OptimizedMultiScaleHasher
        from optimized_clustering.feature_extractor import FeatureExtractor
        from optimized_clustering.clustering_engine import ClusteringEngine
        from optimized_clustering.clusterer import OptimizedLogoClusterer
        from optimized_clustering import config
        
        print("All modules imported successfully")
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False


def test_brand_intelligence():
    """Test brand intelligence module"""
    print("\nTesting brand intelligence...")
    
    try:
        from optimized_clustering.brand_intelligence import BrandIntelligence
        
        bi = BrandIntelligence()
        
        # Test brand family extraction
        assert bi.extract_brand_family('google.com') == 'google'
        assert bi.extract_brand_family('youtube.com') == 'google'
        assert bi.extract_brand_family('unknown-site.com') != 'unknown'
        
        # Test industry classification
        assert bi.classify_industry('apple.com') == 'technology'
        assert bi.classify_industry('jpmorgan.com') == 'finance'
        
        print("✓ Brand intelligence tests passed")
        return True
    except Exception as e:
        print(f"✗ Brand intelligence test failed: {e}")
        return False


def test_visual_analyzer():
    """Test visual analyzer module"""
    print("\nTesting visual analyzer...")
    
    try:
        from optimized_clustering.visual_analyzer import OptimizedVisualAnalyzer
        import cv2
        import numpy as np
        
        va = OptimizedVisualAnalyzer()
        
        # Create a test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test color palette extraction
        palette = va.extract_color_palette(test_image)
        assert 'dominant_colors' in palette
        assert 'color_count' in palette
        
        # Test composition analysis
        composition = va.analyze_logo_composition(test_image)
        assert 'text_score' in composition
        assert 'symbol_score' in composition
        assert 'layout' in composition
        
        print("✓ Visual analyzer tests passed")
        return True
    except Exception as e:
        print(f"✗ Visual analyzer test failed: {e}")
        return False


def test_hashing():
    """Test hashing module"""
    print("\nTesting hashing...")
    
    try:
        from optimized_clustering.hashing import OptimizedMultiScaleHasher
        import cv2
        import numpy as np
        
        hasher = OptimizedMultiScaleHasher()
        
        # Create a test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Test pHash
        phash_data = hasher.compute_phash_with_bucketing(test_image)
        assert 'hash' in phash_data
        assert 'bucket' in phash_data
        assert len(phash_data['hash']) > 0
        
        # Test DCT hash
        dct_hash = hasher.compute_dct_hash(test_image)
        assert len(dct_hash) > 0
        
        # Test FFT hash
        fft_hash = hasher.compute_fft_hash(test_image)
        assert len(fft_hash) > 0
        
        print("✓ Hashing tests passed")
        return True
    except Exception as e:
        print(f"✗ Hashing test failed: {e}")
        return False


def test_clustering_engine():
    """Test clustering engine module"""
    print("\nTesting clustering engine...")
    
    try:
        from optimized_clustering.clustering_engine import ClusteringEngine
        
        engine = ClusteringEngine()
        
        # Test threshold initialization
        assert 'phash' in engine.thresholds
        assert 'orb' in engine.thresholds
        assert 'color' in engine.thresholds
        
        print("✓ Clustering engine tests passed")
        return True
    except Exception as e:
        print(f"✗ Clustering engine test failed: {e}")
        return False


def test_config():
    """Test configuration module"""
    print("\nTesting configuration...")
    
    try:
        from optimized_clustering import config
        
        # Test threshold presets exist
        assert hasattr(config, 'ULTRA_RELAXED_THRESHOLDS')
        assert hasattr(config, 'RELAXED_THRESHOLDS')
        assert hasattr(config, 'MODERATE_THRESHOLDS')
        assert hasattr(config, 'STRICT_THRESHOLDS')
        
        # Test feature weights
        assert hasattr(config, 'FEATURE_WEIGHTS')
        assert 'phash' in config.FEATURE_WEIGHTS
        
        print("✓ Configuration tests passed")
        return True
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("OPTIMIZED CLUSTERING MODULE TESTS")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_brand_intelligence,
        test_visual_analyzer,
        test_hashing,
        test_clustering_engine,
        test_config
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
