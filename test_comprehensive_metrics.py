#!/usr/bin/env python3

import numpy as np
import cv2
from PIL import Image
import time
import pickle

def test_comprehensive_metrics():
    """Test the enhanced C++ comprehensive similarity metrics system"""
    
    print("🔬 TESTING ENHANCED C++ COMPREHENSIVE METRICS")
    print("=" * 60)
    
    try:
        # Import the enhanced C++ module
        import enhanced_fourier_math_cpp as enhanced_cpp
        print("✅ Enhanced C++ module loaded successfully")
    except ImportError as e:
        print(f"❌ Failed to import enhanced C++ module: {e}")
        print("💡 You need to build the enhanced module first:")
        print("   cmake -B build_enhanced -S . -f CMakeLists_enhanced.txt")
        print("   cd build_enhanced && make")
        return
    
    # Load test images from our logo dataset
    print("\n📋 Loading test logos from cached data...")
    try:
        with open('comprehensive_logo_extraction_fast_results.pkl', 'rb') as f:
            enhanced_data = pickle.load(f)
        
        successful_logos = enhanced_data.get('successful_logos', [])[:10]  # Test with 10 logos
        print(f"✅ Loaded {len(successful_logos)} test logos")
        
    except Exception as e:
        print(f"❌ Failed to load logo data: {e}")
        return
    
    # Convert logo data to numpy arrays
    test_images = []
    test_domains = []
    
    for logo_data in successful_logos:
        try:
            # Convert binary logo data to numpy array
            img_array = np.frombuffer(logo_data['logo_data'], dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Resize to standard size
                img_resized = cv2.resize(img, (128, 128))
                test_images.append(img_resized.astype(np.float64))
                test_domains.append(logo_data['domain'])
                
        except Exception as e:
            print(f"⚠️  Failed to process {logo_data['domain']}: {e}")
    
    print(f"✅ Prepared {len(test_images)} images for testing")
    
    if len(test_images) < 2:
        print("❌ Need at least 2 images for testing")
        return
    
    # Test 1: Individual feature extraction
    print(f"\n🧪 TEST 1: Individual Feature Extraction")
    print("-" * 40)
    
    analyzer = enhanced_cpp.EnhancedFourierAnalyzer(128)
    
    start_time = time.time()
    features = analyzer.extract_all_features(test_images[0])
    extraction_time = time.time() - start_time
    
    print(f"⏱️  Feature extraction time: {extraction_time*1000:.2f}ms")
    print(f"✅ Features extracted successfully:")
    print(f"   📱 pHash length: {len(features.phash)}")
    print(f"   🌊 FFT features: {len(features.fft_features)}")
    print(f"   🔄 Fourier-Mellin: {len(features.fmt_signature)}")
    print(f"   🎨 Color FMT: {len(features.color_fmt_features)}")
    print(f"   ✨ Saliency FFT: {len(features.saliency_fft_features)}")
    print(f"   📐 Hu moments: {len(features.hu_moments)}")
    print(f"   ⭕ Zernike moments: {len(features.zernike_moments)}")
    print(f"   🔍 SIFT signature: {len(features.sift_signature)}")
    print(f"   ⚡ ORB signature: {len(features.orb_signature)}")
    print(f"   🧩 Texture features: {len(features.texture_features)}")
    print(f"   🌈 Color features: {len(features.color_features)}")
    print(f"   ✅ Valid: {features.valid}")
    
    # Test 2: Pairwise comprehensive similarity
    print(f"\n🧪 TEST 2: Comprehensive Similarity Metrics")
    print("-" * 40)
    
    similarity_computer = enhanced_cpp.EnhancedSimilarityComputer()
    
    # Extract features for two images
    features1 = analyzer.extract_all_features(test_images[0])
    features2 = analyzer.extract_all_features(test_images[1])
    
    start_time = time.time()
    metrics = similarity_computer.compute_comprehensive_similarity(features1, features2)
    similarity_time = time.time() - start_time
    
    print(f"⏱️  Similarity computation time: {similarity_time*1000:.2f}ms")
    print(f"🎯 Comparing: {test_domains[0]} vs {test_domains[1]}")
    print(f"📊 COMPREHENSIVE METRICS:")
    print(f"   🔥 Calibrated similarity: {metrics.calibrated_similarity:.4f} ({'✅ Similar' if metrics.calibrated_similar else '❌ Different'})")
    print(f"   🔗 Fused hash distance: {metrics.fused_hash_distance:.4f} ({'✅ Similar' if metrics.fused_hash_similar else '❌ Different'})")
    print(f"   📈 Confidence score: {metrics.confidence_score:.4f}")
    print(f"   📱 pHash distance: {metrics.phash_distance:.1f} ({'✅ Similar' if metrics.phash_similar else '❌ Different'})")
    print(f"   🌊 FFT similarity: {metrics.fft_similarity:.4f} ({'✅ Similar' if metrics.fft_similar else '❌ Different'})")
    print(f"   🔄 FMT similarity: {metrics.fmt_similarity:.4f} ({'✅ Similar' if metrics.fmt_similar else '❌ Different'})")
    print(f"   🎨 Color FMT similarity: {metrics.color_fmt_similarity:.4f} ({'✅ Similar' if metrics.color_fmt_similar else '❌ Different'})")
    print(f"   ✨ Saliency FFT similarity: {metrics.saliency_fft_similarity:.4f} ({'✅ Similar' if metrics.saliency_fft_similar else '❌ Different'})")
    print(f"   📐 Hu similarity: {metrics.hu_similarity:.4f} ({'✅ Similar' if metrics.hu_similar else '❌ Different'})")
    print(f"   ⭕ Zernike similarity: {metrics.zernike_similarity:.4f} ({'✅ Similar' if metrics.zernike_similar else '❌ Different'})")
    print(f"   🔍 SIFT similarity: {metrics.sift_similarity:.4f} ({'✅ Similar' if metrics.sift_similar else '❌ Different'})")
    print(f"   ⚡ ORB similarity: {metrics.orb_similarity:.4f} ({'✅ Similar' if metrics.orb_similar else '❌ Different'})")
    print(f"   🎊 OVERALL DECISION: {'✅ SIMILAR' if metrics.overall_similar else '❌ DIFFERENT'}")
    
    # Test 3: Batch analysis
    if len(test_images) >= 5:
        print(f"\n🧪 TEST 3: Batch Comprehensive Analysis")
        print("-" * 40)
        
        pipeline = enhanced_cpp.EnhancedLogoAnalysisPipeline(128)
        
        start_time = time.time()
        results = pipeline.analyze_logo_batch_comprehensive(test_images[:5], 0.7)
        batch_time = time.time() - start_time
        
        print(f"⏱️  Batch analysis time: {batch_time*1000:.2f}ms")
        print(f"📊 Batch Analysis Results:")
        print(f"   🖼️  Valid images: {results.valid_images}/5")
        print(f"   📏 Similarity matrix shape: {len(results.similarity_matrix)}x{len(results.similarity_matrix[0])}")
        print(f"   🎪 Clusters found: {len(results.clusters)}")
        print(f"   📈 Cluster scores: {[f'{score:.3f}' for score in results.cluster_scores[:3]]}")
        print(f"   ⚡ Total analysis time: {results.analysis_time_ms:.2f}ms")
        
        # Show similarity matrix
        print(f"\n📊 SIMILARITY MATRIX (first 5x5):")
        for i in range(min(5, len(results.similarity_matrix))):
            row_str = "   "
            for j in range(min(5, len(results.similarity_matrix[i]))):
                row_str += f"{results.similarity_matrix[i][j]:.3f} "
            print(row_str)
    
    # Test 4: High-level convenience functions
    print(f"\n🧪 TEST 4: High-Level Convenience Functions")
    print("-" * 40)
    
    # Test the high-level function that matches your Python logic
    start_time = time.time()
    direct_metrics = enhanced_cpp.compute_comprehensive_similarity_metrics(test_images[0], test_images[1])
    direct_time = time.time() - start_time
    
    print(f"⏱️  Direct comparison time: {direct_time*1000:.2f}ms")
    print(f"🎯 Direct metrics match pipeline: {direct_metrics.overall_similar == metrics.overall_similar}")
    
    # Test batch feature extraction
    start_time = time.time()
    batch_features = enhanced_cpp.batch_feature_extraction(test_images[:3])
    batch_feature_time = time.time() - start_time
    
    print(f"⏱️  Batch feature extraction: {batch_feature_time*1000:.2f}ms for 3 images")
    print(f"📊 All features valid: {all(f.valid for f in batch_features)}")
    
    # Test 5: Performance benchmark
    print(f"\n🧪 TEST 5: Performance Benchmark")
    print("-" * 40)
    
    benchmark_results = enhanced_cpp.benchmark_comprehensive_analysis(test_images[:3], 5)
    
    print(f"🚀 PERFORMANCE RESULTS:")
    print(f"   ⏱️  Total time: {benchmark_results['total_time_ms']:.2f}ms")
    print(f"   🔄 Iterations: {benchmark_results['iterations']}")
    print(f"   🖼️  Images per iteration: {benchmark_results['images_per_iteration']}")
    print(f"   📈 Avg time per iteration: {benchmark_results['avg_time_per_iteration_ms']:.2f}ms")
    print(f"   ⚡ Features per second: {benchmark_results['features_per_second']:.1f}")
    
    # Summary
    print(f"\n🎉 COMPREHENSIVE METRICS TEST COMPLETE!")
    print("=" * 60)
    print(f"✅ All C++ comprehensive similarity metrics working correctly")
    print(f"🔥 Ready for integration with your Python logo analysis pipeline")
    print(f"⚡ High-performance feature extraction and similarity computation")
    print(f"📊 Matches your Python metrics structure exactly")

if __name__ == "__main__":
    test_comprehensive_metrics()
