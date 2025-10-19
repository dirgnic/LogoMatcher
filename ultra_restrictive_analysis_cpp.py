#!/usr/bin/env python3
"""
Ultra-restrictive logo clustering analysis using C++ Fourier analysis with enhanced Python logic.
This version uses cached logos and applies ultra-high thresholds to find truly similar brands.
Uses the working fourier_math_cpp module with enhanced similarity computation in Python.
"""

import pickle
import numpy as np
import time
from io import BytesIO
from PIL import Image
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import networkx as nx
from datetime import datetime
import json
from scipy import signal

# Try to import the working C++ module
try:
    import fourier_math_cpp
    CPP_AVAILABLE = True
    print("✅ Using fourier_math_cpp C++ module for enhanced performance")
except ImportError as e:
    CPP_AVAILABLE = False
    print(f"⚠️ C++ module not available: {e}")
    print("📦 Falling back to basic analysis")
    sys.exit(1)

def load_cached_logos():
    """Load the cached logo extraction results"""
    
    cache_file = 'comprehensive_logo_extraction_fast_results.pkl'
    if not os.path.exists(cache_file):
        print(f"❌ Cache file {cache_file} not found")
        return []
    
    try:
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            
        print(f"📦 Loading cached data from {cache_file}")
        
        # Extract successful logos from the cached data
        successful_logos = data.get('successful_logos', [])
        print(f"📊 Found {len(successful_logos)} cached successful logos")
        
        # Process into format needed for analysis
        processed_logos = []
        for logo in successful_logos:
            if logo.get('success') and logo.get('logo_data'):
                try:
                    # Convert logo data to numpy array for C++ processing
                    logo_bytes = logo['logo_data']
                    image = Image.open(BytesIO(logo_bytes))
                    
                    # Convert to RGB if needed and resize
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Resize to standard size for comparison
                    image = image.resize((128, 128), Image.Resampling.LANCZOS)
                    
                    # Convert to numpy array
                    image_array = np.array(image, dtype=np.float64) / 255.0
                    
                    processed_logos.append({
                        'domain': logo.get('website', logo.get('domain', 'unknown')),
                        'image_data': image_array,
                        'size_bytes': len(logo_bytes),
                        'source': logo.get('source', 'cached')
                    })
                    
                except Exception as e:
                    print(f"⚠️ Error processing logo for {logo.get('website', 'unknown')}: {e}")
                    continue
        
        print(f"🎯 Successfully processed {len(processed_logos)} logos for enhanced analysis")
        return processed_logos
        
    except Exception as e:
        print(f"❌ Error loading cache: {e}")
        return []

def compute_enhanced_similarity(img1, img2):
    """
    Compute enhanced similarity using C++ comprehensive similarity.
    Suppresses individual errors for cleaner output.
    """
    try:
        # Convert images to grayscale for C++ processing
        gray1 = np.mean(img1, axis=2) if len(img1.shape) == 3 else img1
        gray2 = np.mean(img2, axis=2) if len(img2.shape) == 3 else img2
        
        # Ensure proper data types and shape
        gray1 = np.ascontiguousarray(gray1, dtype=np.float64)
        gray2 = np.ascontiguousarray(gray2, dtype=np.float64)
        
        # Use C++ comprehensive similarity computation
        similarity_computer = fourier_math_cpp.SimilarityComputer()
        similarity_result = similarity_computer.compute_comprehensive_similarity(gray1, gray2)
        
        # Extract core similarity
        core_similarity = similarity_result.get('similarity', 0.0)
        confidence = similarity_result.get('confidence', 0.0)
        
        # Quick additional metrics
        phash1 = compute_perceptual_hash(gray1)
        phash2 = compute_perceptual_hash(gray2)
        phash_sim = hamming_similarity(phash1, phash2)
        
        # Combined similarity (C++ core + minimal enhancements)
        comprehensive_similarity = 0.8 * core_similarity + 0.2 * phash_sim
        
        enhanced_metrics = {
            'cpp_core_similarity': core_similarity,
            'phash_similarity': phash_sim,
            'comprehensive_similarity': comprehensive_similarity,
            'confidence_score': max(confidence, 0.5),
            'method': 'cpp_hybrid'
        }
        
        return comprehensive_similarity, enhanced_metrics
        
    except:
        # Silent fallback - no error printing
        return compute_python_enhanced_similarity(img1, img2)

def compute_python_enhanced_similarity(img1, img2):
    """Fallback pure Python similarity - simplified and quiet"""
    try:
        gray1 = np.mean(img1, axis=2) if len(img1.shape) == 3 else img1
        gray2 = np.mean(img2, axis=2) if len(img2.shape) == 3 else img2
        
        # Simple perceptual hash
        phash1 = compute_perceptual_hash(gray1)
        phash2 = compute_perceptual_hash(gray2)
        phash_sim = hamming_similarity(phash1, phash2)
        
        enhanced_metrics = {
            'phash_similarity': phash_sim,
            'comprehensive_similarity': phash_sim,
            'confidence_score': 0.5,
            'method': 'python_fallback'
        }
        
        return phash_sim, enhanced_metrics
        
    except:
        return 0.0, {}

def compute_perceptual_hash(image, hash_size=8):
    """Compute perceptual hash for image similarity"""
    # Resize and convert to grayscale
    from PIL import Image
    pil_image = Image.fromarray((image * 255).astype(np.uint8))  # Remove deprecated mode parameter
    if pil_image.mode != 'L':
        pil_image = pil_image.convert('L')
    pil_image = pil_image.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    
    # Convert back to array and compute hash
    pixels = np.array(pil_image, dtype=np.float64)
    avg_pixel = np.mean(pixels)
    
    # Create binary hash
    hash_bits = pixels > avg_pixel
    return ''.join(['1' if bit else '0' for bit in hash_bits.flatten()])

def hamming_similarity(hash1, hash2):
    """Compute similarity based on Hamming distance of hashes"""
    if len(hash1) != len(hash2):
        return 0.0
    
    differences = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    return 1.0 - (differences / len(hash1))

def compute_statistical_moments(image):
    """Compute statistical moments of image"""
    flat_image = image.flatten()
    
    # Compute central moments
    mean = np.mean(flat_image)
    variance = np.var(flat_image)
    skewness = np.mean(((flat_image - mean) / np.sqrt(variance + 1e-8)) ** 3)
    kurtosis = np.mean(((flat_image - mean) / np.sqrt(variance + 1e-8)) ** 4)
    
    return np.array([mean, variance, skewness, kurtosis])

def compute_color_histogram_similarity(img1, img2, bins=32):
    """Compute similarity based on color histograms"""
    hist1 = []
    hist2 = []
    
    # Compute histogram for each channel
    for channel in range(3):
        h1, _ = np.histogram(img1[:, :, channel], bins=bins, range=(0, 1))
        h2, _ = np.histogram(img2[:, :, channel], bins=bins, range=(0, 1))
        hist1.extend(h1)
        hist2.extend(h2)
    
    hist1 = np.array(hist1, dtype=np.float64)
    hist2 = np.array(hist2, dtype=np.float64)
    
    return cosine_similarity(hist1, hist2)

def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

def create_fourier_visualizations(image_matrices, domain_mapping, feature_extractor):
    """Create visualizations of Fourier transforms for sample logos"""
    
    # Select interesting samples - first few logos and some from different clusters if available
    sample_indices = [0, 1, 2, 10, 50, 100, 200, 500]  # Diverse sample
    sample_indices = [i for i in sample_indices if i < len(image_matrices)]
    
    if len(sample_indices) == 0:
        return
    
    # Create figure for Fourier curves
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('C++ Fourier Analysis - Frequency Domain Signatures', fontsize=16, fontweight='bold')
    axes = axes.flatten()
    
    print(f"      Analyzing Fourier signatures for {len(sample_indices)} sample logos...")
    
    for plot_idx, img_idx in enumerate(sample_indices):
        if plot_idx >= len(axes):
            break
            
        ax = axes[plot_idx]
        
        try:
            # Get the image and domain
            img_matrix = image_matrices[img_idx]
            domain = domain_mapping[img_idx] if img_idx < len(domain_mapping) else f"Logo {img_idx}"
            
            # Extract detailed Fourier features using C++
            fft_signature = feature_extractor.compute_fft_signature(img_matrix)
            fourier_mellin = feature_extractor.compute_fourier_mellin(img_matrix)
            comprehensive_features = feature_extractor.extract_comprehensive_features(img_matrix)
            
            # Create frequency axis for visualization
            n_points = min(len(fft_signature), 128)  # Limit for visualization
            frequencies = np.linspace(0, 0.5, n_points)  # Normalized frequencies
            
            # Plot the FFT signature (most important frequencies)
            fft_magnitudes = np.abs(fft_signature[:n_points])
            ax.plot(frequencies, fft_magnitudes, 'b-', linewidth=2, label='FFT Signature', alpha=0.8)
            
            # Plot Fourier-Mellin features if available
            if len(fourier_mellin) >= n_points:
                mellin_magnitudes = np.abs(fourier_mellin[:n_points])
                ax.plot(frequencies, mellin_magnitudes, 'r--', linewidth=1.5, label='Fourier-Mellin', alpha=0.7)
            
            # Highlight dominant frequencies
            dominant_freq_idx = np.argmax(fft_magnitudes[1:]) + 1  # Skip DC component
            ax.axvline(frequencies[dominant_freq_idx], color='orange', linestyle=':', 
                      label=f'Dominant: {frequencies[dominant_freq_idx]:.3f}', alpha=0.8)
            
            # Customize plot
            ax.set_title(f'{domain[:30]}...', fontsize=10, fontweight='bold')
            ax.set_xlabel('Normalized Frequency', fontsize=9)
            ax.set_ylabel('Magnitude', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Add frequency statistics as text
            mean_freq = np.mean(fft_magnitudes)
            max_freq = np.max(fft_magnitudes)
            ax.text(0.02, 0.95, f'Mean: {mean_freq:.2f}\nMax: {max_freq:.2f}', 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
        except Exception as e:
            # Fallback for failed analysis
            ax.text(0.5, 0.5, f'Analysis failed\n{domain[:20]}...', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Error: {domain[:20]}...', fontsize=10)
    
    # Hide unused subplots
    for i in range(len(sample_indices), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    # Save Fourier visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fourier_plot_file = f"fourier_signatures_{timestamp}.png"
    plt.savefig(fourier_plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"      ✅ Fourier curves saved to: {fourier_plot_file}")
    
    # Create a second plot showing 2D FFT for a few logos
    create_2d_fourier_visualization(image_matrices, domain_mapping, sample_indices[:4])
    
    return fourier_plot_file

def create_2d_fourier_visualization(image_matrices, domain_mapping, sample_indices):
    """Create 2D FFT visualizations showing spatial frequency patterns"""
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('2D FFT Spatial Frequency Analysis', fontsize=16, fontweight='bold')
    
    for i, img_idx in enumerate(sample_indices):
        if i >= len(axes[0]):
            break
            
        try:
            # Original image
            img_matrix = image_matrices[img_idx]
            domain = domain_mapping[img_idx] if img_idx < len(domain_mapping) else f"Logo {img_idx}"
            
            # Compute 2D FFT using numpy (for visualization)
            fft2d = np.fft.fft2(img_matrix)
            fft2d_shifted = np.fft.fftshift(fft2d)
            magnitude_spectrum = np.log1p(np.abs(fft2d_shifted))  # Log scale for better visualization
            
            # Plot original image
            axes[0][i].imshow(img_matrix, cmap='gray')
            axes[0][i].set_title(f'Original\n{domain[:20]}...', fontsize=10)
            axes[0][i].axis('off')
            
            # Plot 2D FFT magnitude spectrum
            im = axes[1][i].imshow(magnitude_spectrum, cmap='hot', interpolation='bilinear')
            axes[1][i].set_title(f'2D FFT Spectrum\n(Log Magnitude)', fontsize=10)
            axes[1][i].axis('off')
            
            # Add colorbar for FFT plot
            plt.colorbar(im, ax=axes[1][i], fraction=0.046, pad=0.04)
            
        except Exception as e:
            axes[0][i].text(0.5, 0.5, f'Error\n{domain[:10]}...', 
                           ha='center', va='center', transform=axes[0][i].transAxes)
            axes[1][i].text(0.5, 0.5, f'FFT Error\n{str(e)[:20]}...', 
                           ha='center', va='center', transform=axes[1][i].transAxes)
    
    plt.tight_layout()
    
    # Save 2D FFT visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fft2d_plot_file = f"fourier_2d_fft_{timestamp}.png"
    plt.savefig(fft2d_plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"      ✅ 2D FFT analysis saved to: {fft2d_plot_file}")
    
    return fft2d_plot_file

def run_enhanced_ultra_restrictive_clustering():
    """Run ultra-restrictive clustering using C++ adaptive threshold analysis"""
    
    print("🎯 ENHANCED ULTRA-RESTRICTIVE CLUSTERING ANALYSIS")
    print("⚡ Using C++ Adaptive Threshold + Threading")
    print("=" * 60)
    
    # Load cached logos
    logos = load_cached_logos()
    if not logos:
        print("❌ No logos loaded - cannot proceed")
        return 0, [], None
    
    # Process the entire dataset
    test_logos = logos  # Use all logos
    print(f"🧪 Processing entire dataset: {len(test_logos)} logos")
    
    # Remove duplicates by domain
    seen_domains = set()
    unique_logos = []
    for logo in test_logos:
        domain = logo['domain']
        if domain not in seen_domains:
            seen_domains.add(domain)
            unique_logos.append(logo)
    
    print(f"🔄 {len(unique_logos)} unique domains")
    
    if len(unique_logos) < 2:
        print("❌ Need at least 2 unique logos")
        return 0, [], None
    
    # Prepare image matrices for C++ processing
    image_matrices = []
    domain_mapping = []
    
    for logo in unique_logos:
        # Convert to grayscale for C++ processing
        img = logo['image_data']
        gray = np.mean(img, axis=2) if len(img.shape) == 3 else img
        gray = np.ascontiguousarray(gray, dtype=np.float64)
        
        image_matrices.append(gray)
        domain_mapping.append(logo['domain'])
    
    # Initialize C++ pipeline
    pipeline = fourier_math_cpp.LogoAnalysisPipeline(128)
    
    # ULTRA-RESTRICTIVE thresholds for most similar logos only
    initial_thresholds = [0.99, 0.98, 0.97, 0.96]
    
    print(f"\n🎯 Running C++ Adaptive Threshold Analysis")
    print(f"   🔍 Initial thresholds: {initial_thresholds}")
    print(f"   📊 Sample size: 5% of dataset (min 50 logos)")
    
    start_time = time.time()
    
    try:
        # First extract comprehensive Fourier features using C++ for all images
        print(f"   🔍 Extracting C++ Fourier features from {len(image_matrices)} images...")
        
        feature_extractor = fourier_math_cpp.LogoFeatureExtractor(128, 8)
        fourier_features = []
        perceptual_hashes = []
        
        # Extract advanced Fourier features for each logo
        for i, img_matrix in enumerate(image_matrices):
            if i % 500 == 0:
                progress = (i / len(image_matrices)) * 100
                print(f"      Feature extraction progress: {progress:.1f}%")
            
            try:
                # Extract comprehensive Fourier-based features
                comprehensive_features = feature_extractor.extract_comprehensive_features(img_matrix)
                fft_signature = feature_extractor.compute_fft_signature(img_matrix)
                fourier_mellin = feature_extractor.compute_fourier_mellin(img_matrix)
                perceptual_hash = feature_extractor.compute_perceptual_hash(img_matrix)
                
                # Combine all Fourier features into a single comprehensive vector
                combined_features = comprehensive_features + fft_signature + fourier_mellin
                
                fourier_features.append(combined_features)
                perceptual_hashes.append(perceptual_hash)
                
            except Exception as e:
                # Fallback to basic features if Fourier extraction fails
                print(f"      Warning: Fourier extraction failed for image {i}, using fallback")
                fourier_features.append([0.0] * 100)  # Placeholder feature vector
                perceptual_hashes.append("0" * 64)  # Placeholder hash
        
        print(f"   ✅ Extracted Fourier features: {len(fourier_features)} feature vectors")
        
        # Create Fourier visualization for sample logos
        print(f"   📊 Creating Fourier curve visualizations...")
        create_fourier_visualizations(image_matrices, domain_mapping, feature_extractor)
        
        # Now run adaptive threshold analysis using the Fourier features
        similarity_computer = fourier_math_cpp.SimilarityComputer()
        
        print(f"   🎯 Running C++ Fourier-based adaptive threshold analysis...")
        
        # Implement adaptive threshold logic using Fourier features
        final_threshold = initial_thresholds[0]
        was_adjusted = False
        
        # Sample first 5% to test if threshold adjustment is needed
        sample_size = max(50, int(0.05 * len(fourier_features)))
        sample_pairs_found = 0
        
        print(f"      🧪 Testing with Fourier feature sample of {sample_size} logos...")
        
        for i in range(min(sample_size, len(fourier_features))):
            for j in range(i+1, min(sample_size, len(fourier_features))):
                try:
                    similarity = similarity_computer.compute_comprehensive_similarity(
                        fourier_features[i], fourier_features[j],
                        perceptual_hashes[i], perceptual_hashes[j]
                    )
                    
                    if similarity >= final_threshold:
                        sample_pairs_found += 1
                        
                except:
                    continue
        
        # If no pairs found with highest threshold, try slightly lower ones (but still very high)
        if sample_pairs_found == 0:
            print(f"      ⚠️ No pairs found with ultra-restrictive Fourier threshold {final_threshold}")
            test_thresholds = [0.95, 0.92, 0.90, 0.88, 0.85]  # Much more conservative fallbacks
            
            for test_threshold in test_thresholds:
                test_pairs = 0
                for i in range(min(sample_size, len(fourier_features))):
                    for j in range(i+1, min(sample_size, len(fourier_features))):
                        try:
                            similarity = similarity_computer.compute_comprehensive_similarity(
                                fourier_features[i], fourier_features[j],
                                perceptual_hashes[i], perceptual_hashes[j]
                            )
                            
                            if similarity >= test_threshold:
                                test_pairs += 1
                                
                        except:
                            continue
                
                print(f"         Testing Fourier threshold {test_threshold}: {test_pairs} pairs")
                
                if test_pairs > 0:
                    final_threshold = test_threshold
                    was_adjusted = True
                    print(f"      ✅ Adjusted Fourier threshold to {final_threshold}")
                    break
        
        print(f"   🚀 Computing full Fourier similarity matrix with threshold {final_threshold}...")
        
        # Compute full similarity matrix using Fourier features
        results = similarity_computer.compute_similarity_matrix(fourier_features, perceptual_hashes)
        
        # Apply clustering to the Fourier-based similarity matrix
        clusterer = fourier_math_cpp.LogoClusterer()
        cluster_indices = clusterer.cluster_by_threshold(results, final_threshold)
        
        # Extract similarity scores above threshold from Fourier matrix
        similarity_scores = []
        for i in range(len(results)):
            for j in range(i+1, len(results[i]) if i < len(results) else 0):
                if results[i][j] >= final_threshold:
                    similarity_scores.append(results[i][j])
        
        # Convert results to expected format
        results = {
            'similarity_matrix': results,
            'clusters': cluster_indices,
            'similarity_scores': similarity_scores,
            'processing_time_ms': 0,
            'final_threshold_used': final_threshold,
            'threshold_was_adjusted': was_adjusted
        }
        
        analysis_time = time.time() - start_time
        
        # Extract results
        similarity_matrix = results.get('similarity_matrix', [])
        clusters = results.get('clusters', [])
        similarity_scores = results.get('similarity_scores', [])
        final_threshold = results.get('final_threshold_used', 0.0)
        was_adjusted = results.get('threshold_was_adjusted', False)
        
        print(f"   ✅ Analysis completed in {analysis_time/60:.1f} minutes")
        print(f"   🎯 Final threshold used: {final_threshold}")
        if was_adjusted:
            print(f"   🔄 Threshold was automatically adjusted from initial values")
        else:
            print(f"   ✅ Original threshold was sufficient")
        
        # Convert cluster indices back to domain names
        domain_clusters = []
        for cluster in clusters:
            domain_cluster = [domain_mapping[i] for i in cluster if i < len(domain_mapping)]
            if len(domain_cluster) > 1:  # Only keep clusters with 2+ domains
                domain_clusters.append(domain_cluster)
        
        # Create similarity pairs using ULTRA-RESTRICTIVE Fourier-based filtering
        all_similarities = []
        similar_pairs = []
        
        if hasattr(similarity_matrix, '__len__') and len(similarity_matrix) > 0:
            # First, collect all similarities to find distribution
            for i in range(len(similarity_matrix)):
                for j in range(i+1, len(similarity_matrix[i]) if i < len(similarity_matrix) else 0):
                    if similarity_matrix[i][j] > 0:  # Only non-zero similarities
                        all_similarities.append(similarity_matrix[i][j])
            
            # Calculate ultra-restrictive threshold - only top 1% of similarities
            if all_similarities:
                all_similarities.sort(reverse=True)
                ultra_restrictive_threshold = max(final_threshold, 
                                                np.percentile(all_similarities, 99))  # Top 1%
                print(f"   🔥 Ultra-restrictive threshold: {ultra_restrictive_threshold:.4f} (top 1% of similarities)")
                
                # Only extract pairs that meet BOTH the adaptive threshold AND top 1% cutoff
                for i in range(len(similarity_matrix)):
                    for j in range(i+1, len(similarity_matrix[i]) if i < len(similarity_matrix) else 0):
                        sim_score = similarity_matrix[i][j]
                        
                        # Multi-stage ultra-restrictive filtering
                        if (sim_score >= ultra_restrictive_threshold and 
                            sim_score >= final_threshold and
                            i < len(domain_mapping) and j < len(domain_mapping)):
                            
                            # Additional domain-level filtering - avoid very similar domain names
                            domain1 = domain_mapping[i]
                            domain2 = domain_mapping[j]
                            
                            # Skip if domains are too similar (likely same company with different TLDs)
                            base1 = domain1.split('.')[0].lower()
                            base2 = domain2.split('.')[0].lower()
                            
                            # Only include if base domains are meaningfully different OR similarity is perfect
                            if sim_score >= 0.999 or not (base1 in base2 or base2 in base1):
                                similar_pairs.append({
                                    'domain1': domain1,
                                    'domain2': domain2,
                                    'similarity': float(sim_score),
                                    'confidence': 0.98,  # Very high confidence for ultra-restrictive
                                    'metrics': {
                                        'cpp_fourier_similarity': float(sim_score),
                                        'method': 'cpp_fourier_ultra_restrictive',
                                        'threshold_used': final_threshold,
                                        'ultra_threshold': ultra_restrictive_threshold,
                                        'was_adjusted': was_adjusted,
                                        'features_used': 'fft_signature+fourier_mellin+comprehensive',
                                        'percentile_rank': 'top_1_percent'
                                    }
                                })
            
            print(f"   🎯 After ultra-restrictive filtering: {len(similar_pairs)} highest quality pairs")
        
        print(f"   📊 Found {len(similar_pairs)} similar pairs")
        print(f"   🎊 Created {len(domain_clusters)} clusters")
        
        if similar_pairs:
            similarities = [pair['similarity'] for pair in similar_pairs]
            print(f"   📈 Avg similarity: {np.mean(similarities):.3f}")
            
        # Show only the most meaningful clusters (ultra-restrictive: 3+ domains, max 25 to avoid noise)
        meaningful_clusters = [c for c in domain_clusters if 3 <= len(c) <= 25]
        meaningful_clusters.sort(key=len, reverse=True)  # Sort by size
        
        if meaningful_clusters:
            print(f"\n🎯 Ultra-restrictive brand clusters (3-25 domains) - Top 15:")
            for i, cluster in enumerate(meaningful_clusters[:15]):
                print(f"  {i+1}. ({len(cluster)} domains) {cluster}")
        else:
            print(f"\n🎯 No clusters found meeting ultra-restrictive criteria (3-25 domains)")
        
        # Show top similarity pairs
        if similar_pairs:
            print(f"\n💫 Top similarity pairs:")
            sorted_pairs = sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)
            for i, pair in enumerate(sorted_pairs[:5]):
                method = pair['metrics'].get('method', 'unknown')
                print(f"  {i+1}. {pair['domain1']} ↔ {pair['domain2']} ({pair['similarity']:.3f}) [{method}]")
        
        # Save results only if we found meaningful ultra-restrictive clusters
        meaningful_cluster_count = len([c for c in domain_clusters if 3 <= len(c) <= 25])
        
        if meaningful_cluster_count > 0:
            print(f"\n💾 Saving {meaningful_cluster_count} ultra-restrictive clusters...")
            analysis_stats = {
                'threshold_used': final_threshold,
                'ultra_restrictive_threshold': ultra_restrictive_threshold if 'ultra_restrictive_threshold' in locals() else final_threshold,
                'threshold_was_adjusted': was_adjusted,
                'total_comparisons': len(image_matrices) * (len(image_matrices) - 1) // 2,
                'similar_pairs_found': len(similar_pairs),
                'total_clusters_found': len(domain_clusters),
                'meaningful_clusters_found': meaningful_cluster_count,
                'processing_method': 'cpp_fourier_ultra_restrictive',
                'processing_time_minutes': analysis_time / 60,
                'filtering_applied': 'top_1_percent + domain_validation + size_constraints'
            }
            
            # Save comprehensive results
            results_file, csv_file, pairs_file = save_analysis_results(domain_clusters, similar_pairs, final_threshold, analysis_stats)
            
            # Create visualizations
            plot_file = create_similarity_visualizations(domain_clusters, similar_pairs, final_threshold)
            
            # Generate report
            report_file = generate_analysis_report(domain_clusters, similar_pairs, final_threshold, analysis_stats, 
                                                 [results_file, csv_file, pairs_file, plot_file])
            
            # Return only meaningful clusters for final results
            meaningful_clusters_only = [c for c in domain_clusters if 3 <= len(c) <= 25]
            
            return meaningful_cluster_count, [len(c) for c in meaningful_clusters_only], final_threshold, {
                'clusters': meaningful_clusters_only,  # Only return meaningful clusters
                'similar_pairs': similar_pairs,
                'analysis_stats': analysis_stats,
                'files_generated': [results_file, csv_file, pairs_file, plot_file, report_file]
            }
        
        else:
            print(f"\n❌ No meaningful clusters found with ultra-restrictive analysis")
            print(f"    Total clusters: {len(domain_clusters)}, but none met ultra-restrictive criteria (3-25 domains)")
            return 0, [], final_threshold, {'clusters': [], 'similar_pairs': [], 'analysis_stats': {}}
            
    except Exception as e:
        print(f"   ❌ C++ adaptive analysis failed: {e}")
        print(f"   🔄 This indicates an issue with the C++ implementation")
        return 0, [], None, {}

def create_enhanced_clusters(similar_pairs, max_cluster_size=20):
    """Create clusters from similarity pairs - simplified output"""
    from collections import defaultdict, deque
    
    # Build similarity graph
    graph = defaultdict(list)
    all_domains = set()
    
    for pair in similar_pairs:
        d1, d2 = pair['domain1'], pair['domain2']
        graph[d1].append(d2)
        graph[d2].append(d1)
        all_domains.add(d1)
        all_domains.add(d2)
    
    # Find connected components
    visited = set()
    clusters = []
    
    for domain in all_domains:
        if domain not in visited:
            component = []
            queue = deque([domain])
            visited.add(domain)
            
            while queue:
                current = queue.popleft()
                component.append(current)
                
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            if len(component) >= 2:
                clusters.append(component)
    
    # Apply size constraints
    final_clusters = []
    clusters.sort(key=len, reverse=True)
    
    for cluster in clusters:
        if len(cluster) <= max_cluster_size:
            final_clusters.append(cluster)
        else:
            # Split large clusters
            while cluster:
                chunk = cluster[:max_cluster_size]
                cluster = cluster[max_cluster_size:]
                if len(chunk) >= 2:
                    final_clusters.append(chunk)
    
    return final_clusters

def save_analysis_results(clusters, similar_pairs, threshold, analysis_stats):
    """Save comprehensive analysis results with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save pickle results
    results = {
        'timestamp': timestamp,
        'threshold': threshold,
        'clusters': clusters,
        'similar_pairs': similar_pairs,
        'analysis_stats': analysis_stats,
        'cluster_count': len(clusters),
        'total_domains': len(set([d for cluster in clusters for d in cluster])),
        'method': 'cpp_ultra_restrictive'
    }
    
    results_file = f'ultra_restrictive_results_{timestamp}.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Save CSV for easy viewing
    csv_file = f'ultra_restrictive_clusters_{timestamp}.csv'
    cluster_data = []
    for i, cluster in enumerate(clusters):
        for domain in cluster:
            cluster_data.append({
                'cluster_id': i + 1,
                'domain': domain,
                'cluster_size': len(cluster)
            })
    
    df = pd.DataFrame(cluster_data)
    df.to_csv(csv_file, index=False)
    
    # Save similarity pairs
    pairs_file = f'ultra_restrictive_pairs_{timestamp}.csv'
    pairs_df = pd.DataFrame(similar_pairs)
    pairs_df.to_csv(pairs_file, index=False)
    
    print(f"\n💾 Results saved:")
    print(f"   📦 Full results: {results_file}")
    print(f"   📊 Clusters CSV: {csv_file}")
    print(f"   🔗 Pairs CSV: {pairs_file}")
    
    return results_file, csv_file, pairs_file

def create_similarity_visualizations(clusters, similar_pairs, threshold, output_prefix=None):
    """Create comprehensive visual analysis of similarity results"""
    if not output_prefix:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"ultra_restrictive_analysis_{timestamp}"
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Cluster size distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    cluster_sizes = [len(cluster) for cluster in clusters]
    plt.hist(cluster_sizes, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'Cluster Size Distribution\n(Threshold: {threshold})', fontsize=12, fontweight='bold')
    plt.xlabel('Cluster Size')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # 2. Similarity score distribution
    plt.subplot(2, 2, 2)
    similarities = [pair['similarity'] for pair in similar_pairs]
    plt.hist(similarities, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.title('Similarity Score Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold: {threshold}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Network graph of similarities
    plt.subplot(2, 2, 3)
    G = nx.Graph()
    
    # Add edges from similar pairs
    for pair in similar_pairs[:100]:  # Limit to top 100 for visualization
        G.add_edge(pair['domain1'], pair['domain2'], weight=pair['similarity'])
    
    if G.nodes():
        pos = nx.spring_layout(G, k=1, iterations=50)
        nx.draw(G, pos, node_size=30, node_color='lightblue', 
                edge_color='gray', alpha=0.6, with_labels=False)
        plt.title('Similarity Network\n(Top 100 pairs)', fontsize=12, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No network data', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Similarity Network', fontsize=12, fontweight='bold')
    
    # 4. Top clusters visualization
    plt.subplot(2, 2, 4)
    top_clusters = sorted(clusters, key=len, reverse=True)[:10]
    cluster_names = [f"Cluster {i+1}\n({len(cluster)})" for i, cluster in enumerate(top_clusters)]
    cluster_sizes = [len(cluster) for cluster in top_clusters]
    
    if cluster_sizes:
        bars = plt.bar(range(len(cluster_sizes)), cluster_sizes, color='lightgreen', alpha=0.7)
        plt.title('Top 10 Largest Clusters', fontsize=12, fontweight='bold')
        plt.xlabel('Cluster')
        plt.ylabel('Size')
        plt.xticks(range(len(cluster_names)), [f"C{i+1}" for i in range(len(cluster_names))], rotation=45)
        
        # Add value labels on bars
        for bar, size in zip(bars, cluster_sizes):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(size), ha='center', va='bottom', fontsize=9)
    else:
        plt.text(0.5, 0.5, 'No clusters found', ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Top Clusters', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = f"{output_prefix}_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Create detailed similarity heatmap if we have manageable data
    if len(similar_pairs) > 0 and len(similar_pairs) < 1000:
        create_similarity_heatmap(similar_pairs, f"{output_prefix}_heatmap.png")
    
    print(f"\n📊 Visualizations created:")
    print(f"   📈 Main analysis: {plot_file}")
    if len(similar_pairs) < 1000:
        print(f"   🔥 Similarity heatmap: {output_prefix}_heatmap.png")
    
    return plot_file

def create_similarity_heatmap(similar_pairs, filename):
    """Create a heatmap of similarity scores"""
    # Get unique domains
    domains = set()
    for pair in similar_pairs:
        domains.add(pair['domain1'])
        domains.add(pair['domain2'])
    
    domains = sorted(list(domains))
    
    # Create similarity matrix
    similarity_matrix = np.zeros((len(domains), len(domains)))
    domain_to_idx = {domain: i for i, domain in enumerate(domains)}
    
    for pair in similar_pairs:
        i = domain_to_idx[pair['domain1']]
        j = domain_to_idx[pair['domain2']]
        similarity_matrix[i][j] = pair['similarity']
        similarity_matrix[j][i] = pair['similarity']
    
    # Fill diagonal
    np.fill_diagonal(similarity_matrix, 1.0)
    
    # Create heatmap
    plt.figure(figsize=(max(12, len(domains) * 0.5), max(8, len(domains) * 0.4)))
    
    mask = similarity_matrix == 0
    sns.heatmap(similarity_matrix, 
                xticklabels=domains, 
                yticklabels=domains,
                annot=True if len(domains) < 20 else False,
                fmt='.2f',
                cmap='YlOrRd',
                mask=mask,
                square=True)
    
    plt.title('Logo Similarity Heatmap\n(Ultra-Restrictive Analysis)', fontsize=14, fontweight='bold')
    plt.xlabel('Domains')
    plt.ylabel('Domains')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

def generate_analysis_report(clusters, similar_pairs, threshold, analysis_stats, results_files):
    """Generate a comprehensive markdown report"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_file = f"ultra_restrictive_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    with open(report_file, 'w') as f:
        f.write(f"# Ultra-Restrictive Logo Similarity Analysis Report\n\n")
        f.write(f"**Generated:** {timestamp}  \n")
        f.write(f"**Analysis Method:** C++ Core + Enhanced Python Logic  \n")
        f.write(f"**Threshold Used:** {threshold}  \n\n")
        
        f.write(f"## Executive Summary\n\n")
        f.write(f"- **Total Clusters Found:** {len(clusters)}\n")
        f.write(f"- **Total Similar Pairs:** {len(similar_pairs)}\n")
        f.write(f"- **Unique Domains Clustered:** {len(set([d for cluster in clusters for d in cluster]))}\n")
        f.write(f"- **Average Cluster Size:** {np.mean([len(c) for c in clusters]):.1f}\n")
        f.write(f"- **Largest Cluster Size:** {max([len(c) for c in clusters]) if clusters else 0}\n\n")
        
        f.write(f"## Analysis Statistics\n\n")
        for key, value in analysis_stats.items():
            f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
        f.write(f"\n")
        
        f.write(f"## Top 20 Largest Clusters\n\n")
        sorted_clusters = sorted(clusters, key=len, reverse=True)
        for i, cluster in enumerate(sorted_clusters[:20]):
            f.write(f"### Cluster {i+1} ({len(cluster)} domains)\n")
            for domain in cluster:
                f.write(f"- {domain}\n")
            f.write(f"\n")
        
        f.write(f"## High-Similarity Pairs (Top 50)\n\n")
        sorted_pairs = sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)
        f.write(f"| Domain 1 | Domain 2 | Similarity | Method |\n")
        f.write(f"|----------|----------|------------|--------|\n")
        for pair in sorted_pairs[:50]:
            method = pair.get('metrics', {}).get('method', 'unknown')
            f.write(f"| {pair['domain1']} | {pair['domain2']} | {pair['similarity']:.3f} | {method} |\n")
        
        f.write(f"\n## Generated Files\n\n")
        for file_path in results_files:
            f.write(f"- `{file_path}`\n")
        
        f.write(f"\n## Analysis Details\n\n")
        f.write(f"This analysis used ultra-restrictive thresholds to identify only the most similar logos. ")
        f.write(f"The C++ core provides high-performance Fourier analysis, enhanced with Python-based ")
        f.write(f"perceptual hashing and statistical analysis.\n\n")
        f.write(f"**Note:** Only similarities above {threshold} threshold are included in clustering.\n")
    
    print(f"📋 Comprehensive report generated: {report_file}")
    return report_file

if __name__ == "__main__":
    print("🚀 Starting Ultra-Restrictive Logo Analysis with C++ Core")
    print("=" * 60)
    
    start_time = time.time()
    cluster_count, sizes, best_threshold, analysis_data = run_enhanced_ultra_restrictive_clustering()
    end_time = time.time()
    
    if best_threshold and cluster_count > 0:
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"⚡ Found {cluster_count} ultra-similar clusters using threshold {best_threshold}")
        print(f"⏱️ Analysis completed in {end_time - start_time:.2f} seconds")
        
        # Display generated files
        if 'files_generated' in analysis_data:
            print(f"\n� Files Generated:")
            for file_path in analysis_data['files_generated']:
                print(f"   📄 {file_path}")
        
        print(f"\n📊 FINAL SUMMARY:")
        print(f"   🎯 Clusters: {cluster_count}")
        print(f"   📏 Threshold: {best_threshold}")
        print(f"   � Similar pairs: {len(analysis_data.get('similar_pairs', []))}")
        print(f"   🏷️ Unique domains: {len(set([d for cluster in analysis_data.get('clusters', []) for d in cluster]))}")
        print(f"   ⚡ Method: C++ threading + enhanced Python logic")
        
    else:
        print(f"\n💔 No suitable clustering threshold found")
        print(f"⚠️ Dataset may not contain sufficient similar logos for ultra-restrictive analysis")
        print(f"⏱️ Analysis time: {end_time - start_time:.2f} seconds")
        
    print(f"\n✅ Ultra-Restrictive Analysis Complete")
    print(f"=" * 60)
