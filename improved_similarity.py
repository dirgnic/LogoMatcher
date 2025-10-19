#!/usr/bin/env python3
"""
Improved Logo Similarity Analysis with Multiple Thresholds
 Find similar logos with adaptive thresholds and better analysis
"""

import asyncio
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import pickle
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import io
import warnings
warnings.filterwarnings('ignore')

# For Fourier analysis
from scipy.fft import fft2, fftshift
from skimage import filters, transform
from sklearn.metrics.pairwise import cosine_similarity

def analyze_similarity_distribution():
    """Analyze the distribution of similarity scores to find optimal threshold"""
    print(" ANALYZING SIMILARITY SCORE DISTRIBUTION")
    print("=" * 60)
    
    # Load results
    with open('logo_extraction_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    successful_logos = results['successful_logos'][:200]  # Sample for speed
    print(f" Analyzing similarity distribution on {len(successful_logos)} logos")
    
    from similarity_pipeline import FourierLogoAnalyzer
    analyzer = FourierLogoAnalyzer()
    
    # Extract features for sample
    print(" Extracting features...")
    analyzed_logos = []
    for logo in successful_logos:
        features = analyzer.extract_logo_features(logo['logo_data'])
        if features['valid']:
            logo_with_features = logo.copy()
            logo_with_features['features'] = features
            analyzed_logos.append(logo_with_features)
    
    print(f" {len(analyzed_logos)} logos with valid features")
    
    # Compute similarity scores for all pairs
    print(" Computing similarity distribution...")
    similarities = []
    
    for i in range(min(len(analyzed_logos), 50)):  # Limit for speed
        for j in range(i + 1, min(len(analyzed_logos), 50)):
            similarity = analyzer.compute_similarity_score(
                analyzed_logos[i]['features'], 
                analyzed_logos[j]['features']
            )
            similarities.append(similarity)
    
    similarities = np.array(similarities)
    
    print(f"\n SIMILARITY STATISTICS:")
    print(f"   Total pairs analyzed: {len(similarities)}")
    print(f"   Mean similarity: {similarities.mean():.3f}")
    print(f"   Std deviation: {similarities.std():.3f}")
    print(f"   Min similarity: {similarities.min():.3f}")
    print(f"   Max similarity: {similarities.max():.3f}")
    print(f"   95th percentile: {np.percentile(similarities, 95):.3f}")
    print(f"   90th percentile: {np.percentile(similarities, 90):.3f}")
    print(f"   75th percentile: {np.percentile(similarities, 75):.3f}")
    
    # Suggest thresholds
    high_threshold = np.percentile(similarities, 95)
    medium_threshold = np.percentile(similarities, 90)
    low_threshold = np.percentile(similarities, 75)
    
    print(f"\n SUGGESTED THRESHOLDS:")
    print(f"   High precision (95th percentile): {high_threshold:.3f}")
    print(f"   Medium precision (90th percentile): {medium_threshold:.3f}")
    print(f"   High recall (75th percentile): {low_threshold:.3f}")
    
    return {
        'high_threshold': high_threshold,
        'medium_threshold': medium_threshold,
        'low_threshold': low_threshold,
        'similarities': similarities
    }

def find_similar_logos_with_multiple_thresholds():
    """Find similar logos using multiple thresholds"""
    print(" MULTI-THRESHOLD SIMILARITY ANALYSIS")
    print("=" * 60)
    
    # Get threshold recommendations
    threshold_analysis = analyze_similarity_distribution()
    
    # Load full results
    with open('logo_extraction_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    successful_logos = results['successful_logos']
    print(f"\n Processing {len(successful_logos)} logos with multiple thresholds")
    
    from similarity_pipeline import FourierLogoAnalyzer, UnionFind
    analyzer = FourierLogoAnalyzer()
    
    # Extract features (reuse if already computed)
    print(" Extracting features...")
    analyzed_logos = analyzer.analyze_logo_batch(successful_logos)
    valid_logos = [logo for logo in analyzed_logos if logo['features']['valid']]
    
    print(f" {len(valid_logos)} logos with valid features")
    
    # Test multiple thresholds
    thresholds = {
        'high': threshold_analysis['high_threshold'],
        'medium': threshold_analysis['medium_threshold'], 
        'low': threshold_analysis['low_threshold'],
        'very_low': 0.3  # Very permissive
    }
    
    results_by_threshold = {}
    
    for threshold_name, threshold_value in thresholds.items():
        print(f"\n Testing threshold: {threshold_name} ({threshold_value:.3f})")
        
        similar_pairs = []
        comparison_count = 0
        
        # Smart sampling for very large datasets
        if len(valid_logos) > 1000:
            print(f"   Large dataset detected, using smart sampling...")
            step = max(1, len(valid_logos) // 1000)
            sample_logos = valid_logos[::step]
            print(f"   Analyzing {len(sample_logos)} sampled logos")
        else:
            sample_logos = valid_logos
        
        start_time = time.time()
        
        for i in range(len(sample_logos)):
            if i % 50 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = comparison_count / elapsed if elapsed > 0 else 0
                print(f"   Progress: {i}/{len(sample_logos)} ({rate:.0f} comparisons/s)")
            
            for j in range(i + 1, len(sample_logos)):
                comparison_count += 1
                similarity = analyzer.compute_similarity_score(
                    sample_logos[i]['features'], 
                    sample_logos[j]['features']
                )
                
                if similarity >= threshold_value:
                    similar_pairs.append((
                        sample_logos[i]['website'],
                        sample_logos[j]['website'],
                        similarity
                    ))
        
        elapsed = time.time() - start_time
        print(f"   Found {len(similar_pairs)} similar pairs in {elapsed:.1f}s")
        
        # Clustering
        clusters = []
        if similar_pairs:
            all_websites = set()
            for pair in similar_pairs:
                all_websites.add(pair[0])
                all_websites.add(pair[1])
            
            uf = UnionFind(all_websites)
            for website1, website2, similarity in similar_pairs:
                uf.union(website1, website2)
            
            clusters = uf.get_clusters()
            print(f"   Formed {len(clusters)} clusters")
        
        results_by_threshold[threshold_name] = {
            'threshold': threshold_value,
            'similar_pairs': similar_pairs,
            'clusters': clusters,
            'clustered_websites': sum(len(cluster) for cluster in clusters)
        }
    
    # Show results summary
    print(f"\n THRESHOLD COMPARISON RESULTS:")
    print("=" * 60)
    
    for threshold_name, result in results_by_threshold.items():
        print(f" {threshold_name.upper()} threshold ({result['threshold']:.3f}):")
        print(f"   Similar pairs: {len(result['similar_pairs'])}")
        print(f"   Clusters: {len(result['clusters'])}")
        print(f"   Websites in clusters: {result['clustered_websites']}")
        
        # Show sample clusters
        if result['clusters']:
            sorted_clusters = sorted(result['clusters'], key=len, reverse=True)
            print(f"   Top clusters:")
            for i, cluster in enumerate(sorted_clusters[:3]):
                print(f"     Cluster {i+1}: {len(cluster)} websites")
                for website in cluster[:2]:
                    print(f"       - {website}")
                if len(cluster) > 2:
                    print(f"       ... and {len(cluster)-2} more")
        print()
    
    # Save best results
    best_threshold = 'medium'  # Default choice
    if results_by_threshold['medium']['similar_pairs']:
        best_threshold = 'medium'
    elif results_by_threshold['low']['similar_pairs']:
        best_threshold = 'low'
    elif results_by_threshold['very_low']['similar_pairs']:
        best_threshold = 'very_low'
    
    print(f" Saving results using {best_threshold} threshold")
    
    best_results = results_by_threshold[best_threshold]
    
    # Save results
    final_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'analyzed_logos': len(analyzed_logos),
        'valid_logos': len(valid_logos),
        'threshold_used': best_results['threshold'],
        'similar_pairs': best_results['similar_pairs'],
        'clusters': best_results['clusters'],
        'total_clustered_websites': best_results['clustered_websites'],
        'all_threshold_results': results_by_threshold
    }
    
    with open('improved_similarity_results.pkl', 'wb') as f:
        pickle.dump(final_results, f)
    
    # Save as CSV files
    if best_results['similar_pairs']:
        df_pairs = pd.DataFrame(best_results['similar_pairs'], 
                               columns=['website1', 'website2', 'similarity'])
        df_pairs.to_csv('improved_similar_pairs.csv', index=False)
        print(f" Saved {len(best_results['similar_pairs'])} similar pairs to improved_similar_pairs.csv")
    
    if best_results['clusters']:
        cluster_data = []
        for i, cluster in enumerate(best_results['clusters']):
            for website in cluster:
                cluster_data.append({
                    'cluster_id': i,
                    'cluster_size': len(cluster),
                    'website': website
                })
        
        df_clusters = pd.DataFrame(cluster_data)
        df_clusters.to_csv('improved_logo_clusters.csv', index=False)
        print(f" Saved {len(best_results['clusters'])} clusters to improved_logo_clusters.csv")
    
    print(f"\n IMPROVED SIMILARITY ANALYSIS COMPLETE!")
    print(f"   - Best threshold: {best_results['threshold']:.3f}")
    print(f"   - Similar pairs found: {len(best_results['similar_pairs'])}")
    print(f"   - Clusters formed: {len(best_results['clusters'])}")
    print(f"   - Total websites clustered: {best_results['clustered_websites']}")
    
    return final_results

if __name__ == "__main__":
    results = find_similar_logos_with_multiple_thresholds()
