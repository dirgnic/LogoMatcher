#!/usr/bin/env python3

import pickle
import time
import numpy as np
from python_scraping_class import LogoAnalysisPipeline

async def run_adaptive_clustering():
    """Run natural clustering with adaptive threshold to find optimal similarity level"""
    
    print(" ADAPTIVE THRESHOLD CLUSTERING ANALYSIS")
    print("=" * 60)
    
    # Load enhanced logo data
    print(" Loading enhanced logo data...")
    with open('comprehensive_logo_extraction_fast_results.pkl', 'rb') as f:
        enhanced_data = pickle.load(f)
    
    successful_logos = enhanced_data.get('successful_logos', [])
    print(f" Loaded {len(successful_logos)} successful logos ({len(successful_logos)/4384*100:.2f}% success rate)")
    
    # Remove duplicates by domain
    seen_domains = set()
    unique_logos = []
    for logo in successful_logos:
        if logo['domain'] not in seen_domains:
            seen_domains.add(logo['domain'])
            unique_logos.append(logo)
    
    print(f" Removed duplicates: {len(unique_logos)} unique domains")
    
    # Initialize pipeline
    pipeline = LogoAnalysisPipeline()
    
    # Prepare logos for analysis
    pipeline_logos = []
    for logo in unique_logos:
        pipeline_logos.append({
            'domain': logo['domain'],
            'logo_data': logo['logo_data'],
            'size_bytes': logo['size_bytes'],
            'source': logo.get('source', 'unknown')
        })
    
    print(f" Prepared {len(pipeline_logos)} logos for analysis")
    
    # Try different thresholds to find the sweet spot
    thresholds = [0.95, 0.93, 0.91, 0.89, 0.87, 0.85]
    
    for threshold in thresholds:
        print(f"\n Testing threshold: {threshold}")
        start_time = time.time()
        
        # Run similarity analysis
        if hasattr(pipeline, 'cpp_available') and pipeline.cpp_available:
            similarity_results = pipeline._cpp_similarity_analysis(pipeline_logos, threshold)
        else:
            similarity_results = pipeline._python_similarity_analysis(pipeline_logos, threshold)
        
        # Extract results
        valid_logos = similarity_results.get('valid_logos', [])
        similarity_matrix = similarity_results.get('similarity_matrix', [])
        
        # Count similar pairs
        similar_pairs = []
        if similarity_matrix is not None and len(similarity_matrix) > 0 and len(valid_logos) > 1:
            for i in range(len(similarity_matrix)):
                for j in range(i+1, len(similarity_matrix[i])):
                    if similarity_matrix[i][j] >= threshold:
                        similar_pairs.append({
                            'domain1': valid_logos[i]['domain'],
                            'domain2': valid_logos[j]['domain'],
                            'similarity': similarity_matrix[i][j]
                        })
        
        analysis_time = time.time() - start_time
        
        print(f"    Found {len(similar_pairs)} similar pairs in {analysis_time:.1f}s")
        
        # Check if this threshold gives a reasonable number of pairs
        if len(similar_pairs) <= 10000:  # Reasonable number for clustering
            print(f" Threshold {threshold} looks good - proceeding with clustering...")
            
            # Create natural clusters
            if similar_pairs:
                clusters = pipeline._create_natural_similarity_clusters(similar_pairs, max_cluster_size=50)
                
                # Analyze cluster sizes
                cluster_sizes = [len(cluster) for cluster in clusters]
                cluster_sizes.sort(reverse=True)
                
                print(f"\n CLUSTERING RESULTS (Threshold: {threshold})")
                print(f"=" * 50)
                print(f" Total clusters: {len(clusters)}")
                print(f" Cluster sizes: {cluster_sizes[:10]}{'...' if len(cluster_sizes) > 10 else ''}")
                print(f" Largest cluster: {max(cluster_sizes) if cluster_sizes else 0} domains")
                print(f" Average cluster size: {np.mean(cluster_sizes):.1f}")
                
                # Show some example clusters
                print(f"\n EXAMPLE CLUSTERS:")
                for i, cluster in enumerate(clusters[:5]):
                    print(f"Cluster {i+1} ({len(cluster)} domains): {cluster[:5]}{'...' if len(cluster) > 5 else ''}")
                
                return len(clusters), cluster_sizes, threshold
            else:
                print(f"  No clusters found with threshold {threshold}")
        else:
            print(f" Threshold {threshold} produces too many pairs ({len(similar_pairs)}) - trying higher threshold")
    
    print(f"\n Could not find a suitable threshold")
    return 0, [], None

if __name__ == "__main__":
    import asyncio
    cluster_count, sizes, best_threshold = asyncio.run(run_adaptive_clustering())
    
    if best_threshold:
        print(f"\n FINAL RESULT: Found {cluster_count} natural clusters using threshold {best_threshold}")
    else:
        print(f"\n No suitable clustering threshold found")
