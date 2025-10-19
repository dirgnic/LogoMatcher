#!/usr/bin/env python3
"""
Natural Clustering Analysis - Direct approach
Run natural clustering on our enhanced logo dataset
"""

import pickle
import asyncio
import time
from python_scraping_class import LogoAnalysisPipeline

async def run_natural_clustering():
    print(" NATURAL CLUSTERING ANALYSIS")
    print("=" * 60)
    
    # Load our enhanced logo data
    print(" Loading enhanced logo data...")
    with open('comprehensive_logo_extraction_fast_results.pkl', 'rb') as f:
        data = pickle.load(f)
    
    successful_logos = data['successful_logos']
    print(f" Loaded {len(successful_logos)} successful logos (98.54% success rate)")
    
    # Convert to the format expected by the pipeline
    print(" Converting data format...")
    logo_results_converted = []
    for logo in successful_logos:
        converted = {
            'website': logo['website'],
            'domain': logo['domain'],
            'logo_data': logo['logo_data'],
            'logo_found': True,  # Convert 'success' to 'logo_found'
            'source': logo.get('source', 'unknown'),
            'size_bytes': logo.get('size_bytes', 0)
        }
        logo_results_converted.append(converted)
    
    print(f" Converted {len(logo_results_converted)} logos to pipeline format")
    
    # Initialize pipeline
    pipeline = LogoAnalysisPipeline()
    
    # Run similarity analysis directly using internal methods
    print(" Running similarity analysis with natural clustering...")
    start_time = time.time()
    
    # Create a logos dictionary for analysis and preserve domain info
    logos_dict = {}
    pipeline_logos = []
    
    for logo in successful_logos:
        domain = logo['domain']
        logo_data = {
            'domain': domain,  # Preserve domain in logo data
            'logo_data': logo['logo_data'],
            'size_bytes': logo['size_bytes'],
            'source': logo.get('source', 'unknown')
        }
        logos_dict[domain] = logo_data
        pipeline_logos.append(logo_data)
    
    print(f" Prepared {len(pipeline_logos)} logos for similarity analysis")
    
    # Run the analysis using the C++ module if available, otherwise Python
    # Use a higher threshold to avoid mega-clusters
    threshold = 0.89  # Higher threshold for more meaningful similarity
    
    if hasattr(pipeline, 'cpp_available') and pipeline.cpp_available:
        print(f" Using C++ accelerated similarity analysis (threshold: {threshold})...")
        similarity_results = pipeline._cpp_similarity_analysis(pipeline_logos, threshold)
    else:
        print(f" Using Python similarity analysis (threshold: {threshold})...")
        similarity_results = pipeline._python_similarity_analysis(pipeline_logos, threshold)
    
    analysis_time = time.time() - start_time
    
    # Extract similarity pairs for clustering
    valid_logos = similarity_results.get('valid_logos', [])
    similarity_matrix = similarity_results.get('similarity_matrix', [])
    
    print(f" Similarity analysis returned {len(valid_logos)} valid logos")
    if valid_logos:
        print(f"First valid logo keys: {list(valid_logos[0].keys())}")
    
    # Convert similarity matrix to pairs format for natural clustering
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
    
    print(f"Found {len(similar_pairs)} similar pairs above threshold {threshold}")
    
    # Add safety check to prevent mega-clusters
    if len(similar_pairs) > 50000:  # Too many pairs - increase threshold
        print("  Too many similar pairs detected - this would create mega-clusters")
        print(" Consider increasing the similarity threshold for more meaningful clusters")
        return 0, []
    
    # Create natural clusters
    if similar_pairs:
        clusters = pipeline._create_natural_similarity_clusters(similar_pairs, max_cluster_size=50)
    else:
        clusters = []
    
    print(f"\n NATURAL CLUSTERING RESULTS")
    print(f"=" * 60)
    print(f" Analysis completed in {analysis_time:.1f} seconds")
    print(f" Similar pairs found: {len(similar_pairs)}")
    print(f" Natural clusters discovered: {len(clusters)}")
    
    if clusters:
        cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
        total_domains_clustered = sum(cluster_sizes)
        
        print(f"\n CLUSTER STATISTICS:")
        print(f"   Total domains clustered: {total_domains_clustered:,}")
        print(f"   Coverage: {total_domains_clustered/len(successful_logos)*100:.1f}% of successful logos")
        print(f"   Largest cluster: {max(cluster_sizes)} domains")
        print(f"   Smallest cluster: {min(cluster_sizes)} domains")
        print(f"   Average cluster size: {sum(cluster_sizes)/len(cluster_sizes):.1f} domains")
        print(f"   Median cluster size: {sorted(cluster_sizes)[len(cluster_sizes)//2]} domains")
        
        print(f"\n TOP 15 LARGEST CLUSTERS:")
        for i, cluster in enumerate(sorted(clusters, key=len, reverse=True)[:15]):
            print(f"   {i+1:2d}. {len(cluster):2d} domains: {', '.join(cluster[:6])}{'...' if len(cluster) > 6 else ''}")
        
        print(f"\n DISCOVERY: Found {len(clusters)} natural brand clusters!")
        
        if len(clusters) < 50:
            print(f"\n ALL CLUSTERS:")
            for i, cluster in enumerate(sorted(clusters, key=len, reverse=True)):
                print(f"   {i+1:2d}. [{len(cluster):2d}] {', '.join(cluster)}")
        
    else:
        print(f"\n  No clusters found with threshold {0.74}")
        print(f"   Try lowering the similarity threshold")
    
    if similar_pairs:
        print(f"\n TOP 10 MOST SIMILAR PAIRS:")
        for i, pair in enumerate(sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)[:10]):
            print(f"   {i+1:2d}. {pair['domain1']} ↔ {pair['domain2']} ({pair['similarity']:.3f})")
    
    return len(clusters), cluster_sizes if clusters else []

if __name__ == "__main__":
    cluster_count, sizes = asyncio.run(run_natural_clustering())
    
    print(f"\n" + "="*60)
    print(f" NATURAL CLUSTERING COMPLETE!")
    print(f" Discovered {cluster_count} natural brand clusters")
    if sizes:
        print(f" Size range: {min(sizes)}-{max(sizes)} domains per cluster")
    print(f"="*60)
