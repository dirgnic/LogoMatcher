#!/usr/bin/env python3

import asyncio
from python_scraping_class import LogoAnalysisPipeline

async def find_optimal_threshold_for_target_clusters(target_clusters=37):
    """Binary search to find threshold that gives target number of clusters"""
    print(f"Finding optimal threshold for ~{target_clusters} clusters...")
    
    pipeline = LogoAnalysisPipeline()
    
    # Binary search bounds
    low_threshold = 0.990
    high_threshold = 0.999
    best_threshold = None
    best_cluster_count = None
    
    trials = []
    
    for threshold in [0.990, 0.992, 0.994, 0.996, 0.998, 0.999]:
        print(f"\nTesting threshold {threshold}...")
        
        try:
            result = await pipeline.run_analysis_from_parquet(
                parquet_file='logos.snappy.parquet',
                pickle_file='logo_extraction_results.pkl',
                similarity_threshold=threshold,
                create_visualizations=False  # Skip viz to speed up
            )
            
            clusters = result.get('clusters', [])
            similar_pairs = result.get('similar_pairs', [])
            
            cluster_count = len(clusters)
            pairs_count = len(similar_pairs)
            
            trials.append((threshold, cluster_count, pairs_count))
            
            print(f"  Threshold {threshold}: {cluster_count} clusters, {pairs_count} pairs")
            
            # Track best result closest to target
            if best_cluster_count is None or abs(cluster_count - target_clusters) < abs(best_cluster_count - target_clusters):
                best_threshold = threshold
                best_cluster_count = cluster_count
                
            if cluster_count <= target_clusters:
                break
                
        except Exception as e:
            print(f"  Error at threshold {threshold}: {e}")
    
    print(f"\n" + "="*60)
    print(f"THRESHOLD OPTIMIZATION RESULTS")
    print(f"="*60)
    print(f"Target clusters: {target_clusters}")
    print(f"Best threshold: {best_threshold}")
    print(f"Best cluster count: {best_cluster_count}")
    print(f"Difference from target: {abs(best_cluster_count - target_clusters) if best_cluster_count else 'N/A'}")
    
    print(f"\nAll trials:")
    for threshold, clusters, pairs in trials:
        diff = abs(clusters - target_clusters)
        status = "  BEST" if threshold == best_threshold else f" (±{diff})"
        print(f"  {threshold:.3f}: {clusters:3d} clusters, {pairs:5d} pairs{status}")
    
    # Run final analysis with best threshold
    if best_threshold:
        print(f"\n" + "="*60)
        print(f"FINAL ANALYSIS WITH OPTIMAL THRESHOLD {best_threshold}")
        print(f"="*60)
        
        result = await pipeline.run_analysis_from_parquet(
            parquet_file='logos.snappy.parquet',
            pickle_file='logo_extraction_results.pkl',
            similarity_threshold=best_threshold,
            create_visualizations=False
        )
        
        clusters = result.get('clusters', [])
        similar_pairs = result.get('similar_pairs', [])
        
        print(f"Final results:")
        print(f"  Clusters: {len(clusters)}")
        print(f"  Similar pairs: {len(similar_pairs)}")
        print(f"  Clustered websites: {sum(len(cluster) for cluster in clusters)}")
        
        # Show cluster size distribution
        cluster_sizes = sorted([len(cluster) for cluster in clusters], reverse=True)
        print(f"\nTop 10 cluster sizes: {cluster_sizes[:10]}")
        
        if len(clusters) >= 5:
            print(f"\nTop 5 clusters:")
            for i, cluster in enumerate(clusters[:5]):
                domains_preview = ", ".join(cluster[:3]) + ("..." if len(cluster) > 3 else "")
                print(f"  Cluster {i+1}: {len(cluster)} domains - {domains_preview}")

if __name__ == "__main__":
    asyncio.run(find_optimal_threshold_for_target_clusters(37))
