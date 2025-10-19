#!/usr/bin/env python3

import asyncio
from python_scraping_class import LogoAnalysisPipeline

async def test_natural_clustering():
    print(" TESTING NATURAL CLUSTERING WITH ENHANCED LOGO DATA")
    print("=" * 60)
    pipeline = LogoAnalysisPipeline()
    
    # Use our enhanced logo extraction results (4,320 logos, 98.54% success rate)
    result = await pipeline.run_analysis_from_parquet(
        parquet_file='logos.snappy.parquet',
        pickle_file='comprehensive_logo_extraction_fast_results.pkl',  # Updated to use enhanced results
        similarity_threshold=0.74  # Good threshold for meaningful clustering
    )
    
    similar_pairs = result.get('similar_pairs', [])
    clusters = result.get('clusters', [])
    
    print(f'\n NATURAL CLUSTERING RESULTS:')
    print(f'   Similar pairs found: {len(similar_pairs)}')
    print(f'   Natural clusters discovered: {len(clusters)}')
    print(f'   Using 4,320 enhanced logos (98.54% success rate)')
    
    if similar_pairs:
        print(f'\n Top 10 similar pairs:')
        for i, pair in enumerate(similar_pairs[:10]):
            print(f'   {i+1}. {pair["domain1"]} ↔ {pair["domain2"]} (similarity: {pair["similarity"]:.3f})')
    
    if clusters:
        print(f'\n Top 10 natural clusters:')
        cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
        for i, cluster in enumerate(clusters[:10]):
            print(f'   Cluster {i+1}: {len(cluster)} domains - {", ".join(cluster[:5])}{"..." if len(cluster) > 5 else ""}')
        
        print(f'\n Cluster size distribution:')
        print(f'   Largest cluster: {max(cluster_sizes)} domains')
        print(f'   Smallest cluster: {min(cluster_sizes)} domains')
        print(f'   Average cluster size: {sum(cluster_sizes)/len(cluster_sizes):.1f} domains')
        print(f'   Total domains in clusters: {sum(cluster_sizes)}')
        
        print(f'\n FINAL RESULT: {len(clusters)} natural clusters discovered!')
        
    else:
        print(f'\n  No clusters found - try lowering similarity threshold')

if __name__ == "__main__":
    asyncio.run(test_natural_clustering())
