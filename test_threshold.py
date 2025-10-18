#!/usr/bin/env python3

import asyncio
from python_scraping_class import LogoAnalysisPipeline

async def test_lower_threshold():
    print("Testing with lower similarity threshold...")
    pipeline = LogoAnalysisPipeline()
    
    result = await pipeline.run_analysis_from_parquet(
        parquet_file='logos.snappy.parquet',
        pickle_file='logo_extraction_results.pkl',
        similarity_threshold=0.96  # Use good detection threshold with size-constrained clustering
    )
    
    similar_pairs = result.get('similar_pairs', [])
    clusters = result.get('clusters', [])
    
    print(f'Similar pairs found: {len(similar_pairs)}')
    print(f'Clusters found: {len(clusters)}')
    
    if similar_pairs:
        print('\nTop 10 similar pairs:')
        for i, pair in enumerate(similar_pairs[:10]):
            print(f'  {i+1}. {pair["domain1"]} <-> {pair["domain2"]} (similarity: {pair["similarity"]:.3f})')
    
    if clusters:
        print(f'\nTop 5 clusters:')
        for i, cluster in enumerate(clusters[:5]):
            print(f'  Cluster {i+1}: {len(cluster)} domains - {", ".join(cluster[:5])}{"..." if len(cluster) > 5 else ""}')

if __name__ == "__main__":
    asyncio.run(test_lower_threshold())
