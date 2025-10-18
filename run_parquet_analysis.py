#!/usr/bin/env python3
"""
Run logo analysis using existing parquet and pickle data
Enhanced C++ + Python pipeline for large-scale logo analysis
"""

import asyncio
import time
from python_scraping_class import LogoAnalysisPipeline

async def main():
    """Run the complete analysis pipeline using existing data"""
    
    print("LOGO MATCHER - ENHANCED PARQUET ANALYSIS")
    print("=" * 50)
    print("Loading existing parquet domains and pickle logo data...")
    print("Using enhanced C++ threading for similarity computation")
    print()
    
    # Initialize pipeline
    pipeline = LogoAnalysisPipeline()
    
    try:
        # Run analysis from existing data files
        results = await pipeline.run_analysis_from_parquet(
            parquet_file="logos.snappy.parquet",
            pickle_file="logo_extraction_results.pkl",
            create_visualizations=True,
            similarity_threshold=0.45
        )
        
        if results:
            print("\n" + "=" * 60)
            print("FINAL ANALYSIS SUMMARY")
            print("=" * 60)
            
            print(f"Timestamp: {results.get('timestamp', 'N/A')}")
            print(f"Total domains loaded: {results.get('total_domains', 0):,}")
            print(f"Analyzed logos: {results.get('analyzed_logos', 0):,}")
            print(f"Valid feature vectors: {results.get('valid_logos', 0):,}")
            print(f"Similarity threshold: {results.get('threshold_used', 0):.2f}")
            
            print(f"\nSimilarity Results:")
            print(f"  Similar pairs found: {len(results.get('similar_pairs', []))}")
            print(f"  Logo clusters: {len(results.get('clusters', []))}")
            print(f"  Clustered websites: {results.get('total_clustered_websites', 0)}")
            
            # Show success rates
            total_domains = results.get('total_domains', 1)
            analyzed = results.get('analyzed_logos', 0)
            valid = results.get('valid_logos', 0)
            
            print(f"\nSuccess Rates:")
            print(f"  Extraction success: {analyzed/total_domains*100:.1f}% ({analyzed:,}/{total_domains:,})")
            print(f"  Feature extraction: {valid/analyzed*100:.1f}% ({valid:,}/{analyzed:,})")
            
            # Performance metrics
            if 'performance_metrics' in results:
                metrics = results['performance_metrics']
                print(f"\nPerformance Metrics:")
                print(f"  C++ computation time: {metrics.get('cpp_computation_time', 0):.2f}s")
                print(f"  Total pipeline time: {metrics.get('total_pipeline_time', 0):.2f}s")
                print(f"  Processing rate: {metrics.get('logos_per_second', 0):.1f} logos/second")
            
            # Show some example clusters
            clusters = results.get('clusters', [])
            if clusters:
                print(f"\nExample Logo Clusters (showing up to 3):")
                for i, cluster in enumerate(clusters[:3]):
                    print(f"  Cluster {i+1}: {', '.join(cluster[:5])}" + 
                          (f" (+{len(cluster)-5} more)" if len(cluster) > 5 else ""))
            
            # Show some high-similarity pairs
            similar_pairs = results.get('similar_pairs', [])
            if similar_pairs:
                print(f"\nTop Similar Pairs (showing up to 5):")
                sorted_pairs = sorted(similar_pairs, key=lambda x: x['similarity'], reverse=True)
                for i, pair in enumerate(sorted_pairs[:5]):
                    print(f"  {i+1}. {pair['domain1']} ↔ {pair['domain2']} "
                          f"(similarity: {pair['similarity']:.3f})")
        
        else:
            print("ERROR: Analysis failed or returned no results")
            return 1
        
        print("\n" + "=" * 60)
        print("Analysis completed successfully!")
        print("Check 'visualizations/' directory for generated charts")
        
        return 0
        
    except Exception as e:
        print(f"\nERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
