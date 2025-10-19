"""
Main entry point for optimized logo clustering

Usage:
    python -m optimized_clustering.run_clustering <jpeg_folder_path>
    
Example:
    python -m optimized_clustering.run_clustering extracted_logos_20251019_174045
"""

import os
import sys
from .clusterer import OptimizedLogoClusterer


def main():
    """Main function to run optimized clustering"""
    
    # Get JPEG folder from command line or use default
    if len(sys.argv) > 1:
        jpeg_folder = sys.argv[1]
    else:
        jpeg_folder = "/Users/ingridcorobana/Desktop/personal_projs/logo_matcher/extracted_logos_20251019_174045"
    
    if not os.path.exists(jpeg_folder):
        print(f"JPEG folder not found: {jpeg_folder}")
        print(f"\nUsage: python -m optimized_clustering.run_clustering <jpeg_folder_path>")
        return 1
    
    # Optional: custom thresholds
    custom_thresholds = {
        'phash': 60,   # Ultra-relaxed (out of 64 bits)
        'orb': 2,      # Ultra-relaxed (min good matches)
        'color': 0.10  # Ultra-relaxed (min similarity)
    }
    
    # Initialize clusterer
    clusterer = OptimizedLogoClusterer(jpeg_folder, thresholds=custom_thresholds)
    
    # Run clustering
    results = clusterer.run_clustering()
    
    if results:
        print("\nOPTIMIZED CLUSTERING COMPLETED SUCCESSFULLY!")
        print(f"Generated {results['quality_metrics']['total_clusters']} clusters")
        print(f"Singleton rate: {results['quality_metrics']['singleton_rate']:.1f}%") 
        print(f"Brand coherence: {results['quality_metrics']['brand_coherence_rate']:.1f}%")
        return 0
    else:
        print("Clustering failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
