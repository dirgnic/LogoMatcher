"""
Example: Using the Modular Optimized Clustering

This script demonstrates various ways to use the modular clustering system.
"""

import sys
import os

# Add parent directory to path if running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def example_1_basic_usage():
    """Example 1: Basic clustering with default settings"""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 60)
    
    from optimized_clustering import OptimizedLogoClusterer
    
    # Path to your logo folder
    logo_folder = "/Users/ingridcorobana/Desktop/personal_projs/logo_matcher/extracted_logos_20251019_174045"
    
    if not os.path.exists(logo_folder):
        print(f"Folder not found: {logo_folder}")
        return
    
    # Initialize and run
    clusterer = OptimizedLogoClusterer(logo_folder)
    results = clusterer.run_clustering()
    
    # Print summary
    if results:
        print(f"\nSummary:")
        print(f"  Clusters: {results['quality_metrics']['total_clusters']}")
        print(f"  Domains: {results['quality_metrics']['total_domains']}")
        print(f"  Singleton rate: {results['quality_metrics']['singleton_rate']:.1f}%")


def example_2_custom_thresholds():
    """Example 2: Using custom thresholds"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Custom Thresholds")
    print("=" * 60)
    
    from optimized_clustering import OptimizedLogoClusterer
    from optimized_clustering.config import MODERATE_THRESHOLDS
    
    logo_folder = "/Users/ingridcorobana/Desktop/personal_projs/logo_matcher/extracted_logos_20251019_174045"
    
    if not os.path.exists(logo_folder):
        print(f"Folder not found: {logo_folder}")
        return
    
    # Use moderate thresholds for better quality
    print("Using MODERATE_THRESHOLDS for higher quality clustering")
    print(f"Settings: {MODERATE_THRESHOLDS}")
    
    clusterer = OptimizedLogoClusterer(logo_folder, thresholds=MODERATE_THRESHOLDS)
    results = clusterer.run_clustering()
    
    if results:
        print(f"\nWith moderate thresholds:")
        print(f"  Clusters: {results['quality_metrics']['total_clusters']}")
        print(f"  Singleton rate: {results['quality_metrics']['singleton_rate']:.1f}%")


def example_3_individual_components():
    """Example 3: Using individual components"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Individual Components")
    print("=" * 60)
    
    from optimized_clustering.brand_intelligence import BrandIntelligence
    from optimized_clustering.visual_analyzer import OptimizedVisualAnalyzer
    from optimized_clustering.hashing import OptimizedMultiScaleHasher
    import cv2
    
    # Brand intelligence
    print("\nBrand Intelligence:")
    bi = BrandIntelligence()
    
    test_domains = ['google.com', 'youtube.com', 'apple.com', 'jpmorgan.com']
    for domain in test_domains:
        brand = bi.extract_brand_family(domain)
        industry = bi.classify_industry(domain)
        print(f"  {domain}: brand={brand}, industry={industry}")
    
    # Visual analysis
    print("\nVisual Analysis:")
    logo_folder = "/Users/ingridcorobana/Desktop/personal_projs/logo_matcher/extracted_logos_20251019_174045"
    
    if os.path.exists(logo_folder):
        # Get first logo
        logos = [f for f in os.listdir(logo_folder) if f.endswith('.jpg')]
        if logos:
            test_logo = os.path.join(logo_folder, logos[0])
            image = cv2.imread(test_logo)
            
            if image is not None:
                va = OptimizedVisualAnalyzer()
                palette = va.extract_color_palette(image)
                composition = va.analyze_logo_composition(image)
                
                print(f"  Logo: {logos[0]}")
                print(f"  Dominant colors: {palette['color_count']}")
                print(f"  Layout: {composition['layout']}")
                print(f"  Text score: {composition['text_score']:.2f}")
                
                # Hashing
                hasher = OptimizedMultiScaleHasher()
                phash = hasher.compute_phash_with_bucketing(image)
                print(f"  pHash bucket: {phash['bucket']}")


def example_4_analyzing_results():
    """Example 4: Analyzing clustering results"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Analyzing Results")
    print("=" * 60)
    
    import pandas as pd
    import glob
    
    # Find most recent results
    csv_files = glob.glob('/Users/ingridcorobana/Desktop/personal_projs/logo_matcher/optimized_logo_clusters_*.csv')
    
    if not csv_files:
        print("No clustering results found")
        return
    
    # Load most recent
    latest_csv = max(csv_files, key=os.path.getctime)
    print(f"Loading: {os.path.basename(latest_csv)}")
    
    df = pd.read_csv(latest_csv)
    
    # Analysis
    print(f"\nDataset Statistics:")
    print(f"  Total domains: {len(df)}")
    print(f"  Total clusters: {df['cluster_id'].nunique()}")
    print(f"  Average cluster size: {df.groupby('cluster_id').size().mean():.2f}")
    
    # Top clusters
    print(f"\nTop 5 Largest Clusters:")
    top_clusters = df.groupby('cluster_id').size().sort_values(ascending=False).head(5)
    for cluster_id, size in top_clusters.items():
        domains = df[df['cluster_id'] == cluster_id]['domain'].tolist()
        print(f"  Cluster {cluster_id}: {size} logos")
        for domain in domains[:3]:
            print(f"    - {domain}")
        if size > 3:
            print(f"    ... and {size - 3} more")


def main():
    """Run all examples"""
    print("=" * 60)
    print("MODULAR CLUSTERING EXAMPLES")
    print("=" * 60)
    
    # Uncomment the examples you want to run:
    
    # example_1_basic_usage()
    # example_2_custom_thresholds()
    example_3_individual_components()
    example_4_analyzing_results()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
