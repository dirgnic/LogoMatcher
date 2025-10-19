"""
Main Clusterer Module

Orchestrates the entire logo clustering pipeline using modular components.
"""

import os
import time
import pickle
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from .feature_extractor import FeatureExtractor
from .clustering_engine import ClusteringEngine
from .brand_intelligence import BrandIntelligence


class OptimizedLogoClusterer:
    """Main orchestrator for optimized logo clustering pipeline"""
    
    def __init__(self, jpeg_folder_path, thresholds=None):
        """
        Initialize the logo clusterer
        
        Args:
            jpeg_folder_path: Path to folder containing logo JPEGs
            thresholds: Optional dict with 'phash', 'orb', 'color' thresholds
        """
        self.jpeg_folder_path = jpeg_folder_path
        self.jpeg_files = []
        self.batch_size = 50
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.clustering_engine = ClusteringEngine(thresholds)
        self.brand_intelligence = BrandIntelligence()
        
        # Get thresholds for display
        self.thresholds = self.clustering_engine.thresholds
        
        print("OPTIMIZED Advanced Logo Clusterer initialized")
        print(f"Ultra-relaxed thresholds: pHash={self.thresholds['phash']}, ORB={self.thresholds['orb']}, Color={self.thresholds['color']}")
    
    def load_jpeg_files(self):
        """Load and validate JPEG files"""
        print("Loading JPEG files...")
        start_time = time.time()
        
        self.jpeg_files = []
        for filename in os.listdir(self.jpeg_folder_path):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                filepath = os.path.join(self.jpeg_folder_path, filename)
                domain = filename.replace('.jpg', '').replace('.jpeg', '')
                
                self.jpeg_files.append({
                    'filepath': filepath,
                    'filename': filename,
                    'domain': domain,
                    'index': len(self.jpeg_files)
                })
        
        elapsed = time.time() - start_time
        print(f"Loaded {len(self.jpeg_files)} JPEG files in {elapsed:.2f}s")
        return len(self.jpeg_files)
    
    def extract_all_features_parallel(self):
        """Extract features with optimized parallel processing"""
        print("Extracting optimized features...")
        start_time = time.time()
        
        all_features = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(self.feature_extractor.extract_features, jpeg_info): jpeg_info 
                for jpeg_info in self.jpeg_files
            }
            
            for future in as_completed(futures):
                jpeg_info = futures[future]
                try:
                    features = future.result()
                    if features:
                        all_features.append(features)
                        
                        if len(all_features) % 100 == 0:
                            elapsed = time.time() - start_time
                            rate = len(all_features) / elapsed
                            print(f"Processed {len(all_features)} files at {rate:.1f} files/sec")
                            
                except Exception as e:
                    print(f"Feature extraction failed for {jpeg_info['domain']}: {e}")
        
        elapsed = time.time() - start_time
        print(f"Feature extraction completed: {len(all_features)} files in {elapsed:.2f}s ({len(all_features)/elapsed:.1f} files/sec)")
        
        return all_features
    
    def analyze_cluster_quality(self, clusters):
        """Analyze cluster quality with focus on brand coherence"""
        print("\nULTRA-RELAXED CLUSTER ANALYSIS")
        
        total_domains = sum(len(domains) for domains in clusters.values())
        num_clusters = len(clusters)
        
        # Cluster size analysis
        cluster_sizes = [len(domains) for domains in clusters.values()]
        singletons = sum(1 for size in cluster_sizes if size == 1)
        
        # Brand coherence analysis
        brand_coherent_clusters = 0
        total_brand_groups = 0
        
        for cluster_id, domains in clusters.items():
            if len(domains) > 1:
                # Extract brand families
                brand_families = []
                for domain in domains:
                    brand_family = self.brand_intelligence.extract_brand_family(domain)
                    brand_families.append(brand_family)
                
                # Count unique brand families
                unique_brands = set(brand_families)
                
                # Consider coherent if 70% belong to same brand family
                if len(unique_brands) == 1 or (len(domains) > 2 and 
                    max(brand_families.count(brand) for brand in unique_brands) / len(domains) >= 0.7):
                    brand_coherent_clusters += 1
                
                total_brand_groups += 1
        
        # Calculate metrics
        singleton_rate = (singletons / total_domains) * 100 if total_domains > 0 else 0
        brand_coherence_rate = (brand_coherent_clusters / max(total_brand_groups, 1)) * 100
        
        print(f"Total clusters: {num_clusters}")
        print(f"Total domains: {total_domains}")
        print(f"Average cluster size: {total_domains/num_clusters:.2f}")
        print(f"Cluster size range: {min(cluster_sizes)} - {max(cluster_sizes)}")
        print(f"Singletons: {singletons} ({singleton_rate:.1f}%)")
        print(f"Brand coherence: {brand_coherent_clusters}/{total_brand_groups} ({brand_coherence_rate:.1f}%)")
        
        # Size distribution
        size_dist = defaultdict(int)
        for size in cluster_sizes:
            if size == 1:
                size_dist['1 (singleton)'] += 1
            elif size <= 5:
                size_dist['2-5 (small)'] += 1
            elif size <= 10:
                size_dist['6-10 (medium)'] += 1
            elif size <= 20:
                size_dist['11-20 (large)'] += 1
            else:
                size_dist['20+ (huge)'] += 1
        
        print(f"\nCluster size distribution:")
        for size_range, count in size_dist.items():
            percentage = (count / num_clusters) * 100
            print(f"   {size_range}: {count} clusters ({percentage:.1f}%)")
        
        return {
            'total_clusters': num_clusters,
            'total_domains': total_domains,
            'singletons': singletons,
            'singleton_rate': singleton_rate,
            'brand_coherence_rate': brand_coherence_rate,
            'cluster_sizes': cluster_sizes
        }
    
    def save_results(self, clusters, features_list, filename_suffix=""):
        """Save clustering results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save clusters to CSV
        csv_filename = f"optimized_logo_clusters_{timestamp}{filename_suffix}.csv"
        csv_path = os.path.join(os.path.dirname(self.jpeg_folder_path), csv_filename)
        
        rows = []
        for cluster_id, domains in clusters.items():
            for domain in domains:
                rows.append({
                    'cluster_id': cluster_id,
                    'domain': domain,
                    'cluster_size': len(domains)
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        
        # Save full results
        pickle_filename = f"optimized_logo_clustering_results_{timestamp}{filename_suffix}.pkl"
        pickle_path = os.path.join(os.path.dirname(self.jpeg_folder_path), pickle_filename)
        
        results = {
            'clusters': clusters,
            'features': features_list,
            'timestamp': timestamp,
            'thresholds': self.thresholds,
            'num_files': len(self.jpeg_files)
        }
        
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Save detailed text analysis
        txt_filename = f"optimized_cluster_analysis_{timestamp}{filename_suffix}.txt"
        txt_path = os.path.join(os.path.dirname(self.jpeg_folder_path), txt_filename)
        self._save_detailed_text_analysis(clusters, txt_path, timestamp)
        
        print(f"Results saved:")
        print(f"   CSV: {csv_filename}")
        print(f"   Pickle: {pickle_filename}")
        print(f"   Detailed Analysis: {txt_filename}")
        
        return csv_path, pickle_path
    
    def _save_detailed_text_analysis(self, clusters, txt_path, timestamp):
        """Generate detailed cluster analysis text file with brand coherence"""
        
        # Calculate statistics
        total_logos = len(self.jpeg_files)
        total_clusters = len(clusters)
        cluster_sizes = [len(domains) for domains in clusters.values()]
        singletons = sum(1 for size in cluster_sizes if size == 1)
        multi_logo_clusters = total_clusters - singletons
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("OPTIMIZED LOGO CLUSTERING - DETAILED ANALYSIS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Total Logos Analyzed: {total_logos}\n")
            f.write(f"Total Clusters: {total_clusters}\n")
            f.write(f"Multi-Logo Clusters: {multi_logo_clusters}\n")
            f.write(f"Singleton Clusters: {singletons}\n\n")
            
            # Cluster size distribution
            f.write("CLUSTER SIZE DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Largest cluster: {max(cluster_sizes)} logos\n")
            f.write(f"Smallest cluster: {min(cluster_sizes)} logos\n")
            f.write(f"Average cluster size: {sum(cluster_sizes)/len(cluster_sizes):.1f}\n")
            f.write(f"Median cluster size: {sorted(cluster_sizes)[len(cluster_sizes)//2]}\n\n")
            
            # Size categories
            f.write("SIZE CATEGORIES:\n")
            f.write("-" * 20 + "\n")
            tiny = sum(1 for size in cluster_sizes if size <= 5)
            small = sum(1 for size in cluster_sizes if 6 <= size <= 15)
            medium = sum(1 for size in cluster_sizes if 16 <= size <= 50)
            large = sum(1 for size in cluster_sizes if size > 50)
            
            f.write(f"Tiny clusters (≤5 logos): {tiny}\n")
            f.write(f"Small clusters (6-15 logos): {small}\n")
            f.write(f"Medium clusters (16-50 logos): {medium}\n")
            f.write(f"Large clusters (>50 logos): {large}\n\n\n")
            
            # Detailed cluster listings
            f.write("DETAILED CLUSTER LISTINGS:\n")
            f.write("=" * 80 + "\n\n")
            
            # Sort clusters by size (largest first)
            sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
            
            for idx, (cluster_id, domains) in enumerate(sorted_clusters, 1):
                f.write(f"CLUSTER #{idx} (ID: {cluster_id})\n")
                f.write(f"Size: {len(domains)} logos\n")
                f.write("-" * 60 + "\n")
                
                # Analyze brand families
                brand_families = defaultdict(list)
                for domain in domains:
                    brand_family = self.brand_intelligence.extract_brand_family(domain)
                    brand_families[brand_family].append(domain)
                
                # Calculate brand coherence
                num_families = len(brand_families)
                largest_family_size = max(len(members) for members in brand_families.values())
                brand_coherence = largest_family_size / len(domains) * 100
                
                if num_families == 1:
                    f.write(f"BRAND COHERENT CLUSTER (100% same brand family)\n")
                    f.write(f"  Brand family: '{list(brand_families.keys())[0]}'\n")
                elif num_families <= 3:
                    f.write(f"MOSTLY COHERENT ({num_families} brand families, {brand_coherence:.1f}% coherence)\n")
                    for brand, members in sorted(brand_families.items(), key=lambda x: len(x[1]), reverse=True):
                        f.write(f"  Brand family '{brand}': {len(members)} logos\n")
                else:
                    f.write(f"MIXED BRAND CLUSTER ({num_families} different brand families detected):\n")
                    for brand, members in sorted(brand_families.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
                        f.write(f"  Brand family '{brand}': {len(members)} logos\n")
                    if num_families > 10:
                        f.write(f"  ... and {num_families - 10} more brand families\n")
                
                f.write("\nCluster members:\n")
                
                # List all domains
                for i, domain in enumerate(sorted(domains), 1):
                    f.write(f"  {i:3}. {domain}\n")
                    if i >= 100 and len(domains) > 100:  # Limit display for very large clusters
                        f.write(f"  ... and {len(domains) - 100} more logos\n")
                        break
                
                f.write("\n" + "=" * 80 + "\n\n")
            
            # Summary statistics
            f.write("\nBRAND COHERENCE SUMMARY:\n")
            f.write("=" * 80 + "\n")
            
            coherent_clusters = 0
            mixed_clusters = 0
            
            for cluster_id, domains in clusters.items():
                brand_families = defaultdict(list)
                for domain in domains:
                    brand_family = self.brand_intelligence.extract_brand_family(domain)
                    brand_families[brand_family].append(domain)
                
                if len(brand_families) == 1:
                    coherent_clusters += 1
                else:
                    mixed_clusters += 1
            
            f.write(f"Brand-coherent clusters (100% same brand): {coherent_clusters} ({coherent_clusters/total_clusters*100:.1f}%)\n")
            f.write(f"Mixed-brand clusters: {mixed_clusters} ({mixed_clusters/total_clusters*100:.1f}%)\n")
            f.write(f"\nOverall brand coherence rate: {coherent_clusters/total_clusters*100:.1f}%\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Detailed analysis saved to {os.path.basename(txt_path)}")
    
    def run_clustering(self):
        """Main optimized clustering pipeline"""
        print("STARTING OPTIMIZED LOGO CLUSTERING")
        print("=" * 60)
        
        overall_start = time.time()
        
        # Step 1: Load files
        num_files = self.load_jpeg_files()
        if num_files == 0:
            print("No JPEG files found!")
            return None
        
        # Step 2: Extract features
        features_list = self.extract_all_features_parallel()
        if not features_list:
            print("Feature extraction failed!")
            return None
        
        # Step 3: Cluster with candidate pruning
        clusters, similarity_matrix = self.clustering_engine.cluster_with_pruning(features_list)
        
        # Step 4: Aggressive singleton merging
        clusters = self.clustering_engine.merge_singletons_aggressively(clusters, features_list)
        
        # Step 5: Analyze quality
        quality_metrics = self.analyze_cluster_quality(clusters)
        
        # Step 6: Save results
        csv_path, pickle_path = self.save_results(clusters, features_list, "_modular")
        
        overall_elapsed = time.time() - overall_start
        print(f"\nTOTAL PROCESSING TIME: {overall_elapsed:.2f}s")
        print(f"Performance: {len(features_list)/overall_elapsed:.1f} files/sec")
        print("=" * 60)
        
        return {
            'clusters': clusters,
            'features': features_list,
            'quality_metrics': quality_metrics,
            'similarity_matrix': similarity_matrix,
            'csv_path': csv_path,
            'pickle_path': pickle_path
        }
