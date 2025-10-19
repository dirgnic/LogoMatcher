"""
Visualization Module for Optimized Logo Clustering

Creates comprehensive visualizations including:
- Cluster separation in 2D/3D feature space
- pHash similarity heatmaps
- FFT/DCT feature distributions
- Color palette analysis
- ORB descriptor statistics
- Brand coherence visualization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
from collections import defaultdict
import cv2
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)


class ClusteringVisualizer:
    """Comprehensive visualization for clustering results"""
    
    def __init__(self, results_pickle_path):
        """
        Initialize visualizer with clustering results
        
        Args:
            results_pickle_path: Path to the pickle file with clustering results
        """
        print("Loading clustering results...")
        with open(results_pickle_path, 'rb') as f:
            self.results = pickle.load(f)
        
        self.clusters = self.results['clusters']
        self.features = self.results['features']
        self.timestamp = self.results['timestamp']
        
        # Create feature matrix
        self._prepare_feature_matrices()
        
        print(f"Loaded {len(self.features)} logos in {len(self.clusters)} clusters")
    
    def _prepare_feature_matrices(self):
        """Prepare feature matrices for visualization"""
        n = len(self.features)
        
        # Extract various features
        self.phash_matrix = np.zeros(n)
        self.color_features = []
        self.dct_hashes = []
        self.fft_hashes = []
        self.orb_counts = []
        self.domains = []
        self.cluster_labels = []
        
        # Create domain to cluster mapping
        domain_to_cluster = {}
        for cluster_id, domains in self.clusters.items():
            for domain in domains:
                domain_to_cluster[domain] = cluster_id
        
        for i, feature in enumerate(self.features):
            domain = feature['domain']
            self.domains.append(domain)
            self.cluster_labels.append(domain_to_cluster.get(domain, -1))
            
            # pHash numeric values
            self.phash_matrix[i] = feature.get('phash_numeric', 0)
            
            # Color histograms
            color_hist = feature.get('color_histogram', np.zeros(30))
            self.color_features.append(color_hist)
            
            # DCT hash (convert to numeric)
            dct_hash = feature.get('dct_hash', '0' * 64)
            dct_numeric = sum([int(bit) * (2 ** i) for i, bit in enumerate(dct_hash[:32])])
            self.dct_hashes.append(dct_numeric)
            
            # FFT hash (convert to numeric)
            fft_hash = feature.get('fft_hash', '0' * 256)
            fft_numeric = sum([int(bit) * (2 ** i) for i, bit in enumerate(fft_hash[:32])])
            self.fft_hashes.append(fft_numeric)
            
            # ORB keypoint count
            self.orb_counts.append(feature.get('orb_keypoints_count', 0))
        
        self.color_features = np.array(self.color_features)
        self.cluster_labels = np.array(self.cluster_labels)
        
        print(f"Feature matrices prepared: {n} samples")
    
    def visualize_cluster_separation_2d(self, method='pca', save_path=None):
        """
        Visualize cluster separation in 2D using dimensionality reduction
        
        Args:
            method: 'pca' or 'tsne'
            save_path: Optional path to save the figure
        """
        print(f"\nGenerating 2D cluster visualization using {method.upper()}...")
        
        # Combine features for dimensionality reduction
        combined_features = np.column_stack([
            self.color_features,
            np.array(self.dct_hashes).reshape(-1, 1),
            np.array(self.fft_hashes).reshape(-1, 1),
            np.array(self.orb_counts).reshape(-1, 1)
        ])
        
        # Apply dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            coords = reducer.fit_transform(combined_features)
            var_explained = reducer.explained_variance_ratio_
            title = f'PCA Cluster Visualization (Var: {var_explained[0]:.1%}, {var_explained[1]:.1%})'
        else:  # tsne
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            coords = reducer.fit_transform(combined_features)
            title = 't-SNE Cluster Visualization'
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Plot 1: Colored by cluster
        unique_clusters = np.unique(self.cluster_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = self.cluster_labels == cluster_id
            cluster_size = np.sum(mask)
            
            if cluster_size == 1:
                # Singletons in gray
                ax1.scatter(coords[mask, 0], coords[mask, 1], 
                           c='gray', s=30, alpha=0.5, label=f'Singletons')
            else:
                ax1.scatter(coords[mask, 0], coords[mask, 1], 
                           c=[colors[i]], s=50 + cluster_size * 5, 
                           alpha=0.6, label=f'Cluster {cluster_id} (n={cluster_size})')
        
        ax1.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax1.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Colored by cluster size
        cluster_sizes = []
        for label in self.cluster_labels:
            cluster_domains = self.clusters.get(label, [])
            cluster_sizes.append(len(cluster_domains))
        
        scatter = ax2.scatter(coords[:, 0], coords[:, 1], 
                            c=cluster_sizes, s=50, 
                            cmap='viridis', alpha=0.6)
        ax2.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
        ax2.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
        ax2.set_title('Cluster Separation by Size', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Cluster Size', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        
        return coords
    
    def visualize_cluster_separation_3d(self, save_path=None):
        """
        3D visualization of cluster separation using matplotlib
        
        Args:
            save_path: Optional path to save figure
        """
        print("\nGenerating 3D cluster visualization...")
        
        # Combine features
        combined_features = np.column_stack([
            self.color_features,
            np.array(self.dct_hashes).reshape(-1, 1),
            np.array(self.fft_hashes).reshape(-1, 1),
            np.array(self.orb_counts).reshape(-1, 1)
        ])
        
        # PCA to 3D
        pca = PCA(n_components=3, random_state=42)
        coords_3d = pca.fit_transform(combined_features)
        
        # Prepare data
        cluster_sizes = []
        for label in self.cluster_labels:
            cluster_domains = self.clusters.get(label, [])
            cluster_sizes.append(len(cluster_domains))
        
        # Create 3D scatter plot
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot with color by cluster size
        scatter = ax.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2],
                           c=cluster_sizes, s=30, cmap='viridis', alpha=0.6)
        
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_zlabel('PC3', fontsize=12)
        ax.set_title(f'3D Cluster Visualization (PCA)\nVariance Explained: {pca.explained_variance_ratio_.sum():.1%}',
                    fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, pad=0.1)
        cbar.set_label('Cluster Size', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved 3D plot to {save_path}")
        
        plt.show()
    
    def visualize_phash_heatmap(self, max_clusters=20, save_path=None):
        """
        Visualize pHash similarity heatmap across clusters
        
        Args:
            max_clusters: Maximum number of clusters to show
            save_path: Optional path to save the figure
        """
        print("\nGenerating pHash similarity heatmap...")
        
        # Get top clusters by size
        cluster_items = sorted(self.clusters.items(), 
                             key=lambda x: len(x[1]), 
                             reverse=True)[:max_clusters]
        
        # Sample logos from each cluster
        sampled_domains = []
        cluster_ids = []
        
        for cluster_id, domains in cluster_items:
            # Sample up to 5 logos per cluster
            sample_size = min(5, len(domains))
            sampled = domains[:sample_size]
            sampled_domains.extend(sampled)
            cluster_ids.extend([cluster_id] * sample_size)
        
        # Get pHash values for sampled domains
        domain_to_phash = {f['domain']: f['phash'] for f in self.features}
        phash_values = [domain_to_phash.get(d, '0' * 64) for d in sampled_domains]
        
        # Compute pairwise Hamming distances
        n = len(phash_values)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if phash_values[i] and phash_values[j]:
                    hash_diff = bin(int(phash_values[i], 16) ^ int(phash_values[j], 16)).count('1')
                    distance_matrix[i, j] = hash_diff
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # Plot heatmap
        im = ax.imshow(distance_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=64)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('pHash Hamming Distance (bits)', fontsize=12)
        
        # Set ticks and labels
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f"C{cluster_ids[i]}:{sampled_domains[i][:15]}" 
                           for i in range(n)], rotation=90, fontsize=8)
        ax.set_yticklabels([f"C{cluster_ids[i]}:{sampled_domains[i][:15]}" 
                           for i in range(n)], fontsize=8)
        
        # Add title
        ax.set_title('pHash Similarity Heatmap Across Clusters\n(Lower=More Similar)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add grid lines between clusters
        prev_cluster = cluster_ids[0]
        for i, cluster_id in enumerate(cluster_ids[1:], 1):
            if cluster_id != prev_cluster:
                ax.axhline(i - 0.5, color='white', linewidth=2)
                ax.axvline(i - 0.5, color='white', linewidth=2)
            prev_cluster = cluster_id
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    
    def visualize_feature_distributions(self, save_path=None):
        """
        Visualize distributions of FFT, DCT, and other features across clusters
        
        Args:
            save_path: Optional path to save the figure
        """
        print("\nGenerating feature distribution plots...")
        
        # Get cluster sizes
        cluster_sizes = {cid: len(domains) for cid, domains in self.clusters.items()}
        top_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
        top_cluster_ids = [cid for cid, _ in top_clusters]
        
        # Prepare data
        cluster_dct = defaultdict(list)
        cluster_fft = defaultdict(list)
        cluster_orb = defaultdict(list)
        cluster_color_variance = defaultdict(list)
        
        for feature in self.features:
            domain = feature['domain']
            # Find cluster
            cluster_id = None
            for cid, domains in self.clusters.items():
                if domain in domains:
                    cluster_id = cid
                    break
            
            if cluster_id in top_cluster_ids:
                # DCT hash numeric
                dct_hash = feature.get('dct_hash', '0' * 64)
                dct_numeric = sum([int(bit) for bit in dct_hash])
                cluster_dct[cluster_id].append(dct_numeric)
                
                # FFT hash numeric
                fft_hash = feature.get('fft_hash', '0' * 256)
                fft_numeric = sum([int(bit) for bit in fft_hash[:64]])
                cluster_fft[cluster_id].append(fft_numeric)
                
                # ORB count
                cluster_orb[cluster_id].append(feature.get('orb_keypoints_count', 0))
                
                # Color variance
                color_hist = feature.get('color_histogram', np.zeros(30))
                cluster_color_variance[cluster_id].append(np.var(color_hist))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Plot 1: DCT hash distribution
        ax = axes[0, 0]
        for cluster_id in top_cluster_ids:
            values = cluster_dct[cluster_id]
            if values:
                ax.hist(values, bins=20, alpha=0.5, 
                       label=f'Cluster {cluster_id} (n={len(values)})')
        ax.set_xlabel('DCT Hash Bit Count', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('DCT Hash Distribution Across Top Clusters', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: FFT hash distribution
        ax = axes[0, 1]
        for cluster_id in top_cluster_ids:
            values = cluster_fft[cluster_id]
            if values:
                ax.hist(values, bins=20, alpha=0.5, 
                       label=f'Cluster {cluster_id} (n={len(values)})')
        ax.set_xlabel('FFT Hash Bit Count (first 64 bits)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('FFT Hash Distribution Across Top Clusters', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 3: ORB keypoint distribution
        ax = axes[1, 0]
        for cluster_id in top_cluster_ids:
            values = cluster_orb[cluster_id]
            if values:
                ax.hist(values, bins=20, alpha=0.5, 
                       label=f'Cluster {cluster_id} (n={len(values)})')
        ax.set_xlabel('ORB Keypoint Count', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('ORB Keypoint Distribution Across Top Clusters', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Color variance distribution
        ax = axes[1, 1]
        for cluster_id in top_cluster_ids:
            values = cluster_color_variance[cluster_id]
            if values:
                ax.hist(values, bins=20, alpha=0.5, 
                       label=f'Cluster {cluster_id} (n={len(values)})')
        ax.set_xlabel('Color Histogram Variance', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Color Variance Distribution Across Top Clusters', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
    
    def visualize_cluster_statistics(self, save_path=None):
        """
        Visualize mean feature values for all clusters
        
        Args:
            save_path: Optional path to save the figure
        """
        print("\nGenerating cluster statistics visualization...")
        
        # Calculate mean features per cluster
        cluster_stats = []
        
        for cluster_id, domains in self.clusters.items():
            # Get features for this cluster
            cluster_features = [f for f in self.features if f['domain'] in domains]
            
            if not cluster_features:
                continue
            
            # Calculate means
            mean_dct = np.mean([sum([int(bit) for bit in f.get('dct_hash', '0' * 64)]) 
                               for f in cluster_features])
            mean_fft = np.mean([sum([int(bit) for bit in f.get('fft_hash', '0' * 256)[:64]]) 
                               for f in cluster_features])
            mean_orb = np.mean([f.get('orb_keypoints_count', 0) for f in cluster_features])
            mean_color_var = np.mean([np.var(f.get('color_histogram', np.zeros(30))) 
                                     for f in cluster_features])
            
            cluster_stats.append({
                'cluster_id': cluster_id,
                'size': len(domains),
                'mean_dct': mean_dct,
                'mean_fft': mean_fft,
                'mean_orb': mean_orb,
                'mean_color_var': mean_color_var
            })
        
        df_stats = pd.DataFrame(cluster_stats)
        df_stats = df_stats.sort_values('size', ascending=False).head(20)
        
        # Create subplot
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # Plot 1: Mean DCT by cluster size
        ax = axes[0, 0]
        scatter = ax.scatter(df_stats['size'], df_stats['mean_dct'], 
                           c=df_stats['size'], s=100, alpha=0.6, cmap='viridis')
        ax.set_xlabel('Cluster Size', fontsize=11)
        ax.set_ylabel('Mean DCT Hash Bits', fontsize=11)
        ax.set_title('Mean DCT Hash vs Cluster Size', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Cluster Size')
        
        # Plot 2: Mean FFT by cluster size
        ax = axes[0, 1]
        scatter = ax.scatter(df_stats['size'], df_stats['mean_fft'], 
                           c=df_stats['size'], s=100, alpha=0.6, cmap='plasma')
        ax.set_xlabel('Cluster Size', fontsize=11)
        ax.set_ylabel('Mean FFT Hash Bits (first 64)', fontsize=11)
        ax.set_title('Mean FFT Hash vs Cluster Size', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Cluster Size')
        
        # Plot 3: Mean ORB by cluster size
        ax = axes[1, 0]
        scatter = ax.scatter(df_stats['size'], df_stats['mean_orb'], 
                           c=df_stats['size'], s=100, alpha=0.6, cmap='cool')
        ax.set_xlabel('Cluster Size', fontsize=11)
        ax.set_ylabel('Mean ORB Keypoints', fontsize=11)
        ax.set_title('Mean ORB Keypoints vs Cluster Size', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Cluster Size')
        
        # Plot 4: Mean Color Variance by cluster size
        ax = axes[1, 1]
        scatter = ax.scatter(df_stats['size'], df_stats['mean_color_var'], 
                           c=df_stats['size'], s=100, alpha=0.6, cmap='hot')
        ax.set_xlabel('Cluster Size', fontsize=11)
        ax.set_ylabel('Mean Color Variance', fontsize=11)
        ax.set_title('Mean Color Variance vs Cluster Size', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Cluster Size')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        plt.show()
        
        return df_stats
    
    def create_comprehensive_dashboard(self, output_dir='.'):
        """
        Create comprehensive visualization dashboard
        
        Args:
            output_dir: Directory to save all visualizations
        """
        print("\n" + "=" * 70)
        print("CREATING COMPREHENSIVE VISUALIZATION DASHBOARD")
        print("=" * 70)
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 2D PCA visualization
        print("\n[1/7] 2D PCA Cluster Separation...")
        self.visualize_cluster_separation_2d(
            method='pca',
            save_path=f'{output_dir}/cluster_separation_pca_2d.png'
        )
        
        # 2. 2D t-SNE visualization
        print("\n[2/7] 2D t-SNE Cluster Separation...")
        self.visualize_cluster_separation_2d(
            method='tsne',
            save_path=f'{output_dir}/cluster_separation_tsne_2d.png'
        )
        
        # 3. 3D visualization
        print("\n[3/7] 3D Cluster Visualization...")
        self.visualize_cluster_separation_3d(
            save_path=f'{output_dir}/cluster_separation_3d.png'
        )
        
        # 4. pHash heatmap
        print("\n[4/7] pHash Similarity Heatmap...")
        self.visualize_phash_heatmap(
            max_clusters=20,
            save_path=f'{output_dir}/phash_heatmap.png'
        )
        
        # 5. Feature distributions
        print("\n[5/7] Feature Distributions...")
        self.visualize_feature_distributions(
            save_path=f'{output_dir}/feature_distributions.png'
        )
        
        # 6. Cluster statistics
        print("\n[6/7] Cluster Statistics...")
        df_stats = self.visualize_cluster_statistics(
            save_path=f'{output_dir}/cluster_statistics.png'
        )
        
        # Save statistics to CSV
        df_stats.to_csv(f'{output_dir}/cluster_statistics.csv', index=False)
        print(f"Saved cluster statistics to {output_dir}/cluster_statistics.csv")
        
        # 7. Summary report
        print("\n[7/7] Generating Summary Report...")
        self._generate_summary_report(output_dir)
        
        print("\n" + "=" * 70)
        print("VISUALIZATION DASHBOARD COMPLETE!")
        print("=" * 70)
        print(f"\nAll visualizations saved to: {output_dir}/")
        print("\nGenerated files:")
        print("  - cluster_separation_pca_2d.png    (2D PCA visualization)")
        print("  - cluster_separation_tsne_2d.png   (2D t-SNE visualization)")
        print("  - cluster_separation_3d.html       (Interactive 3D plot)")
        print("  - phash_heatmap.png                (pHash similarity heatmap)")
        print("  - feature_distributions.png        (Feature distributions)")
        print("  - cluster_statistics.png           (Cluster statistics)")
        print("  - cluster_statistics.csv           (Statistics data)")
        print("  - visualization_summary.txt        (Summary report)")
    
    def _generate_summary_report(self, output_dir):
        """Generate text summary report"""
        report_path = f'{output_dir}/visualization_summary.txt'
        
        # Calculate statistics
        total_logos = len(self.features)
        total_clusters = len(self.clusters)
        cluster_sizes = [len(domains) for domains in self.clusters.values()]
        singletons = sum(1 for size in cluster_sizes if size == 1)
        
        with open(report_path, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("CLUSTERING VISUALIZATION SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Timestamp: {self.timestamp}\n\n")
            
            f.write("DATASET OVERVIEW\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total Logos:           {total_logos:,}\n")
            f.write(f"Total Clusters:        {total_clusters:,}\n")
            f.write(f"Singleton Count:       {singletons:,} ({singletons/total_logos*100:.1f}%)\n")
            f.write(f"Average Cluster Size:  {np.mean(cluster_sizes):.2f}\n")
            f.write(f"Median Cluster Size:   {np.median(cluster_sizes):.0f}\n")
            f.write(f"Largest Cluster:       {max(cluster_sizes):,} logos\n")
            f.write(f"Smallest Cluster:      {min(cluster_sizes):,} logos\n\n")
            
            f.write("CLUSTER SIZE DISTRIBUTION\n")
            f.write("-" * 70 + "\n")
            size_bins = [(1, 1, 'Singleton'), (2, 5, 'Small'), (6, 10, 'Medium'), 
                        (11, 20, 'Large'), (21, 1000, 'Huge')]
            for min_size, max_size, label in size_bins:
                count = sum(1 for size in cluster_sizes if min_size <= size <= max_size)
                pct = count / total_clusters * 100
                f.write(f"{label:12} ({min_size:3}-{max_size:3}): {count:4} clusters ({pct:5.1f}%)\n")
            
            f.write("\n")
            f.write("VISUALIZATIONS GENERATED\n")
            f.write("-" * 70 + "\n")
            f.write("1. 2D PCA Cluster Separation (cluster_separation_pca_2d.png)\n")
            f.write("2. 2D t-SNE Cluster Separation (cluster_separation_tsne_2d.png)\n")
            f.write("3. Interactive 3D Visualization (cluster_separation_3d.html)\n")
            f.write("4. pHash Similarity Heatmap (phash_heatmap.png)\n")
            f.write("5. Feature Distributions (feature_distributions.png)\n")
            f.write("6. Cluster Statistics (cluster_statistics.png)\n")
            f.write("7. Statistics Data (cluster_statistics.csv)\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"Summary report saved to {report_path}")


def main():
    """Main function to run visualization"""
    import sys
    import glob
    
    # Find most recent results
    pickle_files = glob.glob('optimized_logo_clustering_results_*_modular.pkl')
    
    if not pickle_files:
        print("No clustering results found!")
        print("Please run clustering first:")
        print("  python -m optimized_clustering.run_clustering <logo_folder>")
        return 1
    
    # Use most recent
    latest_pickle = max(pickle_files, key=lambda x: x.split('_')[4])
    print(f"Loading results from: {latest_pickle}")
    
    # Create visualizer
    viz = ClusteringVisualizer(latest_pickle)
    
    # Create comprehensive dashboard
    output_dir = 'clustering_visualizations'
    viz.create_comprehensive_dashboard(output_dir=output_dir)
    
    print(f"\nAll visualizations complete!")
    print(f"Open {output_dir}/cluster_separation_3d.html in your browser for interactive 3D view")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
