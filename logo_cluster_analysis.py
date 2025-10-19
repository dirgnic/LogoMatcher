"""
Logo Clustering Analysis
Cluster all 4000+ logos using comprehensive Fourier features
"""

import pickle
import numpy as np
import cv2
from fourier_logo_analyzer import FourierLogoAnalyzer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
from datetime import datetime
import os
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

class LogoClusterAnalyzer:
    """Comprehensive logo clustering using Fourier analysis"""
    
    def __init__(self):
        self.analyzer = FourierLogoAnalyzer()
        self.scaler = StandardScaler()
        self.feature_matrix = None
        self.domains = None
        self.cluster_labels = None
        
    def load_all_logos(self, cache_path, max_logos=None):
        """Load all logos from cache"""
        
        print(f"Loading logos from {cache_path}...")
        
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        logo_results = cached_data['logo_results']
        print(f"Found {len(logo_results)} logo entries in cache")
        
        logos_dict = {}
        failed_count = 0
        processed_count = 0
        
        for i, logo_entry in enumerate(logo_results):
            if max_logos and processed_count >= max_logos:
                break
                
            if i % 500 == 0:
                print(f"Processing logo {i+1}/{len(logo_results)}...")
                
            try:
                domain = logo_entry.get('domain', 'unknown')
                logo_bytes = logo_entry.get('logo_data')
                success = logo_entry.get('success', False)
                
                if not success or not logo_bytes:
                    failed_count += 1
                    continue
                
                # Convert bytes to image
                img_array = np.frombuffer(logo_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Resize to standard size
                    img_resized = cv2.resize(img, (128, 128))
                    logos_dict[domain] = img_resized
                    processed_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                continue
        
        print(f"Successfully loaded {len(logos_dict)} valid logo images")
        print(f"Failed to load {failed_count} logos")
        return logos_dict
    
    def extract_features_batch(self, logos_dict, batch_size=100):
        """Extract Fourier features from all logos in batches"""
        
        domains = list(logos_dict.keys())
        num_logos = len(domains)
        
        print(f"Extracting Fourier features from {num_logos} logos...")
        
        # Initialize feature lists
        feature_vectors = []
        valid_domains = []
        
        for i in range(0, num_logos, batch_size):
            batch_end = min(i + batch_size, num_logos)
            batch_domains = domains[i:batch_end]
            
            print(f"Processing batch {i//batch_size + 1}/{(num_logos-1)//batch_size + 1} ({i+1}-{batch_end})...")
            
            for domain in batch_domains:
                try:
                    img = logos_dict[domain]
                    
                    # Extract all Fourier features
                    features = self.analyzer.compute_all_features(img)
                    
                    # Combine all numerical features into a single vector
                    feature_vector = self._combine_features(features)
                    
                    feature_vectors.append(feature_vector)
                    valid_domains.append(domain)
                    
                except Exception as e:
                    print(f"Error processing {domain}: {e}")
                    continue
        
        self.feature_matrix = np.array(feature_vectors)
        self.domains = valid_domains
        
        print(f"Extracted features for {len(valid_domains)} logos")
        print(f"Feature matrix shape: {self.feature_matrix.shape}")
        
        return self.feature_matrix, valid_domains
    
    def _combine_features(self, features):
        """Combine all Fourier features into a single vector"""
        
        combined = []
        
        # FFT features (1024 dimensions)
        combined.extend(features['fft_features'])
        
        # Fourier-Mellin signature (64 dimensions)
        combined.extend(features['fmt_signature'])
        
        # Color-aware FMT (48 dimensions)
        combined.extend(features['color_aware_fmt'].flatten())
        
        # Saliency-weighted FFT (1024 dimensions)
        combined.extend(features['saliency_weighted_fft'])
        
        # Hu moments (7 dimensions)
        combined.extend(features['hu_moments'])
        
        # Perceptual hash as numerical (64 dimensions)
        phash_numerical = [int(bit) for bit in features['phash']]
        combined.extend(phash_numerical)
        
        # Color vector (3 dimensions)
        combined.extend(features['color_vector'])
        
        return np.array(combined, dtype=np.float32)
    
    def preprocess_features(self):
        """Standardize features for clustering"""
        
        print("Preprocessing features...")
        
        # Handle NaN and infinite values
        self.feature_matrix = np.nan_to_num(self.feature_matrix, 
                                          nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Standardize features
        self.feature_matrix = self.scaler.fit_transform(self.feature_matrix)
        
        print(f"Features standardized. Shape: {self.feature_matrix.shape}")
        
    def find_optimal_clusters(self, max_clusters=20):
        """Find optimal number of clusters using silhouette score"""
        
        print("Finding optimal number of clusters...")
        
        silhouette_scores = []
        cluster_range = range(2, min(max_clusters + 1, len(self.domains)//2))
        
        for n_clusters in cluster_range:
            print(f"Testing {n_clusters} clusters...")
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.feature_matrix)
            
            silhouette_avg = silhouette_score(self.feature_matrix, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            
            print(f"  Silhouette score: {silhouette_avg:.4f}")
        
        # Find optimal number of clusters
        optimal_idx = np.argmax(silhouette_scores)
        optimal_clusters = cluster_range[optimal_idx]
        
        print(f"Optimal number of clusters: {optimal_clusters}")
        print(f"Best silhouette score: {silhouette_scores[optimal_idx]:.4f}")
        
        return optimal_clusters, silhouette_scores
    
    def perform_clustering(self, n_clusters=None, method='cosine_similarity'):
        """Perform clustering with methods optimized for logo similarity (NO K-NN)"""
        
        if n_clusters is None and method != 'dbscan':
            n_clusters, _ = self.find_optimal_clusters()
        
        print(f"Performing {method} clustering...")
        
        if method == 'cosine_similarity':
            # Use hierarchical clustering with cosine distance (better for high-dimensional features)
            from sklearn.metrics.pairwise import cosine_distances
            distance_matrix = cosine_distances(self.feature_matrix)
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters if n_clusters else 10, 
                metric='precomputed', 
                linkage='average'
            )
            self.cluster_labels = clusterer.fit_predict(distance_matrix)
            
        elif method == 'spectral':
            # Spectral clustering - good for non-convex clusters
            clusterer = SpectralClustering(
                n_clusters=n_clusters if n_clusters else 10,
                random_state=42,
                affinity='cosine'  # Use cosine similarity
            )
            self.cluster_labels = clusterer.fit_predict(self.feature_matrix)
            
        elif method == 'kmeans_cosine':
            # K-means with cosine distance preprocessing
            from sklearn.preprocessing import normalize
            normalized_features = normalize(self.feature_matrix, norm='l2')
            clusterer = KMeans(n_clusters=n_clusters if n_clusters else 10, random_state=42, n_init=10)
            self.cluster_labels = clusterer.fit_predict(normalized_features)
            
        elif method == 'hierarchical_ward':
            # Ward linkage hierarchical (good for similar-sized clusters)
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters if n_clusters else 10, 
                linkage='ward'
            )
            self.cluster_labels = clusterer.fit_predict(self.feature_matrix)
            
        elif method == 'dbscan_adaptive':
            # DBSCAN with adaptive eps based on feature distribution
            from sklearn.neighbors import NearestNeighbors
            # Use 4th nearest neighbor distance for eps estimation
            nbrs = NearestNeighbors(n_neighbors=4, metric='cosine')
            nbrs.fit(self.feature_matrix)
            distances, _ = nbrs.kneighbors(self.feature_matrix)
            eps = np.percentile(distances[:, -1], 80)  # 80th percentile
            
            clusterer = DBSCAN(eps=eps, min_samples=5, metric='cosine')
            self.cluster_labels = clusterer.fit_predict(self.feature_matrix)
            
        else:
            raise ValueError(f"Unknown clustering method: {method}")
        
        # Count clusters
        unique_labels = np.unique(self.cluster_labels)
        n_clusters_found = len(unique_labels)
        
        if 'dbscan' in method:
            n_noise = list(self.cluster_labels).count(-1)
            print(f"Found {n_clusters_found-1} clusters ({n_noise} noise points)")
        else:
            print(f"Created {n_clusters_found} clusters")
        
        return self.cluster_labels
    
    def analyze_clusters(self):
        """Analyze cluster composition and characteristics"""
        
        print("\nCluster Analysis:")
        print("=" * 50)
        
        clusters_df = pd.DataFrame({
            'domain': self.domains,
            'cluster': self.cluster_labels
        })
        
        # Cluster size distribution
        cluster_counts = clusters_df['cluster'].value_counts().sort_index()
        
        print(f"Cluster size distribution:")
        for cluster_id, count in cluster_counts.items():
            if cluster_id == -1:  # Noise in DBSCAN
                print(f"  Noise: {count} logos")
            else:
                print(f"  Cluster {cluster_id}: {count} logos")
        
        # Show sample domains from each cluster
        print(f"\nSample domains from each cluster:")
        for cluster_id in sorted(cluster_counts.index):
            if cluster_id == -1:
                continue
                
            cluster_domains = clusters_df[clusters_df['cluster'] == cluster_id]['domain'].tolist()
            sample_domains = cluster_domains[:5]  # Show first 5
            
            print(f"\nCluster {cluster_id} ({len(cluster_domains)} logos):")
            for i, domain in enumerate(sample_domains):
                brand_name = self._extract_brand_name(domain)
                print(f"  {i+1}. {brand_name} ({domain})")
            
            if len(cluster_domains) > 5:
                print(f"  ... and {len(cluster_domains) - 5} more")
        
        return clusters_df
    
    def _extract_brand_name(self, domain):
        """Extract brand name from domain"""
        try:
            if domain.startswith(('http://', 'https://')):
                parsed = urlparse(domain)
                domain = parsed.netloc
            
            # Remove www. and common TLDs
            domain = domain.replace('www.', '')
            parts = domain.split('.')
            if len(parts) > 1:
                return parts[0].capitalize()
            return domain.capitalize()
        except:
            return domain
    
    def visualize_clusters(self, save_path=None):
        """Create cluster visualization using PCA"""
        
        print("Creating cluster visualization...")
        
        # Reduce dimensionality for visualization
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(self.feature_matrix)
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        unique_labels = np.unique(self.cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, cluster_id in enumerate(unique_labels):
            if cluster_id == -1:  # Noise points
                color = 'black'
                label = 'Noise'
                alpha = 0.5
            else:
                color = colors[i]
                label = f'Cluster {cluster_id}'
                alpha = 0.7
            
            mask = self.cluster_labels == cluster_id
            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                       c=[color], label=label, alpha=alpha, s=30)
        
        plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.title('Logo Clusters - Fourier Feature Space (PCA Projection)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"logo_clusters_analysis_{timestamp}.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Cluster visualization saved to: {save_path}")
        
        return save_path
    
    def save_results(self, output_prefix=None):
        """Save clustering results"""
        
        if output_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"logo_clustering_{timestamp}"
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'domain': self.domains,
            'cluster': self.cluster_labels,
            'brand_name': [self._extract_brand_name(d) for d in self.domains]
        })
        
        # Save to CSV
        csv_path = f"{output_prefix}.csv"
        results_df.to_csv(csv_path, index=False)
        
        # Save detailed results
        results_dict = {
            'domains': self.domains,
            'cluster_labels': self.cluster_labels,
            'feature_matrix': self.feature_matrix,
            'scaler': self.scaler,
            'clustering_params': {
                'n_clusters': len(np.unique(self.cluster_labels)),
                'n_logos': len(self.domains),
                'feature_dimensions': self.feature_matrix.shape[1]
            }
        }
        
        pkl_path = f"{output_prefix}.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(results_dict, f)
        
        print(f"Results saved:")
        print(f"  CSV: {csv_path}")
        print(f"  Detailed: {pkl_path}")
        
        return csv_path, pkl_path

def main():
    """Main clustering pipeline"""
    
    print("=== Logo Clustering Analysis ===\n")
    
    # Initialize analyzer
    analyzer = LogoClusterAnalyzer()
    
    # Load all logos
    cache_path = "comprehensive_logo_extraction_fast_results.pkl"
    logos_dict = analyzer.load_all_logos(cache_path)
    
    # Extract features
    feature_matrix, domains = analyzer.extract_features_batch(logos_dict)
    
    # Preprocess features
    analyzer.preprocess_features()
    
    # Perform clustering
    print("\n" + "="*50)
    print("CLUSTERING ANALYSIS")
    print("="*50)
    
    # Try different clustering methods (NO K-NN - using proper clustering algorithms)
    methods = [
        'cosine_similarity',    # Hierarchical with cosine distance
        'spectral',             # Spectral clustering 
        'kmeans_cosine',        # K-means with cosine preprocessing
        'dbscan_adaptive'       # DBSCAN with adaptive parameters
    ]
    
    for method in methods:
        print(f"\n--- {method.upper().replace('_', ' ')} CLUSTERING ---")
        
        cluster_labels = analyzer.perform_clustering(method=method)
        clusters_df = analyzer.analyze_clusters()
        
        # Visualize
        viz_path = analyzer.visualize_clusters(f"clusters_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        # Save results
        csv_path, pkl_path = analyzer.save_results(f"clustering_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        print(f"\n{method.replace('_', ' ').title()} clustering complete!")
        print(f"Generated {len(np.unique(cluster_labels))} clusters from {len(domains)} logos")
    
    print(f"\n✓ Clustering analysis complete!")
    print(f"✓ Check the generated files for detailed results and visualizations.")

if __name__ == "__main__":
    main()
