"""
Advanced Logo Clustering Analysis
Force more granular clustering to find meaningful logo groups
"""

import pickle
import numpy as np
import cv2
from fourier_logo_analyzer import FourierLogoAnalyzer
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
from datetime import datetime
import seaborn as sns
import os

def load_clustering_results():
    """Load the previous clustering results"""
    
    # Load the most recent results
    pkl_files = [f for f in os.listdir('.') if f.startswith('clustering_') and f.endswith('.pkl')]
    if not pkl_files:
        print("No clustering results found!")
        return None
    
    latest_file = sorted(pkl_files)[-1]
    print(f"Loading results from: {latest_file}")
    
    with open(latest_file, 'rb') as f:
        results = pickle.load(f)
    
    return results

def force_granular_clustering(feature_matrix, domains, n_clusters_list=[5, 10, 15, 20, 25, 30]):
    """Force more granular clustering to find meaningful groups"""
    
    print("Performing forced granular clustering...")
    
    results = {}
    
    for n_clusters in n_clusters_list:
        print(f"\nAnalyzing {n_clusters} clusters...")
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)
        
        # Calculate silhouette score
        if n_clusters < len(domains):
            silhouette_avg = silhouette_score(feature_matrix, cluster_labels)
        else:
            silhouette_avg = -1
        
        # Analyze cluster composition
        clusters_df = pd.DataFrame({
            'domain': domains,
            'cluster': cluster_labels
        })
        
        cluster_counts = clusters_df['cluster'].value_counts().sort_index()
        
        print(f"  Silhouette score: {silhouette_avg:.4f}")
        print(f"  Cluster sizes: {dict(cluster_counts)}")
        
        # Look for interesting clusters (not too big, not too small)
        interesting_clusters = []
        for cluster_id, count in cluster_counts.items():
            if 5 <= count <= 100:  # Reasonable size clusters
                cluster_domains = clusters_df[clusters_df['cluster'] == cluster_id]['domain'].tolist()
                interesting_clusters.append({
                    'cluster_id': cluster_id,
                    'size': count,
                    'domains': cluster_domains[:10]  # First 10 for analysis
                })
        
        print(f"  Found {len(interesting_clusters)} interesting clusters")
        
        results[n_clusters] = {
            'labels': cluster_labels,
            'silhouette_score': silhouette_avg,
            'cluster_counts': dict(cluster_counts),
            'interesting_clusters': interesting_clusters,
            'clusters_df': clusters_df
        }
    
    return results

def analyze_brand_patterns(clustering_results):
    """Analyze clustering patterns to find brand groupings"""
    
    print("\n" + "="*60)
    print("BRAND PATTERN ANALYSIS")
    print("="*60)
    
    best_clustering = None
    best_score = -1
    
    for n_clusters, results in clustering_results.items():
        interesting_clusters = results['interesting_clusters']
        
        if len(interesting_clusters) > best_score:
            best_score = len(interesting_clusters)
            best_clustering = (n_clusters, results)
    
    if best_clustering is None:
        print("No interesting clustering found!")
        return
    
    n_clusters, results = best_clustering
    print(f"\nBest clustering: {n_clusters} clusters with {best_score} interesting groups")
    print(f"Silhouette score: {results['silhouette_score']:.4f}")
    
    print(f"\nInteresting brand clusters:")
    for i, cluster_info in enumerate(results['interesting_clusters']):
        cluster_id = cluster_info['cluster_id']
        size = cluster_info['size']
        domains = cluster_info['domains']
        
        print(f"\nCluster {cluster_id} ({size} logos):")
        
        # Extract brand names and look for patterns
        brand_names = []
        for domain in domains:
            brand_name = extract_brand_name(domain)
            brand_names.append(brand_name)
        
        # Look for common patterns
        common_words = find_common_words(brand_names)
        
        for j, (domain, brand) in enumerate(zip(domains, brand_names)):
            print(f"  {j+1}. {brand} ({domain})")
        
        if common_words:
            print(f"  Common patterns: {', '.join(common_words)}")
    
    return best_clustering

def extract_brand_name(domain):
    """Extract brand name from domain"""
    try:
        if domain.startswith(('http://', 'https://')):
            from urllib.parse import urlparse
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

def find_common_words(brand_names):
    """Find common words in brand names"""
    
    if len(brand_names) < 2:
        return []
    
    # Split brand names into words
    all_words = []
    for brand in brand_names:
        words = brand.lower().replace('-', ' ').split()
        all_words.extend(words)
    
    # Count word frequency
    from collections import Counter
    word_counts = Counter(all_words)
    
    # Find words that appear in multiple brands
    common_words = [word for word, count in word_counts.items() 
                   if count >= 2 and len(word) > 2]
    
    return common_words[:5]  # Top 5 common words

def create_detailed_visualization(clustering_results, feature_matrix, domains):
    """Create detailed visualization of clustering results"""
    
    print("Creating detailed cluster visualizations...")
    
    # Create PCA projection
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(feature_matrix)
    
    # Create subplots for different cluster counts
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Logo Clustering Analysis - Different Granularities', fontsize=16, fontweight='bold')
    
    cluster_counts = [5, 10, 15, 20, 25, 30]
    
    for i, n_clusters in enumerate(cluster_counts):
        row = i // 3
        col = i % 3
        
        if n_clusters in clustering_results:
            labels = clustering_results[n_clusters]['labels']
            silhouette = clustering_results[n_clusters]['silhouette_score']
            
            # Plot clusters
            unique_labels = np.unique(labels)
            colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
            
            for j, cluster_id in enumerate(unique_labels):
                mask = labels == cluster_id
                axes[row, col].scatter(features_2d[mask, 0], features_2d[mask, 1], 
                                     c=[colors[j]], alpha=0.6, s=20, 
                                     label=f'C{cluster_id}' if len(unique_labels) <= 10 else None)
            
            axes[row, col].set_title(f'{n_clusters} Clusters (Silhouette: {silhouette:.3f})')
            axes[row, col].grid(True, alpha=0.3)
            
            if len(unique_labels) <= 10:
                axes[row, col].legend(fontsize=8)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"detailed_clustering_analysis_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Detailed visualization saved to: {save_path}")
    
    return save_path

def main():
    """Main advanced clustering analysis"""
    
    print("=== Advanced Logo Clustering Analysis ===\n")
    
    # Load previous clustering results
    import os
    results = load_clustering_results()
    
    if results is None:
        print("Please run logo_cluster_analysis.py first!")
        return
    
    feature_matrix = results['feature_matrix']
    domains = results['domains']
    
    print(f"Loaded {len(domains)} logos with {feature_matrix.shape[1]} features")
    
    # Perform granular clustering
    clustering_results = force_granular_clustering(feature_matrix, domains)
    
    # Analyze brand patterns
    best_clustering = analyze_brand_patterns(clustering_results)
    
    # Create detailed visualization
    viz_path = create_detailed_visualization(clustering_results, feature_matrix, domains)
    
    # Save the best clustering result
    if best_clustering:
        n_clusters, best_results = best_clustering
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed CSV
        csv_path = f"detailed_clustering_{n_clusters}clusters_{timestamp}.csv"
        best_results['clusters_df'].to_csv(csv_path, index=False)
        
        print(f"\nBest clustering results saved to: {csv_path}")
        
        # Print summary
        print(f"\n" + "="*50)
        print("CLUSTERING SUMMARY")
        print("="*50)
        print(f"Total logos analyzed: {len(domains)}")
        print(f"Feature dimensions: {feature_matrix.shape[1]}")
        print(f"Optimal clusters: {n_clusters}")
        print(f"Silhouette score: {best_results['silhouette_score']:.4f}")
        print(f"Interesting brand groups found: {len(best_results['interesting_clusters'])}")
    
    print(f"\n✓ Advanced clustering analysis complete!")

if __name__ == "__main__":
    main()
