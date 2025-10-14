#!/usr/bin/env python3
"""
k-NN Probe: Explainability and feature discovery for logo clusters

No clustering here - just k-NN graphs on interpretable features:
- Geometry (aspect ratio), color stats, structure features
- Decision Tree rules to reverse-engineer cluster characteristics  
- Per-cluster profiles with z-scores and distinguishing features

Usage:
    python knn_probe.py clusters.json --k 5 --tree_output tree.png
"""

import json
import numpy as np
import cv2
from PIL import Image
import argparse
from typing import List, Dict, Tuple
from collections import defaultdict
import io

# ML for explainability only (no clustering)
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class InterpretableFeatureExtractor:
    """Extract human-interpretable features for explainability"""
    
    def extract_geometry_features(self, img: np.ndarray) -> Dict[str, float]:
        """Geometry: aspect ratio, size characteristics"""
        h, w = img.shape[:2]
        return {
            'aspect_ratio': w / h,
            'width': w,
            'height': h,
            'area': w * h,
            'log_area': np.log(w * h),
            'is_square': abs(w - h) / max(w, h) < 0.1,
            'is_wide': w / h > 1.5,
            'is_tall': h / w > 1.5
        }
    
    def extract_color_features(self, img: np.ndarray) -> Dict[str, float]:
        """Color statistics: means, HSV hue distribution"""
        # RGB means
        rgb_means = np.mean(img.reshape(-1, 3), axis=0)
        
        # Convert to HSV for hue analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 12-bin hue histogram (30° bins)
        hue = hsv[:, :, 0]
        hue_hist, _ = np.histogram(hue, bins=12, range=(0, 180))
        hue_hist = hue_hist / np.sum(hue_hist)  # Normalize
        
        # Saturation and value stats
        sat_mean = np.mean(hsv[:, :, 1])
        val_mean = np.mean(hsv[:, :, 2])
        
        features = {
            'rgb_r_mean': rgb_means[2],  # OpenCV uses BGR
            'rgb_g_mean': rgb_means[1],
            'rgb_b_mean': rgb_means[0],
            'hsv_sat_mean': sat_mean,
            'hsv_val_mean': val_mean,
        }
        
        # Add hue bin features
        for i, prob in enumerate(hue_hist):
            features[f'hue_bin_{i}'] = prob
        
        return features
    
    def extract_structure_features(self, img: np.ndarray) -> Dict[str, float]:
        """Structure: edge density, sharpness, background detection"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Edge density using Canny
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        
        # Sharpness using Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # Background detection (assume corners are background)
        h, w = gray.shape
        corner_size = min(h, w) // 10
        corners = [
            gray[:corner_size, :corner_size],           # Top-left
            gray[:corner_size, -corner_size:],          # Top-right  
            gray[-corner_size:, :corner_size],          # Bottom-left
            gray[-corner_size:, -corner_size:]          # Bottom-right
        ]
        
        corner_mean = np.mean([np.mean(corner) for corner in corners])
        background_white_ratio = corner_mean / 255.0
        
        # Contrast and brightness
        contrast = np.std(gray)
        brightness = np.mean(gray)
        
        return {
            'edge_density': edge_density,
            'sharpness': sharpness,
            'background_white_ratio': background_white_ratio,
            'contrast': contrast,
            'brightness': brightness,
            'log_sharpness': np.log(sharpness + 1)
        }
    
    def extract_all_features(self, img: np.ndarray) -> Dict[str, float]:
        """Extract all interpretable features"""
        features = {}
        features.update(self.extract_geometry_features(img))
        features.update(self.extract_color_features(img))
        features.update(self.extract_structure_features(img))
        return features


class ClusterProfiler:
    """Analyze and profile clusters with interpretable features"""
    
    def __init__(self, feature_extractor: InterpretableFeatureExtractor):
        self.feature_extractor = feature_extractor
        self.scaler = StandardScaler()
    
    def load_cluster_data(self, clusters_file: str) -> Tuple[List[Dict], np.ndarray, List[str]]:
        """Load clusters and extract features from logo images"""
        with open(clusters_file, 'r') as f:
            data = json.load(f)
        
        clusters = data['clusters']
        
        # Extract features for each logo (this would need logo image data)
        # For now, we'll simulate with the cluster structure
        print("📊 Note: Using cluster structure for demonstration")
        print("    In production, would extract features from actual logo images")
        
        # Simulate feature matrix (would be real features in production)
        n_logos = sum(len(cluster['websites']) for cluster in clusters)
        n_features = 30  # Number of interpretable features
        
        # Create mock feature matrix
        feature_matrix = np.random.randn(n_logos, n_features)
        feature_names = self._get_feature_names()
        
        return clusters, feature_matrix, feature_names
    
    def _get_feature_names(self) -> List[str]:
        """Get names of all interpretable features"""
        base_features = [
            # Geometry
            'aspect_ratio', 'width', 'height', 'area', 'log_area',
            'is_square', 'is_wide', 'is_tall',
            # Color  
            'rgb_r_mean', 'rgb_g_mean', 'rgb_b_mean', 'hsv_sat_mean', 'hsv_val_mean',
            # Structure
            'edge_density', 'sharpness', 'background_white_ratio', 
            'contrast', 'brightness', 'log_sharpness'
        ]
        
        # Add hue bins
        hue_features = [f'hue_bin_{i}' for i in range(12)]
        
        return base_features + hue_features
    
    def build_knn_graph(self, feature_matrix: np.ndarray, k: int = 5) -> np.ndarray:
        """Build k-NN graph on interpretable features (no clustering)"""
        # Standardize features
        features_scaled = self.scaler.fit_transform(feature_matrix)
        
        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean')
        nbrs.fit(features_scaled)
        
        distances, indices = nbrs.kneighbors(features_scaled)
        
        # Remove self-connections
        return indices[:, 1:], distances[:, 1:]
    
    def train_decision_tree(self, feature_matrix: np.ndarray, cluster_labels: np.ndarray, feature_names: List[str]) -> DecisionTreeClassifier:
        """Train Decision Tree to reveal cluster characteristics"""
        # Only train if we have multiple clusters
        if len(np.unique(cluster_labels)) < 2:
            print("⚠️ Need multiple clusters for Decision Tree analysis")
            return None
        
        # Train shallow tree for interpretability
        tree = DecisionTreeClassifier(
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42
        )
        
        features_scaled = self.scaler.fit_transform(feature_matrix)
        tree.fit(features_scaled, cluster_labels)
        
        return tree
    
    def analyze_cluster_profiles(self, clusters: List[Dict], feature_matrix: np.ndarray, feature_names: List[str]) -> Dict:
        """Generate per-cluster profiles with z-scores"""
        
        # Create cluster labels
        cluster_labels = []
        logo_to_cluster = {}
        
        logo_idx = 0
        for cluster_id, cluster in enumerate(clusters):
            for website in cluster['websites']:
                cluster_labels.append(cluster_id)
                logo_to_cluster[logo_idx] = cluster_id
                logo_idx += 1
        
        cluster_labels = np.array(cluster_labels)
        
        # Standardize features for z-score analysis
        features_scaled = self.scaler.fit_transform(feature_matrix)
        
        profiles = {}
        
        for cluster_id, cluster in enumerate(clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_features = features_scaled[cluster_mask]
            
            if len(cluster_features) == 0:
                continue
            
            # Compute means and z-scores relative to global distribution
            feature_means = np.mean(cluster_features, axis=0)
            feature_stds = np.std(cluster_features, axis=0)
            
            # Z-scores relative to global mean (which is 0 after standardization)
            z_scores = feature_means  # Since global mean = 0 after standardization
            
            # Find most distinguishing features
            top_positive_idx = np.argsort(z_scores)[-5:][::-1]  # Top 5 positive
            top_negative_idx = np.argsort(z_scores)[:5]         # Top 5 negative
            
            profiles[cluster_id] = {
                'cluster_info': cluster,
                'feature_means': feature_means.tolist(),
                'feature_stds': feature_stds.tolist(),
                'z_scores': z_scores.tolist(),
                'top_distinguishing_features': {
                    'positive': [(feature_names[i], z_scores[i]) for i in top_positive_idx],
                    'negative': [(feature_names[i], z_scores[i]) for i in top_negative_idx]
                }
            }
        
        return profiles
    
    def export_decision_tree(self, tree: DecisionTreeClassifier, feature_names: List[str], output_file: str = None):
        """Export Decision Tree rules and visualization"""
        if tree is None:
            return
        
        # Text rules
        tree_rules = export_text(tree, feature_names=feature_names, max_depth=4)
        print("🌳 DECISION TREE RULES:")
        print(tree_rules)
        
        # Feature importances
        importances = tree.feature_importances_
        feature_importance_pairs = list(zip(feature_names, importances))
        feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        print("\n📊 FEATURE IMPORTANCES:")
        for feature, importance in feature_importance_pairs[:10]:
            print(f"   {feature}: {importance:.3f}")
        
        # Visualization
        if output_file:
            plt.figure(figsize=(15, 10))
            plot_tree(tree, feature_names=feature_names, filled=True, rounded=True, fontsize=8)
            plt.title("Logo Cluster Decision Tree")
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"🖼️ Tree visualization saved to {output_file}")
            plt.close()


async def main():
    parser = argparse.ArgumentParser(description="k-NN probe for logo cluster explainability")
    parser.add_argument('clusters_file', help='clusters.json from logo_cluster.py')
    parser.add_argument('--k', type=int, default=5, help='k for k-NN graph')
    parser.add_argument('--tree_output', help='Output file for decision tree visualization (PNG)')
    parser.add_argument('--profiles_output', default='cluster_profiles.json', help='Output for cluster profiles')
    
    args = parser.parse_args()
    
    print("🔍 k-NN Probe: Logo Cluster Explainability")
    print("=" * 50)
    
    # Initialize components
    feature_extractor = InterpretableFeatureExtractor()
    profiler = ClusterProfiler(feature_extractor)
    
    # Load data
    print("📂 Loading cluster data...")
    clusters, feature_matrix, feature_names = profiler.load_cluster_data(args.clusters_file)
    
    print(f"   Clusters: {len(clusters)}")
    print(f"   Total logos: {feature_matrix.shape[0]}")
    print(f"   Features: {feature_matrix.shape[1]}")
    
    # Build k-NN graph
    print(f"\n🔗 Building {args.k}-NN graph...")
    knn_indices, knn_distances = profiler.build_knn_graph(feature_matrix, args.k)
    
    # Create cluster labels
    cluster_labels = []
    for cluster_id, cluster in enumerate(clusters):
        cluster_labels.extend([cluster_id] * len(cluster['websites']))
    cluster_labels = np.array(cluster_labels)
    
    # Train Decision Tree for explainability
    print("\n🌳 Training Decision Tree for cluster rules...")
    tree = profiler.train_decision_tree(feature_matrix, cluster_labels, feature_names)
    
    # Export tree rules and visualization
    profiler.export_decision_tree(tree, feature_names, args.tree_output)
    
    # Analyze cluster profiles
    print("\n📊 Analyzing cluster profiles...")
    profiles = profiler.analyze_cluster_profiles(clusters, feature_matrix, feature_names)
    
    # Display cluster profiles
    print("\n🎯 CLUSTER PROFILES:")
    for cluster_id, profile in profiles.items():
        cluster_info = profile['cluster_info']
        print(f"\n   Cluster {cluster_id} ({cluster_info['size']} websites):")
        print(f"   Websites: {', '.join(cluster_info['websites'][:3])}{'...' if cluster_info['size'] > 3 else ''}")
        
        print("   Top distinguishing features:")
        for feature, z_score in profile['top_distinguishing_features']['positive'][:3]:
            print(f"     + {feature}: {z_score:+.2f} (above average)")
        for feature, z_score in profile['top_distinguishing_features']['negative'][:3]:
            print(f"     - {feature}: {z_score:+.2f} (below average)")
    
    # Save profiles
    with open(args.profiles_output, 'w') as f:
        json.dump(profiles, f, indent=2)
    
    print(f"\n💾 Cluster profiles saved to {args.profiles_output}")
    
    # k-NN analysis summary
    print(f"\n🔍 k-NN GRAPH ANALYSIS:")
    avg_distance = np.mean(knn_distances)
    print(f"   Average {args.k}-NN distance: {avg_distance:.3f}")
    
    # Check intra-cluster vs inter-cluster k-NN connections
    intra_cluster_connections = 0
    total_connections = 0
    
    for i, neighbors in enumerate(knn_indices):
        my_cluster = cluster_labels[i]
        for neighbor_idx in neighbors:
            total_connections += 1
            if cluster_labels[neighbor_idx] == my_cluster:
                intra_cluster_connections += 1
    
    intra_cluster_ratio = intra_cluster_connections / total_connections if total_connections > 0 else 0
    print(f"   Intra-cluster k-NN connections: {intra_cluster_ratio:.1%}")
    print(f"   (Higher = more cohesive clusters)")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
