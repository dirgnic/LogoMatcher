"""
Logo Similarity Matcher - Non-ML Approach
Solving logo clustering without ML algorithms (no DBSCAN, K-means, etc.)
Using pure mathematical similarity measures and geometric analysis

Challenge Requirements:
- Match and group websites by logo similarity
- Extract logos for >97% of websites  
- No ML clustering algorithms (DBSCAN, K-means, etc.)
- Output groups of similar websites
"""

import pickle
import numpy as np
import cv2
from fourier_logo_analyzer import FourierLogoAnalyzer
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from typing import List, Dict, Tuple
import os
from urllib.parse import urlparse
import hashlib
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class NonMLLogoMatcher:
    """
    Logo similarity matching using pure mathematical approaches
    No machine learning clustering algorithms
    """
    
    def __init__(self):
        self.analyzer = FourierLogoAnalyzer()
        self.logos_features = {}
        self.similarity_matrix = None
        self.similarity_threshold = 0.85  # High threshold for strict matching
        
    def load_logos_from_cache(self, cache_path):
        """Load all logos from cache"""
        
        print(f"Loading logos from {cache_path}...")
        
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        logo_results = cached_data['logo_results']
        print(f"Found {len(logo_results)} logo entries in cache")
        
        logos_dict = {}
        failed_count = 0
        
        for i, logo_entry in enumerate(logo_results):
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
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                continue
        
        success_rate = (len(logos_dict) / len(logo_results)) * 100
        print(f"Successfully loaded {len(logos_dict)} valid logo images")
        print(f"Failed to load {failed_count} logos")
        print(f"Success rate: {success_rate:.2f}% (target: >97%)")
        
        return logos_dict
    
    def extract_comprehensive_features(self, logos_dict):
        """Extract comprehensive features for non-ML comparison"""
        
        print("Extracting comprehensive features...")
        
        for i, (domain, img) in enumerate(logos_dict.items()):
            if i % 100 == 0:
                print(f"Processing features {i+1}/{len(logos_dict)}...")
            
            try:
                # Use Fourier analyzer for comprehensive features
                features = self.analyzer.compute_all_features(img)
                
                # Add additional geometric features
                additional_features = self._compute_geometric_features(img)
                features.update(additional_features)
                
                self.logos_features[domain] = features
                
            except Exception as e:
                print(f"Error processing {domain}: {e}")
                continue
        
        print(f"Extracted features for {len(self.logos_features)} logos")
    
    def _compute_geometric_features(self, img):
        """Compute additional geometric features for similarity"""
        
        features = {}
        
        # Color histogram features
        hist_b = cv2.calcHist([img], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [32], [0, 256])
        hist_r = cv2.calcHist([img], [2], None, [32], [0, 256])
        
        features['color_hist'] = np.concatenate([hist_b.flatten(), 
                                               hist_g.flatten(), 
                                               hist_r.flatten()])
        
        # Edge density features
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Contour features
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            features['contour_area'] = cv2.contourArea(largest_contour)
            features['contour_perimeter'] = cv2.arcLength(largest_contour, True)
        else:
            features['contour_area'] = 0
            features['contour_perimeter'] = 0
        
        # Texture features using Local Binary Pattern
        from skimage.feature import local_binary_pattern
        lbp = local_binary_pattern(gray, 24, 3, method='uniform')
        features['lbp_hist'] = np.histogram(lbp, bins=26)[0]
        
        return features
    
    def compute_pairwise_similarity_matrix(self):
        """Compute similarity matrix using mathematical measures (no ML)"""
        
        domains = list(self.logos_features.keys())
        n_logos = len(domains)
        
        print(f"Computing pairwise similarities for {n_logos} logos...")
        
        # Initialize similarity matrix
        similarity_matrix = np.zeros((n_logos, n_logos))
        
        for i in range(n_logos):
            if i % 100 == 0:
                print(f"Computing similarities {i+1}/{n_logos}...")
            
            for j in range(i, n_logos):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    # Compute comprehensive similarity
                    sim = self._compute_comprehensive_similarity(
                        self.logos_features[domains[i]],
                        self.logos_features[domains[j]]
                    )
                    similarity_matrix[i][j] = sim
                    similarity_matrix[j][i] = sim  # Symmetric matrix
        
        self.similarity_matrix = similarity_matrix
        self.domains = domains
        
        return similarity_matrix
    
    def _compute_comprehensive_similarity(self, features1, features2):
        """Compute comprehensive similarity without ML algorithms"""
        
        similarities = {}
        
        # 1. Perceptual hash similarity (Hamming distance)
        phash1, phash2 = features1['phash'], features2['phash']
        hamming_dist = sum(c1 != c2 for c1, c2 in zip(phash1, phash2))
        similarities['phash'] = 1.0 - (hamming_dist / 64.0)
        
        # 2. Fourier feature similarity (cosine similarity)
        fft1, fft2 = features1['fft_features'], features2['fft_features']
        cos_sim_fft = np.dot(fft1, fft2) / (np.linalg.norm(fft1) * np.linalg.norm(fft2))
        similarities['fft'] = max(0, cos_sim_fft)  # Ensure non-negative
        
        # 3. Fourier-Mellin similarity (correlation-based)
        fmt1, fmt2 = features1['fmt_signature'], features2['fmt_signature']
        correlation = np.corrcoef(fmt1, fmt2)[0, 1]
        similarities['fourier_mellin'] = max(0, correlation) if not np.isnan(correlation) else 0
        
        # 4. Color histogram similarity (intersection method)
        hist1, hist2 = features1['color_hist'], features2['color_hist']
        hist_intersection = np.sum(np.minimum(hist1, hist2)) / np.sum(hist1)
        similarities['color_hist'] = hist_intersection
        
        # 5. Hu moments similarity (Euclidean distance based)
        hu1, hu2 = features1['hu_moments'], features2['hu_moments']
        hu_dist = np.linalg.norm(hu1 - hu2)
        similarities['hu_moments'] = np.exp(-hu_dist)  # Convert distance to similarity
        
        # 6. Edge density similarity
        edge1, edge2 = features1.get('edge_density', 0), features2.get('edge_density', 0)
        edge_sim = 1.0 - abs(edge1 - edge2)
        similarities['edge_density'] = max(0, edge_sim)
        
        # 7. LBP texture similarity
        lbp1, lbp2 = features1.get('lbp_hist', np.zeros(26)), features2.get('lbp_hist', np.zeros(26))
        lbp_intersection = np.sum(np.minimum(lbp1, lbp2)) / max(np.sum(lbp1), 1)
        similarities['lbp_texture'] = lbp_intersection
        
        # Weighted combination (no ML - just mathematical averaging)
        weights = {
            'phash': 0.20,
            'fft': 0.25,
            'fourier_mellin': 0.20,
            'color_hist': 0.15,
            'hu_moments': 0.10,
            'edge_density': 0.05,
            'lbp_texture': 0.05
        }
        
        overall_similarity = sum(similarities[key] * weights[key] for key in similarities.keys())
        
        return overall_similarity
    
    def find_similar_groups_non_ml(self):
        """Find similar logo groups using graph-based approach (no ML clustering)"""
        
        print("Finding similar groups using graph-based approach...")
        
        n_logos = len(self.domains)
        
        # Create adjacency list based on similarity threshold
        adjacency_list = defaultdict(list)
        
        for i in range(n_logos):
            for j in range(i + 1, n_logos):
                if self.similarity_matrix[i][j] >= self.similarity_threshold:
                    adjacency_list[i].append(j)
                    adjacency_list[j].append(i)
        
        # Find connected components using DFS (no ML)
        visited = [False] * n_logos
        groups = []
        
        def dfs(node, current_group):
            visited[node] = True
            current_group.append(node)
            
            for neighbor in adjacency_list[node]:
                if not visited[neighbor]:
                    dfs(neighbor, current_group)
        
        # Find all connected components
        for i in range(n_logos):
            if not visited[i]:
                current_group = []
                dfs(i, current_group)
                groups.append(current_group)
        
        # Convert indices to domains
        domain_groups = []
        for group in groups:
            domain_group = [self.domains[idx] for idx in group]
            domain_groups.append(domain_group)
        
        # Sort groups by size (largest first)
        domain_groups.sort(key=len, reverse=True)
        
        print(f"Found {len(domain_groups)} groups")
        
        return domain_groups
    
    def analyze_groups(self, groups):
        """Analyze the found groups"""
        
        print("\n" + "="*60)
        print("LOGO SIMILARITY GROUPS ANALYSIS")
        print("="*60)
        
        total_logos = sum(len(group) for group in groups)
        multi_logo_groups = [group for group in groups if len(group) > 1]
        single_logo_groups = [group for group in groups if len(group) == 1]
        
        print(f"Total logos processed: {total_logos}")
        print(f"Total groups found: {len(groups)}")
        print(f"Groups with multiple logos: {len(multi_logo_groups)}")
        print(f"Unique logos (single groups): {len(single_logo_groups)}")
        
        # Analyze multi-logo groups
        if multi_logo_groups:
            print(f"\nMulti-logo groups:")
            for i, group in enumerate(multi_logo_groups[:10]):  # Show top 10
                print(f"\nGroup {i+1} ({len(group)} logos):")
                for j, domain in enumerate(group):
                    brand_name = self._extract_brand_name(domain)
                    print(f"  {j+1}. {brand_name} ({domain})")
                
                if i >= 9:  # Show max 10 groups
                    remaining = len(multi_logo_groups) - 10
                    if remaining > 0:
                        print(f"\n... and {remaining} more groups")
                    break
        
        # Group size distribution
        group_sizes = [len(group) for group in groups]
        size_counts = {}
        for size in group_sizes:
            size_counts[size] = size_counts.get(size, 0) + 1
        
        print(f"\nGroup size distribution:")
        for size in sorted(size_counts.keys(), reverse=True):
            count = size_counts[size]
            print(f"  Size {size}: {count} groups")
        
        return {
            'total_groups': len(groups),
            'multi_logo_groups': len(multi_logo_groups),
            'single_logo_groups': len(single_logo_groups),
            'group_sizes': group_sizes
        }
    
    def _extract_brand_name(self, domain):
        """Extract brand name from domain"""
        try:
            if domain.startswith(('http://', 'https://')):
                parsed = urlparse(domain)
                domain = parsed.netloc
            
            domain = domain.replace('www.', '')
            parts = domain.split('.')
            if len(parts) > 1:
                return parts[0].capitalize()
            return domain.capitalize()
        except:
            return domain
    
    def save_results(self, groups, output_prefix=None):
        """Save similarity matching results"""
        
        if output_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"logo_similarity_nonml_{timestamp}"
        
        # Create results DataFrame
        results_data = []
        for group_id, group in enumerate(groups):
            for domain in group:
                brand_name = self._extract_brand_name(domain)
                results_data.append({
                    'group_id': group_id,
                    'group_size': len(group),
                    'domain': domain,
                    'brand_name': brand_name,
                    'is_unique': len(group) == 1
                })
        
        results_df = pd.DataFrame(results_data)
        
        # Save to CSV
        csv_path = f"{output_prefix}.csv"
        results_df.to_csv(csv_path, index=False)
        
        # Save detailed results with similarity matrix
        detailed_results = {
            'groups': groups,
            'similarity_matrix': self.similarity_matrix,
            'domains': self.domains,
            'similarity_threshold': self.similarity_threshold,
            'features': self.logos_features,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        pkl_path = f"{output_prefix}.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(detailed_results, f)
        
        print(f"\nResults saved:")
        print(f"  CSV: {csv_path}")
        print(f"  Detailed: {pkl_path}")
        
        return csv_path, pkl_path
    
    def visualize_similarity_matrix(self, save_path=None):
        """Visualize the similarity matrix"""
        
        plt.figure(figsize=(12, 10))
        
        # Show similarity matrix as heatmap
        plt.imshow(self.similarity_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Similarity Score')
        plt.title('Logo Similarity Matrix (Non-ML Approach)')
        plt.xlabel('Logo Index')
        plt.ylabel('Logo Index')
        
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"similarity_matrix_nonml_{timestamp}.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Similarity matrix visualization saved to: {save_path}")
        
        return save_path

def main():
    """Main logo similarity matching pipeline (non-ML approach)"""
    
    print("="*60)
    print("LOGO SIMILARITY MATCHER - NON-ML APPROACH")
    print("Challenge: Group websites by logo similarity without ML clustering")
    print("="*60)
    
    # Initialize matcher
    matcher = NonMLLogoMatcher()
    
    # Load logos from cache
    cache_path = "comprehensive_logo_extraction_fast_results.pkl"
    
    if not os.path.exists(cache_path):
        print(f"Error: Cache file {cache_path} not found!")
        return
    
    logos_dict = matcher.load_logos_from_cache(cache_path)
    
    # Extract comprehensive features
    matcher.extract_comprehensive_features(logos_dict)
    
    # Compute pairwise similarity matrix (mathematical approach, no ML)
    similarity_matrix = matcher.compute_pairwise_similarity_matrix()
    
    # Find similar groups using graph-based approach (no ML clustering)
    groups = matcher.find_similar_groups_non_ml()
    
    # Analyze results
    analysis = matcher.analyze_groups(groups)
    
    # Visualize similarity matrix
    viz_path = matcher.visualize_similarity_matrix()
    
    # Save results
    csv_path, pkl_path = matcher.save_results(groups)
    
    # Final summary
    print(f"\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    print(f"✓ Logos processed: {len(logos_dict)}")
    print(f"✓ Success rate: {(len(logos_dict)/4384)*100:.2f}% (target: >97%)")
    print(f"✓ Total groups found: {analysis['total_groups']}")
    print(f"✓ Groups with similar logos: {analysis['multi_logo_groups']}")
    print(f"✓ Unique logos: {analysis['single_logo_groups']}")
    print(f"✓ Approach: Pure mathematical similarity (NO ML clustering)")
    print(f"✓ Methods used: Graph theory, Fourier analysis, geometric features")
    
    if (len(logos_dict)/4384)*100 > 97:
        print(f"✅ SUCCESS: Exceeded 97% logo extraction target!")
    else:
        print(f"⚠️  WARNING: Below 97% extraction target")
    
    print(f"\nOutput files generated:")
    print(f"  • {csv_path} - Group assignments")
    print(f"  • {pkl_path} - Detailed results with similarity matrix")
    print(f"  • {viz_path} - Similarity matrix visualization")

if __name__ == "__main__":
    main()
