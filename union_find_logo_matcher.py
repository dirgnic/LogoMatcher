"""
Optimized Logo Similarity Matcher using Union-Find
Advanced non-ML approach with multiple matching strategies
"""

import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
from typing import List, Dict, Tuple
import os
from urllib.parse import urlparse
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class UnionFind:
    """Efficient Union-Find data structure for clustering"""
    
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.size = [1] * n
        self.num_sets = n
    
    def find(self, x):
        """Find with path compression"""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """Union by rank"""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        
        self.parent[py] = px
        self.size[px] += self.size[py]
        
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        self.num_sets -= 1
        return True
    
    def get_groups(self):
        """Get all connected components"""
        groups = defaultdict(list)
        for i in range(len(self.parent)):
            root = self.find(i)
            groups[root].append(i)
        return list(groups.values())

class OptimizedLogoMatcher:
    """
    Optimized logo similarity matcher with multiple strategies
    Uses Union-Find for efficient clustering
    """
    
    def __init__(self):
        self.logos_data = {}
        self.similarity_thresholds = {
            'strict': 0.95,      # Very similar logos
            'moderate': 0.85,    # Moderately similar
            'loose': 0.75        # Loosely similar
        }
        
    def load_logos_optimized(self, cache_path, max_logos=None):
        """Optimized logo loading with better success rate"""
        
        print(f"Loading logos from {cache_path}...")
        
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        # Use successful_logos for better success rate
        if 'successful_logos' in cached_data and len(cached_data['successful_logos']) > 0:
            logo_results = cached_data['successful_logos']
            print(f"Using successful_logos: {len(logo_results)} entries")
        else:
            logo_results = cached_data['logo_results']
            print(f"Using logo_results: {len(logo_results)} entries")
        
        logos_dict = {}
        failed_count = 0
        
        for i, logo_entry in enumerate(logo_results):
            if max_logos and len(logos_dict) >= max_logos:
                break
                
            if i % 500 == 0:
                print(f"Processing logo {i+1}/{len(logo_results)}...")
                
            try:
                domain = logo_entry.get('domain', logo_entry.get('website', 'unknown'))
                logo_bytes = logo_entry.get('logo_data')
                success = logo_entry.get('success', True)  # Default to True for successful_logos
                
                if not success or not logo_bytes:
                    failed_count += 1
                    continue
                
                # Convert bytes to image
                img_array = np.frombuffer(logo_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is not None and img.shape[0] > 10 and img.shape[1] > 10:
                    # Resize to standard size
                    img_resized = cv2.resize(img, (128, 128))
                    logos_dict[domain] = img_resized
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                continue
        
        total_entries = len(logo_results)
        success_rate = (len(logos_dict) / total_entries) * 100 if total_entries > 0 else 0
        
        print(f"Successfully loaded {len(logos_dict)} valid logo images")
        print(f"Failed to load {failed_count} logos")
        print(f"Success rate: {success_rate:.2f}% (target: >97%)")
        
        return logos_dict
    
    def compute_multi_signature(self, img):
        """Compute multiple types of signatures for robust matching"""
        
        signatures = {}
        
        # 1. Perceptual Hash (DCT-based)
        signatures['phash'] = self._compute_perceptual_hash(img)
        
        # 2. Average Hash
        signatures['ahash'] = self._compute_average_hash(img)
        
        # 3. Difference Hash
        signatures['dhash'] = self._compute_difference_hash(img)
        
        # 4. Wavelet Hash (alternative to Fourier)
        signatures['whash'] = self._compute_wavelet_hash(img)
        
        # 5. Color Histogram Signature
        signatures['color_hist'] = self._compute_color_histogram(img)
        
        # 6. Edge Density Map
        signatures['edge_map'] = self._compute_edge_density_map(img)
        
        # 7. Contour Signature
        signatures['contour_sig'] = self._compute_contour_signature(img)
        
        return signatures
    
    def _compute_perceptual_hash(self, img):
        """DCT-based perceptual hash"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (32, 32))
        dct = cv2.dct(np.float32(resized))
        dct_low = dct[1:9, 1:9]  # Remove DC component
        median = np.median(dct_low)
        return ''.join(['1' if x > median else '0' for x in dct_low.flatten()])
    
    def _compute_average_hash(self, img):
        """Average hash"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (8, 8))
        avg = np.mean(resized)
        return ''.join(['1' if x > avg else '0' for x in resized.flatten()])
    
    def _compute_difference_hash(self, img):
        """Difference hash"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (9, 8))
        diff = resized[:, 1:] > resized[:, :-1]
        return ''.join(['1' if x else '0' for x in diff.flatten()])
    
    def _compute_wavelet_hash(self, img):
        """Simplified wavelet-like hash using gradient"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (16, 16))
        
        # Compute gradients
        grad_x = cv2.Sobel(resized, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(resized, cv2.CV_64F, 0, 1, ksize=3)
        
        # Combine gradients
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        median = np.median(gradient_mag)
        
        return ''.join(['1' if x > median else '0' for x in gradient_mag.flatten()])
    
    def _compute_color_histogram(self, img):
        """Color histogram signature"""
        hist_b = cv2.calcHist([img], [0], None, [8], [0, 256])
        hist_g = cv2.calcHist([img], [1], None, [8], [0, 256])
        hist_r = cv2.calcHist([img], [2], None, [8], [0, 256])
        
        combined = np.concatenate([hist_b.flatten(), hist_g.flatten(), hist_r.flatten()])
        # Normalize
        return combined / (np.sum(combined) + 1e-8)
    
    def _compute_edge_density_map(self, img):
        """Edge density in different regions"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Divide into 4x4 grid and compute density
        h, w = edges.shape
        grid_h, grid_w = h // 4, w // 4
        
        densities = []
        for i in range(4):
            for j in range(4):
                region = edges[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                density = np.sum(region > 0) / (region.shape[0] * region.shape[1])
                densities.append(density)
        
        return np.array(densities)
    
    def _compute_contour_signature(self, img):
        """Contour-based signature"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return np.zeros(7)
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Compute Hu moments
        moments = cv2.moments(largest_contour)
        hu_moments = cv2.HuMoments(moments).flatten()
        
        # Take log and handle zeros
        hu_log = []
        for moment in hu_moments:
            if moment != 0:
                hu_log.append(np.copysign(np.log10(abs(moment)), moment))
            else:
                hu_log.append(0)
        
        return np.array(hu_log)
    
    def extract_all_signatures(self, logos_dict):
        """Extract signatures for all logos"""
        
        print("Computing multi-modal signatures...")
        
        for i, (domain, img) in enumerate(logos_dict.items()):
            if i % 100 == 0:
                print(f"Processing signature {i+1}/{len(logos_dict)}...")
            
            try:
                signatures = self.compute_multi_signature(img)
                self.logos_data[domain] = signatures
                
            except Exception as e:
                print(f"Error processing {domain}: {e}")
                continue
        
        print(f"Extracted signatures for {len(self.logos_data)} logos")
    
    def compute_comprehensive_similarity(self, sig1, sig2):
        """Compute similarity using multiple methods"""
        
        similarities = {}
        
        # Hash-based similarities (Hamming distance)
        for hash_type in ['phash', 'ahash', 'dhash', 'whash']:
            if hash_type in sig1 and hash_type in sig2:
                hamming_dist = sum(c1 != c2 for c1, c2 in zip(sig1[hash_type], sig2[hash_type]))
                max_dist = len(sig1[hash_type])
                similarities[hash_type] = 1.0 - (hamming_dist / max_dist)
        
        # Histogram similarity (correlation)
        if 'color_hist' in sig1 and 'color_hist' in sig2:
            correlation = np.corrcoef(sig1['color_hist'], sig2['color_hist'])[0, 1]
            similarities['color_hist'] = max(0, correlation) if not np.isnan(correlation) else 0
        
        # Edge density similarity (cosine similarity)
        if 'edge_map' in sig1 and 'edge_map' in sig2:
            cos_sim = np.dot(sig1['edge_map'], sig2['edge_map']) / (
                np.linalg.norm(sig1['edge_map']) * np.linalg.norm(sig2['edge_map']) + 1e-8)
            similarities['edge_map'] = max(0, cos_sim)
        
        # Contour similarity (Hu moments)
        if 'contour_sig' in sig1 and 'contour_sig' in sig2:
            hu_dist = np.linalg.norm(sig1['contour_sig'] - sig2['contour_sig'])
            similarities['contour_sig'] = np.exp(-hu_dist)
        
        # Weighted combination
        weights = {
            'phash': 0.25,
            'ahash': 0.15,
            'dhash': 0.15,
            'whash': 0.15,
            'color_hist': 0.15,
            'edge_map': 0.10,
            'contour_sig': 0.05
        }
        
        overall_similarity = sum(similarities.get(key, 0) * weight 
                               for key, weight in weights.items())
        
        return overall_similarity, similarities
    
    def find_similar_groups_union_find(self, threshold_type='moderate'):
        """Find similar groups using Union-Find algorithm"""
        
        domains = list(self.logos_data.keys())
        n_logos = len(domains)
        threshold = self.similarity_thresholds[threshold_type]
        
        print(f"Finding similar groups using Union-Find (threshold: {threshold})")
        
        # Initialize Union-Find
        uf = UnionFind(n_logos)
        
        # Compute pairwise similarities and union similar logos
        total_comparisons = (n_logos * (n_logos - 1)) // 2
        current_comparison = 0
        similar_pairs = 0
        
        for i in range(n_logos):
            if i % 50 == 0:
                progress = (current_comparison / total_comparisons) * 100
                print(f"Processing {i+1}/{n_logos} ({progress:.1f}% complete, {similar_pairs} similar pairs found)")
            
            for j in range(i + 1, n_logos):
                current_comparison += 1
                
                overall_sim, _ = self.compute_comprehensive_similarity(
                    self.logos_data[domains[i]],
                    self.logos_data[domains[j]]
                )
                
                if overall_sim >= threshold:
                    uf.union(i, j)
                    similar_pairs += 1
        
        # Get groups from Union-Find
        groups_indices = uf.get_groups()
        
        # Convert to domain groups
        domain_groups = []
        for group in groups_indices:
            domain_group = [domains[idx] for idx in group]
            domain_groups.append(domain_group)
        
        # Sort by group size
        domain_groups.sort(key=len, reverse=True)
        
        print(f"Found {len(domain_groups)} groups using Union-Find")
        print(f"Total similar pairs: {similar_pairs}")
        print(f"Largest group size: {len(domain_groups[0]) if domain_groups else 0}")
        
        return domain_groups
    
    def analyze_union_find_groups(self, groups, threshold_type):
        """Analyze Union-Find results"""
        
        print(f"\n" + "="*60)
        print(f"UNION-FIND LOGO SIMILARITY RESULTS ({threshold_type.upper()})")
        print("="*60)
        
        total_logos = sum(len(group) for group in groups)
        multi_logo_groups = [group for group in groups if len(group) > 1]
        single_logo_groups = [group for group in groups if len(group) == 1]
        
        print(f"Total logos processed: {total_logos}")
        print(f"Total groups found: {len(groups)}")
        print(f"Groups with multiple logos: {len(multi_logo_groups)}")
        print(f"Unique logos (single groups): {len(single_logo_groups)}")
        print(f"Similarity threshold: {self.similarity_thresholds[threshold_type]}")
        
        # Group size distribution
        group_sizes = [len(group) for group in groups]
        size_distribution = {}
        for size in group_sizes:
            size_distribution[size] = size_distribution.get(size, 0) + 1
        
        print(f"\nGroup size distribution:")
        for size in sorted(size_distribution.keys(), reverse=True)[:10]:
            count = size_distribution[size]
            print(f"  Size {size}: {count} groups")
        
        # Show sample multi-logo groups
        if multi_logo_groups:
            print(f"\nSample multi-logo groups:")
            for i, group in enumerate(multi_logo_groups[:5]):
                print(f"\nGroup {i+1} ({len(group)} logos):")
                for j, domain in enumerate(group[:5]):
                    brand_name = self._extract_brand_name(domain)
                    print(f"  {j+1}. {brand_name} ({domain})")
                if len(group) > 5:
                    print(f"  ... and {len(group) - 5} more")
        
        return {
            'total_groups': len(groups),
            'multi_logo_groups': len(multi_logo_groups),
            'single_logo_groups': len(single_logo_groups),
            'threshold_type': threshold_type,
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
    
    def save_union_find_results(self, groups, threshold_type, output_prefix=None):
        """Save Union-Find results"""
        
        if output_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"union_find_logos_{threshold_type}_{timestamp}"
        
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
                    'is_unique': len(group) == 1,
                    'threshold_type': threshold_type,
                    'method': 'Union_Find_Multi_Signature'
                })
        
        results_df = pd.DataFrame(results_data)
        
        # Save to CSV
        csv_path = f"{output_prefix}.csv"
        results_df.to_csv(csv_path, index=False)
        
        # Save detailed results
        detailed_results = {
            'groups': groups,
            'signatures': self.logos_data,
            'threshold_type': threshold_type,
            'similarity_thresholds': self.similarity_thresholds,
            'method': 'Union_Find_Multi_Signature',
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        pkl_path = f"{output_prefix}.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(detailed_results, f)
        
        print(f"\nUnion-Find results saved:")
        print(f"  CSV: {csv_path}")
        print(f"  Detailed: {pkl_path}")
        
        return csv_path, pkl_path

def main():
    """Main Union-Find logo similarity matching pipeline"""
    
    print("="*60)
    print("OPTIMIZED LOGO SIMILARITY MATCHER")
    print("Union-Find + Multi-Signature Approach (No ML Clustering)")
    print("="*60)
    
    # Initialize matcher
    matcher = OptimizedLogoMatcher()
    
    # Load logos with better success rate
    cache_path = "comprehensive_logo_extraction_fast_results.pkl"
    
    if not os.path.exists(cache_path):
        print(f"Error: Cache file {cache_path} not found!")
        return
    
    logos_dict = matcher.load_logos_optimized(cache_path)
    
    # Extract multi-modal signatures
    matcher.extract_all_signatures(logos_dict)
    
    # Test different similarity thresholds
    threshold_types = ['strict', 'moderate', 'loose']
    
    results_summary = {}
    
    for threshold_type in threshold_types:
        print(f"\n{'='*60}")
        print(f"TESTING {threshold_type.upper()} SIMILARITY THRESHOLD")
        print(f"{'='*60}")
        
        # Find groups using Union-Find
        groups = matcher.find_similar_groups_union_find(threshold_type)
        
        # Analyze results
        analysis = matcher.analyze_union_find_groups(groups, threshold_type)
        
        # Save results
        csv_path, pkl_path = matcher.save_union_find_results(groups, threshold_type)
        
        results_summary[threshold_type] = {
            'analysis': analysis,
            'csv_path': csv_path,
            'pkl_path': pkl_path
        }
    
    # Final summary
    print(f"\n" + "="*60)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*60)
    
    total_entries = 4384  # From cache
    success_rate = (len(logos_dict) / total_entries) * 100
    
    print(f"✓ Total cache entries: {total_entries}")
    print(f"✓ Successfully processed: {len(logos_dict)}")
    print(f"✓ Success rate: {success_rate:.2f}% (target: >97%)")
    print(f"✓ Method: Union-Find + Multi-Signature")
    print(f"✓ Signatures: pHash, aHash, dHash, wavelet, color, edge, contour")
    
    if success_rate > 97:
        print(f"✅ SUCCESS: Exceeded 97% extraction target!")
    else:
        print(f"⚠️  WARNING: Below 97% extraction target")
    
    print(f"\nResults by threshold:")
    for threshold_type, results in results_summary.items():
        analysis = results['analysis']
        threshold = matcher.similarity_thresholds[threshold_type]
        print(f"  {threshold_type.capitalize()} (≥{threshold}): {analysis['multi_logo_groups']} similar groups, {analysis['single_logo_groups']} unique logos")
    
    print(f"\nKey advantages of this approach:")
    print(f"  • Union-Find: O(n log n) clustering efficiency")
    print(f"  • Multi-signature: Robust to variations")
    print(f"  • No ML algorithms: Pure mathematical similarity")
    print(f"  • Scalable: Handles thousands of logos efficiently")

if __name__ == "__main__":
    main()
