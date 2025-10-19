"""
Clustering Engine Module

Handles the core clustering logic including:
- Similarity computation between logos
- Hierarchical clustering with candidate pruning
- Aggressive singleton merging
- Cluster quality analysis
"""

import cv2
import numpy as np
from collections import defaultdict
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import time


class ClusteringEngine:
    """Core clustering engine with optimized algorithms"""
    
    def __init__(self, thresholds=None):
        """
        Initialize clustering engine with thresholds
        
        Args:
            thresholds: dict with keys 'phash', 'orb', 'color' for matching thresholds
        """
        self.thresholds = thresholds or {
            'phash': 60,   # Ultra-relaxed
            'orb': 2,      # Ultra-relaxed
            'color': 0.10  # Ultra-relaxed
        }
        self.phash_buckets = defaultdict(list)
    
    def build_phash_buckets(self, features_list):
        """Build pHash buckets for fast candidate pruning"""
        self.phash_buckets.clear()
        for features in features_list:
            bucket = features.get('phash_bucket', '')
            index = features.get('index', 0)
            if bucket:
                self.phash_buckets[bucket].append(index)
    
    def calculate_similarity(self, features1, features2):
        """Calculate similarity between two logo features"""
        try:
            similarities = {}
            match_signals = 0  # Count matching signals for two-channel rule
            
            # 1. pHash similarity (primary channel)
            phash1 = features1['phash']
            phash2 = features2['phash']
            
            if phash1 and phash2:
                # Fast hamming distance for pHash
                hash_diff = bin(int(phash1, 16) ^ int(phash2, 16)).count('1')
                phash_sim = max(0, (64 - hash_diff) / 64)
                similarities['phash'] = phash_sim
                
                # Check if it meets ultra-relaxed threshold
                if hash_diff <= self.thresholds['phash']:
                    match_signals += 1
            
            # 2. Color histogram similarity (secondary channel)
            color_hist1 = features1.get('color_histogram', np.array([]))
            color_hist2 = features2.get('color_histogram', np.array([]))
            
            if len(color_hist1) > 0 and len(color_hist2) > 0:
                color_sim = 1 - np.sum(np.abs(color_hist1 - color_hist2)) / 2
                similarities['color'] = color_sim
                
                # Check if it meets ultra-relaxed threshold
                if color_sim >= self.thresholds['color']:
                    match_signals += 1
            
            # 3. ORB descriptors similarity (tertiary channel)
            orb_desc1 = features1.get('orb_descriptors')
            orb_desc2 = features2.get('orb_descriptors')
            
            if orb_desc1 is not None and orb_desc2 is not None and len(orb_desc1) > 0 and len(orb_desc2) > 0:
                # Use brute-force matcher for ORB descriptors
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                matches = bf.match(orb_desc1, orb_desc2)
                
                # Calculate similarity based on good matches
                if len(matches) > 0:
                    # Sort matches by distance (lower is better)
                    matches = sorted(matches, key=lambda x: x.distance)
                    
                    # Count good matches (distance < 50 is considered good for ORB)
                    good_matches = [m for m in matches if m.distance < 50]
                    
                    # Calculate similarity as ratio of good matches to total possible
                    max_features = min(len(orb_desc1), len(orb_desc2))
                    orb_sim = len(good_matches) / max(max_features, 1) if max_features > 0 else 0
                    similarities['orb'] = orb_sim
                    
                    # Check if it meets ultra-relaxed threshold
                    if len(good_matches) >= self.thresholds['orb']:
                        match_signals += 1
                else:
                    similarities['orb'] = 0.0
            else:
                similarities['orb'] = 0.0
            
            # 4. DCT hash similarity (quaternary channel)
            dct_hash1 = features1.get('dct_hash', '')
            dct_hash2 = features2.get('dct_hash', '')
            
            if dct_hash1 and dct_hash2 and len(dct_hash1) == len(dct_hash2):
                dct_diff = sum(c1 != c2 for c1, c2 in zip(dct_hash1, dct_hash2))
                dct_sim = max(0, (len(dct_hash1) - dct_diff) / len(dct_hash1))
                similarities['dct'] = dct_sim
            
            # 5. Brand family bonus
            brand_bonus = 0
            if (features1.get('brand_family') == features2.get('brand_family') and 
                features1.get('brand_family') != 'unknown'):
                brand_bonus = 0.1
                match_signals += 1  # Brand match counts as a signal
            
            # Calculate weighted similarity
            weights = {'phash': 0.5, 'color': 0.25, 'orb': 0.15, 'dct': 0.1}
            weighted_sim = sum(similarities.get(key, 0) * weight for key, weight in weights.items())
            weighted_sim += brand_bonus
            
            # Relaxed two-channel rule: Allow single strong signals
            if match_signals == 0 and weighted_sim > 0.5:
                weighted_sim *= 0.5  # Only penalize if no signals match
            
            return {
                'similarity': min(weighted_sim, 1.0),
                'details': similarities,
                'match_signals': match_signals,
                'brand_bonus': brand_bonus
            }
            
        except Exception as e:
            print(f"Similarity calculation error: {e}")
            return {'similarity': 0.0, 'details': {}, 'match_signals': 0, 'brand_bonus': 0}
    
    def cluster_with_pruning(self, features_list):
        """Cluster with pHash bucketing for candidate pruning"""
        print("Clustering with optimized candidate pruning...")
        start_time = time.time()
        
        n = len(features_list)
        similarity_matrix = np.zeros((n, n))
        
        # Build pHash buckets
        self.build_phash_buckets(features_list)
        
        # Phase 1: Fast pHash bucketing for candidate pruning
        comparisons_made = 0
        bucket_hits = 0
        
        for i in range(n):
            features1 = features_list[i]
            bucket1 = features1.get('phash_bucket', '')
            
            # Get candidates from same and similar buckets
            candidates = set()
            
            # Same bucket (high priority)
            if bucket1 in self.phash_buckets:
                candidates.update(self.phash_buckets[bucket1])
                bucket_hits += len(self.phash_buckets[bucket1])
            
            # Similar buckets (1-2 bit differences)
            for bucket, indices in self.phash_buckets.items():
                if bucket != bucket1:
                    # Fast hamming check on bucket keys
                    if bin(int(bucket1, 16) ^ int(bucket, 16)).count('1') <= 2:
                        candidates.update(indices)
            
            # Compare with candidates only
            for j in candidates:
                if i < j < len(features_list):
                    features2 = features_list[j]
                    
                    similarity_data = self.calculate_similarity(features1, features2)
                    similarity_matrix[i][j] = similarity_data['similarity']
                    similarity_matrix[j][i] = similarity_data['similarity']
                    
                    comparisons_made += 1
        
        # Phase 2: Hierarchical clustering
        print(f"Made {comparisons_made:,} optimized comparisons (vs {n*(n-1)//2:,} brute force)")
        print(f"Bucket efficiency: {bucket_hits:,} candidate hits")
        
        # Convert to distance matrix
        distance_matrix = 1 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0.0)
        
        # Hierarchical clustering with ultra-relaxed threshold
        try:
            linkage_matrix = linkage(squareform(distance_matrix), method='single')
            ultra_relaxed_threshold = 0.85  # Very low threshold for maximum merging
            cluster_labels = fcluster(linkage_matrix, ultra_relaxed_threshold, criterion='distance')
        except Exception as e:
            print(f"Clustering error: {e}, using fallback")
            cluster_labels = np.arange(n) + 1
        
        # Organize results
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(features_list[idx]['domain'])
        
        elapsed = time.time() - start_time
        print(f"Clustering completed in {elapsed:.2f}s")
        print(f"Performance: {comparisons_made/elapsed:.0f} comparisons/sec")
        
        return dict(clusters), similarity_matrix
    
    def merge_singletons_aggressively(self, clusters, features_list):
        """Force aggressive merging of remaining singletons"""
        print("Applying aggressive singleton merging...")
        
        # Extract singletons
        singletons = []
        multi_clusters = []
        
        for cluster_id, domains in clusters.items():
            if len(domains) == 1:
                singletons.extend(domains)
            else:
                multi_clusters.append(domains)
        
        if len(singletons) < 2:
            return clusters
        
        print(f"Found {len(singletons)} singletons to merge")
        
        # Create features lookup
        domain_to_features = {f['domain']: f for f in features_list}
        
        # Aggressive merging
        remaining_singletons = singletons.copy()
        new_clusters = []
        
        while len(remaining_singletons) >= 2:
            seed = remaining_singletons.pop(0)
            seed_features = domain_to_features.get(seed)
            
            if not seed_features:
                continue
                
            cluster = [seed]
            to_remove = []
            
            # Find matches with extremely relaxed criteria
            for other in remaining_singletons:
                other_features = domain_to_features.get(other)
                if not other_features:
                    continue
                
                # Super relaxed matching
                sim_data = self.calculate_similarity(seed_features, other_features)
                
                accept = False
                
                # pHash check (very relaxed)
                if 'phash' in sim_data['details']:
                    phash1 = seed_features.get('phash', '')
                    phash2 = other_features.get('phash', '')
                    if phash1 and phash2:
                        hash_diff = bin(int(phash1, 16) ^ int(phash2, 16)).count('1')
                        if hash_diff <= 62:
                            accept = True
                
                # Color check
                if not accept and sim_data['details'].get('color', 0) >= 0.05:
                    accept = True
                
                # ORB check
                if not accept and sim_data['details'].get('orb', 0) >= 0.1:
                    accept = True
                
                # Brand family check
                if not accept and (seed_features.get('brand_family') == other_features.get('brand_family') 
                                 and seed_features.get('brand_family') != 'unknown'):
                    accept = True
                
                # Industry check
                if not accept and (seed_features.get('industry') == other_features.get('industry')
                                 and seed_features.get('industry') != 'general'):
                    accept = True
                
                if accept and len(cluster) < 20:  # Limit cluster size
                    cluster.append(other)
                    to_remove.append(other)
            
            # Remove merged items
            for item in to_remove:
                remaining_singletons.remove(item)
            
            new_clusters.append(cluster)
        
        # Add remaining singletons
        for singleton in remaining_singletons:
            new_clusters.append([singleton])
        
        # Combine with original multi-clusters
        all_clusters = multi_clusters + new_clusters
        
        # Convert back to dict
        result_clusters = {}
        for i, cluster in enumerate(all_clusters):
            result_clusters[i + 1] = cluster
        
        print(f"Aggressive merging: {len(singletons)} singletons → {len([c for c in result_clusters.values() if len(c) == 1])} remaining")
        
        return result_clusters
