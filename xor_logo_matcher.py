"""
XOR-Based Logo Similarity Matcher
Using XOR operations on Fourier-transformed areas for logo clustering
Pure mathematical approach without ML algorithms
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

class XORLogoMatcher:
    """
    Logo similarity matching using XOR operations on Fourier areas
    Pure mathematical approach - no ML clustering algorithms
    """
    
    def __init__(self):
        self.logo_signatures = {}
        self.similarity_threshold = 0.75  # XOR similarity threshold
        
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
                    # Resize to standard size for consistent XOR operations
                    img_resized = cv2.resize(img, (64, 64))  # Smaller for faster XOR
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
    
    def compute_fourier_xor_signature(self, img):
        """Compute XOR-based signature from Fourier transformed areas"""
        
        # Convert to grayscale for simpler processing
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        # Apply 2D FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        
        # Get magnitude spectrum
        magnitude = np.abs(fft_shift)
        
        # Normalize to 0-255 range for binary operations
        magnitude_norm = ((magnitude - magnitude.min()) / 
                         (magnitude.max() - magnitude.min()) * 255).astype(np.uint8)
        
        # Create binary representation using multiple thresholds
        signatures = {}
        
        # 1. Low frequency XOR signature (center region)
        center_h, center_w = magnitude_norm.shape[0]//2, magnitude_norm.shape[1]//2
        low_freq_region = magnitude_norm[center_h-8:center_h+8, center_w-8:center_w+8]
        signatures['low_freq_xor'] = self._create_xor_signature(low_freq_region)
        
        # 2. High frequency XOR signature (outer regions)
        high_freq_mask = np.ones_like(magnitude_norm, dtype=bool)
        high_freq_mask[center_h-16:center_h+16, center_w-16:center_w+16] = False
        high_freq_region = magnitude_norm[high_freq_mask].reshape(-1)
        signatures['high_freq_xor'] = self._create_xor_signature_1d(high_freq_region)
        
        # 3. Phase-based XOR signature
        phase = np.angle(fft_shift)
        phase_norm = ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8)
        signatures['phase_xor'] = self._create_xor_signature(phase_norm)
        
        # 4. Radial frequency XOR signature
        radial_profile = self._compute_radial_profile(magnitude_norm)
        signatures['radial_xor'] = self._create_xor_signature_1d(radial_profile)
        
        # 5. Quadrant-based XOR signatures
        h, w = magnitude_norm.shape
        quad1 = magnitude_norm[:h//2, :w//2]
        quad2 = magnitude_norm[:h//2, w//2:]
        quad3 = magnitude_norm[h//2:, :w//2]
        quad4 = magnitude_norm[h//2:, w//2:]
        
        signatures['quad1_xor'] = self._create_xor_signature(quad1)
        signatures['quad2_xor'] = self._create_xor_signature(quad2)
        signatures['quad3_xor'] = self._create_xor_signature(quad3)
        signatures['quad4_xor'] = self._create_xor_signature(quad4)
        
        return signatures
    
    def _create_xor_signature(self, data_2d):
        """Create XOR signature from 2D data"""
        
        # Multiple threshold levels for robust binary representation
        thresholds = [64, 128, 192]
        xor_signature = 0
        
        for i, threshold in enumerate(thresholds):
            # Create binary mask
            binary_mask = (data_2d > threshold).astype(np.uint8)
            
            # XOR rows and columns to create compact signature
            row_xor = np.zeros(binary_mask.shape[0], dtype=np.uint8)
            col_xor = np.zeros(binary_mask.shape[1], dtype=np.uint8)
            
            for row in range(binary_mask.shape[0]):
                row_xor[row] = np.bitwise_xor.reduce(binary_mask[row, :])
            
            for col in range(binary_mask.shape[1]):
                col_xor[col] = np.bitwise_xor.reduce(binary_mask[:, col])
            
            # Combine row and column XOR signatures
            combined = np.concatenate([row_xor, col_xor])
            
            # Convert to integer signature using bit packing
            signature_int = 0
            for j, bit in enumerate(combined):
                if bit:
                    signature_int |= (1 << (j % 64))  # Pack into 64-bit integers
            
            # XOR with previous threshold signature
            xor_signature ^= signature_int
        
        return xor_signature
    
    def _create_xor_signature_1d(self, data_1d):
        """Create XOR signature from 1D data"""
        
        if len(data_1d) == 0:
            return 0
        
        # Reshape to power of 2 for efficient XOR operations
        target_size = 2 ** int(np.log2(len(data_1d)))
        if target_size < len(data_1d):
            data_1d = data_1d[:target_size]
        
        # Multiple threshold levels
        thresholds = [64, 128, 192]
        xor_signature = 0
        
        for threshold in thresholds:
            # Create binary representation
            binary_data = (data_1d > threshold).astype(np.uint8)
            
            # XOR all bits together in chunks
            signature_int = 0
            chunk_size = 64  # Process in 64-bit chunks
            
            for i in range(0, len(binary_data), chunk_size):
                chunk = binary_data[i:i+chunk_size]
                chunk_int = 0
                
                for j, bit in enumerate(chunk):
                    if bit:
                        chunk_int |= (1 << j)
                
                signature_int ^= chunk_int
            
            xor_signature ^= signature_int
        
        return xor_signature
    
    def _compute_radial_profile(self, data_2d):
        """Compute radial frequency profile"""
        
        h, w = data_2d.shape
        center = (h//2, w//2)
        
        # Create coordinate arrays
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        r = r.astype(int)
        
        # Compute radial average
        max_radius = min(center)
        radial_profile = np.zeros(max_radius, dtype=np.uint8)
        
        for radius in range(max_radius):
            mask = (r == radius)
            if np.any(mask):
                radial_profile[radius] = np.mean(data_2d[mask])
        
        return radial_profile
    
    def extract_all_signatures(self, logos_dict):
        """Extract XOR signatures for all logos"""
        
        print("Computing XOR-based Fourier signatures...")
        
        for i, (domain, img) in enumerate(logos_dict.items()):
            if i % 100 == 0:
                print(f"Processing signature {i+1}/{len(logos_dict)}...")
            
            try:
                signature = self.compute_fourier_xor_signature(img)
                self.logo_signatures[domain] = signature
                
            except Exception as e:
                print(f"Error processing {domain}: {e}")
                continue
        
        print(f"Extracted XOR signatures for {len(self.logo_signatures)} logos")
    
    def compute_xor_similarity(self, sig1, sig2):
        """Compute similarity between two XOR signatures"""
        
        similarities = []
        
        # Compare each signature component
        for key in sig1.keys():
            if key in sig2:
                # XOR the signatures to find differences
                xor_diff = sig1[key] ^ sig2[key]
                
                # Count the number of differing bits (Hamming distance)
                hamming_distance = bin(xor_diff).count('1')
                
                # Convert to similarity (0 = identical, higher = more different)
                # Normalize by expected maximum difference (64 bits)
                similarity = 1.0 - (hamming_distance / 64.0)
                similarities.append(max(0, similarity))
        
        # Return average similarity across all signature components
        return np.mean(similarities) if similarities else 0.0
    
    def find_similar_groups_xor(self):
        """Find similar groups using XOR-based graph approach"""
        
        print("Finding similar groups using XOR-based approach...")
        
        domains = list(self.logo_signatures.keys())
        n_logos = len(domains)
        
        # Create adjacency list based on XOR similarity
        adjacency_list = defaultdict(list)
        
        print("Computing pairwise XOR similarities...")
        
        total_comparisons = (n_logos * (n_logos - 1)) // 2
        current_comparison = 0
        
        for i in range(n_logos):
            if i % 50 == 0:  # More frequent updates
                progress = (current_comparison / total_comparisons) * 100
                print(f"Processing similarities {i+1}/{n_logos} ({progress:.1f}% complete)...")
            
            for j in range(i + 1, n_logos):
                current_comparison += 1
                
                similarity = self.compute_xor_similarity(
                    self.logo_signatures[domains[i]],
                    self.logo_signatures[domains[j]]
                )
                
                if similarity >= self.similarity_threshold:
                    adjacency_list[i].append(j)
                    adjacency_list[j].append(i)
        
        # Find connected components using iterative DFS (avoid recursion limit)
        visited = [False] * n_logos
        groups = []
        
        def iterative_dfs(start_node):
            """Iterative DFS to avoid recursion depth issues"""
            current_group = []
            stack = [start_node]
            
            while stack:
                node = stack.pop()
                
                if not visited[node]:
                    visited[node] = True
                    current_group.append(node)
                    
                    # Add unvisited neighbors to stack
                    for neighbor in adjacency_list[node]:
                        if not visited[neighbor]:
                            stack.append(neighbor)
            
            return current_group
        
        # Find all connected components
        for i in range(n_logos):
            if not visited[i]:
                current_group = iterative_dfs(i)
                if current_group:  # Only add non-empty groups
                    groups.append(current_group)
        
        # Convert indices to domains
        domain_groups = []
        for group in groups:
            domain_group = [domains[idx] for idx in group]
            domain_groups.append(domain_group)
        
        # Sort groups by size (largest first)
        domain_groups.sort(key=len, reverse=True)
        
        # Count total connections found
        total_connections = sum(len(neighbors) for neighbors in adjacency_list.values()) // 2
        
        print(f"Found {len(domain_groups)} groups using XOR similarity")
        print(f"Total similar pairs found: {total_connections}")
        
        return domain_groups
    
    def analyze_xor_groups(self, groups):
        """Analyze the XOR-based groups"""
        
        print("\n" + "="*60)
        print("XOR-BASED LOGO SIMILARITY GROUPS")
        print("="*60)
        
        total_logos = sum(len(group) for group in groups)
        multi_logo_groups = [group for group in groups if len(group) > 1]
        single_logo_groups = [group for group in groups if len(group) == 1]
        
        print(f"Total logos processed: {total_logos}")
        print(f"Total groups found: {len(groups)}")
        print(f"Groups with multiple logos: {len(multi_logo_groups)}")
        print(f"Unique logos (single groups): {len(single_logo_groups)}")
        print(f"XOR similarity threshold: {self.similarity_threshold}")
        
        # Show multi-logo groups
        if multi_logo_groups:
            print(f"\nMulti-logo groups (XOR-similar logos):")
            for i, group in enumerate(multi_logo_groups[:10]):
                print(f"\nGroup {i+1} ({len(group)} logos):")
                for j, domain in enumerate(group):
                    brand_name = self._extract_brand_name(domain)
                    print(f"  {j+1}. {brand_name} ({domain})")
                
                if i >= 9:
                    remaining = len(multi_logo_groups) - 10
                    if remaining > 0:
                        print(f"\n... and {remaining} more groups")
                    break
        
        return {
            'total_groups': len(groups),
            'multi_logo_groups': len(multi_logo_groups),
            'single_logo_groups': len(single_logo_groups),
            'xor_threshold': self.similarity_threshold
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
    
    def save_xor_results(self, groups, output_prefix=None):
        """Save XOR-based results"""
        
        if output_prefix is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_prefix = f"xor_logo_similarity_{timestamp}"
        
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
                    'method': 'XOR_Fourier'
                })
        
        results_df = pd.DataFrame(results_data)
        
        # Save to CSV
        csv_path = f"{output_prefix}.csv"
        results_df.to_csv(csv_path, index=False)
        
        # Save detailed results
        detailed_results = {
            'groups': groups,
            'signatures': self.logo_signatures,
            'similarity_threshold': self.similarity_threshold,
            'method': 'XOR_Fourier_Similarity',
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        pkl_path = f"{output_prefix}.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(detailed_results, f)
        
        print(f"\nXOR-based results saved:")
        print(f"  CSV: {csv_path}")
        print(f"  Detailed: {pkl_path}")
        
        return csv_path, pkl_path

def main():
    """Main XOR-based logo similarity matching pipeline"""
    
    print("="*60)
    print("XOR-BASED LOGO SIMILARITY MATCHER")
    print("Challenge: Group websites by logo similarity using XOR operations")
    print("Method: XOR operations on Fourier-transformed areas")
    print("="*60)
    
    # Initialize XOR matcher
    matcher = XORLogoMatcher()
    
    # Load logos from cache
    cache_path = "comprehensive_logo_extraction_fast_results.pkl"
    
    if not os.path.exists(cache_path):
        print(f"Error: Cache file {cache_path} not found!")
        return
    
    logos_dict = matcher.load_logos_from_cache(cache_path)
    
    # Extract XOR-based Fourier signatures
    matcher.extract_all_signatures(logos_dict)
    
    # Find similar groups using XOR approach
    groups = matcher.find_similar_groups_xor()
    
    # Analyze results
    analysis = matcher.analyze_xor_groups(groups)
    
    # Save results
    csv_path, pkl_path = matcher.save_xor_results(groups)
    
    # Final summary
    print(f"\n" + "="*60)
    print("XOR-BASED RESULTS SUMMARY")
    print("="*60)
    print(f" Logos processed: {len(logos_dict)}")
    print(f" Success rate: {(len(logos_dict)/4384)*100:.2f}% (target: >97%)")
    print(f" Total groups found: {analysis['total_groups']}")
    print(f" Groups with similar logos: {analysis['multi_logo_groups']}")
    print(f" Unique logos: {analysis['single_logo_groups']}")
    print(f" Method: XOR operations on Fourier areas")
    print(f" Features: Low/high freq, phase, radial, quadrant XOR signatures")
    
    if (len(logos_dict)/4384)*100 > 97:
        print(f" SUCCESS: Exceeded 97% logo extraction target!")
    else:
        print(f"  WARNING: Below 97% extraction target")
    
    print(f"\nXOR-based approach benefits:")
    print(f"  • Pure mathematical operations (no ML)")
    print(f"  • Fast bit-level operations")
    print(f"  • Robust to small variations")
    print(f"  • Multiple frequency domain signatures")
    
    print(f"\nOutput files generated:")
    print(f"  • {csv_path} - XOR-based group assignments")
    print(f"  • {pkl_path} - Detailed XOR signatures and results")

if __name__ == "__main__":
    main()
