#!/usr/bin/env python3
"""
Logo Similarity Analysis & Clustering Pipeline
🎯 Continue from logo extraction to similarity analysis and clustering
"""

import asyncio
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import pickle
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import io
import warnings
warnings.filterwarnings('ignore')

# For Fourier analysis
from scipy.fft import fft2, fftshift
from skimage import filters, transform
from sklearn.metrics.pairwise import cosine_similarity

class FourierLogoAnalyzer:
    """Advanced Fourier-based logo similarity analysis"""
    
    def __init__(self):
        self.hash_size = 8
        self.similarity_threshold = 0.7
    
    def preprocess_logo(self, logo_data: bytes) -> Optional[np.ndarray]:
        """Convert logo bytes to standardized numpy array"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(logo_data))
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to standard size for comparison
            image = image.resize((128, 128), Image.Resampling.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Convert to grayscale for Fourier analysis
            if len(img_array.shape) == 3:
                img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_array
            
            return img_gray
            
        except Exception as e:
            print(f"❌ Error preprocessing logo: {e}")
            return None
    
    def compute_phash(self, img: np.ndarray) -> str:
        """Compute perceptual hash using DCT"""
        try:
            # Resize to hash_size + 1
            img_resized = cv2.resize(img, (self.hash_size + 1, self.hash_size))
            
            # Apply DCT
            dct_coeffs = cv2.dct(np.float32(img_resized))
            
            # Extract top-left 8x8 coefficients (low frequencies)
            dct_low = dct_coeffs[:self.hash_size, :self.hash_size]
            
            # Compute median
            median = np.median(dct_low)
            
            # Generate binary hash
            binary_hash = dct_low > median
            
            # Convert to string
            return ''.join(['1' if x else '0' for x in binary_hash.flatten()])
            
        except Exception as e:
            return ''
    
    def compute_fft_signature(self, img: np.ndarray) -> np.ndarray:
        """Compute FFT-based signature focusing on low frequencies"""
        try:
            # Apply FFT
            fft = fft2(img)
            fft_shifted = fftshift(fft)
            
            # Get magnitude spectrum
            magnitude = np.abs(fft_shifted)
            
            # Focus on center (low frequencies)
            center = magnitude.shape[0] // 2
            radius = 16  # Extract 32x32 center region
            low_freq = magnitude[center-radius:center+radius, center-radius:center+radius]
            
            # Normalize and flatten
            if low_freq.size > 0:
                normalized = (low_freq - np.mean(low_freq)) / (np.std(low_freq) + 1e-8)
                return normalized.flatten()[:256]  # Take first 256 components
            else:
                return np.zeros(256)
                
        except Exception as e:
            return np.zeros(256)
    
    def compute_fourier_mellin(self, img: np.ndarray) -> np.ndarray:
        """Compute Fourier-Mellin transform for rotation/scale invariance"""
        try:
            # Convert to log-polar coordinates for scale/rotation invariance
            height, width = img.shape
            center = (width // 2, height // 2)
            
            # Create log-polar mapping
            max_radius = min(center[0], center[1])
            log_polar = cv2.logPolar(img, center, max_radius / np.log(max_radius), 
                                   cv2.WARP_FILL_OUTLIERS)
            
            # Apply FFT
            fft_lp = fft2(log_polar)
            magnitude = np.abs(fft_lp)
            
            # Extract features from magnitude spectrum
            features = []
            for i in range(0, min(magnitude.shape[0], 16), 2):
                for j in range(0, min(magnitude.shape[1], 16), 2):
                    features.append(magnitude[i, j])
            
            return np.array(features[:64])  # Return first 64 features
            
        except Exception as e:
            return np.zeros(64)
    
    def extract_logo_features(self, logo_data: bytes) -> Dict:
        """Extract comprehensive features from logo"""
        img = self.preprocess_logo(logo_data)
        if img is None:
            return {
                'phash': '',
                'fft_signature': np.zeros(256),
                'fourier_mellin': np.zeros(64),
                'valid': False
            }
        
        return {
            'phash': self.compute_phash(img),
            'fft_signature': self.compute_fft_signature(img),
            'fourier_mellin': self.compute_fourier_mellin(img),
            'valid': True
        }
    
    def hamming_distance(self, hash1: str, hash2: str) -> float:
        """Compute normalized Hamming distance"""
        if len(hash1) != len(hash2) or len(hash1) == 0:
            return 1.0
        
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2)) / len(hash1)
    
    def compute_similarity_score(self, features1: Dict, features2: Dict) -> float:
        """Compute combined similarity score"""
        if not features1['valid'] or not features2['valid']:
            return 0.0
        
        # 1. pHash similarity (lower Hamming distance = higher similarity)
        phash_sim = 1.0 - self.hamming_distance(features1['phash'], features2['phash'])
        
        # 2. FFT signature similarity
        fft_sim = max(0, cosine_similarity([features1['fft_signature']], 
                                         [features2['fft_signature']])[0][0])
        
        # 3. Fourier-Mellin similarity
        fm_sim = max(0, cosine_similarity([features1['fourier_mellin']], 
                                        [features2['fourier_mellin']])[0][0])
        
        # Combined score with weights (pHash is most reliable)
        combined_score = (0.5 * phash_sim + 0.3 * fft_sim + 0.2 * fm_sim)
        
        return combined_score
    
    def analyze_logo_batch(self, logos: List[Dict]) -> List[Dict]:
        """Extract features from a batch of logos"""
        print(f"🔍 Analyzing {len(logos)} logos...")
        start_time = time.time()
        
        analyzed_logos = []
        successful_analysis = 0
        
        for i, logo in enumerate(logos):
            if i % 500 == 0:
                print(f"   Progress: {i}/{len(logos)}")
            
            try:
                features = self.extract_logo_features(logo['logo_data'])
                logo_with_features = logo.copy()
                logo_with_features['features'] = features
                analyzed_logos.append(logo_with_features)
                
                if features['valid']:
                    successful_analysis += 1
                    
            except Exception as e:
                print(f"❌ Error analyzing {logo.get('website', 'unknown')}: {e}")
                logo_with_features = logo.copy()
                logo_with_features['features'] = {'valid': False}
                analyzed_logos.append(logo_with_features)
        
        elapsed = time.time() - start_time
        print(f"✅ Feature extraction complete: {successful_analysis}/{len(logos)} in {elapsed:.1f}s")
        
        return analyzed_logos
    
    def find_similar_pairs(self, analyzed_logos: List[Dict], threshold: float = 0.7) -> List[Tuple]:
        """Find pairs of similar logos"""
        print(f"🔗 Finding similar pairs (threshold: {threshold})")
        start_time = time.time()
        
        valid_logos = [logo for logo in analyzed_logos if logo['features']['valid']]
        print(f"   Comparing {len(valid_logos)} valid logos...")
        
        similar_pairs = []
        
        for i in range(len(valid_logos)):
            if i % 100 == 0 and i > 0:
                print(f"   Progress: {i}/{len(valid_logos)}")
                
            for j in range(i + 1, len(valid_logos)):
                similarity = self.compute_similarity_score(
                    valid_logos[i]['features'], 
                    valid_logos[j]['features']
                )
                
                if similarity >= threshold:
                    similar_pairs.append((
                        valid_logos[i]['website'],
                        valid_logos[j]['website'],
                        similarity
                    ))
        
        elapsed = time.time() - start_time
        print(f"✅ Found {len(similar_pairs)} similar pairs in {elapsed:.1f}s")
        
        return similar_pairs


class UnionFind:
    """Union-Find data structure for clustering"""
    
    def __init__(self, elements):
        self.parent = {elem: elem for elem in elements}
        self.rank = {elem: 0 for elem in elements}
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
    
    def get_clusters(self):
        clusters = defaultdict(list)
        for elem in self.parent:
            root = self.find(elem)
            clusters[root].append(elem)
        return [cluster for cluster in clusters.values() if len(cluster) > 1]


async def continue_similarity_pipeline():
    """Continue the pipeline with similarity analysis and clustering"""
    
    print("🚀 CONTINUING LOGO SIMILARITY PIPELINE")
    print("=" * 60)
    
    # Step 1: Load extracted logos
    print("📂 Loading extracted logo results...")
    try:
        with open('logo_extraction_results.pkl', 'rb') as f:
            results = pickle.load(f)
        
        successful_logos = results['successful_logos']
        print(f"✅ Loaded {len(successful_logos)} logos for analysis")
        
    except FileNotFoundError:
        print("❌ No logo results found. Run lightning_pipeline.py first.")
        return
    
    # Step 2: Feature extraction and analysis
    print(f"\n🔍 FEATURE EXTRACTION")
    print("-" * 30)
    
    analyzer = FourierLogoAnalyzer()
    analyzed_logos = analyzer.analyze_logo_batch(successful_logos)
    
    # Step 3: Similarity analysis
    print(f"\n🔗 SIMILARITY ANALYSIS")
    print("-" * 30)
    
    similar_pairs = analyzer.find_similar_pairs(analyzed_logos, threshold=0.7)
    
    # Step 4: Clustering
    print(f"\n🎯 CLUSTERING")
    print("-" * 30)
    
    if similar_pairs:
        # Get all websites involved in similarities
        all_websites = set()
        for pair in similar_pairs:
            all_websites.add(pair[0])
            all_websites.add(pair[1])
        
        # Create union-find structure
        uf = UnionFind(all_websites)
        
        # Union similar websites
        for website1, website2, similarity in similar_pairs:
            uf.union(website1, website2)
        
        # Get clusters
        clusters = uf.get_clusters()
        
        print(f"📊 Found {len(clusters)} clusters with 2+ websites")
        
        # Show top clusters
        sorted_clusters = sorted(clusters, key=len, reverse=True)
        for i, cluster in enumerate(sorted_clusters[:10]):
            print(f"   Cluster {i+1}: {len(cluster)} websites")
            for website in cluster[:3]:
                print(f"      - {website}")
            if len(cluster) > 3:
                print(f"      ... and {len(cluster)-3} more")
    
    else:
        print("❌ No similar pairs found with current threshold")
        clusters = []
    
    # Step 5: Save results
    print(f"\n💾 SAVING SIMILARITY RESULTS")
    print("-" * 30)
    
    similarity_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'analyzed_logos': len(analyzed_logos),
        'valid_logos': len([l for l in analyzed_logos if l['features']['valid']]),
        'similar_pairs': similar_pairs,
        'clusters': clusters,
        'total_clustered_websites': sum(len(cluster) for cluster in clusters)
    }
    
    # Save similarity results
    with open('similarity_analysis_results.pkl', 'wb') as f:
        pickle.dump(similarity_results, f)
    
    # Save clusters as CSV
    if clusters:
        cluster_data = []
        for i, cluster in enumerate(clusters):
            for website in cluster:
                cluster_data.append({
                    'cluster_id': i,
                    'cluster_size': len(cluster),
                    'website': website
                })
        
        df_clusters = pd.DataFrame(cluster_data)
        df_clusters.to_csv('logo_clusters.csv', index=False)
        print(f"✅ Saved clusters to logo_clusters.csv")
    
    # Save similar pairs as CSV
    if similar_pairs:
        df_pairs = pd.DataFrame(similar_pairs, columns=['website1', 'website2', 'similarity'])
        df_pairs.to_csv('similar_pairs.csv', index=False)
        print(f"✅ Saved similar pairs to similar_pairs.csv")
    
    print(f"✅ Saved similarity analysis to similarity_analysis_results.pkl")
    
    # Final summary
    print(f"\n🎉 SIMILARITY PIPELINE COMPLETE!")
    print(f"   - Logos analyzed: {len(analyzed_logos)}")
    print(f"   - Valid features: {len([l for l in analyzed_logos if l['features']['valid']])}")
    print(f"   - Similar pairs: {len(similar_pairs)}")
    print(f"   - Clusters found: {len(clusters)}")
    print(f"   - Websites clustered: {sum(len(cluster) for cluster in clusters)}")
    
    return similarity_results


if __name__ == "__main__":
    print("🎯 Starting Logo Similarity Analysis Pipeline")
    results = asyncio.run(continue_similarity_pipeline())
