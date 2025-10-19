"""
Advanced Logo Clustering with Brand Intelligence - OPTIMIZED VERSION
Performance optimizations based on comprehensive audit:
- Removed expensive texture analysis
- Removed redundant SIFT/ORB features  
- Optimized color extraction with histograms
- Implemented pHash bucketing for candidate pruning
- Added two-channel rule for threshold validation
- Optimized cache keying with content-based hashing
"""

import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from collections import defaultdict, deque
import imagehash
from PIL import Image, ImageEnhance, ImageFilter
import networkx as nx
from scipy.spatial.distance import hamming
import time
from skimage import morphology, segmentation, filters, feature, measure
from scipy import ndimage
import re
import pytesseract
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import webcolors
from colorthief import ColorThief
import io
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher
import hashlib

# === ULTRA-RELAXED THRESHOLDS FOR AGGRESSIVE SINGLETON MERGING ===
PHASH_THRESHOLD_ULTRA_RELAXED = 60     # Extremely relaxed for maximum merging (out of 64 bits)
ORB_THRESHOLD_ULTRA_RELAXED = 2        # Ultra-low for aggressive matching (min 2 good matches)
COLOR_THRESHOLD_ULTRA_RELAXED = 0.10   # Very relaxed color matching

class BrandIntelligence:
    """Intelligent brand analysis for logo clustering"""
    
    def __init__(self):
        # Industry keyword mapping for brand classification
        self.industry_keywords = {
            'technology': ['tech', 'software', 'app', 'digital', 'cloud', 'ai', 'data', 
                          'microsoft', 'google', 'apple', 'amazon', 'meta', 'adobe',
                          'oracle', 'salesforce', 'zoom', 'slack', 'dropbox', 'github'],
            'finance': ['bank', 'financial', 'capital', 'investment', 'credit', 'loan',
                       'visa', 'mastercard', 'paypal', 'stripe', 'square', 'amex',
                       'jpmorgan', 'goldman', 'wells', 'chase', 'citibank'],
            'ecommerce': ['shop', 'store', 'market', 'buy', 'sell', 'retail', 'commerce',
                         'amazon', 'ebay', 'etsy', 'shopify', 'walmart', 'target',
                         'bestbuy', 'alibaba', 'wish'],
            'cosmetics': ['cosmetics', 'beauty', 'makeup', 'skincare', 'loreal', 'maybelline',
                         'sephora', 'ulta', 'revlon', 'covergirl', 'olay', 'clinique'],
            'automotive': ['auto', 'car', 'vehicle', 'motor', 'ford', 'toyota', 'honda',
                          'bmw', 'mercedes', 'audi', 'volkswagen', 'tesla', 'uber', 'lyft'],
            'food': ['food', 'restaurant', 'cafe', 'pizza', 'burger', 'starbucks',
                    'mcdonalds', 'subway', 'dominos', 'kfc', 'cocacola', 'pepsi'],
            'fashion': ['fashion', 'clothing', 'apparel', 'style', 'nike', 'adidas',
                       'zara', 'hm', 'gap', 'uniqlo', 'forever21', 'guess'],
            'media': ['news', 'media', 'tv', 'radio', 'streaming', 'netflix', 'youtube',
                     'spotify', 'cnn', 'bbc', 'espn', 'disney', 'hulu'],
            'healthcare': ['health', 'medical', 'hospital', 'care', 'pharmacy', 'cvs',
                          'walgreens', 'pfizer', 'johnson', 'abbott', 'medtronic'],
            'education': ['education', 'school', 'university', 'learning', 'course',
                         'harvard', 'stanford', 'mit', 'coursera', 'udemy', 'khan']
        }
        
        # Common brand family patterns
        self.brand_patterns = {
            'google': ['google', 'youtube', 'gmail', 'chrome', 'android', 'drive'],
            'microsoft': ['microsoft', 'windows', 'office', 'azure', 'teams', 'outlook'],
            'amazon': ['amazon', 'aws', 'prime', 'alexa', 'kindle', 'twitch'],
            'apple': ['apple', 'iphone', 'ipad', 'mac', 'ios', 'safari'],
            'meta': ['facebook', 'instagram', 'whatsapp', 'messenger', 'meta'],
            'disney': ['disney', 'pixar', 'marvel', 'starwars', 'espn', 'hulu']
        }

    def extract_brand_family(self, domain):
        """Extract brand family from domain name"""
        domain_lower = domain.lower()
        
        # Check for exact brand family matches
        for family, brands in self.brand_patterns.items():
            for brand in brands:
                if brand in domain_lower:
                    return family
        
        # Extract base brand from domain (remove common extensions)
        brand_name = domain_lower.replace('www.', '').replace('.com', '').replace('.org', '')
        brand_name = brand_name.replace('.net', '').replace('.co', '').split('.')[0]
        
        return brand_name if brand_name else 'unknown'

    def classify_industry(self, domain):
        """Classify industry based on domain and keywords"""
        domain_lower = domain.lower()
        scores = defaultdict(float)
        
        # Keyword-based classification
        for industry, keywords in self.industry_keywords.items():
            for keyword in keywords:
                if keyword in domain_lower:
                    scores[industry] += 1.0
        
        # Return the highest scoring industry or 'general' if no match
        if scores:
            return max(scores, key=scores.get)
        return 'general'

class OptimizedVisualAnalyzer:
    """Optimized visual analysis focusing on core features"""
    
    def __init__(self):
        self.color_cache = {}
    
    def extract_color_palette(self, image, max_colors=5):
        """Fast color palette extraction using histogram analysis"""
        try:
            # Convert to HSV for better color separation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Fast histogram-based color extraction
            hist_h = cv2.calcHist([hsv], [0], None, [36], [0, 180])  # 36 hue bins
            hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])  # 32 saturation bins
            hist_v = cv2.calcHist([hsv], [2], None, [32], [0, 256])  # 32 value bins
            
            # Find dominant hues
            dominant_hues = np.argsort(hist_h.flatten())[-max_colors:]
            
            # Convert back to RGB colors
            colors = []
            for hue_bin in dominant_hues:
                hue = hue_bin * 5  # Convert bin to hue value
                # Use moderate saturation and value for representative color
                hsv_color = np.uint8([[[hue, 128, 200]]])
                rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0, 0]
                colors.append(tuple(rgb_color))
            
            return {
                'dominant_colors': colors,
                'color_count': len(colors),
                'hist_h': hist_h.flatten()[:10],  # Top 10 hue bins only
                'hist_s': hist_s.flatten()[:10],  # Top 10 saturation bins only
                'hist_v': hist_v.flatten()[:10]   # Top 10 value bins only
            }
            
        except Exception as e:
            print(f"Color palette extraction error: {e}")
            return {'dominant_colors': [], 'color_count': 0, 'hist_h': [], 'hist_s': [], 'hist_v': []}

    def analyze_logo_composition(self, image):
        """Fast logo composition analysis"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Fast edge detection for layout analysis
            edges = cv2.Canny(gray, 50, 150)
            
            # Simple region analysis
            center_region = edges[height//4:3*height//4, width//4:3*width//4]
            edge_density = np.sum(center_region > 0) / (center_region.shape[0] * center_region.shape[1])
            
            # Basic text vs symbol classification
            text_score = 1.0 if edge_density < 0.1 else 0.3  # Low edge density suggests text
            symbol_score = 1.0 - text_score
            
            return {
                'text_score': text_score,
                'symbol_score': symbol_score,
                'edge_density': edge_density,
                'layout': 'horizontal' if width > height * 1.5 else 'square'
            }
            
        except Exception as e:
            print(f"Logo composition error: {e}")
            return {'text_score': 0.5, 'symbol_score': 0.5, 'edge_density': 0.1, 'layout': 'square'}

class OptimizedMultiScaleHasher:
    """Optimized hashing focusing on core perceptual features"""
    
    def __init__(self):
        self.dct_cache = {}
    
    def compute_phash_with_bucketing(self, image):
        """Compute pHash with bucketing for fast candidate pruning"""
        try:
            # Standard pHash computation
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            phash = imagehash.phash(pil_img, hash_size=16)  # 16x16 for higher precision
            
            # Create bucket key from first 8 bits for fast filtering
            bucket_key = str(phash)[:8]
            
            return {
                'hash': str(phash),
                'bucket': bucket_key,
                'numeric': int(str(phash), 16)
            }
            
        except Exception as e:
            print(f"pHash computation error: {e}")
            return {'hash': '0' * 64, 'bucket': '0' * 8, 'numeric': 0}
    
    def compute_dct_hash(self, image):
        """Fast DCT-based hash for frequency domain analysis"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Resize to 32x32 for DCT
            resized = cv2.resize(gray, (32, 32))
            
            # Apply DCT (Discrete Cosine Transform)
            dct = cv2.dct(np.float32(resized))
            
            # Take low-frequency components (top-left 8x8)
            low_freq = dct[:8, :8]
            
            # Compute median for thresholding
            median = np.median(low_freq)
            
            # Create binary hash
            hash_bits = (low_freq > median).flatten()
            
            # Convert to hex string
            hash_str = ''.join(['1' if bit else '0' for bit in hash_bits])
            
            return hash_str
            
        except Exception as e:
            print(f"DCT hash error: {e}")
            return '0' * 64

    def compute_fft_hash(self, image):
        """FFT-based hash for frequency analysis"""
        try:
            # Convert to grayscale and resize
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            
            # Apply 2D FFT
            f_transform = fft2(resized)
            f_shifted = fftshift(f_transform)
            
            # Get magnitude spectrum
            magnitude = np.abs(f_shifted)
            
            # Focus on low frequencies (center region)
            center = 32
            low_freq_region = magnitude[center-8:center+8, center-8:center+8]
            
            # Create hash from frequency magnitudes
            median_mag = np.median(low_freq_region)
            hash_bits = (low_freq_region > median_mag).flatten()
            
            hash_str = ''.join(['1' if bit else '0' for bit in hash_bits])
            
            return hash_str
            
        except Exception as e:
            print(f"FFT hash error: {e}")
            return '0' * 256

class OptimizedLogoClusterer:
    """Optimized logo clustering system with performance improvements"""
    
    def __init__(self, jpeg_folder_path):
        self.jpeg_folder_path = jpeg_folder_path
        self.jpeg_files = []
        self.batch_size = 50
        
        # Initialize optimized components
        self.brand_intelligence = BrandIntelligence()
        self.visual_analyzer = OptimizedVisualAnalyzer()
        self.multiscale_hasher = OptimizedMultiScaleHasher()
        
        # Performance optimizations
        self.feature_cache = {}
        self.phash_buckets = defaultdict(list)  # For fast candidate pruning
        
        # Ultra-relaxed thresholds for aggressive singleton reduction
        self.thresholds = {
            'phash': PHASH_THRESHOLD_ULTRA_RELAXED,
            'orb': ORB_THRESHOLD_ULTRA_RELAXED,  
            'color': COLOR_THRESHOLD_ULTRA_RELAXED
        }
        
        print(f" OPTIMIZED Advanced Logo Clusterer initialized")
        print(f" Ultra-relaxed thresholds: pHash={self.thresholds['phash']}, ORB={self.thresholds['orb']}, Color={self.thresholds['color']}")
    
    def _get_content_hash(self, image_path):
        """Generate content-based hash for stable caching"""
        try:
            with open(image_path, 'rb') as f:
                content = f.read()
            return hashlib.sha256(content).hexdigest()[:16]
        except:
            return str(hash(image_path))
    
    def load_jpeg_files(self):
        """Load and validate JPEG files"""
        print(" Loading JPEG files...")
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
        print(f" Loaded {len(self.jpeg_files)} JPEG files in {elapsed:.2f}s")
        return len(self.jpeg_files)
    
    def compute_perceptual_hash(self, img_path):
        """Optimized perceptual hash computation with bucketing"""
        try:
            image = cv2.imread(img_path)
            if image is None:
                return {'hash': '0'*64, 'bucket': '0'*8}
            
            return self.multiscale_hasher.compute_phash_with_bucketing(image)
            
        except Exception as e:
            print(f"pHash error for {img_path}: {e}")
            return {'hash': '0'*64, 'bucket': '0'*8}
    
    def compute_fast_color_histogram(self, img_path):
        """Fast color histogram computation"""
        try:
            image = cv2.imread(img_path)
            if image is None:
                return np.zeros(30, dtype=np.float32)  # Reduced from 48 to 30
            
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Compute histograms with fewer bins for speed
            hist_h = cv2.calcHist([hsv], [0], None, [10], [0, 180])  # 10 hue bins
            hist_s = cv2.calcHist([hsv], [1], None, [10], [0, 256])  # 10 saturation bins  
            hist_v = cv2.calcHist([hsv], [2], None, [10], [0, 256])  # 10 value bins
            
            # Normalize and combine
            hist_h = hist_h.flatten() / (hist_h.sum() + 1e-7)
            hist_s = hist_s.flatten() / (hist_s.sum() + 1e-7)
            hist_v = hist_v.flatten() / (hist_v.sum() + 1e-7)
            
            return np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)
            
        except Exception as e:
            print(f"Color histogram error for {img_path}: {e}")
            return np.zeros(30, dtype=np.float32)
    
    def compute_orb_descriptors(self, img_path):
        """Compute ORB descriptors for keypoint matching"""
        try:
            image = cv2.imread(img_path)
            if image is None:
                return [], None
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Initialize ORB detector with optimized parameters
            orb = cv2.ORB_create(
                nfeatures=100,      # Reduced from default 500 for speed
                scaleFactor=1.2,    # Default
                nlevels=8,          # Default
                edgeThreshold=15,   # Slightly relaxed for more features
                patchSize=31        # Default
            )
            
            # Detect keypoints and compute descriptors
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            
            return keypoints, descriptors
            
        except Exception as e:
            print(f"ORB computation error for {img_path}: {e}")
            return [], None
    
    def extract_optimized_features(self, jpeg_info):
        """Extract core features optimized for speed and accuracy"""
        filepath = jpeg_info['filepath']
        domain = jpeg_info['domain']
        index = jpeg_info['index']
        
        # Check cache first
        content_hash = self._get_content_hash(filepath)
        if content_hash in self.feature_cache:
            return self.feature_cache[content_hash]
        
        try:
            # Load image once
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError(f"Could not load image: {filepath}")
            
            # === BRAND INTELLIGENCE ===
            brand_family = self.brand_intelligence.extract_brand_family(domain)
            industry = self.brand_intelligence.classify_industry(domain)
            
            # === CORE VISUAL FEATURES (OPTIMIZED) ===
            # 1. Primary: pHash with bucketing
            phash_data = self.compute_perceptual_hash(filepath)
            
            # 2. Secondary: Fast color histogram
            color_hist = self.compute_fast_color_histogram(filepath)
            
            # 3. Tertiary: ORB descriptors for keypoint matching
            keypoints, orb_descriptors = self.compute_orb_descriptors(filepath)
            
            # 4. Supplementary: Logo composition
            composition = self.visual_analyzer.analyze_logo_composition(image)
            
            # 5. Advanced: DCT hash for frequency analysis
            dct_hash = self.multiscale_hasher.compute_dct_hash(image)
            
            # 6. Advanced: FFT hash for pattern recognition
            fft_hash = self.multiscale_hasher.compute_fft_hash(image)
            
            features = {
                # Basic info
                'domain': domain,
                'filepath': filepath,
                'index': index,
                'content_hash': content_hash,
                
                # Brand intelligence
                'brand_family': brand_family,
                'industry': industry,
                
                # Core visual features (optimized)
                'phash': phash_data['hash'],
                'phash_bucket': phash_data['bucket'],
                'phash_numeric': phash_data['numeric'],
                'color_histogram': color_hist,
                'orb_descriptors': orb_descriptors,
                'orb_keypoints_count': len(keypoints) if keypoints else 0,
                'composition': composition,
                'dct_hash': dct_hash,
                'fft_hash': fft_hash
            }
            
            # Cache the results
            self.feature_cache[content_hash] = features
            
            # Add to pHash buckets for fast candidate pruning
            self.phash_buckets[phash_data['bucket']].append(index)
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error for {domain}: {e}")
            return None
    
    def extract_all_features_parallel(self):
        """Extract features with optimized parallel processing"""
        print(" Extracting optimized features...")
        start_time = time.time()
        
        all_features = []
        
        with ThreadPoolExecutor(max_workers=4) as executor:  # Reduced workers to avoid memory issues
            futures = {
                executor.submit(self.extract_optimized_features, jpeg_info): jpeg_info 
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
                            print(f" Processed {len(all_features)} files at {rate:.1f} files/sec")
                            
                except Exception as e:
                    print(f"Feature extraction failed for {jpeg_info['domain']}: {e}")
        
        elapsed = time.time() - start_time
        print(f" Feature extraction completed: {len(all_features)} files in {elapsed:.2f}s ({len(all_features)/elapsed:.1f} files/sec)")
        
        return all_features
    
    def calculate_optimized_similarity(self, features1, features2):
        """Optimized similarity calculation with two-channel rule"""
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
            
            # 4. Brand family bonus
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
    
    def cluster_with_candidate_pruning(self, features_list):
        """Cluster with pHash bucketing for candidate pruning"""
        print(" Clustering with optimized candidate pruning...")
        start_time = time.time()
        
        n = len(features_list)
        similarity_matrix = np.zeros((n, n))
        
        # Build index mapping
        domain_to_index = {f['domain']: i for i, f in enumerate(features_list)}
        
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
                    
                    similarity_data = self.calculate_optimized_similarity(features1, features2)
                    similarity_matrix[i][j] = similarity_data['similarity']
                    similarity_matrix[j][i] = similarity_data['similarity']
                    
                    comparisons_made += 1
        
        # Phase 2: Advanced clustering with ultra-relaxed thresholds
        print(f" Made {comparisons_made:,} optimized comparisons (vs {n*(n-1)//2:,} brute force)")
        print(f" Bucket efficiency: {bucket_hits:,} candidate hits")
        
        # Convert to distance matrix (higher similarity = lower distance)
        distance_matrix = 1 - similarity_matrix
        
        # Ensure diagonal is zero (required by scipy)
        np.fill_diagonal(distance_matrix, 0.0)
        
        # Hierarchical clustering with ultra-relaxed threshold
        try:
            # Use single linkage for aggressive merging
            linkage_matrix = linkage(squareform(distance_matrix), method='single')
            
            # Ultra-relaxed distance threshold for maximum merging (lower = more aggressive)
            ultra_relaxed_threshold = 0.85  # Very low threshold for maximum merging
            cluster_labels = fcluster(linkage_matrix, ultra_relaxed_threshold, criterion='distance')
            
        except Exception as e:
            print(f"Clustering error: {e}, using fallback")
            cluster_labels = np.arange(n) + 1  # Each item in its own cluster
        
        # Organize results
        clusters = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            clusters[label].append(features_list[idx]['domain'])
        
        elapsed = time.time() - start_time
        print(f" Clustering completed in {elapsed:.2f}s")
        print(f" Performance: {comparisons_made/elapsed:.0f} comparisons/sec")
        
        return dict(clusters), similarity_matrix
    
    def analyze_cluster_quality(self, clusters):
        """Analyze cluster quality with focus on brand coherence"""
        print("\n ULTRA-RELAXED CLUSTER ANALYSIS")
        
        total_domains = sum(len(domains) for domains in clusters.values())
        num_clusters = len(clusters)
        
        # Cluster size analysis
        cluster_sizes = [len(domains) for domains in clusters.values()]
        singletons = sum(1 for size in cluster_sizes if size == 1)
        
        # Brand coherence analysis
        brand_coherent_clusters = 0
        total_brand_groups = 0
        
        for cluster_id, domains in clusters.items():
            if len(domains) > 1:  # Only analyze multi-domain clusters
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
        singleton_rate = (singletons / total_domains) * 100
        brand_coherence_rate = (brand_coherent_clusters / max(total_brand_groups, 1)) * 100
        
        print(f" Total clusters: {num_clusters}")
        print(f" Total domains: {total_domains}")
        print(f" Average cluster size: {total_domains/num_clusters:.2f}")
        print(f" Cluster size range: {min(cluster_sizes)} - {max(cluster_sizes)}")
        print(f" Singletons: {singletons} ({singleton_rate:.1f}%)")
        print(f" Brand coherence: {brand_coherent_clusters}/{total_brand_groups} ({brand_coherence_rate:.1f}%)")
        
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
        
        print(f"\n Cluster size distribution:")
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
        
        print(f" Results saved:")
        print(f"    CSV: {csv_filename}")
        print(f"    Pickle: {pickle_filename}")
        
        return csv_path, pickle_path
    
    def aggressive_singleton_merging(self, clusters, features_list):
        """Force aggressive merging of remaining singletons"""
        print(" Applying aggressive singleton merging...")
        
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
        
        print(f" Found {len(singletons)} singletons to merge")
        
        # Create features lookup
        domain_to_features = {f['domain']: f for f in features_list}
        
        # Aggressive merging with very low thresholds
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
                
                # Super relaxed matching criteria
                sim_data = self.calculate_optimized_similarity(seed_features, other_features)
                
                # Accept if ANY of these criteria are met:
                # 1. pHash difference <= 62 bits (very relaxed)
                # 2. Color similarity >= 0.05 (extremely relaxed)
                # 3. ORB similarity >= 0.1 (very relaxed)
                # 4. Same brand family
                # 5. Similar industry
                accept = False
                
                # pHash check
                if 'phash' in sim_data['details']:
                    phash1 = seed_features.get('phash', '')
                    phash2 = other_features.get('phash', '')
                    if phash1 and phash2:
                        hash_diff = bin(int(phash1, 16) ^ int(phash2, 16)).count('1')
                        if hash_diff <= 62:  # Accept almost everything
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
        
        # Add any remaining singletons as individual clusters
        for singleton in remaining_singletons:
            new_clusters.append([singleton])
        
        # Combine with original multi-clusters
        all_clusters = multi_clusters + new_clusters
        
        # Convert back to dict format
        result_clusters = {}
        for i, cluster in enumerate(all_clusters):
            result_clusters[i + 1] = cluster
        
        print(f" Aggressive merging: {len(singletons)} singletons → {len([c for c in result_clusters.values() if len(c) == 1])} remaining")
        
        return result_clusters

    def run_optimized_clustering(self):
        """Main optimized clustering pipeline"""
        print(" STARTING OPTIMIZED LOGO CLUSTERING")
        print("=" * 60)
        
        overall_start = time.time()
        
        # Step 1: Load files
        num_files = self.load_jpeg_files()
        if num_files == 0:
            print(" No JPEG files found!")
            return None
        
        # Step 2: Extract optimized features
        features_list = self.extract_all_features_parallel()
        if not features_list:
            print(" Feature extraction failed!")
            return None
        
        # Step 3: Cluster with candidate pruning
        clusters, similarity_matrix = self.cluster_with_candidate_pruning(features_list)
        
        # Step 3.5: Aggressive singleton merging
        clusters = self.aggressive_singleton_merging(clusters, features_list)
        
        # Step 4: Analyze quality
        quality_metrics = self.analyze_cluster_quality(clusters)
        
        # Step 5: Save results
        csv_path, pickle_path = self.save_results(clusters, features_list, "_optimized")
        
        overall_elapsed = time.time() - overall_start
        print(f"\n TOTAL PROCESSING TIME: {overall_elapsed:.2f}s")
        print(f" Performance: {len(features_list)/overall_elapsed:.1f} files/sec")
        print("=" * 60)
        
        return {
            'clusters': clusters,
            'features': features_list,
            'quality_metrics': quality_metrics,
            'similarity_matrix': similarity_matrix,
            'csv_path': csv_path,
            'pickle_path': pickle_path
        }

def main():
    """Main function to run optimized clustering"""
    # Set path to JPEG folder
    jpeg_folder = "/Users/ingridcorobana/Desktop/personal_projs/logo_matcher/extracted_logos_20251019_174045"
    
    if not os.path.exists(jpeg_folder):
        print(f" JPEG folder not found: {jpeg_folder}")
        return
    
    # Initialize optimized clusterer
    clusterer = OptimizedLogoClusterer(jpeg_folder)
    
    # Run optimized clustering
    results = clusterer.run_optimized_clustering()
    
    if results:
        print("\n OPTIMIZED CLUSTERING COMPLETED SUCCESSFULLY!")
        print(f" Generated {results['quality_metrics']['total_clusters']} clusters")
        print(f" Singleton rate: {results['quality_metrics']['singleton_rate']:.1f}%") 
        print(f" Brand coherence: {results['quality_metrics']['brand_coherence_rate']:.1f}%")
    else:
        print(" Clustering failed!")

if __name__ == "__main__":
    main()
