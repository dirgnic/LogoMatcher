"""
Feature Extractor Module

Extracts comprehensive features from logo images including:
- Perceptual hashes (pHash, DCT, FFT)
- Color histograms
- ORB descriptors
- Logo composition analysis
- Brand intelligence
"""

import cv2
import numpy as np
import hashlib
from .brand_intelligence import BrandIntelligence
from .visual_analyzer import OptimizedVisualAnalyzer
from .hashing import OptimizedMultiScaleHasher


class FeatureExtractor:
    """Extracts comprehensive features from logo images"""
    
    def __init__(self):
        self.brand_intelligence = BrandIntelligence()
        self.visual_analyzer = OptimizedVisualAnalyzer()
        self.multiscale_hasher = OptimizedMultiScaleHasher()
        self.feature_cache = {}
    
    def _get_content_hash(self, image_path):
        """Generate content-based hash for stable caching"""
        try:
            with open(image_path, 'rb') as f:
                content = f.read()
            return hashlib.sha256(content).hexdigest()[:16]
        except:
            return str(hash(image_path))
    
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
                return np.zeros(30, dtype=np.float32)
            
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
    
    def extract_features(self, jpeg_info):
        """Extract all features from a logo image"""
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
            
            # === CORE VISUAL FEATURES ===
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
                
                # Core visual features
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
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error for {domain}: {e}")
            return None
