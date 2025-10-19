"""
Hashing Module

Provides optimized perceptual hashing for logo matching.
Computes pHash, DCT hash, and FFT hash for robust comparison.
"""

import cv2
import numpy as np
import imagehash
from PIL import Image
from scipy.fft import fft2, fftshift


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
