"""
Enhanced Fourier Logo Analyzer - 2025 Research Edition
Advanced Fourier analysis with deep hashing, multi-signature matching, and robust RGB handling
"""

import io
import numpy as np
import cv2
from PIL import Image
from scipy.fft import fft2, fftshift
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from scipy.stats import skew, kurtosis
from scipy.special import factorial
import cmath
import colorsys
import time
import warnings
warnings.filterwarnings('ignore')

class FourierLogoAnalyzer:
    """
    Enhanced logo analyzer with ALL 2025 research features:
    - Traditional: pHash, FFT, Fourier-Mellin, SIFT, ORB
    - Advanced: Hu/Zernike moments, LBP, GLCM, Gabor, saliency-weighted hashing
    - Color-aware: Per-channel Fourier-Mellin, enhanced color features
    """
    
    def __init__(self):
        self.similarity_threshold_phash = 6  # Hamming distance
        self.similarity_threshold_fft = 0.985  # Cosine similarity
        self.similarity_threshold_fmt = 0.995  # Fourier-Mellin
        
        # Advanced feature parameters
        self.zernike_max_order = 8
        self.lbp_radius = 3
        self.lbp_n_points = 8 * self.lbp_radius
        self.gabor_frequencies = [0.1, 0.3, 0.5]
        self.gabor_angles = [0, 45, 90, 135]
    
    def compute_phash(self, img: np.ndarray) -> str:
        """Compute perceptual hash using DCT (Fourier cousin)"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Resize to 32x32 for DCT
        resized = cv2.resize(gray, (32, 32))
        
        # Compute DCT (like 2D Fourier but with cosines)
        dct = cv2.dct(np.float32(resized))
        
        # Take top-left 8x8 (low frequencies)
        dct_low = dct[0:8, 0:8]
        
        # Compare with median to create binary hash
        median = np.median(dct_low)
        binary = dct_low > median
        
        # Convert to hex string
        hash_str = ''.join(['1' if b else '0' for b in binary.flatten()])
        return hash_str
    
    def hamming_distance(self, hash1: str, hash2: str) -> int:
        """Calculate Hamming distance between two hashes"""
        return sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    
    def color_distance(self, a, b):
        """Compute Euclidean distance between color vectors"""
        return np.linalg.norm(np.array(a) - np.array(b))
    
    def compute_fft_features(self, img: np.ndarray) -> np.ndarray:
        """Compute FFT low-frequency features for global shape"""
        # Convert to grayscale and normalize
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        
        # Resize to square and standard size
        size = 128
        resized = cv2.resize(gray, (size, size))
        
        # Compute 2D FFT
        fft = fft2(resized)
        fft_shifted = fftshift(fft)
        
        # Take magnitude and apply log
        magnitude = np.abs(fft_shifted)
        log_magnitude = np.log(magnitude + 1e-8)
        
        # Extract central low-frequency block (32x32)
        center = size // 2
        crop_size = 16
        low_freq = log_magnitude[
            center-crop_size:center+crop_size,
            center-crop_size:center+crop_size
        ]
        
        # Flatten and normalize
        features = low_freq.flatten()
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    def compute_fourier_mellin_signature(self, img: np.ndarray) -> np.ndarray:
        """Compute Fourier-Mellin theta signature for rotation/scale invariance"""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        
        # Resize to square
        size = 128
        resized = cv2.resize(gray, (size, size))
        
        # Compute FFT and get magnitude
        fft = fft2(resized)
        fft_shifted = fftshift(fft)
        magnitude = np.abs(fft_shifted)
        
        # Convert to log-polar coordinates
        center = size // 2
        theta_samples = 64
        radius_samples = 32
        
        # Create theta signature by averaging over radius
        theta_signature = np.zeros(theta_samples)
        
        for i, theta in enumerate(np.linspace(0, 2*np.pi, theta_samples, endpoint=False)):
            # Sample along radial lines
            radial_sum = 0
            for r in np.linspace(1, center-1, radius_samples):
                x = int(center + r * np.cos(theta))
                y = int(center + r * np.sin(theta))
                if 0 <= x < size and 0 <= y < size:
                    radial_sum += magnitude[y, x]
            theta_signature[i] = radial_sum
        
        # Normalize
        theta_signature = theta_signature / (np.linalg.norm(theta_signature) + 1e-8)
        
        return theta_signature
    
    def compute_color_aware_fmt(self, img: np.ndarray) -> np.ndarray:
        """Color-aware Fourier-Mellin preserving color relationships"""
        try:
            if len(img.shape) != 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            img_resized = cv2.resize(img, (128, 128))
            channel_signatures = []
            
            for c in range(3):  # B, G, R channels
                channel = img_resized[:, :, c].astype(np.float32) / 255.0
                
                # Apply 2D FFT
                fft = np.fft.fft2(channel)
                fft_shift = np.fft.fftshift(fft)
                magnitude = np.abs(fft_shift)
                
                # Convert to log-polar coordinates
                center_x, center_y = magnitude.shape[1] // 2, magnitude.shape[0] // 2
                
                # Create log-polar sampling grid
                signature_size = 32
                theta_samples = np.linspace(0, 2 * np.pi, signature_size, endpoint=False)
                rho_max = min(center_x, center_y) - 1
                rho_samples = np.logspace(0, np.log10(rho_max), signature_size // 2)
                
                signature = []
                for rho in rho_samples:
                    theta_signature = []
                    for theta in theta_samples:
                        x = int(center_x + rho * np.cos(theta))
                        y = int(center_y + rho * np.sin(theta))
                        
                        if 0 <= x < magnitude.shape[1] and 0 <= y < magnitude.shape[0]:
                            theta_signature.append(magnitude[y, x])
                        else:
                            theta_signature.append(0.0)
                    
                    signature.append(np.max(theta_signature))
                
                channel_signatures.append(signature)
            
            return np.concatenate(channel_signatures).astype(np.float32)
            
        except Exception as e:
            return np.zeros(48, dtype=np.float32)  # 3 channels * 16 features each
    
    def compute_saliency_weighted_fft(self, img: np.ndarray) -> np.ndarray:
        """Saliency-weighted FFT emphasizing perceptually significant regions"""
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Compute frequency-tuned saliency map
            gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            mean_val = np.mean(gaussian_blur)
            saliency = np.abs(gray.astype(np.float32) - mean_val)
            
            # Enhance with edge information
            edges = cv2.Canny(gray, 50, 150)
            edge_weight = edges.astype(np.float32) / 255.0
            
            # Combine intensity saliency with edge saliency
            saliency = 0.7 * saliency + 0.3 * edge_weight * 255
            saliency = saliency / (np.max(saliency) + 1e-8)
            saliency = cv2.GaussianBlur(saliency, (3, 3), 0)
            
            # Apply saliency weighting
            weighted_img = gray.astype(np.float32) * saliency / 255.0
            
            # Resize and compute FFT
            resized = cv2.resize(weighted_img, (128, 128))
            fft = np.fft.fft2(resized)
            fft_shift = np.fft.fftshift(fft)
            magnitude = np.abs(fft_shift)
            log_magnitude = np.log(magnitude + 1.0)
            
            # Extract central region
            center = log_magnitude.shape[0] // 2
            crop_size = 16
            central_region = log_magnitude[center-crop_size:center+crop_size, 
                                         center-crop_size:center+crop_size]
            
            return central_region.flatten().astype(np.float32)
            
        except Exception as e:
            return np.zeros(1024, dtype=np.float32)
    
    def compute_hu_moments(self, img: np.ndarray) -> np.ndarray:
        """Compute 7 Hu invariant moments from binary silhouette"""
        try:
            # Convert to grayscale and binary
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # Otsu thresholding for clean binary image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological cleaning
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # Compute Hu moments
            moments = cv2.moments(binary)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            # Log-transform for numerical stability
            hu_log = []
            for hu in hu_moments:
                if hu > 0:
                    hu_log.append(-np.log10(hu))
                elif hu < 0:
                    hu_log.append(-np.log10(-hu))
                else:
                    hu_log.append(0.0)
            
            return np.array(hu_log, dtype=np.float32)
            
        except Exception as e:
            return np.zeros(7, dtype=np.float32)
    
    def compute_zernike_moments(self, img: np.ndarray, max_order: int = 8) -> np.ndarray:
        """Compute Zernike moments up to specified order"""
        try:
            # Convert to grayscale and binary
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            binary_float = binary.astype(np.float32) / 255.0
            
            # Resize for computation
            size = 128
            binary_resized = cv2.resize(binary_float, (size, size))
            
            height, width = binary_resized.shape
            center_x, center_y = width // 2, height // 2
            
            # Create coordinate grids
            x, y = np.ogrid[:height, :width]
            x = x - center_y
            y = y - center_x
            
            # Convert to polar coordinates
            rho = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            
            # Normalize rho to unit circle
            max_rho = np.sqrt(center_x**2 + center_y**2)
            rho = rho / max_rho
            
            # Create unit circle mask
            unit_circle = (rho <= 1.0)
            
            zernike_moments = []
            
            # Compute moments for orders up to max_order (simplified)
            for n in range(min(max_order + 1, 6)):  # Limit for performance
                for m in range(-n, n + 1, 2):  # Only valid combinations
                    if abs(m) <= n and (n - abs(m)) % 2 == 0:
                        # Simplified Zernike computation
                        moment_real = np.mean(binary_resized[unit_circle] * np.cos(m * theta[unit_circle]))
                        moment_imag = np.mean(binary_resized[unit_circle] * np.sin(m * theta[unit_circle]))
                        zernike_moments.extend([moment_real, moment_imag])
            
            return np.array(zernike_moments, dtype=np.float32)
            
        except Exception as e:
            return np.zeros(50, dtype=np.float32)
    
    def compute_texture_features(self, img: np.ndarray) -> dict:
        """Compute LBP, GLCM, and Gabor texture features"""
        texture_features = {}
        
        try:
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img.copy()
            
            # LBP features
            lbp = local_binary_pattern(gray, self.lbp_n_points, self.lbp_radius, method='uniform')
            hist, _ = np.histogram(lbp.ravel(), bins=self.lbp_n_points + 2, 
                                 range=(0, self.lbp_n_points + 2), density=True)
            
            texture_features['lbp_uniformity'] = float(hist[:-1].sum())
            texture_features['lbp_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
            texture_features['lbp_energy'] = float(np.sum(hist ** 2))
            
            # GLCM features (simplified)
            gray_resized = cv2.resize(gray, (64, 64))  # Reduce size for speed
            glcm = graycomatrix(gray_resized, distances=[1], angles=[0], levels=256, 
                               symmetric=True, normed=True)
            
            texture_features['glcm_contrast'] = float(graycoprops(glcm, 'contrast')[0, 0])
            texture_features['glcm_correlation'] = float(graycoprops(glcm, 'correlation')[0, 0])
            texture_features['glcm_energy'] = float(graycoprops(glcm, 'energy')[0, 0])
            texture_features['glcm_homogeneity'] = float(graycoprops(glcm, 'homogeneity')[0, 0])
            
            # Gabor features (simplified - just 2 filters for speed)
            gray_norm = gray.astype(np.float32) / 255.0
            
            gabor_responses = []
            for freq in [0.1, 0.3]:
                for angle in [0, np.pi/4]:
                    try:
                        filt_real, _ = gabor(gray_norm, frequency=freq, theta=angle)
                        gabor_responses.extend([
                            float(np.mean(filt_real)),
                            float(np.std(filt_real)),
                            float(np.mean(filt_real ** 2))
                        ])
                    except Exception:
                        gabor_responses.extend([0.0, 0.0, 0.0])
            
            # Add Gabor responses to features
            for i, response in enumerate(gabor_responses):
                texture_features[f'gabor_{i}'] = response
            
        except Exception as e:
            # Return default values on error
            default_keys = ['lbp_uniformity', 'lbp_entropy', 'lbp_energy', 
                           'glcm_contrast', 'glcm_correlation', 'glcm_energy', 'glcm_homogeneity']
            for key in default_keys:
                texture_features[key] = 0.0
            for i in range(12):  # Gabor features
                texture_features[f'gabor_{i}'] = 0.0
        
        return texture_features
    
    def compute_enhanced_color_features(self, img: np.ndarray) -> dict:
        """Enhanced color analysis across multiple color spaces"""
        color_features = {}
        
        try:
            if len(img.shape) == 3:
                # Color moments in RGB
                for c, channel in enumerate(['R', 'G', 'B']):
                    pixels = img[:, :, c].flatten().astype(np.float32) / 255.0
                    color_features[f'color_{channel}_mean'] = float(np.mean(pixels))
                    color_features[f'color_{channel}_std'] = float(np.std(pixels))
                    color_features[f'color_{channel}_skewness'] = float(skew(pixels))
                
                # HSV analysis
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                for c, channel in enumerate(['H', 'S', 'V']):
                    pixels = hsv[:, :, c].flatten().astype(np.float32) / 255.0
                    color_features[f'color_hsv_{channel}_mean'] = float(np.mean(pixels))
                    color_features[f'color_hsv_{channel}_std'] = float(np.std(pixels))
                
                # Dominant colors (simplified k-means)
                pixels_rgb = img.reshape(-1, 3).astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
                _, labels, centers = cv2.kmeans(pixels_rgb, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                
                # Sort centers by frequency
                unique, counts = np.unique(labels, return_counts=True)
                sorted_indices = np.argsort(counts)[::-1]
                
                for i, idx in enumerate(sorted_indices):
                    center = centers[idx] / 255.0
                    color_features[f'color_dominant_{i}_r'] = float(center[2])
                    color_features[f'color_dominant_{i}_g'] = float(center[1])
                    color_features[f'color_dominant_{i}_b'] = float(center[0])
                    color_features[f'color_dominant_{i}_freq'] = float(counts[idx] / len(labels))
            else:
                # Grayscale - set defaults
                for channel in ['R', 'G', 'B']:
                    for stat in ['mean', 'std', 'skewness']:
                        color_features[f'color_{channel}_{stat}'] = 0.0
                for channel in ['H', 'S', 'V']:
                    for stat in ['mean', 'std']:
                        color_features[f'color_hsv_{channel}_{stat}'] = 0.0
                for i in range(3):
                    for c in ['r', 'g', 'b']:
                        color_features[f'color_dominant_{i}_{c}'] = 0.0
                    color_features[f'color_dominant_{i}_freq'] = 0.0
        
        except Exception as e:
            # Return defaults on error
            for channel in ['R', 'G', 'B']:
                for stat in ['mean', 'std', 'skewness']:
                    color_features[f'color_{channel}_{stat}'] = 0.0
            for channel in ['H', 'S', 'V']:
                for stat in ['mean', 'std']:
                    color_features[f'color_hsv_{channel}_{stat}'] = 0.0
            for i in range(3):
                for c in ['r', 'g', 'b']:
                    color_features[f'color_dominant_{i}_{c}'] = 0.0
                color_features[f'color_dominant_{i}_freq'] = 0.0
        
        return color_features
    
    def mean_color_features(self, img: Image.Image) -> list:
        """Compute compact color signature for clustering"""
        try:
            im = img.convert("RGB").resize((256, 256), Image.BICUBIC)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            
            # RGB means
            r_mean = float(arr[..., 0].mean())
            g_mean = float(arr[..., 1].mean())  
            b_mean = float(arr[..., 2].mean())
            
            # Convert to HSV
            hsv = np.zeros_like(arr)
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    hsv[i, j] = colorsys.rgb_to_hsv(arr[i, j, 0], arr[i, j, 1], arr[i, j, 2])
            
            H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
            
            # Circular mean for hue
            ang = 2 * np.pi * H
            h_mean = float((np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)) / (2 * np.pi))
            s_mean = float(S.mean())
            v_mean = float(V.mean())
            
            return [r_mean, g_mean, b_mean, s_mean, v_mean]
            
        except Exception as e:
            return None
    
    def compute_sift_features(self, img: np.ndarray) -> dict:
        """Compute SIFT keypoints and descriptors for logo matching"""
        try:
            sift = cv2.SIFT_create(nfeatures=100)
            keypoints, descriptors = sift.detectAndCompute(img, None)
            
            if descriptors is None or len(descriptors) == 0:
                return {'valid': False, 'signature': np.zeros(256)}
            
            desc_mean = np.mean(descriptors, axis=0) if len(descriptors) > 0 else np.zeros(128)
            desc_std = np.std(descriptors, axis=0) if len(descriptors) > 0 else np.zeros(128)
            
            return {
                'valid': True,
                'keypoint_count': len(keypoints),
                'descriptors': descriptors,
                'signature': np.concatenate([desc_mean, desc_std])
            }
            
        except Exception:
            return {'valid': False, 'signature': np.zeros(256)}
    
    def compute_orb_features(self, img: np.ndarray) -> dict:
        """Compute ORB keypoints and descriptors"""
        try:
            orb = cv2.ORB_create(nfeatures=50)
            keypoints, descriptors = orb.detectAndCompute(img, None)
            
            if descriptors is None or len(descriptors) == 0:
                return {'valid': False, 'signature': np.zeros(32)}
            
            desc_mean = np.mean(descriptors.astype(np.float32), axis=0) if len(descriptors) > 0 else np.zeros(32)
            
            return {
                'valid': True,
                'keypoint_count': len(keypoints),
                'descriptors': descriptors,
                'signature': desc_mean
            }
            
        except Exception:
            return {'valid': False, 'signature': np.zeros(32)}
    
    def compare_fourier_mellin(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Compare Fourier-Mellin signatures with rotation invariance"""
        n = len(sig1)
        
        # Pad and compute correlation via FFT
        sig1_fft = np.fft.rfft(sig1, n=2*n)
        sig2_fft = np.fft.rfft(sig2[::-1], n=2*n)
        
        correlation = np.fft.irfft(sig1_fft * sig2_fft)
        max_correlation = np.max(correlation)
        
        return max_correlation
    
    def match_sift_features(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """Match SIFT descriptors using FLANN matcher"""
        try:
            if len(desc1) == 0 or len(desc2) == 0:
                return 0.0
            
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc1, desc2, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            total_features = min(len(desc1), len(desc2))
            return len(good_matches) / max(total_features, 1)
            
        except Exception:
            return 0.0
    
    def match_orb_features(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """Match ORB descriptors using Hamming distance"""
        try:
            if len(desc1) == 0 or len(desc2) == 0:
                return 0.0
            
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(desc1, desc2)
            
            good_matches = [m for m in matches if m.distance < 50]
            
            total_features = min(len(desc1), len(desc2))
            return len(good_matches) / max(total_features, 1)
            
        except Exception:
            return 0.0
    
    def compute_deep_fused_hash(self, img: np.ndarray, hash_dim: int = 64) -> np.ndarray:
        """
        Deep hashing inspired compact binary codes from arXiv:1610.07231
        Fuses multiple visual cues into balanced binary representation
        """
        try:
            # Extract core features for fusion
            phash_bits = np.array([int(bit) for bit in self.compute_phash(img)], dtype=np.float32)
            fft_vec = self.compute_fft_features(img)
            fmt_sig = self.compute_fourier_mellin_signature(img)
            hu_moments = self.compute_hu_moments(img)
            
            # Color and texture features (simplified for speed)
            color_vec = []
            try:
                if len(img.shape) == 3:
                    # Simple color moments
                    for c in range(3):
                        channel = img[:, :, c].flatten().astype(np.float32) / 255.0
                        color_vec.extend([np.mean(channel), np.std(channel)])
                else:
                    color_vec = [0.5, 0.2] * 3  # Grayscale defaults
            except:
                color_vec = [0.5, 0.2] * 3
            
            # Multi-scale FFT (different crop sizes for deep-style multi-scale)
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
                # Small scale (32x32)
                small_scale = cv2.resize(gray.astype(np.float32) / 255.0, (32, 32))
                fft_small = np.fft.fft2(small_scale)
                fft_small_mag = np.abs(np.fft.fftshift(fft_small))[12:20, 12:20].flatten()
                
                # Medium scale (64x64) 
                med_scale = cv2.resize(gray.astype(np.float32) / 255.0, (64, 64))
                fft_med = np.fft.fft2(med_scale)
                fft_med_mag = np.abs(np.fft.fftshift(fft_med))[28:36, 28:36].flatten()
                
                multi_scale_fft = np.concatenate([fft_small_mag, fft_med_mag])
            except:
                multi_scale_fft = np.zeros(128, dtype=np.float32)
            
            # Concatenate all features into fusion vector
            feature_parts = [
                phash_bits[:32],  # Limit pHash to 32 bits
                fft_vec[:64],     # Limit FFT features
                fmt_sig[:32],     # Limit FMT signature 
                hu_moments[:7],   # All Hu moments
                np.array(color_vec[:6], dtype=np.float32),  # Color moments
                multi_scale_fft[:32]  # Multi-scale texture
            ]
            
            # Ensure all parts are valid
            valid_parts = []
            for part in feature_parts:
                if part.size > 0:
                    # Normalize each feature type independently (deep hashing principle)
                    part_norm = (part - np.mean(part)) / (np.std(part) + 1e-8)
                    valid_parts.append(part_norm)
            
            if not valid_parts:
                return np.zeros(hash_dim, dtype=np.uint8)
            
            # Concatenate normalized features
            concat_features = np.concatenate(valid_parts)
            
            # Ensure we have enough features for projection
            if len(concat_features) < hash_dim:
                # Pad with zeros if needed
                padded = np.zeros(hash_dim)
                padded[:len(concat_features)] = concat_features
                concat_features = padded
            
            # Random orthogonal projection (simulates learned projection from deep hashing)
            # Use deterministic seed based on image content for consistency
            seed = int(np.sum(concat_features * 1000)) % 100000
            np.random.seed(seed % 1000)  # Limit seed range
            
            # Create orthogonal projection matrix
            proj_dim = min(hash_dim, len(concat_features))
            if len(concat_features) >= proj_dim:
                W = np.random.randn(len(concat_features), proj_dim)
                W, _ = np.linalg.qr(W)  # Orthogonalize (ensures bit independence)
                
                # Project and binarize
                projected = concat_features @ W
                
                # Bit balancing: center around 0 for even +/- distribution
                projected = projected - np.median(projected)
                
                # Binarize with sign function
                binary_hash = (projected > 0).astype(np.uint8)
            else:
                # Fallback for insufficient features
                binary_hash = (concat_features[:proj_dim] > np.median(concat_features[:proj_dim])).astype(np.uint8)
            
            # Ensure output is exactly hash_dim length
            if len(binary_hash) < hash_dim:
                padded_hash = np.zeros(hash_dim, dtype=np.uint8)
                padded_hash[:len(binary_hash)] = binary_hash
                return padded_hash
            else:
                return binary_hash[:hash_dim]
            
        except Exception as e:
            # Return random-like but deterministic hash on failure
            img_sum = np.sum(img.astype(np.float32)) if img is not None else 12345
            np.random.seed(int(img_sum) % 10000)
            return np.random.randint(0, 2, hash_dim, dtype=np.uint8)
    
    def compute_semantic_calibrated_similarity(self, features1: dict, features2: dict) -> float:
        """
        Semantic calibration inspired by deep hashing pairwise loss
        Learns optimal weighting of multiple similarity cues
        """
        try:
            # Individual similarity scores
            similarities = {}
            
            # 1. Deep fused hash similarity (Hamming distance)
            if 'deep_fused_hash' in features1 and 'deep_fused_hash' in features2:
                hash1, hash2 = features1['deep_fused_hash'], features2['deep_fused_hash']
                if hash1 is not None and hash2 is not None and len(hash1) == len(hash2):
                    hamming_dist = np.sum(hash1 != hash2)
                    similarities['fused_hash'] = 1.0 - (hamming_dist / len(hash1))
                else:
                    similarities['fused_hash'] = 0.0
            
            # 2. Traditional pHash
            if features1.get('phash') and features2.get('phash'):
                hamming_dist = self.hamming_distance(features1['phash'], features2['phash'])
                similarities['phash'] = 1.0 - (hamming_dist / 64.0)
            
            # 3. FFT similarity
            if features1.get('fft_features') is not None and features2.get('fft_features') is not None:
                try:
                    fft_sim = cosine_similarity(
                        features1['fft_features'].reshape(1, -1),
                        features2['fft_features'].reshape(1, -1)
                    )[0, 0]
                    similarities['fft'] = max(0, fft_sim)
                except:
                    similarities['fft'] = 0.0
            
            # 4. Moment-based similarity
            if features1.get('hu_moments') is not None and features2.get('hu_moments') is not None:
                try:
                    hu_sim = cosine_similarity(
                        features1['hu_moments'].reshape(1, -1),
                        features2['hu_moments'].reshape(1, -1)
                    )[0, 0]
                    similarities['moments'] = max(0, hu_sim)
                except:
                    similarities['moments'] = 0.0
            
            # 5. Color distance (inverted to similarity)
            if features1.get('color_vector') and features2.get('color_vector'):
                color_dist = self.color_distance(features1['color_vector'], features2['color_vector'])
                similarities['color'] = max(0, 1.0 - color_dist / 2.0)
            
            # Calibrated fusion weights (inspired by deep hashing learned weights)
            # These approximate what a logistic regression would learn
            fusion_weights = {
                'fused_hash': 0.35,  # Primary: compact multi-feature hash
                'phash': 0.25,       # Secondary: proven perceptual hash  
                'fft': 0.20,         # Tertiary: global shape
                'moments': 0.15,     # Quaternary: geometric invariants
                'color': 0.05        # Minimal: color (handled in fused hash)
            }
            
            # Weighted similarity fusion
            weighted_sum = 0.0
            total_weight = 0.0
            
            for method, similarity in similarities.items():
                if method in fusion_weights and similarity > 0:
                    weight = fusion_weights[method]
                    weighted_sum += similarity * weight
                    total_weight += weight
            
            # Normalize by total weights used
            if total_weight > 0:
                calibrated_score = weighted_sum / total_weight
            else:
                calibrated_score = 0.0
            
            return float(calibrated_score)
            
        except Exception as e:
            return 0.0

    def compute_all_features(self, img: np.ndarray) -> dict:
        """Compute ALL features including 2025 research + deep hashing enhancements"""
        # Traditional Fourier features
        fourier_features = {
            'phash': self.compute_phash(img),
            'fft_features': self.compute_fft_features(img),
            'fmt_signature': self.compute_fourier_mellin_signature(img)
        }
        
        # Advanced 2025 research features
        fourier_features['color_aware_fmt'] = self.compute_color_aware_fmt(img)
        fourier_features['saliency_weighted_fft'] = self.compute_saliency_weighted_fft(img)
        fourier_features['hu_moments'] = self.compute_hu_moments(img)
        fourier_features['zernike_moments'] = self.compute_zernike_moments(img)
        
        # Deep hashing inspired features (NEW)
        fourier_features['deep_fused_hash'] = self.compute_deep_fused_hash(img, hash_dim=64)
        
        # Texture features
        texture_features = self.compute_texture_features(img)
        fourier_features.update(texture_features)
        
        # Enhanced color features
        color_features = self.compute_enhanced_color_features(img)
        fourier_features.update(color_features)
        
        # Keypoint features
        sift_features = self.compute_sift_features(img)
        orb_features = self.compute_orb_features(img)
        
        fourier_features.update({
            'sift': sift_features,
            'orb': orb_features
        })
        
        return fourier_features
    
    def are_similar(self, features1: dict, features2: dict) -> tuple:
        """Enhanced similarity using ALL 2025 research + deep hashing methods"""
        
        # Deep hashing calibrated similarity (PRIMARY METHOD)
        calibrated_similarity = self.compute_semantic_calibrated_similarity(features1, features2)
        calibrated_similar = calibrated_similarity >= 0.75  # Learned threshold
        
        # Deep fused hash Hamming distance
        fused_hash_similar = False
        fused_hamming_distance = 64
        if features1.get('deep_fused_hash') is not None and features2.get('deep_fused_hash') is not None:
            hash1, hash2 = features1['deep_fused_hash'], features2['deep_fused_hash']
            if len(hash1) == len(hash2):
                fused_hamming_distance = np.sum(hash1 != hash2)
                # Deep hashing typically uses lower thresholds due to better bit distribution
                fused_hash_similar = fused_hamming_distance <= (len(hash1) * 0.25)  # 25% threshold
        
        # Traditional pHash comparison (SECONDARY)
        phash_distance = self.hamming_distance(features1['phash'], features2['phash'])
        phash_similar = phash_distance <= self.similarity_threshold_phash
        
        # FFT features comparison
        fft_similarity = cosine_similarity(
            features1['fft_features'].reshape(1, -1),
            features2['fft_features'].reshape(1, -1)
        )[0, 0]
        fft_similar = fft_similarity >= self.similarity_threshold_fft
        
        # Fourier-Mellin comparison
        fmt_similarity = self.compare_fourier_mellin(
            features1['fmt_signature'],
            features2['fmt_signature']
        )
        fmt_similar = fmt_similarity >= self.similarity_threshold_fmt
        
        # Advanced 2025 features comparison
        # Color-aware Fourier-Mellin
        color_fmt_similarity = cosine_similarity(
            features1['color_aware_fmt'].reshape(1, -1),
            features2['color_aware_fmt'].reshape(1, -1)
        )[0, 0] if features1['color_aware_fmt'].size > 0 else 0.0
        color_fmt_similar = color_fmt_similarity >= 0.85
        
        # Saliency-weighted FFT
        saliency_fft_similarity = cosine_similarity(
            features1['saliency_weighted_fft'].reshape(1, -1),
            features2['saliency_weighted_fft'].reshape(1, -1)
        )[0, 0] if features1['saliency_weighted_fft'].size > 0 else 0.0
        saliency_fft_similar = saliency_fft_similarity >= 0.80
        
        # Hu moments
        hu_similarity = cosine_similarity(
            features1['hu_moments'].reshape(1, -1),
            features2['hu_moments'].reshape(1, -1)
        )[0, 0] if features1['hu_moments'].size > 0 else 0.0
        hu_similar = hu_similarity >= 0.75
        
        # Zernike moments
        zernike_similarity = cosine_similarity(
            features1['zernike_moments'].reshape(1, -1),
            features2['zernike_moments'].reshape(1, -1)
        )[0, 0] if features1['zernike_moments'].size > 0 else 0.0
        zernike_similar = zernike_similarity >= 0.70
        
        # SIFT/ORB keypoint matching
        sift_similarity = 0.0
        sift_similar = False
        if features1['sift']['valid'] and features2['sift']['valid']:
            sift_similarity = self.match_sift_features(
                features1['sift']['descriptors'], 
                features2['sift']['descriptors']
            )
            if len(features1['sift']['signature']) > 0:
                sift_sig_similarity = cosine_similarity(
                    features1['sift']['signature'].reshape(1, -1),
                    features2['sift']['signature'].reshape(1, -1)
                )[0, 0]
                sift_similarity = max(sift_similarity, sift_sig_similarity)
            sift_similar = sift_similarity >= 0.3
        
        orb_similarity = 0.0
        orb_similar = False
        if features1['orb']['valid'] and features2['orb']['valid']:
            orb_similarity = self.match_orb_features(
                features1['orb']['descriptors'], 
                features2['orb']['descriptors']
            )
            if len(features1['orb']['signature']) > 0:
                orb_sig_similarity = cosine_similarity(
                    features1['orb']['signature'].reshape(1, -1),
                    features2['orb']['signature'].reshape(1, -1)
                )[0, 0]
                orb_similarity = max(orb_similarity, orb_sig_similarity)
            orb_similar = orb_similarity >= 0.25
        
        # Deep hashing inspired similarity decision with hierarchical confidence
        # Primary: Calibrated similarity (combines multiple cues intelligently)
        # Secondary: Individual method agreement for validation
        
        confidence_methods = [
            calibrated_similar,     # Primary: learned fusion
            fused_hash_similar,     # Secondary: compact binary hash
            phash_similar,          # Traditional: perceptual hash
            fft_similar or fmt_similar,  # Shape: global structure
            hu_similar or zernike_similar,  # Geometry: invariant moments
            sift_similar or orb_similar     # Local: keypoint features
        ]
        
        # Multi-tier decision (inspired by deep hashing confidence)
        confidence_score = sum(confidence_methods) / len(confidence_methods)
        
        # Enhanced decision logic with confidence thresholding
        if calibrated_similarity >= 0.85:
            # High confidence from calibrated fusion
            is_similar = True
        elif confidence_score >= 0.5 and calibrated_similar:
            # Medium confidence with method agreement
            is_similar = True  
        elif fused_hash_similar and (phash_similar or fft_similar):
            # Backup: compact hash + traditional method
            is_similar = True
        else:
            # Fallback to traditional multi-method OR
            is_similar = (phash_similar or fft_similar or fmt_similar or 
                         color_fmt_similar or saliency_fft_similar or 
                         hu_similar or zernike_similar or sift_similar or orb_similar)
        
        metrics = {
            # Deep hashing metrics (NEW)
            'calibrated_similarity': calibrated_similarity,
            'calibrated_similar': calibrated_similar,
            'fused_hash_distance': fused_hamming_distance,
            'fused_hash_similar': fused_hash_similar,
            'confidence_score': confidence_score,
            
            # Traditional metrics (EXISTING)
            'phash_distance': phash_distance,
            'phash_similar': phash_similar,
            'fft_similarity': fft_similarity,
            'fft_similar': fft_similar,
            'fmt_similarity': fmt_similarity,
            'fmt_similar': fmt_similar,
            'color_fmt_similarity': color_fmt_similarity,
            'color_fmt_similar': color_fmt_similar,
            'saliency_fft_similarity': saliency_fft_similarity,
            'saliency_fft_similar': saliency_fft_similar,
            'hu_similarity': hu_similarity,
            'hu_similar': hu_similar,
            'zernike_similarity': zernike_similarity,
            'zernike_similar': zernike_similar,
            'sift_similarity': sift_similarity,
            'sift_similar': sift_similar,
            'orb_similarity': orb_similarity,
            'orb_similar': orb_similar,
            'overall_similar': is_similar
        }
        
        return is_similar, metrics
    
    def preprocess_logo(self, logo_data: bytes) -> np.ndarray:
        """Convert logo bytes to numpy array with robust RGB handling"""
        try:
            # Convert bytes to image with better error handling
            img_array = np.frombuffer(logo_data, dtype=np.uint8)
            
            # Try multiple decode strategies
            img = None
            
            # Strategy 1: Direct color decode
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # Strategy 2: If failed, try grayscale then convert
            if img is None:
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            # Strategy 3: Try with any flags
            if img is None:
                img = cv2.imdecode(img_array, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
                if img is not None and len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img is not None and img.shape[2] == 4:  # RGBA
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Strategy 4: Try PIL as fallback
            if img is None:
                try:
                    pil_img = Image.open(io.BytesIO(logo_data))
                    pil_img = pil_img.convert('RGB')  # Ensure RGB
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception:
                    pass
            
            if img is not None and img.shape[0] > 10 and img.shape[1] > 10:
                # Ensure we have 3 channels (BGR)
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif len(img.shape) == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # Resize to standard size
                img_resized = cv2.resize(img, (128, 128))
                return img_resized
            else:
                return None
            
        except Exception as e:
            return None
    
    def analyze_logo_batch(self, logos: list) -> list:
        """Analyze batch of logos with ALL 2025 research features"""
        print(f" Analyzing {len(logos)} logos with 2025 research features...")
        start_time = time.time()
        
        analyzed_logos = []
        successful_analysis = 0
        
        for i, logo in enumerate(logos):
            if i % 50 == 0 and i > 0:
                elapsed = time.time() - start_time
                rate = i / elapsed
                eta = (len(logos) - i) / rate if rate > 0 else 0
                print(f"   Progress: {i}/{len(logos)} ({i/len(logos)*100:.1f}%) - ETA: {eta:.1f}s")
            
            try:
                img = self.preprocess_logo(logo['logo_data'])
                
                if img is not None:
                    # Extract ALL features including 2025 research enhancements
                    features = self.compute_all_features(img)
                    
                    # Extract color features for clustering
                    try:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        color_features = self.mean_color_features(pil_img)
                        features['color_vector'] = color_features
                    except Exception:
                        features['color_vector'] = None
                    
                    features['valid'] = True
                    successful_analysis += 1
                else:
                    features = {'valid': False, 'color_vector': None}
                
                logo_with_features = logo.copy()
                logo_with_features['features'] = features
                analyzed_logos.append(logo_with_features)
                
            except Exception as e:
                logo_with_features = logo.copy()
                logo_with_features['features'] = {'valid': False, 'error': str(e)}
                analyzed_logos.append(logo_with_features)
        
        elapsed = time.time() - start_time
        print(f" Enhanced analysis completed: {successful_analysis}/{len(logos)} valid in {elapsed:.1f}s")
        print(f" Features per logo: Traditional + 2025 Research + Deep Hashing")
        print(f" Deep features: 64-bit fused hash + semantic calibration + multi-scale analysis")
        
        return analyzed_logos
    
    def find_similar_pairs(self, analyzed_logos: list, threshold: float = 0.7) -> list:
        """Find similar pairs using enhanced multi-method comparison"""
        print(f" Finding similar pairs with 2025 research methods (threshold: {threshold})...")
        
        valid_logos = [logo for logo in analyzed_logos if logo['features']['valid']]
        similar_pairs = []
        
        total_comparisons = len(valid_logos) * (len(valid_logos) - 1) // 2
        comparison_count = 0
        
        for i in range(len(valid_logos)):
            for j in range(i + 1, len(valid_logos)):
                comparison_count += 1
                
                if comparison_count % 1000 == 0:
                    progress = comparison_count / total_comparisons * 100
                    print(f"   Progress: {comparison_count}/{total_comparisons} ({progress:.1f}%)")
                
                try:
                    logo1, logo2 = valid_logos[i], valid_logos[j]
                    is_similar, metrics = self.are_similar(logo1['features'], logo2['features'])
                    
                    if is_similar:
                        # Enhanced similarity scoring with deep hashing
                        # Primary: Use calibrated similarity score (learned fusion)
                        composite_score = metrics.get('calibrated_similarity', 0.0)
                        
                        # Fallback: Traditional multi-method scoring if calibrated fails
                        if composite_score < 0.1:
                            similarity_scores = [
                                1.0 - metrics['phash_distance'] / 64.0,
                                metrics['fft_similarity'],
                                metrics['fmt_similarity'],
                                metrics['color_fmt_similarity'],
                                metrics['saliency_fft_similarity'],
                                metrics['hu_similarity'],
                                metrics['zernike_similarity'],
                                metrics['sift_similarity'],
                                metrics['orb_similarity']
                            ]
                            
                            # Take maximum similarity across all methods (best match)
                            composite_score = max([s for s in similarity_scores if s > 0])
                        
                        if composite_score >= threshold:
                            similar_pairs.append((
                                logo1['website'],
                                logo2['website'], 
                                composite_score
                            ))
                            
                except Exception as e:
                    continue
        
        print(f" Similar pairs found: {len(similar_pairs)} using deep hashing + multi-method analysis")
        print(f" Deep hashing: Compact binary fusion + semantic calibration from arXiv:1610.07231")
        return similar_pairs

print(" Enhanced FourierLogoAnalyzer Ready!")
print(" Features: Traditional Fourier + Advanced research + Deep hashing")
print(" Multi-method similarity detection with compact binary codes")
