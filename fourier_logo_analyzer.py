"""
Fourier Logo Analyzer - Extracted from Colab Notebook
Advanced Fourier analysis for logo comparison with visualization capabilities
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft2, fftshift
from scipy.stats import skew
from sklearn.metrics.pairwise import cosine_similarity
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
import colorsys
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class FourierLogoAnalyzer:
    """
    Advanced Fourier analysis for logo similarity detection
    Combines multiple Fourier techniques + deep hashing inspired methods
    """
    
    def __init__(self):
        # Configuration parameters
        self.similarity_threshold_phash = 15
        self.similarity_threshold_fft = 0.75
        self.similarity_threshold_fmt = 0.80
        
        # Texture analysis parameters
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
    
    def compute_all_features(self, img: np.ndarray) -> Dict:
        """Compute comprehensive Fourier-based features"""
        features = {}
        
        # Core Fourier features
        features['phash'] = self.compute_phash(img)
        features['fft_features'] = self.compute_fft_features(img)
        features['fmt_signature'] = self.compute_fourier_mellin_signature(img)
        
        # Advanced Fourier features
        features['color_aware_fmt'] = self.compute_color_aware_fmt(img)
        features['saliency_weighted_fft'] = self.compute_saliency_weighted_fft(img)
        features['hu_moments'] = self.compute_hu_moments(img)
        
        # Color vector for similarity comparison
        try:
            if len(img.shape) == 3:
                # RGB means
                features['color_vector'] = [
                    float(np.mean(img[:,:,2])/255),  # R
                    float(np.mean(img[:,:,1])/255),  # G
                    float(np.mean(img[:,:,0])/255)   # B
                ]
            else:
                features['color_vector'] = [0.5, 0.5, 0.5]  # Grayscale
        except:
            features['color_vector'] = [0.5, 0.5, 0.5]
        
        return features
    
    def compare_fourier_mellin(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Compare Fourier-Mellin signatures with rotation invariance"""
        n = len(sig1)
        
        # Pad and compute correlation via FFT
        sig1_fft = np.fft.rfft(sig1, n=2*n)
        sig2_fft = np.fft.rfft(sig2[::-1], n=2*n)
        
        correlation = np.fft.irfft(sig1_fft * sig2_fft)
        max_correlation = np.max(correlation)
        
        return max_correlation
    
    def compute_similarity(self, features1: Dict, features2: Dict) -> Tuple[float, Dict]:
        """Compute comprehensive similarity between two feature sets"""
        
        similarities = {}
        
        # 1. Perceptual hash similarity
        phash_distance = self.hamming_distance(features1['phash'], features2['phash'])
        similarities['phash'] = 1.0 - (phash_distance / 64.0)
        
        # 2. FFT similarity
        try:
            fft_sim = cosine_similarity(
                features1['fft_features'].reshape(1, -1),
                features2['fft_features'].reshape(1, -1)
            )[0, 0]
            similarities['fft'] = max(0, fft_sim)
        except:
            similarities['fft'] = 0.0
        
        # 3. Fourier-Mellin similarity
        try:
            fmt_sim = self.compare_fourier_mellin(
                features1['fmt_signature'],
                features2['fmt_signature']
            )
            similarities['fourier_mellin'] = max(0, fmt_sim)
        except:
            similarities['fourier_mellin'] = 0.0
        
        # 4. Color-aware Fourier-Mellin similarity
        try:
            color_fmt_sim = cosine_similarity(
                features1['color_aware_fmt'].reshape(1, -1),
                features2['color_aware_fmt'].reshape(1, -1)
            )[0, 0]
            similarities['color_aware_fmt'] = max(0, color_fmt_sim)
        except:
            similarities['color_aware_fmt'] = 0.0
        
        # 5. Saliency-weighted FFT similarity
        try:
            saliency_fft_sim = cosine_similarity(
                features1['saliency_weighted_fft'].reshape(1, -1),
                features2['saliency_weighted_fft'].reshape(1, -1)
            )[0, 0]
            similarities['saliency_fft'] = max(0, saliency_fft_sim)
        except:
            similarities['saliency_fft'] = 0.0
        
        # 6. Hu moments similarity
        try:
            hu_sim = cosine_similarity(
                features1['hu_moments'].reshape(1, -1),
                features2['hu_moments'].reshape(1, -1)
            )[0, 0]
            similarities['hu_moments'] = max(0, hu_sim)
        except:
            similarities['hu_moments'] = 0.0
        
        # 7. Color similarity
        try:
            color_dist = self.color_distance(features1['color_vector'], features2['color_vector'])
            similarities['color'] = max(0, 1.0 - color_dist / 2.0)
        except:
            similarities['color'] = 0.0
        
        # Weighted average (emphasizing Fourier methods)
        weights = {
            'phash': 0.20,
            'fft': 0.25,
            'fourier_mellin': 0.25,
            'color_aware_fmt': 0.15,
            'saliency_fft': 0.10,
            'hu_moments': 0.03,
            'color': 0.02
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for method, similarity in similarities.items():
            if method in weights:
                weight = weights[method]
                weighted_sum += similarity * weight
                total_weight += weight
        
        overall_similarity = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return overall_similarity, similarities
    
    def visualize_fourier_curves(self, img: np.ndarray, domain_name: str = "Logo", 
                                save_path: Optional[str] = None) -> None:
        """Visualize the Fourier analysis curves and features"""
        
        # Extract features
        features = self.compute_all_features(img)
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle(f'Fourier Analysis for {domain_name}', fontsize=16, fontweight='bold')
        
        # 1. Original image
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Logo')
        axes[0, 0].axis('off')
        
        # 2. FFT magnitude spectrum
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (128, 128)).astype(np.float32) / 255.0
        fft = fft2(gray_resized)
        fft_shifted = fftshift(fft)
        magnitude = np.abs(fft_shifted)
        log_magnitude = np.log(magnitude + 1e-8)
        
        im = axes[0, 1].imshow(log_magnitude, cmap='jet')
        axes[0, 1].set_title('FFT Magnitude Spectrum (Log)')
        plt.colorbar(im, ax=axes[0, 1])
        
        # 3. FFT features plot
        fft_features = features['fft_features']
        axes[0, 2].plot(fft_features)
        axes[0, 2].set_title('FFT Features (32x32 Low Freq)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Fourier-Mellin theta signature
        fmt_signature = features['fmt_signature']
        theta_range = np.linspace(0, 2*np.pi, len(fmt_signature))
        axes[1, 0].plot(theta_range, fmt_signature, 'b-', linewidth=2)
        axes[1, 0].set_title('Fourier-Mellin Theta Signature')
        axes[1, 0].set_xlabel('Theta (radians)')
        axes[1, 0].set_ylabel('Magnitude')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Color-aware FMT (per channel)
        color_fmt = features['color_aware_fmt'].reshape(3, -1)
        for i, (color, label) in enumerate(zip(['b', 'g', 'r'], ['Blue', 'Green', 'Red'])):
            axes[1, 1].plot(color_fmt[i], color=color, label=f'{label} Channel', linewidth=2)
        axes[1, 1].set_title('Color-Aware Fourier-Mellin')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Saliency-weighted FFT
        saliency_fft = features['saliency_weighted_fft']
        saliency_2d = saliency_fft.reshape(32, 32)
        im = axes[1, 2].imshow(saliency_2d, cmap='viridis')
        axes[1, 2].set_title('Saliency-Weighted FFT')
        plt.colorbar(im, ax=axes[1, 2])
        
        # 7. Hu moments bar chart
        hu_moments = features['hu_moments']
        axes[2, 0].bar(range(len(hu_moments)), hu_moments, color='orange', alpha=0.7)
        axes[2, 0].set_title('Hu Invariant Moments')
        axes[2, 0].set_xlabel('Moment Index')
        axes[2, 0].set_ylabel('Value')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Perceptual hash visualization
        phash_bits = [int(bit) for bit in features['phash']]
        phash_2d = np.array(phash_bits).reshape(8, 8)
        im = axes[2, 1].imshow(phash_2d, cmap='RdBu', vmin=0, vmax=1)
        axes[2, 1].set_title('Perceptual Hash (8x8 DCT)')
        plt.colorbar(im, ax=axes[2, 1])
        
        # 9. Summary feature comparison
        feature_names = ['pHash', 'FFT', 'F-Mellin', 'Color-FMT', 'Saliency', 'Hu Mom.']
        feature_magnitudes = [
            1.0,  # pHash baseline
            np.linalg.norm(fft_features),
            np.linalg.norm(fmt_signature),
            np.linalg.norm(color_fmt),
            np.linalg.norm(saliency_fft),
            np.linalg.norm(hu_moments)
        ]
        
        bars = axes[2, 2].bar(feature_names, feature_magnitudes, 
                             color=['red', 'blue', 'green', 'purple', 'orange', 'brown'],
                             alpha=0.7)
        axes[2, 2].set_title('Feature Magnitude Comparison')
        axes[2, 2].tick_params(axis='x', rotation=45)
        axes[2, 2].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, mag in zip(bars, feature_magnitudes):
            axes[2, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{mag:.2f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Fourier visualization saved to: {save_path}")
        
        plt.show()
        
        return features
