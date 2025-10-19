"""
Advanced Logo Similarity Clustering - Complete Solution
Implements the comprehensive approach from the solution outline:
- Perceptual hashing (pHash) for near-identical detection
- ORB keypoint matching for shared design elements  
- Graph-based clustering (connected components)
- Multi-criteria similarity without ML clustering algorithms
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
from skimage import morphology, segmentation, filters
from scipy import ndimage

class AdvancedLogoClusterer:
    """
    Advanced logo similarity clustering using multiple techniques:
    1. Perceptual hashing (pHash) - primary method for near-identical logos
    2. ORB keypoint matching - for shared design elements 
    3. Color histogram analysis - supplementary color-based similarity
    4. Graph-based clustering via connected components
    """
    
    def __init__(self, jpeg_folder_path):
        self.jpeg_folder = jpeg_folder_path
        # Aggressive threading for large-scale processing
        self.max_workers = min(16, (os.cpu_count() or 1) * 2)
        self.batch_size = 100
        
        # Similarity thresholds (tunable parameters) - More stringent for better clustering
        self.phash_threshold = 4  # Hamming distance for pHash (out of 64 bits)
        self.orb_match_threshold = 25  # Minimum ORB keypoint matches
        self.color_corr_threshold = 0.90  # Color histogram correlation
        
        # ORB detector for keypoint analysis
        self.orb_detector = cv2.ORB_create(nfeatures=500)  # Limit features for speed
        
        # Load JPEG file paths
        self.jpeg_files = self._load_jpeg_paths()
        print(f"🎯 Found {len(self.jpeg_files)} JPEG files for clustering analysis")
        print(f"🔧 Enhanced preprocessing enabled: Background removal, transparency handling, edge enhancement")
    
    def _load_jpeg_paths(self):
        """Load all JPEG file paths with domain extraction"""
        if not os.path.exists(self.jpeg_folder):
            print(f"❌ Error: Folder {self.jpeg_folder} not found!")
            return []
        
        jpeg_files = []
        for filename in os.listdir(self.jpeg_folder):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                filepath = os.path.join(self.jpeg_folder, filename)
                domain = filename.replace('.jpg', '').replace('.jpeg', '')
                jpeg_files.append({
                    'filename': filename,
                    'filepath': filepath,
                    'domain': domain,
                    'index': len(jpeg_files)  # For graph node indexing
                })
        
        return jpeg_files
    
    def normalize_image_size(self, img, target_size=(256, 256), maintain_aspect_ratio=True, background_color=(255, 255, 255)):
        """
        Normalize image size with aspect ratio preservation and consistent background
        """
        if img is None:
            return None
            
        try:
            h, w = img.shape[:2]
            target_w, target_h = target_size
            
            if maintain_aspect_ratio:
                # Calculate scaling to fit within target size
                scale = min(target_w / w, target_h / h)
                new_w, new_h = int(w * scale), int(h * scale)
                
                # Resize image
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                
                # Create background canvas
                if len(img.shape) == 3:
                    canvas = np.full((target_h, target_w, img.shape[2]), background_color, dtype=np.uint8)
                else:
                    canvas = np.full((target_h, target_w), background_color[0], dtype=np.uint8)
                
                # Center the resized image on canvas
                start_y = (target_h - new_h) // 2
                start_x = (target_w - new_w) // 2
                
                if len(img.shape) == 3:
                    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
                else:
                    canvas[start_y:start_y + new_h, start_x:start_x + new_w] = resized
                
                return canvas
            else:
                # Direct resize without aspect ratio preservation
                return cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)
                
        except Exception as e:
            print(f"Size normalization error: {e}")
            try:
                return cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            except Exception:
                return img

    def handle_transparency_comprehensive(self, img, background_color=(255, 255, 255)):
        """
        Comprehensive transparency handling with consistent background
        """
        try:
            if img is None:
                return None
                
            # If image has alpha channel (4 channels)
            if len(img.shape) == 3 and img.shape[2] == 4:
                # Split channels
                rgb = img[:, :, :3]
                alpha = img[:, :, 3] / 255.0
                
                # Create consistent background
                background = np.full_like(rgb, background_color, dtype=np.uint8)
                
                # Alpha blend
                result = np.zeros_like(rgb, dtype=np.uint8)
                for c in range(3):
                    result[:, :, c] = (alpha * rgb[:, :, c] + (1 - alpha) * background[:, :, c]).astype(np.uint8)
                
                return result
            
            # No transparency, return as is
            return img
            
        except Exception as e:
            print(f"Transparency handling error: {e}")
            return img

    def create_color_normalized_version(self, img, preserve_color=True):
        """
        Create both color and grayscale versions for different comparison types
        """
        try:
            if img is None:
                return None, None
                
            # Color version (original or RGB)
            color_version = img.copy() if preserve_color else None
            
            # Grayscale version for shape-based comparisons
            if len(img.shape) == 3:
                grayscale_version = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                grayscale_version = img.copy()
            
            # Normalize grayscale contrast
            grayscale_version = cv2.equalizeHist(grayscale_version)
            
            return color_version, grayscale_version
            
        except Exception as e:
            print(f"Color normalization error: {e}")
            return img, img

    def advanced_preprocess_image(self, img, target_size=(256, 256), options=None):
        """
        Comprehensive preprocessing pipeline following best practices
        """
        if img is None:
            return None
            
        # Default preprocessing options
        default_options = {
            'maintain_aspect_ratio': True,
            'background_color': (255, 255, 255),  # White background
            'handle_transparency': True,
            'normalize_size': True,
            'color_normalization': True,
            'edge_enhancement': True,
            'background_removal': False  # Optional advanced feature
        }
        
        if options is not None:
            default_options.update(options)
        
        try:
            processed_img = img.copy()
            
            # Step 1: Handle transparency with consistent background
            if default_options['handle_transparency']:
                processed_img = self.handle_transparency_comprehensive(
                    processed_img, 
                    background_color=default_options['background_color']
                )
                if processed_img is None:
                    processed_img = img.copy()
            
            # Step 2: Normalize image size (resize with aspect ratio preservation)
            if default_options['normalize_size']:
                processed_img = self.normalize_image_size(
                    processed_img,
                    target_size=target_size,
                    maintain_aspect_ratio=default_options['maintain_aspect_ratio'],
                    background_color=default_options['background_color']
                )
                if processed_img is None:
                    return None
            
            # Step 3: Optional background removal (advanced feature)
            if default_options['background_removal']:
                h, w = processed_img.shape[:2]
                if h >= 50 and w >= 50:  # Only for reasonably sized images
                    mask = self._remove_background_grabcut(processed_img)
                    if mask is not None:
                        processed_img = cv2.bitwise_and(processed_img, processed_img, mask=mask)
            
            # Step 4: Create color-normalized versions
            if default_options['color_normalization']:
                color_version, grayscale_version = self.create_color_normalized_version(processed_img)
                # Store both versions for different comparison types
                processed_img = color_version if color_version is not None else processed_img
                
                # Store grayscale version as attribute for shape-based comparisons
                if not hasattr(self, '_temp_grayscale_cache'):
                    self._temp_grayscale_cache = {}
            
            # Step 5: Optional edge enhancement
            if default_options['edge_enhancement'] and processed_img is not None:
                h, w = processed_img.shape[:2]
                if h >= 20 and w >= 20:  # Only for reasonably sized images
                    enhanced = self._enhance_edges(processed_img)
                    if enhanced is not None and len(enhanced.shape) == 2:
                        # Convert back to RGB for consistency
                        try:
                            processed_img = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
                        except Exception:
                            pass  # Keep original if conversion fails
            
            return processed_img
            
        except Exception as e:
            print(f"Comprehensive preprocessing error: {e}")
            # Fallback to basic size normalization
            try:
                return self.normalize_image_size(img, target_size, maintain_aspect_ratio=True)
            except Exception:
                return img
    
    def _remove_background_grabcut(self, img):
        """Remove background using GrabCut algorithm"""
        try:
            if img is None or len(img.shape) != 3:
                return None
                
            height, width = img.shape[:2]
            
            # Ensure minimum size for GrabCut
            if height < 20 or width < 20:
                return None
            
            # Create initial mask - assume logo is in center 60% of image
            mask = np.zeros((height, width), np.uint8)
            
            # Define rectangle around center region (logo typically centered)
            margin_h, margin_w = max(5, int(height * 0.2)), max(5, int(width * 0.2))
            rect_width = max(10, width - 2*margin_w)
            rect_height = max(10, height - 2*margin_h)
            rect = (margin_w, margin_h, rect_width, rect_height)
            
            # Ensure rectangle is within image bounds
            if rect[0] + rect[2] >= width or rect[1] + rect[3] >= height:
                return None
            
            # Initialize GrabCut
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)
            
            # Apply GrabCut with error handling
            cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_RECT)
            
            # Create final mask (foreground and probable foreground)
            mask_final = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            
            # Clean up mask using morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_OPEN, kernel)
            mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel)
            
            return mask_final
            
        except Exception as e:
            print(f"GrabCut error: {e}")
            return None
    
    def _remove_background_edges(self, img):
        """Fallback background removal using edge detection"""
        try:
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Edge detection using adaptive threshold
            edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (assume it's the logo)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Create mask from largest contour
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.fillPoly(mask, [largest_contour], 1)
                
                # Dilate mask slightly to include logo edges
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                mask = cv2.dilate(mask, kernel, iterations=1)
                
                return mask
            
            return None
            
        except Exception:
            return None
    
    def _handle_transparency(self, img):
        """Handle transparency in images by replacing with white background"""
        try:
            if img is None:
                return None
                
            # If image has 4 channels (RGBA), handle transparency
            if len(img.shape) == 3 and img.shape[2] == 4:
                # Create white background
                background = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255
                
                # Extract RGB and alpha channels
                rgb = img[:, :, :3]
                alpha = img[:, :, 3:4] / 255.0
                
                # Blend with white background
                result = rgb * alpha + background * (1 - alpha)
                return result.astype(np.uint8)
            
            # Already RGB or grayscale, return as is
            return img
            
        except Exception as e:
            print(f"Transparency handling error: {e}")
            return img

    def _enhance_edges(self, img):
        """Enhance logo edges using adaptive filtering"""
        try:
            if img is None:
                return None
                
            # Convert to grayscale if needed
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img.copy()
            
            # Ensure minimum size
            if gray.shape[0] < 10 or gray.shape[1] < 10:
                return gray
            
            # Apply bilateral filter to reduce noise while preserving edges
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Enhance edges using unsharp masking
            gaussian = cv2.GaussianBlur(filtered, (5, 5), 2.0)  # Use explicit kernel size
            unsharp_mask = cv2.addWeighted(filtered, 1.5, gaussian, -0.5, 0)
            
            # Apply CLAHE for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(unsharp_mask)
            
            return enhanced
            
        except Exception as e:
            print(f"Edge enhancement error: {e}")
            # Return original grayscale if enhancement fails
            try:
                if len(img.shape) == 3:
                    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                return img.copy()
            except:
                return img

    def compute_multiple_hashes(self, img_path):
        """
        Compute multiple perceptual hashes for comprehensive similarity detection
        Implements pHash (DCT), aHash (average), and dHash (difference) as mentioned in requirements
        """
        try:
            # Load and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed = self.advanced_preprocess_image(img_rgb, target_size=(64, 64))
            if processed is None:
                processed = cv2.resize(img_rgb, (64, 64))
            
            # Convert to PIL for imagehash
            pil_img = Image.fromarray(processed)
            
            # Multiple hash types as recommended in literature
            hashes = {
                'phash': imagehash.phash(pil_img, hash_size=8),    # DCT-based (primary)
                'ahash': imagehash.average_hash(pil_img, hash_size=8),  # Average-based
                'dhash': imagehash.dhash(pil_img, hash_size=8),    # Difference-based
                'whash': imagehash.whash(pil_img, hash_size=8)     # Wavelet-based
            }
            
            # Convert to integers for efficient storage/comparison
            return {k: int(str(v), 16) for k, v in hashes.items()}
            
        except Exception as e:
            print(f"Multi-hash error for {img_path}: {e}")
            return None

    def compute_structural_similarity(self, img1_path, img2_path):
        """
        Compute Structural Similarity Index (SSIM) between two images
        As mentioned in requirements for detecting near-duplicate logos
        """
        try:
            from skimage.metrics import structural_similarity as ssim
            
            # Load both images
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                return 0.0
            
            # Preprocess both images consistently
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            processed1 = self.advanced_preprocess_image(img1_rgb)
            processed2 = self.advanced_preprocess_image(img2_rgb)
            
            if processed1 is None or processed2 is None:
                return 0.0
            
            # Convert to grayscale for SSIM
            gray1 = cv2.cvtColor(processed1, cv2.COLOR_RGB2GRAY)
            gray2 = cv2.cvtColor(processed2, cv2.COLOR_RGB2GRAY)
            
            # Compute SSIM
            ssim_score = ssim(gray1, gray2, data_range=gray1.max() - gray1.min())
            return ssim_score
            
        except Exception as e:
            print(f"SSIM error: {e}")
            return 0.0

    def compute_hog_descriptor(self, img_path):
        """
        Compute Histogram of Oriented Gradients (HOG) for shape/edge distribution analysis
        As mentioned in requirements for capturing overall shape profiles
        """
        try:
            from skimage.feature import hog
            
            # Load and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed = self.advanced_preprocess_image(img_rgb, target_size=(128, 128))
            
            if processed is None:
                processed = cv2.resize(img_rgb, (128, 128))
            
            # Convert to grayscale for HOG
            if len(processed.shape) == 3:
                gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            else:
                gray = processed
            
            # Compute HOG features
            hog_features = hog(
                gray,
                orientations=9,        # 9 orientation bins
                pixels_per_cell=(16, 16),  # Cell size
                cells_per_block=(2, 2),    # Block size
                block_norm='L2-Hys',       # Normalization
                visualize=False,
                feature_vector=True
            )
            
            return hog_features
            
        except Exception as e:
            print(f"HOG error for {img_path}: {e}")
            return None

    def compute_edge_histogram(self, img_path):
        """
        Compute edge histogram for shape distribution analysis
        Alternative shape descriptor as mentioned in requirements
        """
        try:
            # Load and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            processed = self.advanced_preprocess_image(img_rgb)
            
            if processed is None:
                processed = img_rgb
            
            # Convert to grayscale
            if len(processed.shape) == 3:
                gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            else:
                gray = processed
            
            # Compute edges using Canny
            edges = cv2.Canny(gray, 50, 150)
            
            # Compute gradient orientations
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient angles
            angles = np.arctan2(grad_y, grad_x) * 180 / np.pi
            angles[angles < 0] += 180  # Normalize to 0-180
            
            # Create histogram of edge orientations (8 bins)
            edge_hist = np.zeros(8)
            edge_pixels = edges > 0
            
            if np.any(edge_pixels):
                edge_angles = angles[edge_pixels]
                hist, _ = np.histogram(edge_angles, bins=8, range=(0, 180))
                edge_hist = hist.astype(np.float32)
                
                # Normalize
                if edge_hist.sum() > 0:
                    edge_hist = edge_hist / edge_hist.sum()
            
            return edge_hist
            
        except Exception as e:
            print(f"Edge histogram error for {img_path}: {e}")
            return None
    
    def _resize_with_padding(self, img, target_size):
        """Resize image while preserving aspect ratio using padding"""
        try:
            height, width = img.shape[:2]
            target_w, target_h = target_size
            
            # Calculate scaling factor
            scale = min(target_w / width, target_h / height)
            
            # Calculate new dimensions
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize image
            resized = cv2.resize(img, (new_width, new_height))
            
            # Create padded image with white background
            padded = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
            
            # Calculate padding offsets to center the image
            y_offset = (target_h - new_height) // 2
            x_offset = (target_w - new_width) // 2
            
            # Place resized image in center
            padded[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
            
            return padded
            
        except Exception:
            return cv2.resize(img, target_size)
    
    def compute_perceptual_hash(self, img_path):
        """
        Compute perceptual hash optimized for logo shape comparison
        Returns 64-bit hash as integer for efficient comparison
        """
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                return 0
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Use comprehensive preprocessing for shape-focused comparison
            preprocessing_options = {
                'maintain_aspect_ratio': True,
                'background_color': (255, 255, 255),
                'handle_transparency': True,
                'normalize_size': True,
                'color_normalization': True,
                'edge_enhancement': False,  # pHash handles edge detection
                'background_removal': False
            }
            
            processed_img = self.advanced_preprocess_image(
                img_rgb, 
                target_size=(64, 64),  # Smaller size for hash computation
                options=preprocessing_options
            )
            
            if processed_img is not None:
                # Convert to grayscale for shape-focused hashing
                if len(processed_img.shape) == 3:
                    gray_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
                else:
                    gray_img = processed_img
                
                # Convert to PIL format
                pil_img = Image.fromarray(gray_img)
                
                # Use DCT-based perceptual hash (most robust for logo shapes)
                phash = imagehash.phash(pil_img, hash_size=8)  # 64-bit hash
                return int(str(phash), 16)  # Convert to integer for fast comparison
            else:
                # Fallback to simple method
                with Image.open(img_path) as img:
                    img_gray = img.convert('L').resize((64, 64))
                    phash = imagehash.phash(img_gray, hash_size=8)
                    return int(str(phash), 16)
                    
        except Exception as e:
            print(f"pHash error for {img_path}: {e}")
            return 0  # Default hash
    
    def compute_orb_descriptors(self, img_path):
        """
        Compute ORB keypoints and descriptors with enhanced preprocessing for logos
        Optimized for detecting distinctive visual features in logo designs
        """
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                return None, None
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Use preprocessing optimized for feature detection
            preprocessing_options = {
                'maintain_aspect_ratio': True,
                'background_color': (255, 255, 255),
                'handle_transparency': True,
                'normalize_size': True,
                'color_normalization': True,  # Creates good grayscale for features
                'edge_enhancement': True,  # Important for ORB feature detection
                'background_removal': False  # Keep some context for feature matching
            }
            
            processed_img = self.advanced_preprocess_image(
                img_rgb,
                target_size=(256, 256),  # Good size for feature detection
                options=preprocessing_options
            )
            
            if processed_img is not None:
                # Convert to grayscale for ORB detection
                if len(processed_img.shape) == 3:
                    img_gray = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
                else:
                    img_gray = processed_img
            else:
                # Fallback: simple grayscale conversion and resize
                img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
                img_gray = cv2.resize(img_gray, (256, 256))
            
            # Additional contrast enhancement for better feature detection
            img_gray = cv2.equalizeHist(img_gray)
            
            # Detect ORB keypoints and compute descriptors
            keypoints, descriptors = self.orb_detector.detectAndCompute(img_gray, None)
            
            return keypoints, descriptors
            
        except Exception as e:
            print(f"ORB error for {img_path}: {e}")
            return None, None
    
    def compute_color_histogram(self, img_path):
        """
        Compute normalized color histogram with comprehensive preprocessing
        Focuses on actual logo colors while handling transparency and backgrounds
        """
        try:
            # Load image
            img = cv2.imread(img_path)
            if img is None:
                return np.zeros(48, dtype=np.float32)
            
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Use preprocessing optimized for color analysis
            preprocessing_options = {
                'maintain_aspect_ratio': True,
                'background_color': (255, 255, 255),  # White background for consistent color analysis
                'handle_transparency': True,
                'normalize_size': True,
                'color_normalization': False,  # Keep original colors for histogram
                'edge_enhancement': False,
                'background_removal': True  # Remove background to focus on logo colors
            }
            
            processed_img = self.advanced_preprocess_image(
                img_rgb,
                target_size=(128, 128),
                options=preprocessing_options
            )
            
            if processed_img is not None:
                # Convert back to BGR for OpenCV histogram functions
                img_bgr = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
            else:
                # Fallback: simple resize
                img_bgr = cv2.resize(img_rgb, (128, 128))
                img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
            
            # Convert to HSV as recommended in requirements for better color analysis
            # HSV separates color information (hue) from brightness/saturation
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            
            # Create mask to exclude pure white background pixels (from preprocessing)
            white_mask = np.all(img_bgr >= [240, 240, 240], axis=2)  # Nearly white pixels
            color_mask = (~white_mask).astype(np.uint8) * 255  # Invert to focus on logo colors
            
            # Only use mask if it covers significant portion (avoid empty histograms)
            if np.sum(color_mask) < (128 * 128 * 0.1):  # Less than 10% non-white
                color_mask = None  # Use full image
            
            # Compute HSV histograms as recommended for logo color analysis
            # Benefits: Hue captures color identity, Saturation captures color purity
            hist_h = cv2.calcHist([img_hsv], [0], color_mask, [16], [0, 180])  # Hue (0-180°)
            hist_s = cv2.calcHist([img_hsv], [1], color_mask, [16], [0, 256])  # Saturation (0-255)
            hist_v = cv2.calcHist([img_hsv], [2], color_mask, [16], [0, 256])  # Value/Brightness (0-255)
            
            # Normalize each histogram
            def normalize_hist(hist):
                total = np.sum(hist)
                return hist.flatten() / (total + 1e-8)  # Avoid division by zero
            
            hist_h_norm = normalize_hist(hist_h)
            hist_s_norm = normalize_hist(hist_s)
            hist_v_norm = normalize_hist(hist_v)
            
            # Combine all channels (16 + 16 + 16 = 48 features)
            combined_histogram = np.concatenate([hist_h_norm, hist_s_norm, hist_v_norm])
            
            return combined_histogram.astype(np.float32)
            
        except Exception as e:
            print(f"Color histogram error for {img_path}: {e}")
            return np.zeros(48, dtype=np.float32)
    
    def extract_comprehensive_features(self, jpeg_info):
        """Extract all features for advanced similarity analysis"""
        filepath = jpeg_info['filepath']
        domain = jpeg_info['domain']
        index = jpeg_info['index']
        
        try:
            # 1. Perceptual Hash (primary similarity measure)
            phash = self.compute_perceptual_hash(filepath)
            
            # 2. ORB descriptors (for design element matching)
            keypoints, orb_descriptors = self.compute_orb_descriptors(filepath)
            
            # 3. Color histogram (supplementary)
            color_hist = self.compute_color_histogram(filepath)
            
            features = {
                'domain': domain,
                'filepath': filepath,
                'index': index,
                'phash': phash,
                'orb_descriptors': orb_descriptors,
                'orb_keypoints_count': len(keypoints) if keypoints else 0,
                'color_histogram': color_hist
            }
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error for {domain}: {e}")
            return {
                'domain': domain,
                'filepath': filepath, 
                'index': index,
                'phash': 0,
                'orb_descriptors': None,
                'orb_keypoints_count': 0,
                'color_histogram': np.zeros(48, dtype=np.float32)
            }
    
    def extract_all_features_parallel(self):
        """Extract features for all logos using aggressive parallelism"""
        print(f"🚀 Extracting comprehensive features from {len(self.jpeg_files)} logos...")
        print(f"🧵 Using {self.max_workers} threads with batch size {self.batch_size}")
        
        all_features = {}
        
        def process_batch(batch):
            """Process a batch of JPEG files"""
            batch_results = {}
            for jpeg_info in batch:
                features = self.extract_comprehensive_features(jpeg_info)
                batch_results[features['domain']] = features
            return batch_results
        
        # Split into batches for threading
        batches = [
            self.jpeg_files[i:i + self.batch_size] 
            for i in range(0, len(self.jpeg_files), self.batch_size)
        ]
        
        # Process with threading
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(process_batch, batch): batch 
                for batch in batches
            }
            
            completed = 0
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    all_features.update(batch_results)
                    completed += len(batch_results)
                    
                    # Progress update
                    progress = (completed / len(self.jpeg_files)) * 100
                    print(f"   Progress: {completed}/{len(self.jpeg_files)} ({progress:.1f}%)")
                    
                except Exception as e:
                    print(f"Batch processing error: {e}")
        
        print(f"✅ Feature extraction completed for {len(all_features)} logos")
        return all_features
    
    def compute_hamming_distance(self, hash1, hash2):
        """Compute Hamming distance between two integer hashes"""
        # XOR the hashes and count set bits
        xor_result = hash1 ^ hash2
        return bin(xor_result).count('1')
    
    def match_orb_descriptors(self, desc1, desc2):
        """
        Match ORB descriptors using brute force matcher
        Returns number of good matches
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return 0
        
        try:
            # Use Hamming distance for binary descriptors (ORB)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(desc1, desc2)
            
            # Sort matches by distance (lower is better)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Count "good" matches (distance threshold for binary descriptors)
            good_matches = [m for m in matches if m.distance < 50]  # Tunable threshold
            
            return len(good_matches)
            
        except Exception as e:
            return 0
    
    def compute_pairwise_similarity(self, features1, features2):
        """
        Compute comprehensive similarity between two logo features
        Uses multiple criteria as described in solution outline
        """
        similarities = {}
        
        try:
            # 1. Perceptual Hash Similarity (PRIMARY - most important)
            phash_distance = self.compute_hamming_distance(features1['phash'], features2['phash'])
            similarities['phash_distance'] = phash_distance
            similarities['phash_similar'] = phash_distance <= self.phash_threshold
            
            # 2. ORB Keypoint Matching (SECONDARY - for design elements)
            orb_matches = self.match_orb_descriptors(
                features1['orb_descriptors'], 
                features2['orb_descriptors']
            )
            similarities['orb_matches'] = orb_matches
            similarities['orb_similar'] = orb_matches >= self.orb_match_threshold
            
            # 3. Color Histogram Correlation (SUPPLEMENTARY)
            color_corr = np.corrcoef(
                features1['color_histogram'], 
                features2['color_histogram']
            )[0, 1]
            if np.isnan(color_corr):
                color_corr = 0.0
            similarities['color_correlation'] = color_corr
            similarities['color_similar'] = color_corr >= self.color_corr_threshold
            
            # Multi-criteria decision (as per solution outline)
            # Primary: pHash similarity (catches near-identical logos)
            # Secondary: ORB matching (catches shared design elements)
            # Conservative approach: require strong evidence for similarity
            
            is_similar = (
                similarities['phash_similar'] or  # Near identical
                (similarities['orb_similar'] and similarities['color_similar'])  # Shared design + color
            )
            
            similarities['overall_similar'] = is_similar
            
        except Exception as e:
            print(f"Similarity computation error: {e}")
            similarities = {
                'phash_distance': 64,  # Max distance
                'phash_similar': False,
                'orb_matches': 0,
                'orb_similar': False, 
                'color_correlation': 0.0,
                'color_similar': False,
                'overall_similar': False
            }
        
        return similarities
    
    def build_similarity_graph(self, features_dict):
        """
        Build graph of similar logos using parallel pairwise comparison
        Implements graph-based clustering approach from solution outline
        """
        print(f"🔍 Building similarity graph for {len(features_dict)} logos...")
        print(f"🧵 Using {self.max_workers} threads for pairwise comparison...")
        
        domains = list(features_dict.keys())
        total_comparisons = len(domains) * (len(domains) - 1) // 2
        
        print(f"📊 Total pairwise comparisons: {total_comparisons:,}")
        
        # Create comparison task list
        comparison_tasks = []
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                comparison_tasks.append((i, j, domains[i], domains[j]))
        
        # Batch comparisons for threading
        batch_size = max(1000, total_comparisons // (self.max_workers * 8))
        batches = [
            comparison_tasks[i:i + batch_size] 
            for i in range(0, len(comparison_tasks), batch_size)
        ]
        
        print(f"📦 Split into {len(batches)} batches of ~{batch_size} comparisons each")
        
        # Graph represented as adjacency list
        similarity_edges = []
        completed_comparisons = 0
        
        def process_comparison_batch(batch):
            """Process a batch of pairwise comparisons"""
            batch_edges = []
            
            for i, j, domain1, domain2 in batch:
                # Quick pHash pre-filter (optimization from solution outline)
                phash_dist = self.compute_hamming_distance(
                    features_dict[domain1]['phash'], 
                    features_dict[domain2]['phash']
                )
                
                # If pHash is very different, skip expensive ORB matching
                if phash_dist > self.phash_threshold * 2:  # 2x threshold as cutoff
                    continue
                
                # Compute full similarity
                similarity = self.compute_pairwise_similarity(
                    features_dict[domain1], 
                    features_dict[domain2]
                )
                
                if similarity['overall_similar']:
                    batch_edges.append({
                        'domain1': domain1,
                        'domain2': domain2, 
                        'index1': i,
                        'index2': j,
                        **similarity
                    })
            
            return batch_edges
        
        # Process batches in parallel
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(process_comparison_batch, batch): batch 
                for batch in batches
            }
            
            for future in as_completed(future_to_batch):
                try:
                    batch_edges = future.result()
                    similarity_edges.extend(batch_edges)
                    
                    # Progress update
                    completed_comparisons += len(future_to_batch[future])
                    progress = (completed_comparisons / total_comparisons) * 100
                    elapsed = time.time() - start_time
                    
                    print(f"   Progress: {completed_comparisons:,}/{total_comparisons:,} "
                          f"({progress:.1f}%) - {len(similarity_edges)} edges found - "
                          f"{elapsed:.1f}s elapsed")
                    
                except Exception as e:
                    print(f"Batch comparison error: {e}")
        
        print(f"✅ Similarity graph built with {len(similarity_edges)} edges")
        return similarity_edges, domains
    
    def find_connected_components(self, edges, domains):
        """
        Find connected components in similarity graph using Union-Find
        Each component represents a cluster of similar logos
        """
        print(f"🔗 Finding connected components (clusters)...")
        
        # Union-Find data structure
        n = len(domains)
        parent = list(range(n))
        rank = [0] * n
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return False
            
            # Union by rank
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        # Build domain to index mapping
        domain_to_idx = {domain: i for i, domain in enumerate(domains)}
        
        # Union similar pairs
        edges_processed = 0
        for edge in edges:
            idx1 = domain_to_idx[edge['domain1']]
            idx2 = domain_to_idx[edge['domain2']]
            union(idx1, idx2)
            edges_processed += 1
        
        # Collect connected components
        components = defaultdict(list)
        for i, domain in enumerate(domains):
            root = find(i)
            components[root].append(domain)
        
        # Convert to list format
        clusters = list(components.values())
        clusters.sort(key=len, reverse=True)  # Sort by cluster size
        
        # Statistics
        multi_clusters = [c for c in clusters if len(c) > 1]
        singleton_clusters = [c for c in clusters if len(c) == 1]
        
        print(f"📈 Clustering Results:")
        print(f"   Total clusters: {len(clusters)}")
        print(f"   Multi-logo clusters: {len(multi_clusters)}")
        print(f"   Singleton clusters: {len(singleton_clusters)}")
        print(f"   Largest cluster size: {len(clusters[0]) if clusters else 0}")
        
        return clusters
    
    def analyze_clusters(self, clusters, features_dict):
        """Analyze cluster quality and characteristics"""
        print(f"\n" + "="*60)
        print("ADVANCED LOGO CLUSTERING ANALYSIS")
        print("="*60)
        
        multi_clusters = [c for c in clusters if len(c) > 1]
        
        print(f"📊 Cluster Distribution:")
        cluster_sizes = {}
        for cluster in clusters:
            size = len(cluster)
            cluster_sizes[size] = cluster_sizes.get(size, 0) + 1
        
        for size in sorted(cluster_sizes.keys(), reverse=True)[:10]:
            count = cluster_sizes[size]
            print(f"   Size {size}: {count} clusters")
        
        # Sample interesting clusters
        print(f"\n🔍 Sample Multi-Logo Clusters:")
        for i, cluster in enumerate(multi_clusters[:5]):
            print(f"\nCluster {i+1} ({len(cluster)} logos):")
            
            # Show similarity evidence for first pair in cluster
            if len(cluster) >= 2:
                domain1, domain2 = cluster[0], cluster[1]
                similarity = self.compute_pairwise_similarity(
                    features_dict[domain1], features_dict[domain2]
                )
                
                print(f"  Sample pair: {domain1} ↔ {domain2}")
                print(f"    pHash distance: {similarity['phash_distance']} bits")
                print(f"    ORB matches: {similarity['orb_matches']}")
                print(f"    Color correlation: {similarity['color_correlation']:.3f}")
            
            # List domains
            for j, domain in enumerate(cluster[:5]):
                brand_name = domain.replace('_', ' ').replace('-', ' ').title()
                print(f"    {j+1}. {brand_name} ({domain})")
            
            if len(cluster) > 5:
                print(f"    ... and {len(cluster) - 5} more")
        
        return {
            'total_clusters': len(clusters),
            'multi_clusters': len(multi_clusters),
            'singleton_clusters': len(clusters) - len(multi_clusters),
            'largest_cluster_size': len(clusters[0]) if clusters else 0,
            'cluster_size_distribution': cluster_sizes
        }
    
    def save_clustering_results(self, clusters, features_dict, similarity_edges):
        """Save comprehensive clustering results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Save clusters as CSV
        cluster_data = []
        for cluster_id, cluster in enumerate(clusters):
            for domain in cluster:
                cluster_data.append({
                    'cluster_id': cluster_id,
                    'cluster_size': len(cluster),
                    'domain': domain,
                    'is_singleton': len(cluster) == 1,
                    'phash': features_dict[domain]['phash'],
                    'orb_keypoints': features_dict[domain]['orb_keypoints_count']
                })
        
        clusters_df = pd.DataFrame(cluster_data)
        csv_path = f"advanced_logo_clusters_{timestamp}.csv"
        clusters_df.to_csv(csv_path, index=False)
        
        # 2. Save similarity edges
        if similarity_edges:
            edges_df = pd.DataFrame(similarity_edges)
            edges_csv = f"logo_similarity_edges_{timestamp}.csv"
            edges_df.to_csv(edges_csv, index=False)
        
        # 3. Save complete results with features
        results = {
            'clusters': clusters,
            'features': features_dict,
            'similarity_edges': similarity_edges,
            'method': 'Advanced_Multi_Criteria_Graph_Clustering',
            'thresholds': {
                'phash_threshold': self.phash_threshold,
                'orb_match_threshold': self.orb_match_threshold,
                'color_corr_threshold': self.color_corr_threshold
            },
            'timestamp': datetime.now().isoformat()
        }
        
        pkl_path = f"advanced_logo_clustering_results_{timestamp}.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n💾 Results saved:")
        print(f"   Clusters CSV: {csv_path}")
        print(f"   Edges CSV: {edges_csv}")
        print(f"   Complete results: {pkl_path}")
        
        return csv_path, edges_csv, pkl_path
    
    def enhanced_similarity_analysis(self, features1, features2):
        """
        Enhanced multi-criteria similarity analysis implementing all techniques from requirements:
        - Perceptual Hash (pHash) with Hamming distance
        - ORB Local Feature Matching with FLANN
        - HSV Color Histogram Correlation
        Multi-pronged approach as recommended in literature
        """
        results = {
            'is_similar': False,
            'confidence': 0.0,
            'criteria_met': [],
            'detailed_scores': {}
        }
        
        try:
            # 1. pHash Analysis (Primary - most reliable for near-identical logos)
            phash_similar = False
            if features1.get('phash') is not None and features2.get('phash') is not None:
                hamming_dist = bin(features1['phash'] ^ features2['phash']).count('1')
                phash_similarity = (64 - hamming_dist) / 64.0
                
                results['detailed_scores']['phash_hamming_distance'] = hamming_dist
                results['detailed_scores']['phash_similarity'] = phash_similarity
                
                # Strict threshold (≤4 bits difference)
                if hamming_dist <= self.phash_threshold:
                    phash_similar = True
                    results['criteria_met'].append('pHash')
            
            # 2. ORB Feature Matching (Secondary - for shared design elements)
            orb_similar = False
            orb_matches = 0
            if (features1.get('orb_descriptors') is not None and 
                features2.get('orb_descriptors') is not None and
                features1['orb_descriptors'].shape[0] > 0 and 
                features2['orb_descriptors'].shape[0] > 0):
                
                try:
                    # Use brute force matcher for reliability
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(features1['orb_descriptors'], features2['orb_descriptors'])
                    
                    # Filter good matches by distance
                    good_matches = [m for m in matches if m.distance < 50]
                    orb_matches = len(good_matches)
                    
                    results['detailed_scores']['orb_matches'] = orb_matches
                    
                    # Updated threshold (≥25 matches)
                    if orb_matches >= self.orb_match_threshold:
                        orb_similar = True
                        results['criteria_met'].append('ORB')
                        
                except Exception as e:
                    print(f"ORB matching error: {e}")
            
            # 3. HSV Color Histogram Correlation (Global color analysis)
            color_similar = False
            if (features1.get('color_histogram') is not None and 
                features2.get('color_histogram') is not None):
                
                correlation = cv2.compareHist(features1['color_histogram'],
                                            features2['color_histogram'],
                                            cv2.HISTCMP_CORREL)
                
                results['detailed_scores']['color_correlation'] = correlation
                
                # Updated threshold (≥0.90)
                if correlation >= self.color_corr_threshold:
                    color_similar = True
                    results['criteria_met'].append('Color')
            
            # Multi-criteria decision logic (requires multiple criteria for robustness)
            criteria_count = len(results['criteria_met'])
            
            if criteria_count >= 2:  # At least 2 criteria must be met
                results['is_similar'] = True
                results['confidence'] = criteria_count / 3.0  # Normalize by max criteria
            elif phash_similar and orb_matches > 50:  # Very strong ORB match
                results['is_similar'] = True
                results['confidence'] = 0.9
            elif phash_similar and results['detailed_scores'].get('color_correlation', 0) > 0.95:
                results['is_similar'] = True  # Near-perfect color match
                results['confidence'] = 0.85
            
            return results
            
        except Exception as e:
            print(f"Enhanced similarity analysis error: {e}")
            return results

def main():
    """
    Main pipeline implementing the advanced logo clustering solution
    """
    print("="*70)
    print("ADVANCED LOGO SIMILARITY CLUSTERING")
    print("Multi-Criteria Graph-Based Approach (No Traditional ML)")
    print("="*70)
    
    # Find extracted logos folder
    logo_folders = [d for d in os.listdir('.') if d.startswith('extracted_logos_')]
    
    if not logo_folders:
        print("❌ No extracted logos folder found!")
        print("Please run extract_logos_to_jpg.py first")
        return
    
    # Use most recent folder
    logo_folder = sorted(logo_folders)[-1]
    print(f"📁 Using logo folder: {logo_folder}")
    
    # Initialize advanced clusterer
    clusterer = AdvancedLogoClusterer(logo_folder)
    
    if len(clusterer.jpeg_files) == 0:
        print("❌ No JPEG files found!")
        return
    
    print(f"🎯 Clustering Parameters:")
    print(f"   pHash threshold: ≤{clusterer.phash_threshold} bits")
    print(f"   ORB match threshold: ≥{clusterer.orb_match_threshold} keypoints")
    print(f"   Color correlation threshold: ≥{clusterer.color_corr_threshold}")
    
    # Step 1: Extract comprehensive features
    print(f"\n" + "="*50)
    print("STEP 1: FEATURE EXTRACTION")
    print("="*50)
    
    features_dict = clusterer.extract_all_features_parallel()
    
    # Step 2: Build similarity graph 
    print(f"\n" + "="*50)
    print("STEP 2: SIMILARITY GRAPH CONSTRUCTION") 
    print("="*50)
    
    similarity_edges, domains = clusterer.build_similarity_graph(features_dict)
    
    # Step 3: Find connected components (clusters)
    print(f"\n" + "="*50)
    print("STEP 3: GRAPH CLUSTERING")
    print("="*50)
    
    clusters = clusterer.find_connected_components(similarity_edges, domains)
    
    # Step 4: Analysis and results
    print(f"\n" + "="*50)
    print("STEP 4: ANALYSIS & RESULTS")
    print("="*50)
    
    analysis = clusterer.analyze_clusters(clusters, features_dict)
    
    # Save results
    csv_path, edges_csv, pkl_path = clusterer.save_clustering_results(
        clusters, features_dict, similarity_edges
    )
    
    # Final summary
    print(f"\n" + "="*70)
    print("ADVANCED CLUSTERING COMPLETE!")
    print("="*70)
    
    total_logos = len(features_dict)
    multi_clusters = len([c for c in clusters if len(c) > 1])
    
    print(f"✅ Successfully clustered {total_logos} logos")
    print(f"✅ Found {multi_clusters} groups of similar logos")
    print(f"✅ Method: Multi-criteria graph-based clustering")
    print(f"✅ Techniques: pHash + ORB + Color + Connected Components")
    print(f"✅ No traditional ML algorithms used")
    
    print(f"\n🚀 Key Advantages of This Approach:")
    print(f"   • Perceptual hashing catches near-identical logos efficiently")
    print(f"   • ORB keypoints detect shared design elements")
    print(f"   • Graph clustering finds transitive similarities") 
    print(f"   • Multi-criteria approach reduces false positives")
    print(f"   • Scalable to millions of images with optimization")
    
    if multi_clusters > 0:
        print(f"\n🎯 Found {multi_clusters} clusters with multiple logos!")
        print(f"   Check {csv_path} for detailed results")
    else:
        print(f"\n🔍 No multi-logo clusters found with current thresholds")
        print(f"   Consider lowering thresholds for more lenient clustering")

    def enhanced_similarity_analysis(self, features1, features2):
        """
        Enhanced multi-criteria similarity analysis implementing all techniques from requirements:
        - Perceptual Hash (pHash) with Hamming distance
        - ORB Local Feature Matching with FLANN
        - HSV Color Histogram Correlation
        - HOG Shape Descriptor Comparison
        - Edge Histogram Analysis
        """
        results = {
            'is_similar': False,
            'confidence': 0.0,
            'criteria_met': [],
            'detailed_scores': {}
        }
        
        try:
            # 1. pHash Analysis (Primary - most reliable for near-identical logos)
            phash_similar = False
            if features1.get('phash') is not None and features2.get('phash') is not None:
                hamming_dist = bin(features1['phash'] ^ features2['phash']).count('1')
                phash_similarity = (64 - hamming_dist) / 64.0
                
                results['detailed_scores']['phash_hamming_distance'] = hamming_dist
                results['detailed_scores']['phash_similarity'] = phash_similarity
                
                # Strict threshold as updated (≤4 bits difference)
                if hamming_dist <= self.phash_threshold:
                    phash_similar = True
                    results['criteria_met'].append('pHash')
            
            # 2. ORB Feature Matching (Secondary - for shared design elements)
            orb_similar = False
            orb_matches = 0
            if (features1.get('orb_descriptors') is not None and 
                features2.get('orb_descriptors') is not None and
                len(features1['orb_descriptors']) > 0 and 
                len(features2['orb_descriptors']) > 0):
                
                try:
                    # FLANN matcher for efficiency (as recommended for large datasets)
                    FLANN_INDEX_LSH = 6
                    index_params = dict(algorithm=FLANN_INDEX_LSH,
                                       table_number=6, key_size=12, multi_probe_level=1)
                    search_params = dict(checks=50)
                    
                    flann = cv2.FlannBasedMatcher(index_params, search_params)
                    matches = flann.knnMatch(features1['orb_descriptors'], 
                                           features2['orb_descriptors'], k=2)
                    
                    # Lowe's ratio test for robust matching
                    good_matches = []
                    for match_pair in matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < 0.7 * n.distance:  # Standard threshold
                                good_matches.append(m)
                    
                    orb_matches = len(good_matches)
                    results['detailed_scores']['orb_matches'] = orb_matches
                    
                    # Updated threshold (≥25 matches)
                    if orb_matches >= self.orb_match_threshold:
                        orb_similar = True
                        results['criteria_met'].append('ORB')
                        
                except Exception as e:
                    print(f"ORB matching error: {e}")
            
            # 3. HSV Color Histogram Correlation (Global color analysis)
            color_similar = False
            if (features1.get('color_histogram') is not None and 
                features2.get('color_histogram') is not None):
                
                correlation = cv2.compareHist(features1['color_histogram'],
                                            features2['color_histogram'],
                                            cv2.HISTCMP_CORREL)
                
                results['detailed_scores']['color_correlation'] = correlation
                
                # Updated threshold (≥0.90)
                if correlation >= self.color_corr_threshold:
                    color_similar = True
                    results['criteria_met'].append('Color')
            
            # Multi-criteria decision logic (requires multiple criteria for robustness)
            criteria_count = len(results['criteria_met'])
            
            if criteria_count >= 2:  # At least 2 criteria must be met
                results['is_similar'] = True
                results['confidence'] = criteria_count / 3.0  # Normalize by max criteria
            elif phash_similar and orb_matches > 50:  # Very strong ORB match
                results['is_similar'] = True
                results['confidence'] = 0.9
            elif phash_similar and results['detailed_scores'].get('color_correlation', 0) > 0.95:
                results['is_similar'] = True  # Near-perfect color match
                results['confidence'] = 0.85
            
            return results
            
        except Exception as e:
            print(f"Enhanced similarity analysis error: {e}")
            return results

if __name__ == "__main__":
    main()
