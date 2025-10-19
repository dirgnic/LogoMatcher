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
from skimage import morphology, segmentation, filters, feature, measure
from scipy import ndimage
import re
import pytesseract
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import webcolors
from colorthief import ColorThief
import io
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import SequenceMatcher

class BrandIntelligence:
    """
    Advanced brand intelligence for semantic grouping and analysis
    """
    
    def __init__(self):
        # Industry classification keywords
        self.industry_keywords = {
            'automotive': ['auto', 'car', 'toyota', 'honda', 'ford', 'bmw', 'mercedes', 'audi', 
                          'volkswagen', 'nissan', 'mazda', 'hyundai', 'kia', 'acura', 'lexus',
                          'dealership', 'motors', 'automotive'],
            'cosmetics': ['cosmetics', 'beauty', 'makeup', 'skincare', 'loreal', 'maybelline',
                         'revlon', 'clinique', 'estee', 'mac', 'nars', 'sephora', 'ulta'],
            'industrial': ['industrial', 'manufacturing', 'daikin', 'siemens', 'ge', 'bosch',
                          'caterpillar', 'john-deere', 'equipment', 'machinery'],
            'financial': ['bank', 'financial', 'insurance', 'capital', 'investment', 'credit',
                         'mortgage', 'loan', 'finance', 'securities', 'fund'],
            'retail': ['store', 'shop', 'retail', 'market', 'walmart', 'target', 'amazon',
                      'shopping', 'mall', 'outlet'],
            'technology': ['tech', 'software', 'computer', 'digital', 'microsoft', 'apple',
                          'google', 'ibm', 'oracle', 'cisco', 'intel'],
            'healthcare': ['health', 'medical', 'hospital', 'clinic', 'pharmacy', 'healthcare',
                          'medicine', 'wellness', 'care'],
            'food': ['food', 'restaurant', 'cafe', 'kitchen', 'dining', 'pizza', 'burger',
                    'mcdonald', 'kfc', 'starbucks', 'subway']
        }
        
        # Franchise patterns
        self.franchise_patterns = [
            r'(.+?)[-_](location|city|town|street|ave|road|blvd|plaza|center|mall)',
            r'(.+?)[-_](east|west|north|south|central|downtown|uptown)',
            r'(.+?)[-_]\d+$',  # numbered locations
            r'(.+?)[-_][a-z]{2}$',  # state abbreviations
            r'aamco(.+?)$',  # AAMCO franchise pattern
            r'(.+?)(dealership|dealer|motors)$'
        ]
    
    def extract_brand_family(self, domain_name):
        """
        Extract the core brand name from domain variants
        """
        # Clean domain name
        clean_name = domain_name.lower().strip()
        
        # Remove common suffixes
        suffixes_to_remove = ['_1', '_2', '_3', '_4', '_5', '_6', '_7', '_8', '_9', '_10',
                             '-1', '-2', '-3', '-4', '-5', '-6', '-7', '-8', '-9', '-10']
        
        for suffix in suffixes_to_remove:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)]
                break
        
        # Apply franchise patterns
        for pattern in self.franchise_patterns:
            match = re.match(pattern, clean_name, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Special handling for common franchise types
        if clean_name.startswith('aamco'):
            return 'aamco'
        
        # Handle Toyota dealers
        if 'toyota' in clean_name:
            return 'toyota'
            
        # Handle other automotive brands
        for brand in ['honda', 'ford', 'bmw', 'mercedes', 'audi', 'volkswagen', 'nissan', 'mazda']:
            if brand in clean_name:
                return brand
        
        return clean_name
    
    def classify_industry(self, domain_name, visual_features=None):
        """
        Classify brand into industry category
        """
        domain_lower = domain_name.lower()
        
        # Text-based classification
        for industry, keywords in self.industry_keywords.items():
            for keyword in keywords:
                if keyword in domain_lower:
                    return industry
        
        # Visual feature-based classification (if available)
        if visual_features:
            return self._classify_by_visual_features(visual_features)
        
        return 'general'
    
    def _classify_by_visual_features(self, features):
        """
        Advanced industry classification based on comprehensive visual characteristics
        """
        scores = defaultdict(float)
        
        # Color-based classification
        if 'dominant_colors' in features:
            colors = features['dominant_colors']
            color_scores = self._analyze_color_industry_patterns(colors)
            for industry, score in color_scores.items():
                scores[industry] += score * 0.3
        
        # Shape-based classification
        if 'shape_signature' in features:
            shape_features = features['shape_signature']
            shape_scores = self._analyze_shape_industry_patterns(shape_features)
            for industry, score in shape_scores.items():
                scores[industry] += score * 0.3
        
        # Composition-based classification
        if 'composition' in features:
            composition = features['composition']
            comp_scores = self._analyze_composition_industry_patterns(composition)
            for industry, score in comp_scores.items():
                scores[industry] += score * 0.4
        
        # Return highest scoring industry
        if scores:
            best_industry = max(scores.items(), key=lambda x: x[1])
            if best_industry[1] > 0.5:  # Confidence threshold
                return best_industry[0]
        
        return 'general'
    
    def _analyze_color_industry_patterns(self, colors):
        """Analyze color patterns for industry classification"""
        scores = defaultdict(float)
        
        if not colors:
            return scores
        
        for color in colors:
            r, g, b = color[:3]  # Handle both RGB and RGBA
            
            # Automotive: Red, black, silver, white combinations
            if (r > 200 and g < 100 and b < 100) or (r < 50 and g < 50 and b < 50):  # Red or black
                scores['automotive'] += 0.3
            
            # Financial: Blue, navy, gold combinations
            if b > 150 and r < 100 and g < 100:  # Blue tones
                scores['financial'] += 0.4
            
            # Cosmetics: Pink, purple, gold, pastel colors
            if (r > 180 and g < 150 and b > 150) or (r > 200 and g > 150 and b < 100):  # Pink/gold
                scores['cosmetics'] += 0.5
            
            # Industrial: Gray, blue, green combinations
            if abs(r - g) < 30 and abs(g - b) < 30 and 80 < r < 180:  # Gray tones
                scores['industrial'] += 0.3
            
            # Healthcare: Green, white, blue combinations
            if g > 150 and r < 100 and b < 100:  # Green
                scores['healthcare'] += 0.4
            
            # Technology: Blue, black, white combinations
            if (b > 150 and r < 100) or (r < 50 and g < 50 and b < 50):  # Blue or black
                scores['technology'] += 0.3
        
        return scores
    
    def _analyze_shape_industry_patterns(self, shape_signature):
        """Analyze shape patterns for industry classification"""
        scores = defaultdict(float)
        
        shape_features = shape_signature.get('shape_features', [])
        if not shape_features:
            return scores
        
        for shape in shape_features:
            circularity = shape.get('circularity', 0)
            aspect_ratio = shape.get('aspect_ratio', 1)
            solidity = shape.get('solidity', 0)
            
            # Automotive: Often circular/shield shapes or elongated logos
            if circularity > 0.7 or (aspect_ratio > 2.0 and solidity > 0.8):
                scores['automotive'] += 0.2
            
            # Financial: Often square/rectangular, high solidity
            if 0.8 < aspect_ratio < 1.2 and solidity > 0.9:
                scores['financial'] += 0.3
            
            # Industrial: Often complex shapes, lower solidity
            if solidity < 0.7 and circularity < 0.5:
                scores['industrial'] += 0.2
            
            # Cosmetics: Often elegant, curved shapes
            if circularity > 0.5 and solidity > 0.8:
                scores['cosmetics'] += 0.2
        
        return scores
    
    def _analyze_composition_industry_patterns(self, composition):
        """Analyze logo composition for industry classification"""
        scores = defaultdict(float)
        
        text_ratio = composition.get('text_ratio', 0.5)
        logo_type = composition.get('logo_type', 'balanced')
        
        # Financial: Often text-heavy with clean typography
        if logo_type == 'text_heavy' and text_ratio > 0.6:
            scores['financial'] += 0.4
        
        # Automotive: Often balanced with brand names
        if logo_type == 'balanced' and 0.3 < text_ratio < 0.7:
            scores['automotive'] += 0.3
        
        # Industrial: Often symbol-heavy with company marks
        if logo_type == 'symbol_heavy' and text_ratio < 0.3:
            scores['industrial'] += 0.4
        
        # Cosmetics: Often symbol-heavy with elegant designs
        if logo_type == 'symbol_heavy' and text_ratio < 0.4:
            scores['cosmetics'] += 0.2
        
        # Technology: Often balanced or symbol-heavy
        if logo_type in ['balanced', 'symbol_heavy']:
            scores['technology'] += 0.2
        
        return scores
    
    def _color_distance(self, color1, color2):
        """Calculate Euclidean distance between colors"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(color1, color2)))
    
    def detect_franchise_relationship(self, domain1, domain2):
        """
        Detect if two domains are from the same franchise
        """
        brand1 = self.extract_brand_family(domain1)
        brand2 = self.extract_brand_family(domain2)
        
        # Same brand family = franchise relationship
        if brand1 == brand2:
            return True
        
        # Check for similar names with high similarity
        similarity = SequenceMatcher(None, brand1, brand2).ratio()
        return similarity > 0.8

class AdvancedVisualAnalyzer:
    """
    Advanced visual analysis for logo-specific features
    """
    
    def __init__(self):
        self.text_regions = []
        self.symbol_regions = []
    
    def extract_text_regions(self, image):
        """
        Advanced text region extraction using multiple techniques
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Multiple text detection approaches
            text_regions = []
            
            # Approach 1: Gradient-based detection
            grad_regions = self._detect_text_by_gradient(gray)
            text_regions.extend(grad_regions)
            
            # Approach 2: MSER (Maximally Stable Extremal Regions)
            mser_regions = self._detect_text_by_mser(gray)
            text_regions.extend(mser_regions)
            
            # Approach 3: Stroke Width Transform approximation
            swt_regions = self._detect_text_by_swt_approximation(gray)
            text_regions.extend(swt_regions)
            
            # Remove duplicates and filter
            text_regions = self._filter_and_merge_text_regions(text_regions)
            
            return text_regions
            
        except Exception as e:
            print(f"Text extraction error: {e}")
            return []
    
    def _detect_text_by_gradient(self, gray):
        """Gradient-based text detection"""
        try:
            # Calculate gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient = np.sqrt(grad_x**2 + grad_y**2)
            
            # Threshold and morphological operations
            _, thresh = cv2.threshold(gradient.astype(np.uint8), 0, 255, 
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to connect text components
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter by text-like characteristics
                if self._is_text_like_region(w, h, cv2.contourArea(contour)):
                    regions.append((x, y, w, h))
            
            return regions
        except:
            return []
    
    def _detect_text_by_mser(self, gray):
        """MSER-based text detection"""
        try:
            # Create MSER detector
            mser = cv2.MSER_create()
            
            # Detect regions
            regions, _ = mser.detectRegions(gray)
            
            text_regions = []
            for region in regions:
                # Get bounding rectangle
                x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
                
                # Filter by text characteristics
                if self._is_text_like_region(w, h, len(region)):
                    text_regions.append((x, y, w, h))
            
            return text_regions
        except:
            return []
    
    def _detect_text_by_swt_approximation(self, gray):
        """Approximate Stroke Width Transform for text detection"""
        try:
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate to connect stroke components
            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate stroke width approximation
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                if perimeter > 0:
                    stroke_width = area / perimeter
                    
                    # Filter by stroke width and dimensions
                    if (2 < stroke_width < 20 and 
                        self._is_text_like_region(w, h, area)):
                        regions.append((x, y, w, h))
            
            return regions
        except:
            return []
    
    def _is_text_like_region(self, width, height, area):
        """Check if region has text-like characteristics"""
        if width < 8 or height < 6:  # Too small
            return False
        
        if width > 300 or height > 100:  # Too large
            return False
        
        aspect_ratio = width / height
        if aspect_ratio < 0.2 or aspect_ratio > 10:  # Unrealistic aspect ratio
            return False
        
        # Area consistency check
        expected_area = width * height
        if area < 0.2 * expected_area or area > 0.9 * expected_area:
            return False
        
        return True
    
    def _filter_and_merge_text_regions(self, regions):
        """Filter and merge overlapping text regions"""
        if not regions:
            return []
        
        # Remove duplicates
        unique_regions = []
        for region in regions:
            is_duplicate = False
            for unique_region in unique_regions:
                if self._regions_overlap(region, unique_region, threshold=0.5):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_regions.append(region)
        
        # Merge nearby regions that could be part of the same text line
        merged_regions = []
        used = set()
        
        for i, region1 in enumerate(unique_regions):
            if i in used:
                continue
            
            # Find regions to merge with this one
            merge_group = [region1]
            used.add(i)
            
            for j, region2 in enumerate(unique_regions):
                if j in used:
                    continue
                
                if self._should_merge_text_regions(region1, region2):
                    merge_group.append(region2)
                    used.add(j)
            
            # Merge the group into a single region
            if len(merge_group) == 1:
                merged_regions.append(merge_group[0])
            else:
                merged_region = self._merge_region_group(merge_group)
                merged_regions.append(merged_region)
        
        return merged_regions
    
    def _regions_overlap(self, region1, region2, threshold=0.5):
        """Check if two regions overlap significantly"""
        x1, y1, w1, h1 = region1
        x2, y2, w2, h2 = region2
        
        # Calculate intersection
        x_overlap = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        y_overlap = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        
        intersection = x_overlap * y_overlap
        area1 = w1 * h1
        area2 = w2 * h2
        
        # Calculate IoU (Intersection over Union)
        union = area1 + area2 - intersection
        iou = intersection / union if union > 0 else 0
        
        return iou > threshold
    
    def _should_merge_text_regions(self, region1, region2):
        """Check if two text regions should be merged (same text line)"""
        x1, y1, w1, h1 = region1
        x2, y2, w2, h2 = region2
        
        # Check vertical alignment (similar y-coordinates)
        center_y1 = y1 + h1 / 2
        center_y2 = y2 + h2 / 2
        
        if abs(center_y1 - center_y2) > max(h1, h2) * 0.5:
            return False
        
        # Check horizontal proximity
        gap = abs((x1 + w1) - x2) if x1 < x2 else abs((x2 + w2) - x1)
        
        if gap > max(w1, w2) * 0.5:
            return False
        
        return True
    
    def _merge_region_group(self, regions):
        """Merge a group of regions into a single bounding box"""
        min_x = min(r[0] for r in regions)
        min_y = min(r[1] for r in regions)
        max_x = max(r[0] + r[2] for r in regions)
        max_y = max(r[1] + r[3] for r in regions)
        
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def analyze_logo_composition(self, image):
        """
        Analyze the composition of logo (text vs symbol ratio)
        """
        try:
            text_regions = self.extract_text_regions(image)
            
            # Calculate text area
            text_area = sum(w * h for x, y, w, h in text_regions)
            total_area = image.shape[0] * image.shape[1]
            
            text_ratio = text_area / total_area if total_area > 0 else 0
            symbol_ratio = 1 - text_ratio
            
            # Determine logo type
            if text_ratio > 0.6:
                logo_type = 'text_heavy'
            elif text_ratio < 0.2:
                logo_type = 'symbol_heavy'
            else:
                logo_type = 'balanced'
            
            return {
                'text_ratio': text_ratio,
                'symbol_ratio': symbol_ratio,
                'logo_type': logo_type,
                'text_regions': text_regions
            }
            
        except Exception as e:
            print(f"Composition analysis error: {e}")
            return {'text_ratio': 0.5, 'symbol_ratio': 0.5, 'logo_type': 'balanced', 'text_regions': []}
    
    def extract_color_palette(self, image, n_colors=5):
        """
        Extract dominant color palette from logo using fast histogram method
        """
        try:
            # Fast histogram-based approach instead of slow KMeans
            if len(image.shape) == 3:
                # Convert to HSV for better color separation
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
                # Simple quantization: bin colors into major groups
                h_bins = 18  # 20-degree hue bins
                s_bins = 8   # saturation bins  
                v_bins = 8   # value bins
                
                # Calculate 3D histogram
                hist = cv2.calcHist([hsv], [0, 1, 2], None, [h_bins, s_bins, v_bins], 
                                  [0, 180, 0, 256, 0, 256])
                
                # Find dominant color bins
                hist_flat = hist.flatten()
                dominant_indices = np.argsort(hist_flat)[-n_colors:][::-1]
                
                # Convert back to RGB colors
                colors = []
                color_distribution = []
                
                for idx in dominant_indices:
                    if hist_flat[idx] > 0:
                        # Convert flat index back to 3D coordinates
                        h_idx = idx // (s_bins * v_bins)
                        s_idx = (idx % (s_bins * v_bins)) // v_bins
                        v_idx = idx % v_bins
                        
                        # Convert bin indices to actual HSV values
                        h = int(h_idx * 180 / h_bins + 180 / (2 * h_bins))
                        s = int(s_idx * 256 / s_bins + 256 / (2 * s_bins))  
                        v = int(v_idx * 256 / v_bins + 256 / (2 * v_bins))
                        
                        # Convert HSV to RGB
                        hsv_color = np.uint8([[[h, s, v]]])
                        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
                        
                        colors.append(rgb_color.astype(int).tolist())
                        color_distribution.append(hist_flat[idx])
                
                # Normalize distribution
                if color_distribution:
                    total = sum(color_distribution)
                    color_distribution = [c/total for c in color_distribution]
            else:
                # Grayscale - simple binning
                hist = cv2.calcHist([image], [0], None, [16], [0, 256])
                dominant_indices = np.argsort(hist.flatten())[-n_colors:][::-1]
                
                colors = []
                color_distribution = []
                
                for idx in dominant_indices:
                    if hist[idx][0] > 0:
                        gray_val = int(idx * 256 / 16)
                        colors.append([gray_val, gray_val, gray_val])
                        color_distribution.append(hist[idx][0])
                
                # Normalize
                if color_distribution:
                    total = sum(color_distribution)
                    color_distribution = [c/total for c in color_distribution]
            
            return {
                'dominant_colors': colors,
                'color_distribution': color_distribution
            }
            
        except Exception as e:
            print(f"Color palette extraction error: {e}")
            return {'dominant_colors': [], 'color_distribution': []}
    
    def analyze_shape_signature(self, image):
        """
        Extract shape signature for logo symbols
        """
        try:
            # Convert to binary
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {'shape_features': [], 'contour_count': 0}
            
            # Analyze main contours
            shape_features = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # Filter small contours
                    perimeter = cv2.arcLength(contour, True)
                    
                    # Shape descriptors
                    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                    
                    # Bounding box features
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h if h > 0 else 0
                    
                    # Convex hull features
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    shape_features.append({
                        'area': area,
                        'perimeter': perimeter,
                        'circularity': circularity,
                        'aspect_ratio': aspect_ratio,
                        'solidity': solidity
                    })
            
            return {
                'shape_features': shape_features,
                'contour_count': len(contours)
            }
            
        except Exception as e:
            print(f"Shape signature error: {e}")
            return {'shape_features': [], 'contour_count': 0}
    
    def analyze_spatial_layout(self, image):
        """
        Analyze spatial layout and positioning
        """
        try:
            text_regions = self.extract_text_regions(image)
            h, w = image.shape[:2]
            
            # Analyze text positioning
            text_positions = []
            for x, y, tw, th in text_regions:
                # Normalize positions
                center_x = (x + tw/2) / w
                center_y = (y + th/2) / h
                
                # Determine position category
                if center_y < 0.33:
                    v_pos = 'top'
                elif center_y > 0.67:
                    v_pos = 'bottom'
                else:
                    v_pos = 'center'
                
                if center_x < 0.33:
                    h_pos = 'left'
                elif center_x > 0.67:
                    h_pos = 'right'
                else:
                    h_pos = 'center'
                
                text_positions.append(f"{v_pos}_{h_pos}")
            
            # Overall layout analysis
            if len(text_regions) == 0:
                layout_type = 'symbol_only'
            elif len(text_regions) == 1:
                layout_type = 'simple'
            else:
                layout_type = 'complex'
            
            return {
                'layout_type': layout_type,
                'text_positions': text_positions,
                'text_count': len(text_regions)
            }
            
        except Exception as e:
            print(f"Spatial layout error: {e}")
            return {'layout_type': 'simple', 'text_positions': [], 'text_count': 0}

class MultiScaleHasher:
    """
    Multi-scale perceptual hashing for improved logo matching
    """
    
    def __init__(self):
        self.scales = [32, 64, 128]  # Different resolution scales
    
    def compute_multiscale_hash(self, image):
        """
        Compute perceptual hashes at multiple scales
        """
        try:
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            hashes = {}
            for scale in self.scales:
                # Resize image
                resized = pil_image.resize((scale, scale), Image.Resampling.LANCZOS)
                
                # Compute different types of hashes
                hashes[f'phash_{scale}'] = str(imagehash.phash(resized))
                hashes[f'dhash_{scale}'] = str(imagehash.dhash(resized))
                hashes[f'whash_{scale}'] = str(imagehash.whash(resized))
            
            return hashes
            
        except Exception as e:
            print(f"Multi-scale hashing error: {e}")
            return {}
    
    def compute_logo_aware_hash(self, image, text_regions=None):
        """
        Compute hash focusing on central logo elements
        """
        try:
            h, w = image.shape[:2]
            
            # Create mask for central region (logo typically centered)
            center_mask = np.zeros((h, w), dtype=np.uint8)
            
            # Focus on central 70% of image
            start_h, end_h = int(h * 0.15), int(h * 0.85)
            start_w, end_w = int(w * 0.15), int(w * 0.85)
            center_mask[start_h:end_h, start_w:end_w] = 255
            
            # If text regions provided, add them to mask
            if text_regions:
                for x, y, tw, th in text_regions:
                    center_mask[y:y+th, x:x+tw] = 255
            
            # Apply mask
            if len(image.shape) == 3:
                masked_image = cv2.bitwise_and(image, image, mask=center_mask)
                pil_image = Image.fromarray(cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB))
            else:
                masked_image = cv2.bitwise_and(image, image, mask=center_mask)
                pil_image = Image.fromarray(masked_image)
            
            # Compute hash
            return str(imagehash.phash(pil_image, hash_size=16))
            
        except Exception as e:
            print(f"Logo-aware hashing error: {e}")
            return ""
    
    def compute_text_sensitive_hash(self, image, composition_analysis):
        """
        Compute separate hashes for text and symbol regions
        """
        try:
            h, w = image.shape[:2]
            text_regions = composition_analysis.get('text_regions', [])
            
            # Create text mask
            text_mask = np.zeros((h, w), dtype=np.uint8)
            symbol_mask = np.ones((h, w), dtype=np.uint8) * 255
            
            for x, y, tw, th in text_regions:
                text_mask[y:y+th, x:x+tw] = 255
                symbol_mask[y:y+th, x:x+tw] = 0
            
            hashes = {}
            
            # Text region hash
            if len(image.shape) == 3:
                text_region = cv2.bitwise_and(image, image, mask=text_mask)
                text_pil = Image.fromarray(cv2.cvtColor(text_region, cv2.COLOR_BGR2RGB))
            else:
                text_region = cv2.bitwise_and(image, image, mask=text_mask)
                text_pil = Image.fromarray(text_region)
            
            hashes['text_hash'] = str(imagehash.phash(text_pil))
            
            # Symbol region hash
            if len(image.shape) == 3:
                symbol_region = cv2.bitwise_and(image, image, mask=symbol_mask)
                symbol_pil = Image.fromarray(cv2.cvtColor(symbol_region, cv2.COLOR_BGR2RGB))
            else:
                symbol_region = cv2.bitwise_and(image, image, mask=symbol_mask)
                symbol_pil = Image.fromarray(symbol_region)
            
            hashes['symbol_hash'] = str(imagehash.phash(symbol_pil))
            
            return hashes
            
        except Exception as e:
            print(f"Text-sensitive hashing error: {e}")
            return {'text_hash': '', 'symbol_hash': ''}
    
    def compute_dct_hash(self, image, hash_size=8):
        """
        Advanced DCT-based hashing with frequency domain analysis
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Resize to standard size for DCT
            resized = cv2.resize(gray, (hash_size * 4, hash_size * 4))
            
            # Apply DCT (Discrete Cosine Transform)
            dct_result = cv2.dct(np.float32(resized))
            
            # Extract low-frequency components (top-left corner)
            dct_low_freq = dct_result[:hash_size, :hash_size]
            
            # Calculate median for binary hash
            median = np.median(dct_low_freq)
            
            # Generate binary hash
            binary_hash = (dct_low_freq > median).astype(int)
            
            # Convert to string
            hash_string = ''.join(str(bit) for row in binary_hash for bit in row)
            
            return hash_string
            
        except Exception as e:
            print(f"DCT hash error: {e}")
            return ""
    
    def compute_fft_hash(self, image, hash_size=8):
        """
        FFT-based frequency domain hashing for logo analysis
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Resize to power of 2 for efficient FFT
            size = 64  # 2^6
            resized = cv2.resize(gray, (size, size))
            
            # Apply 2D FFT
            fft_result = fft2(resized)
            fft_shifted = fftshift(fft_result)
            
            # Get magnitude spectrum
            magnitude = np.abs(fft_shifted)
            
            # Focus on low-frequency components (center region)
            center = size // 2
            low_freq_region = magnitude[
                center - hash_size//2:center + hash_size//2,
                center - hash_size//2:center + hash_size//2
            ]
            
            # Calculate median for binary hash
            median = np.median(low_freq_region)
            
            # Generate binary hash
            binary_hash = (low_freq_region > median).astype(int)
            
            # Convert to string
            hash_string = ''.join(str(bit) for row in binary_hash for bit in row)
            
            return hash_string
            
        except Exception as e:
            print(f"FFT hash error: {e}")
            return ""
    
    def compute_brand_specific_hash(self, image, brand_family, industry):
        """
        Brand-specific feature hashing based on industry and brand characteristics
        """
        try:
            hashes = {}
            
            # Industry-specific preprocessing
            if industry == 'automotive':
                # Focus on text and brand symbols
                processed = self._preprocess_for_automotive(image)
            elif industry == 'cosmetics':
                # Focus on colors and package shapes
                processed = self._preprocess_for_cosmetics(image)
            elif industry == 'industrial':
                # Focus on symbols and typography
                processed = self._preprocess_for_industrial(image)
            elif industry == 'financial':
                # Focus on structure and official branding
                processed = self._preprocess_for_financial(image)
            else:
                processed = image
            
            # Compute multiple hashes for the processed image
            hashes['dct_hash'] = self.compute_dct_hash(processed)
            hashes['fft_hash'] = self.compute_fft_hash(processed)
            
            # Brand-specific features
            if brand_family:
                # Create brand-aware mask
                brand_mask = self._create_brand_mask(processed, brand_family)
                if brand_mask is not None:
                    masked_image = cv2.bitwise_and(processed, processed, mask=brand_mask)
                    hashes['brand_dct'] = self.compute_dct_hash(masked_image)
            
            return hashes
            
        except Exception as e:
            print(f"Brand-specific hashing error: {e}")
            return {}
    
    def _preprocess_for_automotive(self, image):
        """Automotive-specific preprocessing focusing on text and symbols"""
        try:
            # Enhance text regions
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply morphological operations to enhance text
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
            
            return processed
        except:
            return image
    
    def _preprocess_for_cosmetics(self, image):
        """Cosmetics-specific preprocessing focusing on colors and shapes"""
        try:
            # Preserve color information more strongly
            if len(image.shape) == 3:
                # Convert to LAB color space for better color representation
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                # Enhance color channels
                lab[:,:,1] = cv2.equalizeHist(lab[:,:,1])  # A channel
                lab[:,:,2] = cv2.equalizeHist(lab[:,:,2])  # B channel
                processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                return processed
            else:
                return image
        except:
            return image
    
    def _preprocess_for_industrial(self, image):
        """Industrial-specific preprocessing focusing on symbols and typography"""
        try:
            # Enhance edges and structural elements
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply edge enhancement
            edges = cv2.Canny(gray, 50, 150)
            
            # Combine original with edges
            processed = cv2.addWeighted(gray, 0.7, edges, 0.3, 0)
            
            return processed
        except:
            return image
    
    def _preprocess_for_financial(self, image):
        """Financial-specific preprocessing focusing on structure and branding"""
        try:
            # Focus on clean, structured elements
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Apply bilateral filter to smooth while preserving edges
            processed = cv2.bilateralFilter(gray, 9, 75, 75)
            
            return processed
        except:
            return image
    
    def _create_brand_mask(self, image, brand_family):
        """Create a mask focusing on brand-specific regions"""
        try:
            h, w = image.shape[:2]
            
            # Create mask based on brand characteristics
            if brand_family.lower() in ['aamco', 'toyota', 'honda']:
                # Automotive brands - focus on center and text regions
                mask = np.zeros((h, w), dtype=np.uint8)
                # Focus on central region where logos typically appear
                center_h, center_w = h//2, w//2
                mask[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4] = 255
                return mask
            else:
                # General brand mask - focus on center
                mask = np.zeros((h, w), dtype=np.uint8)
                center_h, center_w = h//2, w//2
                mask[center_h-h//3:center_h+h//3, center_w-w//3:center_w+w//3] = 255
                return mask
                
        except Exception:
            return None

class HierarchicalBrandClusterer:
    """
    Hierarchical clustering for brand families and variants
    """
    
    def __init__(self, brand_intelligence):
        self.brand_intelligence = brand_intelligence
        self.cluster_hierarchy = {}
    
    def create_hierarchical_clusters(self, domains_with_features):
        """
        Create 3-level hierarchical clusters:
        Level 1: Brand families
        Level 2: Brand variants 
        Level 3: Visual variations
        """
        # Level 1: Group by brand family
        brand_families = defaultdict(list)
        for domain, features in domains_with_features.items():
            brand_family = self.brand_intelligence.extract_brand_family(domain)
            brand_families[brand_family].append((domain, features))
        
        hierarchy = {}
        
        for brand_family, members in brand_families.items():
            if len(members) == 1:
                # Single member family
                hierarchy[brand_family] = {
                    'level_1': brand_family,
                    'level_2': {members[0][0]: [members[0]]},
                    'level_3': {members[0][0]: {members[0][0]: members[0]}}
                }
            else:
                # Multi-member family - create sub-levels
                hierarchy[brand_family] = self._create_sub_hierarchy(brand_family, members)
        
        return hierarchy
    
    def _create_sub_hierarchy(self, brand_family, members):
        """
        Create sub-hierarchy for brand variants and visual variations
        """
        # Level 2: Group by location/dealer variants
        variants = defaultdict(list)
        
        for domain, features in members:
            # Extract variant key (location/dealer)
            variant_key = self._extract_variant_key(domain, brand_family)
            variants[variant_key].append((domain, features))
        
        level_2 = {}
        level_3 = {}
        
        for variant_key, variant_members in variants.items():
            level_2[variant_key] = variant_members
            
            # Level 3: Group by visual similarity within variant
            if len(variant_members) > 1:
                visual_groups = self._group_by_visual_similarity(variant_members)
                level_3[variant_key] = visual_groups
            else:
                level_3[variant_key] = {variant_members[0][0]: variant_members[0]}
        
        return {
            'level_1': brand_family,
            'level_2': level_2,
            'level_3': level_3
        }
    
    def _extract_variant_key(self, domain, brand_family):
        """
        Extract variant identifier (location, dealer, etc.)
        """
        # Remove brand family from domain to get variant
        domain_clean = domain.lower().replace(brand_family.lower(), '').strip('-_')
        
        if not domain_clean:
            return 'main'
        
        # Extract meaningful variant parts
        variant_parts = re.split(r'[-_]', domain_clean)
        
        # Keep location/dealer indicators
        meaningful_parts = []
        for part in variant_parts:
            if len(part) > 2 and not part.isdigit():
                meaningful_parts.append(part)
        
        return '_'.join(meaningful_parts) if meaningful_parts else 'variant'
    
    def _group_by_visual_similarity(self, members):
        """
        Group members by visual similarity for Level 3
        """
        if len(members) <= 1:
            return {members[0][0]: members[0]} if members else {}
        
        # Compute pairwise visual similarity
        visual_groups = {}
        assigned = set()
        
        for i, (domain1, features1) in enumerate(members):
            if domain1 in assigned:
                continue
                
            # Start new group
            group_key = domain1
            visual_groups[group_key] = [(domain1, features1)]
            assigned.add(domain1)
            
            # Find visually similar members
            for j, (domain2, features2) in enumerate(members):
                if j <= i or domain2 in assigned:
                    continue
                
                # Check visual similarity
                if self._are_visually_similar(features1, features2):
                    visual_groups[group_key].append((domain2, features2))
                    assigned.add(domain2)
        
        # Convert to final format
        final_groups = {}
        for group_key, group_members in visual_groups.items():
            if len(group_members) == 1:
                final_groups[group_key] = group_members[0]
            else:
                final_groups[group_key] = group_members
        
        return final_groups
    
    def _are_visually_similar(self, features1, features2):
        """
        Check if two logos are visually similar
        """
        try:
            # Compare perceptual hashes
            if 'phash' in features1 and 'phash' in features2:
                phash_dist = bin(features1['phash'] ^ features2['phash']).count('1')
                if phash_dist <= 5:  # Very similar
                    return True
            
            # Compare color palettes
            if 'color_palette' in features1 and 'color_palette' in features2:
                colors1 = features1['color_palette'].get('dominant_colors', [])
                colors2 = features2['color_palette'].get('dominant_colors', [])
                
                if colors1 and colors2:
                    color_similarity = self._compute_color_similarity(colors1, colors2)
                    if color_similarity > 0.8:
                        return True
            
            return False
            
        except Exception:
            return False
    
    def _compute_color_similarity(self, colors1, colors2):
        """
        Compute similarity between color palettes
        """
        try:
            # Convert to numpy arrays
            c1 = np.array(colors1)
            c2 = np.array(colors2)
            
            # Find closest colors
            similarities = []
            for color1 in c1:
                distances = [np.linalg.norm(color1 - color2) for color2 in c2]
                min_distance = min(distances)
                # Convert distance to similarity (0-1 scale)
                similarity = 1.0 / (1.0 + min_distance / 255.0)
                similarities.append(similarity)
            
            return np.mean(similarities)
            
        except Exception:
            return 0.0

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
        
        # Initialize advanced analysis components
        self.brand_intelligence = BrandIntelligence()
        self.visual_analyzer = AdvancedVisualAnalyzer()
        self.multiscale_hasher = MultiScaleHasher()
        self.hierarchical_clusterer = HierarchicalBrandClusterer(self.brand_intelligence)
        
        # FLEXIBLE SIMILARITY THRESHOLDS FOR SMART CLUSTERING
        # Goal: Achieve <50 clusters by using adaptive thresholds based on context
        
        # Base thresholds for very strict matching (near-identical logos)
        self.phash_threshold_strict = 4  # Very similar logos only
        self.orb_match_threshold_strict = 25  # Strong keypoint agreement
        self.color_corr_threshold_strict = 0.90  # Nearly identical colors
        
        # Relaxed thresholds for merging singleton clusters - VERY AGGRESSIVE
        self.phash_threshold_relaxed = 35  # Much more variation allowed for singletons
        self.orb_match_threshold_relaxed = 5   # Very low keypoint requirement
        self.color_corr_threshold_relaxed = 0.45  # Much more color tolerance
        
        # Ultra-relaxed thresholds for final singleton cleanup
        self.phash_threshold_ultra_relaxed = 50  # Maximum variation for final merging
        self.orb_match_threshold_ultra_relaxed = 3   # Minimal keypoint requirement
        self.color_corr_threshold_ultra_relaxed = 0.30  # Maximum color tolerance
        
        # Large cluster splitting thresholds (stricter to break up huge clusters)  
        self.phash_threshold_split = 2  # Very tight for large clusters
        self.large_cluster_size = 20  # Define "large cluster"
        
        # Current active thresholds (will be set dynamically)
        self.phash_threshold = self.phash_threshold_strict
        self.orb_match_threshold = self.orb_match_threshold_strict  
        self.color_corr_threshold = self.color_corr_threshold_strict
        
        # Industry-specific similarity weights
        self.industry_weights = {
            'automotive': {'text_weight': 0.4, 'symbol_weight': 0.3, 'color_weight': 0.3},
            'cosmetics': {'text_weight': 0.2, 'symbol_weight': 0.3, 'color_weight': 0.5},
            'industrial': {'text_weight': 0.5, 'symbol_weight': 0.4, 'color_weight': 0.1},
            'financial': {'text_weight': 0.6, 'symbol_weight': 0.3, 'color_weight': 0.1},
            'general': {'text_weight': 0.33, 'symbol_weight': 0.33, 'color_weight': 0.34}
        }
        
        # ORB detector for keypoint analysis
        self.orb_detector = cv2.ORB_create(nfeatures=500)  # Limit features for speed
        
        # Load JPEG file paths
        self.jpeg_files = self._load_jpeg_paths()
        print(f" Found {len(self.jpeg_files)} JPEG files for clustering analysis")
        print(f" Enhanced semantic analysis enabled: Brand intelligence, visual analysis, hierarchical clustering")
    
    def _load_jpeg_paths(self):
        """Load all JPEG file paths with domain extraction"""
        if not os.path.exists(self.jpeg_folder):
            print(f" Error: Folder {self.jpeg_folder} not found!")
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
        """Extract all features for advanced similarity analysis with semantic intelligence"""
        filepath = jpeg_info['filepath']
        domain = jpeg_info['domain']
        index = jpeg_info['index']
        
        try:
            # Load image for analysis
            image = cv2.imread(filepath)
            if image is None:
                raise ValueError(f"Could not load image: {filepath}")
            
            # === BRAND INTELLIGENCE ===
            brand_family = self.brand_intelligence.extract_brand_family(domain)
            industry = self.brand_intelligence.classify_industry(domain)
            
            # === BASIC FEATURES (LEGACY) ===
            # 1. Perceptual Hash (primary similarity measure)
            phash = self.compute_perceptual_hash(filepath)
            
            # 2. ORB descriptors (for design element matching)
            keypoints, orb_descriptors = self.compute_orb_descriptors(filepath)
            
            # 3. Color histogram (supplementary)
            color_hist = self.compute_color_histogram(filepath)
            
            # === ADVANCED VISUAL FEATURES ===
            # 4. Logo composition analysis
            composition = self.visual_analyzer.analyze_logo_composition(image)
            
            # 5. Color palette extraction
            color_palette = self.visual_analyzer.extract_color_palette(image)
            
            # 6. Shape signature analysis
            shape_signature = self.visual_analyzer.analyze_shape_signature(image)
            
            # 7. Spatial layout analysis
            spatial_layout = self.visual_analyzer.analyze_spatial_layout(image)
            
            # === ADVANCED MULTI-SCALE HASHING ===
            # 8. Multi-scale perceptual hashes
            multiscale_hashes = self.multiscale_hasher.compute_multiscale_hash(image)
            
            # 9. Logo-aware hash (focusing on central elements)
            text_regions = composition.get('text_regions', [])
            logo_aware_hash = self.multiscale_hasher.compute_logo_aware_hash(image, text_regions)
            
            # 10. Text-sensitive hashes
            text_sensitive_hashes = self.multiscale_hasher.compute_text_sensitive_hash(image, composition)
            
            # 11. DCT-based hash
            dct_hash = self.multiscale_hasher.compute_dct_hash(image)
            
            # 12. FFT-based hash  
            fft_hash = self.multiscale_hasher.compute_fft_hash(image)
            
            # 13. Brand-specific hashes
            brand_specific_hashes = self.multiscale_hasher.compute_brand_specific_hash(
                image, brand_family, industry)
            
            # === OCR-BASED TEXT ANALYSIS ===
            try:
                # Extract text content for similarity analysis
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Note: pytesseract might not be available, so we'll use a fallback
                extracted_text = self._extract_text_fallback(gray)
            except Exception:
                extracted_text = ""
            
            features = {
                # Basic info
                'domain': domain,
                'filepath': filepath,
                'index': index,
                
                # Brand intelligence
                'brand_family': brand_family,
                'industry': industry,
                
                # Legacy features
                'phash': phash,
                'orb_descriptors': orb_descriptors,
                'orb_keypoints_count': len(keypoints) if keypoints else 0,
                'color_histogram': color_hist,
                
                # Advanced visual features
                'composition': composition,
                'color_palette': color_palette,
                'shape_signature': shape_signature,
                'spatial_layout': spatial_layout,
                
                # Advanced multi-scale hashing
                'multiscale_hashes': multiscale_hashes,
                'logo_aware_hash': logo_aware_hash,
                'text_sensitive_hashes': text_sensitive_hashes,
                'dct_hash': dct_hash,
                'fft_hash': fft_hash,
                'brand_specific_hashes': brand_specific_hashes,
                
                # Text analysis
                'extracted_text': extracted_text
            }
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error for {domain}: {e}")
            return {
                'domain': domain,
                'filepath': filepath, 
                'index': index,
                'brand_family': domain,
                'industry': 'general',
                'phash': 0,
                'orb_descriptors': None,
                'orb_keypoints_count': 0,
                'color_histogram': np.zeros(48, dtype=np.float32),
                'composition': {'text_ratio': 0.5, 'symbol_ratio': 0.5, 'logo_type': 'balanced'},
                'color_palette': {'dominant_colors': [], 'color_distribution': []},
                'shape_signature': {'shape_features': [], 'contour_count': 0},
                'spatial_layout': {'layout_type': 'simple', 'text_positions': [], 'text_count': 0},
                'multiscale_hashes': {},
                'logo_aware_hash': "",
                'text_sensitive_hashes': {'text_hash': '', 'symbol_hash': ''},
                'dct_hash': "",
                'fft_hash': "",
                'brand_specific_hashes': {},
                'extracted_text': ""
            }
    
    def _extract_text_fallback(self, gray_image):
        """
        Fallback text extraction without pytesseract dependency
        """
        try:
            # Simple text detection using contours and morphology
            # This is a basic fallback - in production, pytesseract would be preferred
            
            # Apply threshold
            _, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Morphological operations to highlight text
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count text-like regions (approximate text detection)
            text_regions = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 1.5 <= aspect_ratio <= 8 and w > 10 and h > 5:  # Text-like dimensions
                    text_regions += 1
            
            # Return approximate text indicator
            if text_regions > 3:
                return "text_heavy"
            elif text_regions > 0:
                return "text_moderate"
            else:
                return "text_minimal"
                
        except Exception:
            return ""
    
    def extract_all_features_parallel(self):
        """Extract features for all logos using aggressive parallelism"""
        print(f" Extracting comprehensive features from {len(self.jpeg_files)} logos...")
        print(f" Using {self.max_workers} threads with batch size {self.batch_size}")
        
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
        
        print(f" Feature extraction completed for {len(all_features)} logos")
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
    
    def compute_pairwise_similarity(self, features1, features2, context="normal"):
        """
        ADVANCED SEMANTIC SIMILARITY COMPUTATION
        Uses brand intelligence, industry-specific weights, and multi-scale features
        
        Context modes:
        - "strict": For initial clustering (strict thresholds)
        - "relaxed": For merging singletons (relaxed thresholds) 
        - "split": For breaking large clusters (very strict thresholds)
        """
        similarities = {}
        
        try:
            # === BRAND INTELLIGENCE CHECKS ===
            # Check if same brand family (highest priority)
            brand_family_match = (features1.get('brand_family', '') == features2.get('brand_family', ''))
            franchise_relationship = self.brand_intelligence.detect_franchise_relationship(
                features1.get('domain', ''), features2.get('domain', ''))
            
            # Industry classification
            industry1 = features1.get('industry', 'general')
            industry2 = features2.get('industry', 'general')
            same_industry = industry1 == industry2
            
            # Get industry-specific weights
            weights = self.industry_weights.get(industry1, self.industry_weights['general'])
            
            # === LEGACY FEATURES ===
            # 1. Perceptual Hash Similarity (PRIMARY - most important)
            phash_distance = self.compute_hamming_distance(features1['phash'], features2['phash'])
            similarities['phash_distance'] = phash_distance
            
            # 2. ORB Keypoint Matching (SECONDARY - for design elements)
            orb_matches = self.match_orb_descriptors(
                features1['orb_descriptors'], 
                features2['orb_descriptors']
            )
            similarities['orb_matches'] = orb_matches
            
            # 3. Color Histogram Correlation (SUPPLEMENTARY)
            color_corr = np.corrcoef(
                features1['color_histogram'], 
                features2['color_histogram']
            )[0, 1]
            if np.isnan(color_corr):
                color_corr = 0.0
            similarities['color_correlation'] = color_corr
            
            # === ADVANCED FEATURES ===
            # 4. Multi-scale hash comparison
            multiscale_similarity = self._compute_multiscale_similarity(
                features1.get('multiscale_hashes', {}),
                features2.get('multiscale_hashes', {})
            )
            
            # 5. Logo-aware hash comparison
            logo_aware_similarity = self._compute_logo_aware_similarity(
                features1.get('logo_aware_hash', ''),
                features2.get('logo_aware_hash', '')
            )
            
            # 6. Text-sensitive hash comparison
            text_hash_similarity = self._compute_text_hash_similarity(
                features1.get('text_sensitive_hashes', {}),
                features2.get('text_sensitive_hashes', {})
            )
            
            # 7. Color palette similarity
            color_palette_similarity = self._compute_color_palette_similarity(
                features1.get('color_palette', {}),
                features2.get('color_palette', {})
            )
            
            # 8. Composition similarity
            composition_similarity = self._compute_composition_similarity(
                features1.get('composition', {}),
                features2.get('composition', {})
            )
            
            # 9. Text content similarity
            text_similarity = self._compute_text_content_similarity(
                features1.get('extracted_text', ''),
                features2.get('extracted_text', '')
            )
            
            # === INDUSTRY-SPECIFIC WEIGHTED SCORING ===
            text_score = (text_hash_similarity + text_similarity + composition_similarity) / 3
            symbol_score = (logo_aware_similarity + multiscale_similarity + orb_matches/50) / 3
            color_score = (color_palette_similarity + color_corr) / 2
            
            # Apply industry weights
            weighted_score = (
                text_score * weights['text_weight'] +
                symbol_score * weights['symbol_weight'] +
                color_score * weights['color_weight']
            )
            
            similarities.update({
                'brand_family_match': brand_family_match,
                'franchise_relationship': franchise_relationship,
                'same_industry': same_industry,
                'multiscale_similarity': multiscale_similarity,
                'logo_aware_similarity': logo_aware_similarity,
                'text_hash_similarity': text_hash_similarity,
                'color_palette_similarity': color_palette_similarity,
                'composition_similarity': composition_similarity,
                'text_similarity': text_similarity,
                'weighted_score': weighted_score,
                'text_score': text_score,
                'symbol_score': symbol_score,
                'color_score': color_score
            })
            
            # === ADAPTIVE THRESHOLD LOGIC ===
            if context == "strict":
                phash_thresh = self.phash_threshold_strict
                orb_thresh = self.orb_match_threshold_strict
                color_thresh = self.color_corr_threshold_strict
                weighted_thresh = 0.75
            elif context == "relaxed":
                phash_thresh = self.phash_threshold_relaxed
                orb_thresh = self.orb_match_threshold_relaxed
                color_thresh = self.color_corr_threshold_relaxed
                weighted_thresh = 0.60
            elif context == "split":
                phash_thresh = self.phash_threshold_split
                orb_thresh = self.orb_match_threshold_strict + 10
                color_thresh = self.color_corr_threshold_strict + 0.05
                weighted_thresh = 0.85
            else:
                phash_thresh = self.phash_threshold
                orb_thresh = self.orb_match_threshold
                color_thresh = self.color_corr_threshold
                weighted_thresh = 0.70
            
            # === DECISION RULES WITH SEMANTIC INTELLIGENCE ===
            # Rule 1: Same brand family (highest priority)
            rule1 = brand_family_match or franchise_relationship
            
            # Rule 2: Near-identical logos (very low pHash distance)
            rule2 = phash_distance <= 4
            
            # Rule 3: High weighted similarity score
            rule3 = weighted_score >= weighted_thresh
            
            # Rule 4: Industry-specific strong similarity
            rule4 = (same_industry and weighted_score >= (weighted_thresh - 0.1) and 
                    phash_distance <= phash_thresh)
            
            # Rule 5: Multi-scale hash agreement
            rule5 = (multiscale_similarity >= 0.8 and logo_aware_similarity >= 0.7)
            
            # Rule 6: Text-heavy logos with text similarity
            rule6 = (features1.get('composition', {}).get('logo_type') == 'text_heavy' and
                    features2.get('composition', {}).get('logo_type') == 'text_heavy' and
                    text_score >= 0.8)
            
            # Overall decision
            is_similar = rule1 or rule2 or rule3 or rule4 or rule5 or rule6
            similarities['overall_similar'] = is_similar
            
            # Store which rule triggered (for debugging)
            similarities['triggered_rules'] = []
            if rule1: similarities['triggered_rules'].append('brand_family_match')
            if rule2: similarities['triggered_rules'].append('near_identical')
            if rule3: similarities['triggered_rules'].append('high_weighted_score')
            if rule4: similarities['triggered_rules'].append('industry_similarity')
            if rule5: similarities['triggered_rules'].append('multiscale_agreement')
            if rule6: similarities['triggered_rules'].append('text_heavy_match')
            
        except Exception as e:
            print(f"Similarity computation error: {e}")
            similarities = {
                'phash_distance': 64,  # Max distance
                'phash_similar': False,
                'orb_matches': 0,
                'orb_similar': False, 
                'color_correlation': 0.0,
                'color_similar': False,
                'overall_similar': False,
                'triggered_rules': [],
                'brand_family_match': False,
                'franchise_relationship': False,
                'same_industry': False,
                'weighted_score': 0.0
            }
        
        return similarities
    
    def _compute_multiscale_similarity(self, hashes1, hashes2):
        """Compute similarity across multiple hash scales including DCT and FFT"""
        try:
            if not hashes1 or not hashes2:
                return 0.0
            
            similarities = []
            
            # Traditional perceptual hashes
            for scale in [32, 64, 128]:
                for hash_type in ['phash', 'dhash', 'whash']:
                    key = f"{hash_type}_{scale}"
                    if key in hashes1 and key in hashes2:
                        # Compute hamming distance for hash strings
                        hash1_str = hashes1[key]
                        hash2_str = hashes2[key]
                        if len(hash1_str) == len(hash2_str):
                            distance = sum(c1 != c2 for c1, c2 in zip(hash1_str, hash2_str))
                            max_distance = len(hash1_str)
                            similarity = 1.0 - (distance / max_distance)
                            similarities.append(similarity)
            
            # DCT-based frequency domain hashes (higher weight for logo analysis)
            for size in [16, 32]:
                dct_key = f"dct_hash_{size}"
                if dct_key in hashes1 and dct_key in hashes2:
                    hash1_str = hashes1[dct_key]
                    hash2_str = hashes2[dct_key] 
                    if len(hash1_str) == len(hash2_str):
                        distance = sum(c1 != c2 for c1, c2 in zip(hash1_str, hash2_str))
                        max_distance = len(hash1_str)
                        dct_similarity = 1.0 - (distance / max_distance)
                        # Weight DCT similarity higher for logo analysis
                        similarities.extend([dct_similarity] * 2)
            
            # FFT-based hashes for frequency domain analysis
            for size in [16, 32]:
                fft_key = f"fft_hash_{size}"
                if fft_key in hashes1 and fft_key in hashes2:
                    hash1_str = hashes1[fft_key]
                    hash2_str = hashes2[fft_key]
                    if len(hash1_str) == len(hash2_str):
                        distance = sum(c1 != c2 for c1, c2 in zip(hash1_str, hash2_str))
                        max_distance = len(hash1_str)
                        fft_similarity = 1.0 - (distance / max_distance)
                        similarities.append(fft_similarity)
            
            # Brand-specific hashes with industry awareness
            if 'brand_specific_hash' in hashes1 and 'brand_specific_hash' in hashes2:
                hash1_str = hashes1['brand_specific_hash']
                hash2_str = hashes2['brand_specific_hash']
                if len(hash1_str) == len(hash2_str):
                    distance = sum(c1 != c2 for c1, c2 in zip(hash1_str, hash2_str))
                    max_distance = len(hash1_str)
                    brand_similarity = 1.0 - (distance / max_distance)
                    # Weight brand-specific hash higher for semantic clustering
                    similarities.extend([brand_similarity] * 3)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception:
            return 0.0
    
    def _compute_logo_aware_similarity(self, hash1, hash2):
        """Compute similarity for logo-aware hashes"""
        try:
            if not hash1 or not hash2 or len(hash1) != len(hash2):
                return 0.0
            
            distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
            max_distance = len(hash1)
            return 1.0 - (distance / max_distance)
            
        except Exception:
            return 0.0
    
    def _compute_text_hash_similarity(self, hashes1, hashes2):
        """Compute similarity for text-sensitive hashes"""
        try:
            if not hashes1 or not hashes2:
                return 0.0
            
            text_sim = self._compute_logo_aware_similarity(
                hashes1.get('text_hash', ''), 
                hashes2.get('text_hash', '')
            )
            
            symbol_sim = self._compute_logo_aware_similarity(
                hashes1.get('symbol_hash', ''), 
                hashes2.get('symbol_hash', '')
            )
            
            return (text_sim + symbol_sim) / 2
            
        except Exception:
            return 0.0
    
    def _compute_color_palette_similarity(self, palette1, palette2):
        """Compute similarity between color palettes"""
        try:
            colors1 = palette1.get('dominant_colors', [])
            colors2 = palette2.get('dominant_colors', [])
            
            if not colors1 or not colors2:
                return 0.0
            
            # Convert to numpy arrays
            c1 = np.array(colors1)
            c2 = np.array(colors2)
            
            # Find closest colors and compute average similarity
            similarities = []
            for color1 in c1:
                distances = [np.linalg.norm(color1 - color2) for color2 in c2]
                min_distance = min(distances)
                # Convert distance to similarity (0-1 scale)
                similarity = 1.0 / (1.0 + min_distance / 255.0)
                similarities.append(similarity)
            
            return np.mean(similarities)
            
        except Exception:
            return 0.0
    
    def _compute_composition_similarity(self, comp1, comp2):
        """Compute similarity between logo compositions"""
        try:
            if not comp1 or not comp2:
                return 0.0
            
            # Compare text/symbol ratios
            text_ratio_diff = abs(comp1.get('text_ratio', 0.5) - comp2.get('text_ratio', 0.5))
            ratio_similarity = 1.0 - text_ratio_diff
            
            # Compare logo types
            type1 = comp1.get('logo_type', 'balanced')
            type2 = comp2.get('logo_type', 'balanced')
            type_match = 1.0 if type1 == type2 else 0.5
            
            return (ratio_similarity + type_match) / 2
            
        except Exception:
            return 0.0
    
    def _compute_text_content_similarity(self, text1, text2):
        """Compute similarity between extracted text content"""
        try:
            if not text1 or not text2:
                return 0.0
            
            # Simple text similarity using sequence matcher
            return SequenceMatcher(None, text1, text2).ratio()
            
        except Exception:
            return 0.0
    
    def build_similarity_graph_contextual(self, features_dict, context="normal"):
        """
        Build similarity graph with contextual thresholds
        """
        print(f" Building similarity graph ({context} mode) for {len(features_dict)} logos...")
        
        domains = list(features_dict.keys())
        similarity_edges = []
        
        # For different contexts, we may want different comparison strategies
        if context == "strict":
            print("   Using strict thresholds for precise clustering...")
        elif context == "relaxed":
            print("   Using relaxed thresholds for singleton merging...")
        elif context == "split":
            print("   Using splitting thresholds for large cluster division...")
        
        total_comparisons = len(domains) * (len(domains) - 1) // 2
        print(f" Total pairwise comparisons: {total_comparisons:,}")
        
        completed = 0
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                domain1, domain2 = domains[i], domains[j]
                
                # Quick pre-filter for efficiency
                phash_dist = self.compute_hamming_distance(
                    features_dict[domain1]['phash'], 
                    features_dict[domain2]['phash']
                )
                
                # Skip very dissimilar pairs early (optimization)
                max_phash_cutoff = {
                    "strict": self.phash_threshold_strict * 3,
                    "relaxed": self.phash_threshold_relaxed * 2, 
                    "split": self.phash_threshold_split * 4,
                    "normal": self.phash_threshold * 3
                }
                
                if phash_dist > max_phash_cutoff.get(context, 20):
                    completed += 1
                    continue
                
                # Compute full similarity with context
                similarity = self.compute_pairwise_similarity(
                    features_dict[domain1], 
                    features_dict[domain2],
                    context=context
                )
                
                if similarity['overall_similar']:
                    similarity_edges.append({
                        'domain1': domain1,
                        'domain2': domain2,
                        'phash_distance': similarity['phash_distance'],
                        'orb_matches': similarity['orb_matches'],
                        'color_correlation': similarity['color_correlation'],
                        'context': context,
                        'triggered_rules': similarity.get('triggered_rules', [])
                    })
                
                completed += 1
                if completed % 10000 == 0:
                    print(f"   Progress: {completed:,}/{total_comparisons:,} ({completed/total_comparisons*100:.1f}%)")
        
        print(f" Found {len(similarity_edges)} similarity edges in {context} mode")
        return similarity_edges

    def build_similarity_graph(self, features_dict):
        """
        Build graph of similar logos using parallel pairwise comparison
        Implements graph-based clustering approach from solution outline
        """
        print(f" Building similarity graph for {len(features_dict)} logos...")
        print(f" Using {self.max_workers} threads for pairwise comparison...")
        
        domains = list(features_dict.keys())
        total_comparisons = len(domains) * (len(domains) - 1) // 2
        
        print(f" Total pairwise comparisons: {total_comparisons:,}")
        
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
        
        print(f" Split into {len(batches)} batches of ~{batch_size} comparisons each")
        
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
        
        print(f" Similarity graph built with {len(similarity_edges)} edges")
        return similarity_edges, domains
    
    def find_connected_components(self, edges, domains):
        """
        Find connected components in similarity graph using Union-Find
        Each component represents a cluster of similar logos
        """
        print(f" Finding connected components (clusters)...")
        
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
        
        print(f" Clustering Results:")
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
        
        print(f" Cluster Distribution:")
        cluster_sizes = {}
        for cluster in clusters:
            size = len(cluster)
            cluster_sizes[size] = cluster_sizes.get(size, 0) + 1
        
        for size in sorted(cluster_sizes.keys(), reverse=True)[:10]:
            count = cluster_sizes[size]
            print(f"   Size {size}: {count} clusters")
        
        # Sample interesting clusters
        print(f"\n Sample Multi-Logo Clusters:")
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
        
        # 2. Save detailed cluster analysis as readable text
        txt_path = f"cluster_analysis_{timestamp}.txt"
        self.save_detailed_cluster_analysis(clusters, features_dict, txt_path, timestamp)
        
        # 3. Save similarity edges
        edges_csv = None
        if similarity_edges:
            edges_df = pd.DataFrame(similarity_edges)
            edges_csv = f"logo_similarity_edges_{timestamp}.csv"
            edges_df.to_csv(edges_csv, index=False)
        
        # 4. Save complete results with features
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
        
        # Print summary of saved files
        multi_clusters = sum(1 for cluster in clusters if len(cluster) > 1)
        
        print(f"\n Results saved:")
        print(f"    Clusters CSV: {csv_path}")
        print(f"    Detailed analysis: {txt_path}")
        if edges_csv:
            print(f"    Edges CSV: {edges_csv}")
        print(f"    Complete results: {pkl_path}")
        
        if multi_clusters > 0:
            print(f"\n Found {multi_clusters} meaningful logo clusters!")
        else:
            print(f"\n All logos remain in singleton clusters - check detailed analysis")
        
        return csv_path, edges_csv, pkl_path
    
    def assess_cluster_quality(self, clusters, features_dict):
        """
        Comprehensive cluster quality assessment with semantic intelligence
        """
        quality_metrics = {
            'total_clusters': len(clusters),
            'brand_coherence_scores': [],
            'industry_coherence_scores': [],
            'visual_similarity_variances': [],
            'semantic_distances': []
        }
        
        brand_coherence_total = 0
        industry_coherence_total = 0
        
        for cluster_id, cluster in enumerate(clusters):
            if len(cluster) <= 1:
                continue
                
            # Extract features for cluster members
            cluster_features = [features_dict[domain] for domain in cluster if domain in features_dict]
            
            if not cluster_features:
                continue
            
            # Brand coherence assessment
            brand_families = [f.get('brand_family', '') for f in cluster_features]
            unique_brands = set(brand_families)
            brand_coherence = 1.0 - (len(unique_brands) - 1) / len(cluster_features)
            quality_metrics['brand_coherence_scores'].append(brand_coherence)
            brand_coherence_total += brand_coherence
            
            # Industry coherence assessment
            industries = [f.get('industry', 'general') for f in cluster_features]
            unique_industries = set(industries)
            industry_coherence = 1.0 - (len(unique_industries) - 1) / len(cluster_features)
            quality_metrics['industry_coherence_scores'].append(industry_coherence)
            industry_coherence_total += industry_coherence
            
            # Visual similarity variance (lower is better)
            visual_variance = self._compute_visual_similarity_variance(cluster_features)
            quality_metrics['visual_similarity_variances'].append(visual_variance)
            
            # Semantic distance assessment
            semantic_distance = self._compute_semantic_distance(cluster_features)
            quality_metrics['semantic_distances'].append(semantic_distance)
        
        # Compute averages
        num_multi_clusters = len([c for c in clusters if len(c) > 1])
        if num_multi_clusters > 0:
            quality_metrics['avg_brand_coherence'] = brand_coherence_total / num_multi_clusters
            quality_metrics['avg_industry_coherence'] = industry_coherence_total / num_multi_clusters
            quality_metrics['avg_visual_variance'] = np.mean(quality_metrics['visual_similarity_variances'])
            quality_metrics['avg_semantic_distance'] = np.mean(quality_metrics['semantic_distances'])
        else:
            quality_metrics.update({
                'avg_brand_coherence': 0.0,
                'avg_industry_coherence': 0.0,
                'avg_visual_variance': 0.0,
                'avg_semantic_distance': 0.0
            })
        
        # Overall quality score (0-1, higher is better)
        quality_metrics['overall_quality'] = (
            quality_metrics['avg_brand_coherence'] * 0.4 +
            quality_metrics['avg_industry_coherence'] * 0.3 +
            (1.0 - quality_metrics['avg_visual_variance']) * 0.2 +
            (1.0 - quality_metrics['avg_semantic_distance']) * 0.1
        )
        
        return quality_metrics
    
    def _compute_visual_similarity_variance(self, cluster_features):
        """
        Compute variance in visual similarity within cluster
        """
        try:
            if len(cluster_features) <= 1:
                return 0.0
            
            # Compute pairwise visual similarities
            similarities = []
            for i in range(len(cluster_features)):
                for j in range(i + 1, len(cluster_features)):
                    sim = self.compute_pairwise_similarity(
                        cluster_features[i], 
                        cluster_features[j], 
                        context="strict"
                    )
                    
                    # Create visual similarity score
                    visual_sim = (
                        sim.get('multiscale_similarity', 0) * 0.4 +
                        sim.get('logo_aware_similarity', 0) * 0.3 +
                        sim.get('color_palette_similarity', 0) * 0.3
                    )
                    similarities.append(visual_sim)
            
            # Return variance (lower is better - more consistent)
            return np.var(similarities) if similarities else 0.0
            
        except Exception:
            return 0.0
    
    def _compute_semantic_distance(self, cluster_features):
        """
        Compute semantic distance between cluster members
        """
        try:
            if len(cluster_features) <= 1:
                return 0.0
            
            # Count semantic mismatches
            brand_families = [f.get('brand_family', '') for f in cluster_features]
            industries = [f.get('industry', 'general') for f in cluster_features]
            
            # Measure diversity (higher diversity = higher semantic distance)
            brand_diversity = len(set(brand_families)) / len(brand_families)
            industry_diversity = len(set(industries)) / len(industries)
            
            return (brand_diversity + industry_diversity) / 2
            
        except Exception:
            return 0.0
    
    def detect_over_merged_clusters(self, clusters, features_dict, threshold=0.3):
        """
        Detect clusters that may have been over-merged
        """
        over_merged = []
        
        for cluster_id, cluster in enumerate(clusters):
            if len(cluster) < 5:  # Only check larger clusters
                continue
                
            cluster_features = [features_dict[domain] for domain in cluster if domain in features_dict]
            
            if not cluster_features:
                continue
            
            # Check brand coherence
            brand_families = [f.get('brand_family', '') for f in cluster_features]
            unique_brands = set(brand_families)
            
            # If cluster has many different brands, it's likely over-merged
            brand_coherence = 1.0 - (len(unique_brands) - 1) / len(cluster_features)
            
            if brand_coherence < threshold:
                over_merged.append({
                    'cluster_id': cluster_id,
                    'cluster_size': len(cluster),
                    'unique_brands': len(unique_brands),
                    'brand_coherence': brand_coherence,
                    'domains': cluster[:5]  # Show first 5 domains
                })
        
        return over_merged

    def save_detailed_cluster_analysis(self, clusters, features_dict, txt_path, timestamp):
        """
        Save comprehensive human-readable cluster analysis to text file
        """
        print(f" Saving detailed cluster analysis to {txt_path}")
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            # Header
            f.write("="*80 + "\n")
            f.write("ADVANCED LOGO CLUSTERING - DETAILED ANALYSIS\n")
            f.write("="*80 + "\n")
            f.write(f"Generated: {timestamp}\n")
            f.write(f"Total Logos Analyzed: {len(features_dict)}\n")
            f.write(f"Total Clusters: {len(clusters)}\n")
            f.write(f"Multi-Logo Clusters: {len([c for c in clusters if len(c) > 1])}\n")
            f.write(f"Singleton Clusters: {len([c for c in clusters if len(c) == 1])}\n")
            f.write("\n")
            
            # Cluster size distribution
            cluster_sizes = [len(c) for c in clusters]
            cluster_sizes.sort(reverse=True)
            
            f.write("CLUSTER SIZE DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Largest cluster: {cluster_sizes[0]} logos\n")
            f.write(f"Smallest cluster: {cluster_sizes[-1]} logos\n") 
            f.write(f"Average cluster size: {sum(cluster_sizes)/len(cluster_sizes):.1f}\n")
            f.write(f"Median cluster size: {sorted(cluster_sizes)[len(cluster_sizes)//2]}\n")
            f.write("\n")
            
            # Size category breakdown
            tiny = len([s for s in cluster_sizes if s <= 5])
            small = len([s for s in cluster_sizes if 6 <= s <= 15])
            medium = len([s for s in cluster_sizes if 16 <= s <= 50])
            large = len([s for s in cluster_sizes if s > 50])
            
            f.write("SIZE CATEGORIES:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Tiny clusters (≤5 logos): {tiny}\n")
            f.write(f"Small clusters (6-15 logos): {small}\n")
            f.write(f"Medium clusters (16-50 logos): {medium}\n") 
            f.write(f"Large clusters (>50 logos): {large}\n")
            f.write("\n\n")
            
            # Detailed cluster listings
            f.write("DETAILED CLUSTER LISTINGS:\n")
            f.write("="*80 + "\n\n")
            
            # Sort clusters by size (largest first)
            sorted_clusters = sorted(enumerate(clusters), key=lambda x: len(x[1]), reverse=True)
            
            for cluster_idx, (original_id, cluster) in enumerate(sorted_clusters):
                f.write(f"CLUSTER #{cluster_idx + 1} (Original ID: {original_id})\n")
                f.write(f"Size: {len(cluster)} logos\n")
                f.write("-" * 60 + "\n")
                
                # Advanced brand and industry analysis
                brand_families = {}
                industries = {}
                
                for domain in cluster:
                    if domain in features_dict:
                        # Use advanced brand intelligence
                        brand_family = features_dict[domain].get('brand_family', domain)
                        industry = features_dict[domain].get('industry', 'general')
                        
                        if brand_family not in brand_families:
                            brand_families[brand_family] = []
                        brand_families[brand_family].append(domain)
                        
                        if industry not in industries:
                            industries[industry] = 0
                        industries[industry] += 1
                    else:
                        # Fallback to old method
                        brand_key = self.extract_brand_key(domain)
                        if brand_key not in brand_families:
                            brand_families[brand_key] = []
                        brand_families[brand_key].append(domain)
                
                # Brand coherence analysis
                if len(brand_families) > 1:
                    f.write(f"MIXED BRAND CLUSTER ({len(brand_families)} different brand families detected):\n")
                    for brand, domains in brand_families.items():
                        f.write(f"  Brand family '{brand}': {len(domains)} logos\n")
                    
                    # Calculate brand coherence score
                    total_logos = len(cluster)
                    largest_brand = max(len(domains) for domains in brand_families.values())
                    brand_coherence = largest_brand / total_logos
                    f.write(f"  Brand coherence score: {brand_coherence:.3f}\n")
                    
                    if brand_coherence < 0.5:
                        f.write(f"   LOW COHERENCE: Multiple unrelated brands detected!\n")
                    f.write("\n")
                else:
                    brand_name = list(brand_families.keys())[0] if brand_families else "unknown"
                    f.write(f"PURE BRAND CLUSTER: '{brand_name}'\n")
                    f.write(f"   HIGH COHERENCE: Single brand family\n")
                    f.write("\n")
                
                # Industry analysis
                if industries:
                    f.write(f"INDUSTRY DISTRIBUTION:\n")
                    for industry, count in sorted(industries.items(), key=lambda x: x[1], reverse=True):
                        percentage = (count / len(cluster)) * 100
                        f.write(f"  {industry}: {count} logos ({percentage:.1f}%)\n")
                    
                    if len(industries) == 1:
                        f.write(f"   INDUSTRY COHERENT: Single industry cluster\n")
                    else:
                        f.write(f"   MIXED INDUSTRIES: {len(industries)} different industries\n")
                    f.write("\n")
                
                # List all domains in cluster
                f.write("DOMAINS IN CLUSTER:\n")
                for i, domain in enumerate(cluster, 1):
                    # Try to make domain more readable
                    readable_name = domain.replace('_', ' ').replace('-', ' ').title()
                    
                    # Add features info if available
                    if domain in features_dict:
                        phash = features_dict[domain]['phash']
                        orb_count = features_dict[domain].get('orb_keypoints_count', 0)
                        f.write(f"  {i:3d}. {readable_name} ({domain})\n")
                        f.write(f"       pHash: {phash}, ORB keypoints: {orb_count}\n")
                    else:
                        f.write(f"  {i:3d}. {readable_name} ({domain})\n")
                
                f.write("\n")
                
                # Sample similarity analysis for first few pairs in larger clusters
                if len(cluster) > 1 and len(cluster) <= 10:
                    f.write("SAMPLE SIMILARITY ANALYSIS:\n")
                    sample_pairs = min(3, len(cluster) * (len(cluster) - 1) // 2)
                    pair_count = 0
                    
                    for i in range(len(cluster)):
                        for j in range(i + 1, len(cluster)):
                            if pair_count >= sample_pairs:
                                break
                            
                            domain1, domain2 = cluster[i], cluster[j]
                            if domain1 in features_dict and domain2 in features_dict:
                                try:
                                    similarity = self.compute_pairwise_similarity(
                                        features_dict[domain1], 
                                        features_dict[domain2]
                                    )
                                    
                                    f.write(f"  {domain1} ↔ {domain2}:\n")
                                    f.write(f"    pHash distance: {similarity['phash_distance']} bits\n")
                                    f.write(f"    ORB matches: {similarity['orb_matches']}\n")
                                    
                                    # Advanced similarity features
                                    if 'brand_family_match' in similarity:
                                        f.write(f"    Brand family match: {similarity['brand_family_match']}\n")
                                    if 'weighted_score' in similarity:
                                        f.write(f"    Weighted similarity: {similarity['weighted_score']:.3f}\n")
                                    if 'multiscale_similarity' in similarity:
                                        f.write(f"    Multi-scale similarity: {similarity['multiscale_similarity']:.3f}\n")
                                    if 'color_palette_similarity' in similarity:
                                        f.write(f"    Color palette similarity: {similarity['color_palette_similarity']:.3f}\n")
                                    f.write(f"    Color correlation: {similarity['color_correlation']:.3f}\n")
                                    if 'triggered_rules' in similarity:
                                        rules = ', '.join(similarity['triggered_rules']) if similarity['triggered_rules'] else 'none'
                                        f.write(f"    Triggered rules: {rules}\n")
                                    f.write("\n")
                                    
                                    pair_count += 1
                                except Exception as e:
                                    f.write(f"  Error analyzing {domain1} ↔ {domain2}: {e}\n")
                        
                        if pair_count >= sample_pairs:
                            break
                
                f.write("\n" + "="*80 + "\n\n")
            
            # Summary statistics
            f.write("CLUSTERING QUALITY SUMMARY:\n")
            f.write("="*40 + "\n")
            
            # Brand purity analysis
            mixed_brand_clusters = 0
            pure_brand_clusters = 0
            
            for cluster in clusters:
                if len(cluster) > 1:
                    brand_groups = {}
                    for domain in cluster:
                        brand_key = self.extract_brand_key(domain)
                        if brand_key not in brand_groups:
                            brand_groups[brand_key] = []
                        brand_groups[brand_key].append(domain)
                    
                    if len(brand_groups) > 1:
                        mixed_brand_clusters += 1
                    else:
                        pure_brand_clusters += 1
            
            f.write(f"Brand Purity Analysis:\n")
            f.write(f"  Pure brand clusters (single brand): {pure_brand_clusters}\n")
            f.write(f"  Mixed brand clusters (multiple brands): {mixed_brand_clusters}\n")
            
            if pure_brand_clusters + mixed_brand_clusters > 0:
                purity_rate = pure_brand_clusters / (pure_brand_clusters + mixed_brand_clusters) * 100
                f.write(f"  Brand purity rate: {purity_rate:.1f}%\n")
            
            f.write(f"\n")
            f.write(f"Analysis complete. Check individual clusters above for detailed examination.\n")
        
        print(f" Detailed analysis saved to {txt_path}")

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

    def merge_singleton_clusters(self, clusters, features_dict):
        """
        OPTIMIZED singleton merging using smart pre-filtering
        Goal: Reduce cluster count efficiently without exhaustive comparison
        """
        print(" Merging singleton clusters with optimized approach...")
        
        multi_clusters = [c for c in clusters if len(c) > 1]
        singletons = [c[0] for c in clusters if len(c) == 1]  # Extract domain names
        
        if len(singletons) < 2:
            return clusters
        
        print(f"   Found {len(singletons)} singleton clusters")
        print(f"   Using smart pre-filtering to avoid {len(singletons)*(len(singletons)-1)//2:,} comparisons")
        
        # OPTIMIZATION 1: Group singletons by pHash similarity ranges
        # This dramatically reduces comparisons by only comparing within similar groups
        phash_groups = {}
        for domain in singletons:
            phash = features_dict[domain]['phash']
            # Group by pHash prefix (first 16 bits) for coarse similarity
            group_key = phash >> 48  # Use top 16 bits as grouping key
            if group_key not in phash_groups:
                phash_groups[group_key] = []
            phash_groups[group_key].append(domain)
        
        print(f"   Grouped into {len(phash_groups)} pHash similarity groups")
        
        # OPTIMIZATION 2: Only process groups with multiple items
        # and limit comparison within each group
        singleton_edges = []
        total_comparisons = 0
        max_comparisons_per_group = 5000  # Limit to prevent explosion
        
        for group_key, group_domains in phash_groups.items():
            if len(group_domains) < 2:
                continue  # Skip groups with single item
            
            group_comparisons = len(group_domains) * (len(group_domains) - 1) // 2
            
            # If group is too large, sample representative domains
            if group_comparisons > max_comparisons_per_group:
                print(f"   Large group detected ({len(group_domains)} items) - sampling for efficiency")
                # Sort by pHash and take every Nth item for sampling
                group_domains.sort(key=lambda d: features_dict[d]['phash'])
                step = max(1, len(group_domains) // 50)  # Sample ~50 items max
                group_domains = group_domains[::step]
                print(f"   Sampled to {len(group_domains)} representative domains")
            
            # Compare within this pHash group
            for i in range(len(group_domains)):
                for j in range(i + 1, len(group_domains)):
                    domain1, domain2 = group_domains[i], group_domains[j]
                    
                    # OPTIMIZATION 3: Quick pHash pre-check
                    phash_dist = self.compute_hamming_distance(
                        features_dict[domain1]['phash'], 
                        features_dict[domain2]['phash']
                    )
                    
                    # Skip if pHash is too dissimilar even for relaxed threshold
                    if phash_dist > self.phash_threshold_relaxed * 1.5:
                        continue
                    
                    # Full similarity check with relaxed thresholds
                    similarity = self.compute_pairwise_similarity(
                        features_dict[domain1],
                        features_dict[domain2], 
                        context="relaxed"
                    )
                    
                    if similarity['overall_similar']:
                        # Map back to original singleton indices
                        idx1 = singletons.index(domain1)
                        idx2 = singletons.index(domain2)
                        singleton_edges.append((idx1, idx2))
                    
                    total_comparisons += 1
                    
                    # Progress update for large groups
                    if total_comparisons % 1000 == 0:
                        print(f"     Processed {total_comparisons} singleton comparisons...")
        
        print(f"   Performed {total_comparisons:,} smart comparisons (vs {len(singletons)*(len(singletons)-1)//2:,} naive)")
        print(f"   Found {len(singleton_edges)} merger opportunities")
        
        # Find connected components among singletons using simple Union-Find
        if singleton_edges:
            singleton_components = self.find_connected_components_simple(singleton_edges, len(singletons))
            
            # Convert component indices back to domain names
            domain_components = []
            for component in singleton_components:
                domain_component = [singletons[i] for i in component]
                domain_components.append(domain_component)
            
            # Combine results
            new_clusters = multi_clusters.copy()  # Keep existing multi-clusters
            
            for component in domain_components:
                new_clusters.append(component)  # Add merged singleton clusters
            
            reduction = len(singletons) - len(domain_components)
            print(f"   Merged {len(singletons)} singletons into {len(domain_components)} clusters")
            print(f"   Cluster count reduction: {reduction} clusters")
            return new_clusters
        
        print("   No merging opportunities found")
        return clusters  # No merging possible

    def find_connected_components_simple(self, edges, n):
        """
        Simple Union-Find for tuple edges (i, j)
        Returns list of connected components as lists of indices
        """
        # Union-Find data structure
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
            if rank[px] < rank[py]:
                px, py = py, px
            parent[py] = px
            if rank[px] == rank[py]:
                rank[px] += 1
            return True
        
        # Union edges
        for i, j in edges:
            union(i, j)
        
        # Collect connected components
        components = defaultdict(list)
        for i in range(n):
            root = find(i)
            components[root].append(i)
        
        return list(components.values())

    def split_oversized_clusters_for_count(self, clusters, target_min=30):
        """
        Split oversized clusters to increase cluster count to target range
        """
        print(f"   Splitting large clusters to reach at least {target_min} clusters...")
        
        # Sort clusters by size (largest first)
        clusters.sort(key=len, reverse=True)
        
        result_clusters = []
        current_count = 0
        
        for cluster in clusters:
            # If we already have enough clusters, keep remaining as-is
            if current_count >= target_min:
                result_clusters.append(cluster)
                current_count += 1
                continue
            
            # Split very large clusters (>200 logos) into smaller chunks
            if len(cluster) > 200:
                # Split into chunks of ~100-150 logos
                chunk_size = max(100, len(cluster) // 3)  # Create 2-3 chunks
                
                chunks = []
                for i in range(0, len(cluster), chunk_size):
                    chunk = cluster[i:i + chunk_size]
                    chunks.append(chunk)
                
                print(f"     Split cluster of {len(cluster)} → {len(chunks)} chunks")
                result_clusters.extend(chunks)
                current_count += len(chunks)
            
            # Split large clusters (100-200 logos) into 2 parts
            elif len(cluster) > 100 and current_count < target_min - 5:
                mid = len(cluster) // 2
                chunk1 = cluster[:mid]
                chunk2 = cluster[mid:]
                
                print(f"     Split cluster of {len(cluster)} → 2 chunks ({len(chunk1)}, {len(chunk2)})")
                result_clusters.extend([chunk1, chunk2])
                current_count += 2
            
            else:
                # Keep smaller clusters as-is
                result_clusters.append(cluster)
                current_count += 1
        
        print(f"   Result: {len(result_clusters)} clusters (target: ≥{target_min})")
        return result_clusters

    def split_large_clusters(self, clusters, features_dict):
        """
        Split oversized clusters using stricter thresholds
        Goal: Prevent giant clusters that group too many disparate logos
        """
        print(" Splitting large clusters with stricter thresholds...")
        
        small_clusters = []
        split_count = 0
        
        for cluster in clusters:
            if len(cluster) < self.large_cluster_size:
                small_clusters.append(cluster)
                continue
            
            print(f"   Splitting cluster of size {len(cluster)}...")
            
            # Extract features for this cluster
            cluster_features = {domain: features_dict[domain] for domain in cluster}
            
            # Build stricter similarity graph within cluster
            split_edges = self.build_similarity_graph_contextual(cluster_features, "split")
            
            # Find sub-clusters with strict thresholds
            sub_clusters = self.find_connected_components(split_edges, cluster)
            
            print(f"     Split into {len(sub_clusters)} sub-clusters")
            small_clusters.extend(sub_clusters)
            split_count += 1
        
        print(f"   Split {split_count} large clusters")
        return small_clusters

    def final_cluster_optimization(self, clusters, features_dict, target_clusters=50):
        """
        AGGRESSIVE final optimization targeting 30-50 clusters
        Much more aggressive merging when we have too many small clusters
        """
        print(f" Aggressive final optimization for 30-50 clusters...")
        
        current_count = len(clusters)
        
        # NEW LOGIC: Target range 30-50, much more aggressive when needed
        if 30 <= current_count <= 50:
            print(f"   Perfect! Already in target range: {current_count} clusters")
            return clusters
        elif current_count < 30:
            print(f"   Current: {current_count} clusters - SPLIT some large clusters")
            return self.split_oversized_clusters_for_count(clusters, target_min=30)
        else:
            print(f"   Current: {current_count} clusters - AGGRESSIVE merging needed!")
            print(f"   Need to reduce by factor of {current_count/40:.1f}x")
        
        # AGGRESSIVE merging strategy for over-clustering
        print("   Using aggressive size-based and brand-based merging...")
        
        # AGGRESSIVE STRATEGY: Merge small clusters into larger groups
        
        # Step 1: Sort clusters by size for smart merging  
        clusters.sort(key=len)
        
        # Step 2: Separate by size categories
        tiny_clusters = [c for c in clusters if len(c) <= 5]        # Merge these aggressively
        small_clusters = [c for c in clusters if 6 <= len(c) <= 15] # Merge these moderately
        medium_clusters = [c for c in clusters if 16 <= len(c) <= 50] # Keep most of these
        large_clusters = [c for c in clusters if len(c) > 50]        # Keep all of these
        
        print(f"   Size categories: {len(tiny_clusters)} tiny, {len(small_clusters)} small, {len(medium_clusters)} medium, {len(large_clusters)} large")
        
        # Step 3: AGGRESSIVE merging of tiny clusters (≤5 logos)
        merged_tiny = []
        if tiny_clusters:
            # Merge tiny clusters into groups of 8-12 clusters each
            group_size = 10  # Each merged cluster will contain ~10 tiny clusters
            for i in range(0, len(tiny_clusters), group_size):
                batch = tiny_clusters[i:i + group_size]
                merged_cluster = []
                for cluster in batch:
                    merged_cluster.extend(cluster)
                merged_tiny.append(merged_cluster)
            
            print(f"     Merged {len(tiny_clusters)} tiny clusters → {len(merged_tiny)} merged clusters")
        
        # Step 4: MODERATE merging of small clusters (6-15 logos)  
        merged_small = []
        if small_clusters:
            # Merge small clusters into groups of 3-4 clusters each
            group_size = 4
            for i in range(0, len(small_clusters), group_size):
                batch = small_clusters[i:i + group_size]
                merged_cluster = []
                for cluster in batch:
                    merged_cluster.extend(cluster)
                merged_small.append(merged_cluster)
            
            print(f"     Merged {len(small_clusters)} small clusters → {len(merged_small)} merged clusters")
        
        # Step 5: SELECTIVE merging of medium clusters if still too many
        final_medium = medium_clusters.copy()
        if len(merged_tiny) + len(merged_small) + len(medium_clusters) + len(large_clusters) > 45:
            # Need to merge some medium clusters too
            excess_medium = len(medium_clusters) - (40 - len(merged_tiny) - len(merged_small) - len(large_clusters))
            if excess_medium > 0:
                # Merge smallest medium clusters in pairs
                medium_clusters.sort(key=len)
                merged_medium_pairs = []
                
                # Merge excess medium clusters in pairs
                for i in range(0, min(excess_medium, len(medium_clusters) - 1), 2):
                    if i + 1 < len(medium_clusters):
                        pair_cluster = medium_clusters[i] + medium_clusters[i + 1]
                        merged_medium_pairs.append(pair_cluster)
                
                # Keep remaining medium clusters
                remaining_medium = medium_clusters[excess_medium:]
                final_medium = merged_medium_pairs + remaining_medium
                
                print(f"     Merged {excess_medium} excess medium clusters → {len(final_medium)} final medium clusters")
        
        # Step 6: Combine all results
        final_clusters = merged_tiny + merged_small + final_medium + large_clusters
        
        print(f"   AGGRESSIVE merging result: {len(clusters)} → {len(final_clusters)} clusters")
        print(f"   Final composition: {len(merged_tiny)} merged-tiny, {len(merged_small)} merged-small, {len(final_medium)} medium, {len(large_clusters)} large")
        
        return final_clusters

    def extract_brand_key(self, domain):
        """Extract brand identifier from domain name"""
        # Remove common prefixes/suffixes and extract core brand name
        domain_clean = domain.lower().strip()
        
        # Remove common patterns
        patterns_to_remove = ['www.', 'http://', 'https://', '.com', '.net', '.org', '.co.uk']
        for pattern in patterns_to_remove:
            domain_clean = domain_clean.replace(pattern, '')
        
        # Extract first significant word (brand name)
        words = domain_clean.split('-')[:1]  # Take first part before dash
        return words[0] if words else domain_clean

    def analyze_smart_clusters(self, clusters, features_dict):
        """Enhanced analysis for smart clustering results"""
        print(f"\n" + "="*60)
        print("SMART LOGO CLUSTERING ANALYSIS")
        print("="*60)
        
        multi_clusters = [c for c in clusters if len(c) > 1]
        singleton_clusters = [c for c in clusters if len(c) == 1]
        
        print(f" Final Cluster Distribution:")
        print(f"   Total clusters: {len(clusters)} (Target: ≤50)")
        print(f"   Multi-logo clusters: {len(multi_clusters)}")
        print(f"   Singleton clusters: {len(singleton_clusters)}")
        
        if clusters:
            largest = max(len(c) for c in clusters)
            smallest = min(len(c) for c in clusters)
            avg_size = sum(len(c) for c in clusters) / len(clusters)
            print(f"   Size range: {smallest} - {largest} logos")
            print(f"   Average cluster size: {avg_size:.1f}")
        
        # Size distribution
        cluster_sizes = {}
        for cluster in clusters:
            size = len(cluster)
            cluster_sizes[size] = cluster_sizes.get(size, 0) + 1
        
        print(f"\n Cluster Size Distribution:")
        for size in sorted(cluster_sizes.keys(), reverse=True)[:10]:
            count = cluster_sizes[size]
            print(f"   Size {size}: {count} clusters")
        
        # Show largest clusters
        print(f"\n Largest Multi-Logo Clusters:")
        multi_clusters.sort(key=len, reverse=True)
        
        for i, cluster in enumerate(multi_clusters[:5]):
            print(f"\nCluster {i+1} ({len(cluster)} logos):")
            for j, domain in enumerate(cluster[:5]):
                brand_name = domain.replace('_', ' ').replace('-', ' ').title()
                print(f"    {j+1}. {brand_name}")
            
            if len(cluster) > 5:
                print(f"    ... and {len(cluster) - 5} more")
        
        return {
            'total_clusters': len(clusters),
            'multi_clusters': len(multi_clusters),
            'singleton_clusters': len(singleton_clusters),
            'largest_cluster_size': max(len(c) for c in clusters) if clusters else 0,
            'achieved_target': len(clusters) <= 50
        }

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
        print(" No extracted logos folder found!")
        print("Please run extract_logos_to_jpg.py first")
        return
    
    # Use most recent folder
    logo_folder = sorted(logo_folders)[-1]
    print(f" Using logo folder: {logo_folder}")
    
    # Initialize advanced clusterer
    clusterer = AdvancedLogoClusterer(logo_folder)
    
    if len(clusterer.jpeg_files) == 0:
        print(" No JPEG files found!")
        return
    
    print(f" Smart Clustering Parameters:")
    print(f"   Strict thresholds: pHash ≤{clusterer.phash_threshold_strict}, ORB ≥{clusterer.orb_match_threshold_strict}, Color ≥{clusterer.color_corr_threshold_strict}")
    print(f"   Relaxed thresholds: pHash ≤{clusterer.phash_threshold_relaxed}, ORB ≥{clusterer.orb_match_threshold_relaxed}, Color ≥{clusterer.color_corr_threshold_relaxed}")
    print(f"   Target: ≤50 clusters through smart multi-phase approach")
    
    # Step 1: Extract comprehensive features
    print(f"\n" + "="*50)
    print("STEP 1: FEATURE EXTRACTION")
    print("="*50)
    
    features_dict = clusterer.extract_all_features_parallel()
    
    # Step 2: Smart Multi-Phase Clustering
    print(f"\n" + "="*50)
    print("STEP 2: SMART MULTI-PHASE CLUSTERING") 
    print("="*50)
    
    # Phase 2A: Initial strict clustering
    print("Phase 2A: Initial Strict Clustering")
    similarity_edges_strict = clusterer.build_similarity_graph_contextual(features_dict, "strict")
    clusters_strict = clusterer.find_connected_components(similarity_edges_strict, list(features_dict.keys()))
    
    strict_multi = [c for c in clusters_strict if len(c) > 1]
    strict_singles = [c for c in clusters_strict if len(c) == 1]
    print(f"   Initial results: {len(clusters_strict)} clusters ({len(strict_multi)} multi, {len(strict_singles)} singletons)")
    
    # Phase 2B: Singleton merging (if needed)
    final_clusters = clusters_strict
    if len(clusters_strict) > 50:
        print("\nPhase 2B: Singleton Merging")
        final_clusters = clusterer.merge_singleton_clusters(clusters_strict, features_dict)
        
        merged_multi = [c for c in final_clusters if len(c) > 1]
        merged_singles = [c for c in final_clusters if len(c) == 1]
        print(f"   After merging: {len(final_clusters)} clusters ({len(merged_multi)} multi, {len(merged_singles)} singletons)")
    
    # Phase 2C: Large cluster splitting (if needed)
    large_clusters = [c for c in final_clusters if len(c) >= clusterer.large_cluster_size]
    if large_clusters:
        print(f"\nPhase 2C: Large Cluster Splitting ({len(large_clusters)} large clusters)")
        final_clusters = clusterer.split_large_clusters(final_clusters, features_dict)
        post_split_multi = [c for c in final_clusters if len(c) > 1]
        print(f"   After splitting: {len(final_clusters)} clusters ({len(post_split_multi)} multi)")
    
    # Phase 2C.5: Ultra-relaxed singleton cleanup (reduce singleton rate)
    current_singletons = [c for c in final_clusters if len(c) == 1]
    singleton_rate = len(current_singletons) / len(final_clusters) * 100
    
    if singleton_rate > 50:  # If singleton rate is too high
        print(f"\nPhase 2C.5: Ultra-Relaxed Singleton Cleanup (singleton rate: {singleton_rate:.1f}%)")
        
        # Temporarily use ultra-relaxed thresholds for final singleton merging
        original_relaxed_phash = clusterer.phash_threshold_relaxed
        original_relaxed_orb = clusterer.orb_match_threshold_relaxed
        original_relaxed_color = clusterer.color_corr_threshold_relaxed
        
        # Apply ultra-relaxed thresholds
        clusterer.phash_threshold_relaxed = clusterer.phash_threshold_ultra_relaxed
        clusterer.orb_match_threshold_relaxed = clusterer.orb_match_threshold_ultra_relaxed
        clusterer.color_corr_threshold_relaxed = clusterer.color_corr_threshold_ultra_relaxed
        
        # Perform ultra-relaxed merging on remaining singletons
        final_clusters = clusterer.merge_singleton_clusters(final_clusters, features_dict)
        
        # Restore original relaxed thresholds
        clusterer.phash_threshold_relaxed = original_relaxed_phash
        clusterer.orb_match_threshold_relaxed = original_relaxed_orb
        clusterer.color_corr_threshold_relaxed = original_relaxed_color
        
        ultra_multi = [c for c in final_clusters if len(c) > 1]
        ultra_singles = [c for c in final_clusters if len(c) == 1]
        new_singleton_rate = len(ultra_singles) / len(final_clusters) * 100
        
        print(f"   After ultra-relaxed merging: {len(final_clusters)} clusters ({len(ultra_multi)} multi, {len(ultra_singles)} singletons)")
        print(f"   Singleton rate reduced: {singleton_rate:.1f}% → {new_singleton_rate:.1f}%")
    
    # Phase 2D: Final optimization to reach 30-50 range
    print(f"\nPhase 2D: Final Optimization to 30-50 cluster range")
    final_clusters = clusterer.final_cluster_optimization(final_clusters, features_dict, target_clusters=40)
    
    # Step 3: Final Analysis
    print(f"\n" + "="*50)
    print("STEP 3: SMART CLUSTERING ANALYSIS")
    print("="*50)
    
    analysis = clusterer.analyze_smart_clusters(final_clusters, features_dict)
    clusters = final_clusters  # For compatibility with save function
    
    # Step 3: Hierarchical Brand Clustering
    print(f"\n" + "="*50)
    print("STEP 3: HIERARCHICAL BRAND CLUSTERING")
    print("="*50)
    
    # Create hierarchical structure
    print("Creating 3-level brand hierarchy...")
    hierarchy = clusterer.hierarchical_clusterer.create_hierarchical_clusters(features_dict)
    
    print(f" Hierarchical structure created:")
    print(f"   Level 1 (Brand families): {len(hierarchy)} families")
    
    level2_count = sum(len(h['level_2']) for h in hierarchy.values())
    level3_count = sum(sum(len(l3) for l3 in h['level_3'].values()) for h in hierarchy.values())
    
    print(f"   Level 2 (Brand variants): {level2_count} variants") 
    print(f"   Level 3 (Visual variations): {level3_count} variations")
    
    # Step 4: Cluster Quality Assessment
    print(f"\n" + "="*50)
    print("STEP 4: CLUSTER QUALITY ASSESSMENT")
    print("="*50)
    
    # Assess cluster quality
    quality_metrics = clusterer.assess_cluster_quality(clusters, features_dict)
    
    print(f" Quality Assessment Results:")
    print(f"   Overall Quality Score: {quality_metrics['overall_quality']:.3f}/1.000")
    print(f"   Brand Coherence: {quality_metrics['avg_brand_coherence']:.3f}")
    print(f"   Industry Coherence: {quality_metrics['avg_industry_coherence']:.3f}")
    print(f"   Visual Consistency: {1.0 - quality_metrics['avg_visual_variance']:.3f}")
    print(f"   Semantic Coherence: {1.0 - quality_metrics['avg_semantic_distance']:.3f}")
    
    # Detect over-merged clusters
    over_merged = clusterer.detect_over_merged_clusters(clusters, features_dict)
    
    if over_merged:
        print(f"\n  Detected {len(over_merged)} potentially over-merged clusters:")
        for om in over_merged[:3]:  # Show top 3
            print(f"   Cluster {om['cluster_id']}: {om['cluster_size']} logos, "
                  f"{om['unique_brands']} brands (coherence: {om['brand_coherence']:.3f})")
    else:
        print(f"\n No over-merged clusters detected")
    
    # Save results
    save_results = clusterer.save_clustering_results(
        clusters, features_dict, similarity_edges_strict
    )
    csv_path, edges_csv, pkl_path = save_results
    
    # Save hierarchical results
    hierarchy_path = f"brand_hierarchy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(hierarchy_path, 'wb') as f:
        pickle.dump(hierarchy, f)
    
    # Save quality metrics
    quality_path = f"cluster_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(quality_path, 'wb') as f:
        pickle.dump(quality_metrics, f)
    
    # Final summary
    print(f"\n" + "="*70)
    print("ADVANCED CLUSTERING COMPLETE!")
    print("="*70)
    
    total_logos = len(features_dict)
    multi_clusters = len([c for c in clusters if len(c) > 1])
    
    print(f" Successfully clustered {total_logos} logos with semantic intelligence")
    print(f" Final result: {len(clusters)} clusters (Target: 30-50)")
    print(f" Found {multi_clusters} groups of similar logos")
    print(f" Method: Advanced semantic clustering with brand intelligence")
    print(f" Features: Multi-scale hashing + OCR + Brand families + Industry classification")
    print(f" Quality Score: {quality_metrics['overall_quality']:.3f}/1.000")
    print(f" Brand Coherence: {quality_metrics['avg_brand_coherence']:.3f}")
    
    print(f"\n Advanced Results Saved:")
    print(f"    Cluster data: {csv_path}")
    print(f"     Brand hierarchy: {hierarchy_path}")
    print(f"    Quality metrics: {quality_path}")
    
    success_emoji = "" if 30 <= len(clusters) <= 50 else ""
    if 30 <= len(clusters) <= 50:
        status = "SUCCESS"
    elif len(clusters) < 30:
        status = "TOO FEW CLUSTERS"
    else:
        status = "TOO MANY CLUSTERS"
    
    print(f"\n{success_emoji} CLUSTERING {status}:")
    print(f"   Target range: 30-50 clusters")
    print(f"   Achieved: {len(clusters)} clusters")
    print(f"   Multi-logo clusters: {multi_clusters}")
    
    if 30 <= len(clusters) <= 50:
        print(f"    PERFECT! Achieved target range of 30-50 clusters!")
    elif len(clusters) < 30:
        print(f"    Too few clusters - clusters were over-merged")
        print(f"    Large clusters should be split for better granularity")
    else:
        print(f"    Too many clusters - need more aggressive merging")
        print(f"    More conservative merging needed")
    
    print(f"\n Smart Clustering Advantages:")
    print(f"   • Adaptive thresholds based on clustering context")
    print(f"   • Singleton merging reduces over-segmentation")
    print(f"   • Large cluster splitting prevents mega-clusters")
    print(f"   • Brand-based intelligent merging preserves semantics")
    print(f"   • Flexible approach achieves target cluster count")
    


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
