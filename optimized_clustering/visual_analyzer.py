"""
Visual Analyzer Module

Provides optimized visual analysis focusing on core features.
Extracts color palettes and analyzes logo composition.
"""

import cv2
import numpy as np


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
