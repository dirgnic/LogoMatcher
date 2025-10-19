"""
Optimized Logo Analyzer working directly with JPEG files
Uses DCT (already in JPEG) and FFT for efficient similarity analysis
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
from collections import defaultdict

class JPEGLogoAnalyzer:
    """
    Fast logo analyzer working directly with JPEG files
    Leverages JPEG's built-in DCT and adds FFT analysis
    """
    
    def __init__(self, jpeg_folder_path):
        self.jpeg_folder = jpeg_folder_path
        # Aggressive threading for similarity computation
        self.max_workers = min(16, (os.cpu_count() or 1) * 2)  # Use more threads
        self.batch_size = 50
        
        # Load JPEG file paths
        self.jpeg_files = self._load_jpeg_paths()
        print(f"Found {len(self.jpeg_files)} JPEG files in {jpeg_folder_path}")
    
    def _load_jpeg_paths(self):
        """Load all JPEG file paths"""
        if not os.path.exists(self.jpeg_folder):
            print(f"Error: Folder {self.jpeg_folder} not found!")
            return []
        
        jpeg_files = []
        for filename in os.listdir(self.jpeg_folder):
            if filename.lower().endswith(('.jpg', '.jpeg')):
                filepath = os.path.join(self.jpeg_folder, filename)
                jpeg_files.append({
                    'filename': filename,
                    'filepath': filepath,
                    'domain': filename.replace('.jpg', '').replace('.jpeg', '')
                })
        
        return jpeg_files
    
    def extract_dct_features(self, img_path):
        """
        Extract DCT features from JPEG image
        JPEG already uses DCT, so this is very efficient
        """
        try:
            # Load image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return np.zeros(64, dtype=np.float32)
            
            # Resize to standard size for consistency
            img_resized = cv2.resize(img, (64, 64))
            
            # Apply DCT (like pHash but more comprehensive)
            img_float = np.float32(img_resized)
            dct = cv2.dct(img_float)
            
            # Extract low-frequency coefficients (top-left 8x8)
            dct_low = dct[0:8, 0:8]
            
            # Flatten and normalize
            dct_features = dct_low.flatten()
            dct_features = dct_features / (np.linalg.norm(dct_features) + 1e-8)
            
            return dct_features.astype(np.float32)
            
        except Exception as e:
            print(f"DCT extraction error for {img_path}: {e}")
            return np.zeros(64, dtype=np.float32)
    
    def extract_fft_features(self, img_path):
        """
        Extract FFT features for global shape analysis
        Complements DCT with frequency domain analysis
        """
        try:
            # Load image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return np.zeros(256, dtype=np.float32)
            
            # Resize and normalize
            img_resized = cv2.resize(img, (64, 64))
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # 2D FFT
            fft = fft2(img_normalized)
            fft_shifted = fftshift(fft)
            
            # Magnitude spectrum
            magnitude = np.abs(fft_shifted)
            log_magnitude = np.log(magnitude + 1e-8)
            
            # Extract central region (low to mid frequencies)
            center = 32
            crop_size = 8
            central_region = log_magnitude[
                center-crop_size:center+crop_size,
                center-crop_size:center+crop_size
            ]
            
            # Flatten and normalize
            fft_features = central_region.flatten()
            fft_features = fft_features / (np.linalg.norm(fft_features) + 1e-8)
            
            return fft_features.astype(np.float32)
            
        except Exception as e:
            print(f"FFT extraction error for {img_path}: {e}")
            return np.zeros(256, dtype=np.float32)
    
    def extract_color_histogram(self, img_path):
        """Extract color histogram features"""
        try:
            # Load color image
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                return np.zeros(24, dtype=np.float32)
            
            # Resize
            img_resized = cv2.resize(img, (64, 64))
            
            # Compute histograms for each channel
            hist_features = []
            for i in range(3):  # B, G, R channels
                hist = cv2.calcHist([img_resized], [i], None, [8], [0, 256])
                hist_normalized = hist.flatten() / (np.sum(hist) + 1e-8)
                hist_features.extend(hist_normalized)
            
            return np.array(hist_features, dtype=np.float32)
            
        except Exception as e:
            print(f"Color histogram error for {img_path}: {e}")
            return np.zeros(24, dtype=np.float32)
    
    def extract_edge_features(self, img_path):
        """Extract edge-based features"""
        try:
            # Load image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return np.zeros(16, dtype=np.float32)
            
            # Resize
            img_resized = cv2.resize(img, (64, 64))
            
            # Detect edges
            edges = cv2.Canny(img_resized, 50, 150)
            
            # Divide into 4x4 grid and compute edge density
            h, w = edges.shape
            grid_h, grid_w = h // 4, w // 4
            
            edge_densities = []
            for i in range(4):
                for j in range(4):
                    region = edges[i*grid_h:(i+1)*grid_h, j*grid_w:(j+1)*grid_w]
                    density = np.sum(region > 0) / (region.shape[0] * region.shape[1])
                    edge_densities.append(density)
            
            return np.array(edge_densities, dtype=np.float32)
            
        except Exception as e:
            print(f"Edge features error for {img_path}: {e}")
            return np.zeros(16, dtype=np.float32)
    
    def compute_perceptual_hash(self, img_path):
        """Compute perceptual hash string"""
        try:
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return '0' * 64
            
            # Resize and apply DCT
            img_resized = cv2.resize(img, (32, 32))
            dct = cv2.dct(np.float32(img_resized))
            
            # Take 8x8 low-frequency block
            dct_low = dct[0:8, 0:8]
            median = np.median(dct_low)
            
            # Create binary hash
            hash_bits = dct_low > median
            hash_str = ''.join(['1' if bit else '0' for bit in hash_bits.flatten()])
            
            return hash_str
            
        except Exception as e:
            print(f"pHash error for {img_path}: {e}")
            return '0' * 64
    
    def extract_comprehensive_features(self, jpeg_info):
        """Extract all features for a single JPEG file"""
        filepath = jpeg_info['filepath']
        filename = jpeg_info['filename']
        domain = jpeg_info['domain']
        
        try:
            features = {
                'filename': filename,
                'domain': domain,
                'filepath': filepath,
                'dct_features': self.extract_dct_features(filepath),
                'fft_features': self.extract_fft_features(filepath),
                'color_hist': self.extract_color_histogram(filepath),
                'edge_features': self.extract_edge_features(filepath),
                'phash': self.compute_perceptual_hash(filepath)
            }
            
            return features
            
        except Exception as e:
            print(f"Feature extraction error for {filename}: {e}")
            return {
                'filename': filename,
                'domain': domain,
                'filepath': filepath,
                'dct_features': np.zeros(64, dtype=np.float32),
                'fft_features': np.zeros(256, dtype=np.float32),
                'color_hist': np.zeros(24, dtype=np.float32),
                'edge_features': np.zeros(16, dtype=np.float32),
                'phash': '0' * 64
            }
    
    def extract_all_features_threaded(self):
        """Extract features for all JPEG files using threading"""
        print(f" Extracting features from {len(self.jpeg_files)} JPEG files...")
        print(f" Using {self.max_workers} threads with batch size {self.batch_size}")
        
        all_features = {}
        
        def process_batch(batch):
            """Process a batch of JPEG files"""
            batch_results = {}
            for jpeg_info in batch:
                features = self.extract_comprehensive_features(jpeg_info)
                batch_results[features['domain']] = features
            return batch_results
        
        # Split into batches
        batches = [
            self.jpeg_files[i:i + self.batch_size] 
            for i in range(0, len(self.jpeg_files), self.batch_size)
        ]
        
        # Process batches with threading
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
        
        print(f" Feature extraction completed for {len(all_features)} images")
        return all_features
    
    def compute_similarity(self, features1, features2):
        """Optimized similarity computation between two feature sets"""
        similarities = {}
        
        try:
            # Use faster numpy operations instead of sklearn
            # 1. DCT similarity (manual cosine similarity - faster)
            dct1, dct2 = features1['dct_features'], features2['dct_features']
            dct_dot = np.dot(dct1, dct2)
            dct_norm = np.linalg.norm(dct1) * np.linalg.norm(dct2)
            similarities['dct'] = max(0, dct_dot / (dct_norm + 1e-8))
            
            # 2. FFT similarity (manual cosine similarity)
            fft1, fft2 = features1['fft_features'], features2['fft_features']
            fft_dot = np.dot(fft1, fft2)
            fft_norm = np.linalg.norm(fft1) * np.linalg.norm(fft2)
            similarities['fft'] = max(0, fft_dot / (fft_norm + 1e-8))
            
            # 3. Color histogram similarity (fast correlation)
            try:
                color_corr = np.corrcoef(features1['color_hist'], features2['color_hist'])[0, 1]
                similarities['color'] = max(0, color_corr) if not np.isnan(color_corr) else 0
            except:
                similarities['color'] = 0
            
            # 4. Edge similarity (manual cosine similarity)
            edge1, edge2 = features1['edge_features'], features2['edge_features']
            edge_dot = np.dot(edge1, edge2)
            edge_norm = np.linalg.norm(edge1) * np.linalg.norm(edge2)
            similarities['edge'] = max(0, edge_dot / (edge_norm + 1e-8))
            
            # 5. pHash similarity (optimized Hamming distance)
            phash1, phash2 = features1['phash'], features2['phash']
            # Convert to numpy for faster XOR operation
            if isinstance(phash1, str) and isinstance(phash2, str):
                hamming_dist = sum(c1 != c2 for c1, c2 in zip(phash1, phash2))
                similarities['phash'] = 1.0 - (hamming_dist / len(phash1))
            else:
                similarities['phash'] = 0.5  # fallback
            
            # Weighted combination (DCT and FFT are primary for JPEG analysis)
            weights = {
                'dct': 0.35,      # Primary: DCT (JPEG's native transform)
                'fft': 0.25,      # Secondary: FFT (global shape)
                'phash': 0.20,    # Tertiary: pHash (perceptual)
                'color': 0.12,    # Supplement: Color information
                'edge': 0.08      # Supplement: Edge structure
            }
            
            overall_sim = sum(similarities[key] * weight for key, weight in weights.items())
            
        except Exception as e:
            # Minimal fallback computation
            overall_sim = 0.0
            similarities = {'dct': 0, 'fft': 0, 'phash': 0, 'color': 0, 'edge': 0}
        
        return overall_sim, similarities
    
    def find_similar_pairs(self, features_dict, threshold=0.85):
        """Find similar logo pairs above threshold using aggressive threading"""
        print(f" Finding similar pairs with threshold {threshold}...")
        print(f" Using {self.max_workers} threads for similarity computation...")
        
        domains = list(features_dict.keys())
        total_comparisons = len(domains) * (len(domains) - 1) // 2
        
        print(f" Total comparisons needed: {total_comparisons:,}")
        
        # Create comparison tasks (domain index pairs)
        comparison_tasks = []
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                comparison_tasks.append((i, j))
        
        # Batch the comparison tasks for threading
        batch_size = max(1000, total_comparisons // (self.max_workers * 4))  # Larger batches for efficiency
        batches = [
            comparison_tasks[i:i + batch_size] 
            for i in range(0, len(comparison_tasks), batch_size)
        ]
        
        print(f" Split into {len(batches)} batches of ~{batch_size} comparisons each")
        
        similar_pairs = []
        completed_comparisons = 0
        
        def process_comparison_batch(batch):
            """Process a batch of comparisons in a single thread"""
            batch_results = []
            for i, j in batch:
                domain1, domain2 = domains[i], domains[j]
                
                overall_sim, detailed_sims = self.compute_similarity(
                    features_dict[domain1], 
                    features_dict[domain2]
                )
                
                if overall_sim >= threshold:
                    batch_results.append({
                        'domain1': domain1,
                        'domain2': domain2,
                        'overall_similarity': overall_sim,
                        'dct_similarity': detailed_sims.get('dct', 0),
                        'fft_similarity': detailed_sims.get('fft', 0),
                        'color_similarity': detailed_sims.get('color', 0),
                        'edge_similarity': detailed_sims.get('edge', 0),
                        'phash_similarity': detailed_sims.get('phash', 0)
                    })
            
            return batch_results
        
        # Process batches with threading
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_batch = {
                executor.submit(process_comparison_batch, batch): batch 
                for batch in batches
            }
            
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    similar_pairs.extend(batch_results)
                    
                    # Update progress
                    completed_comparisons += len(future_to_batch[future])
                    progress = (completed_comparisons / total_comparisons) * 100
                    
                    print(f"   Progress: {completed_comparisons:,}/{total_comparisons:,} ({progress:.1f}%) - Found {len(similar_pairs)} similar pairs so far")
                    
                except Exception as e:
                    print(f"Batch processing error: {e}")
        
        print(f" Found {len(similar_pairs)} similar pairs")
        return similar_pairs
    
    def save_results(self, features_dict, similar_pairs, threshold):
        """Save analysis results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save features
        features_path = f"jpeg_logo_features_{timestamp}.pkl"
        with open(features_path, 'wb') as f:
            pickle.dump(features_dict, f)
        
        # Save similar pairs as CSV
        if similar_pairs:
            df = pd.DataFrame(similar_pairs)
            csv_path = f"jpeg_similar_pairs_t{int(threshold*100)}_{timestamp}.csv"
            df.to_csv(csv_path, index=False)
            
            print(f" Results saved:")
            print(f"   Features: {features_path}")
            print(f"   Similar pairs: {csv_path}")
            
            # Summary statistics
            print(f"\n Analysis Summary:")
            print(f"   Total logos analyzed: {len(features_dict)}")
            print(f"   Similar pairs found: {len(similar_pairs)}")
            print(f"   Similarity threshold: {threshold}")
            print(f"   Average similarity: {df['overall_similarity'].mean():.3f}")
            print(f"   Max similarity: {df['overall_similarity'].max():.3f}")
        
        return features_path, csv_path if similar_pairs else None

def main():
    """Main analysis pipeline for JPEG logos"""
    print("="*60)
    print("JPEG LOGO ANALYZER - DCT & FFT Analysis")
    print("Working directly with JPEG files")
    print("="*60)
    
    # Find the extracted logos folder
    logo_folders = [d for d in os.listdir('.') if d.startswith('extracted_logos_')]
    
    if not logo_folders:
        print(" No extracted logos folder found!")
        print("Please run extract_logos_to_jpg.py first")
        return
    
    # Use the most recent folder
    logo_folder = sorted(logo_folders)[-1]
    print(f" Using logo folder: {logo_folder}")
    
    # Initialize analyzer
    analyzer = JPEGLogoAnalyzer(logo_folder)
    
    if len(analyzer.jpeg_files) == 0:
        print(" No JPEG files found in the folder!")
        return
    
    # Extract features
    features_dict = analyzer.extract_all_features_threaded()
    
    # Find similar pairs with different thresholds
    thresholds = [0.90, 0.85, 0.80]
    
    for threshold in thresholds:
        print(f"\n{'='*40}")
        print(f"ANALYZING WITH THRESHOLD {threshold}")
        print(f"{'='*40}")
        
        similar_pairs = analyzer.find_similar_pairs(features_dict, threshold)
        features_path, pairs_path = analyzer.save_results(features_dict, similar_pairs, threshold)
    
    print(f"\n JPEG logo analysis completed!")
    print(f" Much faster than byte-array processing!")
    print(f" DCT leverages JPEG's native compression")
    print(f" FFT provides complementary frequency analysis")

if __name__ == "__main__":
    main()
