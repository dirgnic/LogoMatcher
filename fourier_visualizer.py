"""
Fourier Visualization Runner
Uses cached logo data to generate Fourier curve visualizations
"""

import pickle
import numpy as np
import cv2
from fourier_logo_analyzer import FourierLogoAnalyzer
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import os
from datetime import datetime

def load_cached_logos(cache_path, max_logos=50):
    """Load cached logo images from pickle file"""
    
    if not os.path.exists(cache_path):
        print(f"Cache file not found: {cache_path}")
        return {}
    
    try:
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        # Extract logo_results from the data structure
        if isinstance(cached_data, dict) and 'logo_results' in cached_data:
            logo_results = cached_data['logo_results']
            print(f"Found {len(logo_results)} logo entries in cache")
        else:
            print("Unexpected data format in cache file")
            return {}
        
        logos_dict = {}
        count = 0
        failed_count = 0
        
        for logo_entry in logo_results:
            if count >= max_logos:
                break
                
            try:
                # Extract data from entry
                if not isinstance(logo_entry, dict):
                    continue
                    
                domain = logo_entry.get('domain', 'unknown')
                logo_bytes = logo_entry.get('logo_data')
                success = logo_entry.get('success', False)
                
                if not success or not logo_bytes:
                    failed_count += 1
                    continue
                
                # Convert bytes to image
                img_array = np.frombuffer(logo_bytes, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Resize to standard size
                    img_resized = cv2.resize(img, (128, 128))
                    logos_dict[domain] = img_resized
                    count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                continue
        
        print(f"Successfully loaded {len(logos_dict)} valid logo images")
        print(f"Failed to load {failed_count} logos")
        return logos_dict
        
    except Exception as e:
        print(f"Error loading cached data: {e}")
        return {}

def extract_brand_name(domain: str) -> str:
    """Extract clean brand name from domain"""
    try:
        # Parse domain
        if not domain.startswith('http'):
            domain = f"https://{domain}"
        
        parsed = urlparse(domain)
        hostname = parsed.hostname or domain
        
        # Remove www. prefix
        if hostname.startswith('www.'):
            hostname = hostname[4:]
        
        # Split by dots and take the main part
        parts = hostname.split('.')
        if len(parts) >= 2:
            brand = parts[0]
        else:
            brand = hostname
        
        # Capitalize first letter
        return brand.capitalize()
        
    except:
        return domain.split('.')[0].capitalize()

def analyze_sample_logos(analyzer, logos_dict, num_samples=3):
    """Analyze sample logos with comprehensive Fourier features"""
    
    sample_domains = list(logos_dict.keys())[:num_samples]
    results = {}
    
    for i, domain in enumerate(sample_domains):
        print(f"\nAnalyzing logo {i+1}/{num_samples}: {domain}")
        
        img = logos_dict[domain]
        brand_name = extract_brand_name(domain)
        
        # Compute all features
        features = analyzer.compute_all_features(img)
        
        # Create detailed visualization
        create_detailed_fourier_analysis(analyzer, img, features, brand_name, domain)
        
        results[domain] = features
        
        print(f" Analysis complete for {brand_name}")
        print(f"  - FFT features: {len(features['fft_features'])} dimensions")
        print(f"  - Fourier-Mellin signature: {len(features['fmt_signature'])} dimensions")
        print(f"  - Color-aware FMT: {len(features['color_aware_fmt'])} dimensions")
        print(f"  - Saliency-weighted FFT: {len(features['saliency_weighted_fft'])} dimensions")
        print(f"  - Perceptual hash: {features['phash'][:16]}...")
        
    return results

def create_detailed_fourier_analysis(analyzer, img, features, brand_name, domain):
    """Create comprehensive Fourier analysis visualization"""
    
    # Create a larger figure with more subplots for detailed analysis
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f'Comprehensive Fourier Analysis: {brand_name}', fontsize=18, fontweight='bold')
    
    # Row 1: Original image and frequency domain analysis
    # Original image
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original Logo')
    axes[0, 0].axis('off')
    
    # FFT magnitude spectrum
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude_spectrum = np.log(np.abs(fft_shift) + 1)
    
    im1 = axes[0, 1].imshow(magnitude_spectrum, cmap='hot')
    axes[0, 1].set_title('FFT Magnitude Spectrum')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # Phase spectrum
    phase_spectrum = np.angle(fft_shift)
    im2 = axes[0, 2].imshow(phase_spectrum, cmap='hsv')
    axes[0, 2].set_title('FFT Phase Spectrum')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2])
    
    # Radial frequency profile
    h, w = magnitude_spectrum.shape
    center = (h//2, w//2)
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)
    
    # Calculate radial average
    tbin = np.bincount(r.ravel(), magnitude_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    
    axes[0, 3].plot(radialprofile[:min(50, len(radialprofile))], 'b-', linewidth=2)
    axes[0, 3].set_title('Radial Frequency Profile')
    axes[0, 3].set_xlabel('Frequency (cycles/pixel)')
    axes[0, 3].set_ylabel('Log Magnitude')
    axes[0, 3].grid(True, alpha=0.3)
    
    # Row 2: Feature vectors and transforms
    # FFT features plot
    axes[1, 0].plot(features['fft_features'], 'b-', alpha=0.8, linewidth=1.5)
    axes[1, 0].set_title(f'FFT Feature Vector ({len(features["fft_features"])} dims)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlabel('Feature Index')
    axes[1, 0].set_ylabel('Normalized Magnitude')
    
    # Fourier-Mellin signature
    theta_range = np.linspace(0, 2*np.pi, len(features['fmt_signature']))
    axes[1, 1].plot(theta_range, features['fmt_signature'], 'g-', alpha=0.8, linewidth=2)
    axes[1, 1].set_title('Fourier-Mellin Transform')
    axes[1, 1].set_xlabel('Angle (radians)')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Saliency-weighted FFT
    axes[1, 2].plot(features['saliency_weighted_fft'], 'r-', alpha=0.8, linewidth=1.5)
    axes[1, 2].set_title(f'Saliency-Weighted FFT ({len(features["saliency_weighted_fft"])} dims)')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_xlabel('Feature Index')
    axes[1, 2].set_ylabel('Weighted Magnitude')
    
    # Color-aware FMT (separate channels)
    color_fmt = features['color_aware_fmt'].reshape(3, -1)
    for i, (color, label) in enumerate(zip(['b', 'g', 'r'], ['Blue', 'Green', 'Red'])):
        axes[1, 3].plot(color_fmt[i], color=color, alpha=0.7, linewidth=1.5, label=label)
    axes[1, 3].set_title('Color-Aware Fourier-Mellin')
    axes[1, 3].legend()
    axes[1, 3].grid(True, alpha=0.3)
    axes[1, 3].set_xlabel('Feature Index')
    axes[1, 3].set_ylabel('Color Channel Magnitude')
    
    # Row 3: Hash and moments analysis
    # Perceptual hash visualization
    phash_bits = np.array([int(bit) for bit in features['phash']]).reshape(8, 8)
    im3 = axes[2, 0].imshow(phash_bits, cmap='RdBu', interpolation='nearest')
    axes[2, 0].set_title('Perceptual Hash (64-bit DCT)')
    axes[2, 0].set_xticks(range(8))
    axes[2, 0].set_yticks(range(8))
    axes[2, 0].grid(True, alpha=0.3)
    plt.colorbar(im3, ax=axes[2, 0])
    
    # Hu moments
    hu_moments = features['hu_moments']
    axes[2, 1].bar(range(len(hu_moments)), hu_moments, alpha=0.7, color='purple')
    axes[2, 1].set_title('Hu Moments (Shape Invariants)')
    axes[2, 1].set_xlabel('Moment Index')
    axes[2, 1].set_ylabel('Log Value')
    axes[2, 1].grid(True, alpha=0.3)
    
    # FFT vs Saliency FFT comparison
    axes[2, 2].plot(features['fft_features'][:50], 'b-', alpha=0.7, label='Standard FFT', linewidth=1.5)
    axes[2, 2].plot(features['saliency_weighted_fft'][:50], 'r-', alpha=0.7, label='Saliency FFT', linewidth=1.5)
    axes[2, 2].set_title('FFT Comparison (First 50 features)')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    axes[2, 2].set_xlabel('Feature Index')
    axes[2, 2].set_ylabel('Magnitude')
    
    # Feature statistics
    stats_text = f"""Feature Statistics:
FFT Mean: {np.mean(features['fft_features']):.4f}
FFT Std: {np.std(features['fft_features']):.4f}
FMT Mean: {np.mean(features['fmt_signature']):.4f}
FMT Std: {np.std(features['fmt_signature']):.4f}
Color FMT Mean: {np.mean(features['color_aware_fmt']):.4f}
Saliency FFT Mean: {np.mean(features['saliency_weighted_fft']):.4f}
Hash Hamming Weight: {features['phash'].count('1')}/64
"""
    
    axes[2, 3].text(0.1, 0.9, stats_text, transform=axes[2, 3].transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[2, 3].set_title('Feature Statistics')
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    
    # Save comprehensive analysis
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"comprehensive_fourier_analysis_{brand_name}_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comprehensive analysis: {save_path}")
    
    plt.show()
    
    return save_path

def compare_logo_pair(analyzer, logos_dict, domain1, domain2):
    """Compare two specific logos with detailed Fourier analysis"""
    
    if domain1 not in logos_dict or domain2 not in logos_dict:
        print("One or both domains not found in cached data")
        return
    
    print(f"\nComparing {domain1} vs {domain2}")
    
    img1 = logos_dict[domain1]
    img2 = logos_dict[domain2]
    
    # Extract features
    features1 = analyzer.compute_all_features(img1)
    features2 = analyzer.compute_all_features(img2)
    
    # Compute similarities
    overall_sim, detailed_sims = analyzer.compute_similarity(features1, features2)
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Fourier Comparison: {extract_brand_name(domain1)} vs {extract_brand_name(domain2)}', 
                 fontsize=14, fontweight='bold')
    
    # Show original images
    axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title(f'{extract_brand_name(domain1)}')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(f'{extract_brand_name(domain2)}')
    axes[1, 0].axis('off')
    
    # FFT features comparison
    axes[0, 1].plot(features1['fft_features'], 'b-', label='Logo 1', alpha=0.7)
    axes[0, 1].plot(features2['fft_features'], 'r-', label='Logo 2', alpha=0.7)
    axes[0, 1].set_title(f"FFT Features (sim: {detailed_sims['fft']:.3f})")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Fourier-Mellin comparison
    theta_range = np.linspace(0, 2*np.pi, len(features1['fmt_signature']))
    axes[0, 2].plot(theta_range, features1['fmt_signature'], 'b-', label='Logo 1', alpha=0.7)
    axes[0, 2].plot(theta_range, features2['fmt_signature'], 'r-', label='Logo 2', alpha=0.7)
    axes[0, 2].set_title(f"Fourier-Mellin (sim: {detailed_sims['fourier_mellin']:.3f})")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Similarity breakdown
    methods = list(detailed_sims.keys())
    similarities = list(detailed_sims.values())
    
    bars = axes[0, 3].bar(range(len(methods)), similarities, 
                         color=plt.cm.viridis(np.linspace(0, 1, len(methods))))
    axes[0, 3].set_title(f'Similarity Breakdown (Overall: {overall_sim:.3f})')
    axes[0, 3].set_xticks(range(len(methods)))
    axes[0, 3].set_xticklabels(methods, rotation=45, ha='right')
    axes[0, 3].set_ylim(0, 1)
    axes[0, 3].grid(True, alpha=0.3)
    
    # Add value labels
    for i, (bar, sim) in enumerate(zip(bars, similarities)):
        axes[0, 3].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{sim:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Color-aware FMT comparison
    color_fmt1 = features1['color_aware_fmt'].reshape(3, -1)
    color_fmt2 = features2['color_aware_fmt'].reshape(3, -1)
    
    for i, (color, label) in enumerate(zip(['b', 'g', 'r'], ['Blue', 'Green', 'Red'])):
        axes[1, 1].plot(color_fmt1[i], color=color, alpha=0.7, linestyle='-', label=f'Logo 1 {label}')
        axes[1, 1].plot(color_fmt2[i], color=color, alpha=0.7, linestyle='--', label=f'Logo 2 {label}')
    
    axes[1, 1].set_title(f"Color-Aware FMT (sim: {detailed_sims['color_aware_fmt']:.3f})")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)
    
    # Hu moments comparison
    axes[1, 2].bar(np.arange(len(features1['hu_moments'])) - 0.2, features1['hu_moments'], 
                   width=0.4, label='Logo 1', alpha=0.7)
    axes[1, 2].bar(np.arange(len(features2['hu_moments'])) + 0.2, features2['hu_moments'], 
                   width=0.4, label='Logo 2', alpha=0.7)
    axes[1, 2].set_title(f"Hu Moments (sim: {detailed_sims['hu_moments']:.3f})")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Perceptual hash comparison
    phash1_bits = np.array([int(bit) for bit in features1['phash']]).reshape(8, 8)
    phash2_bits = np.array([int(bit) for bit in features2['phash']]).reshape(8, 8)
    
    # Show hash difference
    hash_diff = np.abs(phash1_bits - phash2_bits)
    im = axes[1, 3].imshow(hash_diff, cmap='RdYlBu_r', vmin=0, vmax=1)
    axes[1, 3].set_title(f"pHash Difference (sim: {detailed_sims['phash']:.3f})")
    plt.colorbar(im, ax=axes[1, 3])
    
    plt.tight_layout()
    
    # Save comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"fourier_comparison_{extract_brand_name(domain1)}_vs_{extract_brand_name(domain2)}_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Comparison saved to: {save_path}")
    
    plt.show()
    
    return overall_sim, detailed_sims

def main():
    """Main analysis pipeline"""
    
    print("=== Fourier Logo Analysis Visualization ===\n")
    
    # Initialize analyzer
    analyzer = FourierLogoAnalyzer()
    
    # Load cached logos
    cache_path = "comprehensive_logo_extraction_fast_results.pkl"
    
    if not os.path.exists(cache_path):
        print(f"Error: Cache file {cache_path} not found!")
        print("Available files:")
        for f in os.listdir('.'):
            if f.endswith('.pkl'):
                print(f"  - {f}")
        return
    
    logos_dict = load_cached_logos(cache_path)
    
    if not logos_dict:
        print("No valid logos found in cache!")
        return
    
    # Show some sample domains
    sample_domains = list(logos_dict.keys())[:10]
    print("\nSample domains in cache:")
    for i, domain in enumerate(sample_domains):
        print(f"{i+1}. {domain}")
    
    # Analyze sample logos
    print("\n" + "="*50)
    print("ANALYZING SAMPLE LOGOS")
    print("="*50)
    
    sample_results = analyze_sample_logos(analyzer, logos_dict, num_samples=3)
    
    # Compare specific logo pairs
    print("\n" + "="*50)
    print("COMPARING LOGO PAIRS")
    print("="*50)
    
    # Find some interesting pairs to compare
    domains_list = list(logos_dict.keys())
    
    if len(domains_list) >= 2:
        # Compare first two logos
        overall_sim, detailed_sims = compare_logo_pair(
            analyzer, logos_dict, 
            domains_list[0], domains_list[1]
        )
        
        print(f"\nOverall similarity: {overall_sim:.4f}")
        print("Detailed similarities:")
        for method, sim in detailed_sims.items():
            print(f"  {method}: {sim:.4f}")
    
    print(f"\n Analysis complete! Generated visualizations for Fourier analysis.")
    print(f" Check the generated PNG files for detailed Fourier curve plots.")

if __name__ == "__main__":
    main()
