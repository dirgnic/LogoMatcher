#!/usr/bin/env python3
"""
Real-time Logo Fourier Feature Visualizer
🌊 Visualize actual Fourier features from extracted logos
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import pickle
import io
import random
from similarity_pipeline import FourierLogoAnalyzer

def visualize_real_logo_features(num_examples=6):
    """Visualize Fourier features from actual extracted logos"""
    
    print("🌊 REAL LOGO FOURIER FEATURE VISUALIZATION")
    print("=" * 60)
    
    # Load extracted logos
    try:
        with open('logo_extraction_results.pkl', 'rb') as f:
            results = pickle.load(f)
        successful_logos = results['successful_logos'][:50]  # Take first 50 for speed
        print(f"📊 Loaded {len(successful_logos)} logos for analysis")
    except FileNotFoundError:
        print("❌ No logo results found. Run the extraction pipeline first.")
        return
    
    # Initialize analyzer
    analyzer = FourierLogoAnalyzer()
    
    # Select random logos for visualization
    selected_logos = random.sample(successful_logos, min(num_examples, len(successful_logos)))
    
    fig, axes = plt.subplots(num_examples, 5, figsize=(20, 4*num_examples))
    if num_examples == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('🌊 Real Logo Fourier Feature Analysis', fontsize=16, fontweight='bold')
    
    for idx, logo in enumerate(selected_logos):
        try:
            # Process the logo
            img_gray = analyzer.preprocess_logo(logo['logo_data'])
            if img_gray is None:
                continue
                
            features = analyzer.extract_logo_features(logo['logo_data'])
            
            # Column 1: Original Logo
            ax = axes[idx, 0]
            ax.imshow(img_gray, cmap='gray')
            ax.set_title(f"Original Logo\n{logo['website'][:20]}...")
            ax.axis('off')
            
            # Column 2: pHash Visualization
            ax = axes[idx, 1]
            if features['phash']:
                # Convert hash string to 8x8 binary array
                hash_bits = [int(b) for b in features['phash']]
                hash_array = np.array(hash_bits).reshape(8, 8)
                ax.imshow(hash_array, cmap='RdYlBu', interpolation='nearest')
                ax.set_title(f"pHash\n{features['phash'][:16]}...")
            else:
                ax.text(0.5, 0.5, 'pHash\nFailed', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            
            # Column 3: FFT Magnitude Spectrum
            ax = axes[idx, 2]
            fft = np.fft.fft2(img_gray)
            fft_shifted = np.fft.fftshift(fft)
            magnitude = np.log(np.abs(fft_shifted) + 1)
            
            # Show center region (low frequencies)
            center = magnitude.shape[0] // 2
            radius = 32
            mag_crop = magnitude[center-radius:center+radius, center-radius:center+radius]
            
            im = ax.imshow(mag_crop, cmap='viridis')
            ax.set_title('FFT Magnitude\n(Log Scale)')
            ax.axis('off')
            
            # Column 4: FFT Feature Vector Visualization
            ax = axes[idx, 3]
            if len(features['fft_signature']) > 0:
                # Show first 64 components as a heatmap
                fft_features = features['fft_signature'][:64].reshape(8, 8)
                im = ax.imshow(fft_features, cmap='plasma', aspect='auto')
                ax.set_title('FFT Features\n(64 components)')
            else:
                ax.text(0.5, 0.5, 'FFT Features\nFailed', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            
            # Column 5: Fourier-Mellin Features
            ax = axes[idx, 4]
            if len(features['fourier_mellin']) > 0:
                # Show Fourier-Mellin features as a bar plot
                fm_features = features['fourier_mellin']
                ax.bar(range(len(fm_features)), fm_features, alpha=0.7)
                ax.set_title('Fourier-Mellin\nFeatures')
                ax.set_xlabel('Feature Index')
                ax.set_ylabel('Value')
            else:
                ax.text(0.5, 0.5, 'F-M Features\nFailed', ha='center', va='center', transform=ax.transAxes)
            
        except Exception as e:
            print(f"⚠️ Error processing {logo['website']}: {e}")
            # Fill row with error messages
            for col in range(5):
                axes[idx, col].text(0.5, 0.5, f'Error\n{logo["website"][:15]}...', 
                                  ha='center', va='center', transform=axes[idx, col].transAxes)
                axes[idx, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('real_logo_fourier_features.png', dpi=300, bbox_inches='tight')
    print("💾 Saved: real_logo_fourier_features.png")
    plt.show()

def create_similarity_comparison_visualization():
    """Create visualization comparing similar vs dissimilar logo pairs"""
    
    print("\n🔍 SIMILARITY COMPARISON VISUALIZATION")
    print("=" * 60)
    
    try:
        # Load similarity results
        with open('logo_extraction_results.pkl', 'rb') as f:
            extraction_results = pickle.load(f)
        
        try:
            with open('improved_similar_pairs.csv', 'r') as f:
                import pandas as pd
                pairs_df = pd.read_csv('improved_similar_pairs.csv')
                
                if len(pairs_df) == 0:
                    print("❌ No similar pairs found in results")
                    return
                    
        except FileNotFoundError:
            print("❌ No similarity results found. Run similarity analysis first.")
            return
        
        # Get logos by website name for lookup
        logos_by_website = {logo['website']: logo for logo in extraction_results['successful_logos']}
        
        # Select examples: highest similarity, medium similarity, lowest similarity
        sorted_pairs = pairs_df.sort_values('similarity', ascending=False)
        
        examples = []
        if len(sorted_pairs) >= 3:
            # High similarity
            high_sim = sorted_pairs.iloc[0]
            examples.append(('High Similarity', high_sim, high_sim['similarity']))
            
            # Medium similarity  
            mid_idx = len(sorted_pairs) // 2
            med_sim = sorted_pairs.iloc[mid_idx]
            examples.append(('Medium Similarity', med_sim, med_sim['similarity']))
            
            # Low similarity
            low_sim = sorted_pairs.iloc[-1]
            examples.append(('Low Similarity', low_sim, low_sim['similarity']))
        else:
            # Just use what we have
            for i, (_, pair) in enumerate(sorted_pairs.head(3).iterrows()):
                examples.append((f'Pair {i+1}', pair, pair['similarity']))
        
        # Create visualization
        fig, axes = plt.subplots(len(examples), 5, figsize=(20, 4*len(examples)))
        if len(examples) == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('🔍 Logo Similarity Comparison Analysis', fontsize=16, fontweight='bold')
        
        analyzer = FourierLogoAnalyzer()
        
        for idx, (label, pair, similarity) in enumerate(examples):
            website1, website2 = pair['website1'], pair['website2']
            
            # Get logo data
            logo1 = logos_by_website.get(website1)
            logo2 = logos_by_website.get(website2)
            
            if not logo1 or not logo2:
                continue
            
            # Process both logos
            img1 = analyzer.preprocess_logo(logo1['logo_data'])
            img2 = analyzer.preprocess_logo(logo2['logo_data'])
            
            if img1 is None or img2 is None:
                continue
            
            features1 = analyzer.extract_logo_features(logo1['logo_data'])
            features2 = analyzer.extract_logo_features(logo2['logo_data'])
            
            # Column 1: Logo 1
            ax = axes[idx, 0]
            ax.imshow(img1, cmap='gray')
            ax.set_title(f"Logo A\n{website1[:20]}...")
            ax.axis('off')
            
            # Column 2: Logo 2  
            ax = axes[idx, 1]
            ax.imshow(img2, cmap='gray')
            ax.set_title(f"Logo B\n{website2[:20]}...")
            ax.axis('off')
            
            # Column 3: Hash Comparison
            ax = axes[idx, 2]
            if features1['phash'] and features2['phash']:
                # Show both hashes side by side
                hash1_bits = np.array([int(b) for b in features1['phash']]).reshape(8, 8)
                hash2_bits = np.array([int(b) for b in features2['phash']]).reshape(8, 8)
                
                # Combine with separator
                combined = np.concatenate([hash1_bits, np.ones((8, 1))*0.5, hash2_bits], axis=1)
                ax.imshow(combined, cmap='RdYlBu', interpolation='nearest')
                
                # Calculate Hamming distance
                hamming_dist = analyzer.hamming_distance(features1['phash'], features2['phash'])
                ax.set_title(f'pHash Comparison\nHamming: {hamming_dist:.3f}')
            else:
                ax.text(0.5, 0.5, 'Hash\nComparison\nFailed', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            
            # Column 4: Feature Vector Comparison
            ax = axes[idx, 3]
            if len(features1['fft_signature']) > 0 and len(features2['fft_signature']) > 0:
                # Plot both feature vectors
                x = range(min(32, len(features1['fft_signature'])))
                ax.plot(x, features1['fft_signature'][:32], 'b-', alpha=0.7, label='Logo A')
                ax.plot(x, features2['fft_signature'][:32], 'r-', alpha=0.7, label='Logo B')
                ax.set_title('FFT Features\n(First 32)')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'FFT\nComparison\nFailed', ha='center', va='center', transform=ax.transAxes)
            
            # Column 5: Similarity Metrics
            ax = axes[idx, 4]
            ax.axis('off')
            
            # Calculate individual similarities
            phash_sim = 1.0 - analyzer.hamming_distance(features1['phash'], features2['phash']) if features1['phash'] and features2['phash'] else 0
            
            from sklearn.metrics.pairwise import cosine_similarity
            fft_sim = cosine_similarity([features1['fft_signature']], [features2['fft_signature']])[0][0] if len(features1['fft_signature']) > 0 and len(features2['fft_signature']) > 0 else 0
            fm_sim = cosine_similarity([features1['fourier_mellin']], [features2['fourier_mellin']])[0][0] if len(features1['fourier_mellin']) > 0 and len(features2['fourier_mellin']) > 0 else 0
            
            # Create bar chart of similarities
            methods = ['pHash', 'FFT', 'F-M', 'Combined']
            scores = [phash_sim, fft_sim, fm_sim, similarity]
            colors = ['red', 'blue', 'green', 'purple']
            
            bars = ax.bar(methods, scores, color=colors, alpha=0.7)
            ax.set_ylim(0, 1)
            ax.set_title(f'{label}\nScore: {similarity:.3f}')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('logo_similarity_comparison.png', dpi=300, bbox_inches='tight')
        print("💾 Saved: logo_similarity_comparison.png")
        plt.show()
        
    except Exception as e:
        print(f"❌ Error creating similarity visualization: {e}")

def main():
    """Run real logo feature visualizations"""
    print("🎨 REAL LOGO FOURIER VISUALIZATION PIPELINE")
    print("=" * 60)
    
    # Create real logo feature visualization
    visualize_real_logo_features(num_examples=6)
    
    # Create similarity comparison
    create_similarity_comparison_visualization()
    
    print("\n✅ Real logo visualizations completed!")
    print("📁 Generated files:")
    print("   • real_logo_fourier_features.png")
    print("   • logo_similarity_comparison.png")

if __name__ == "__main__":
    main()
