#!/usr/bin/env python3

import numpy as np
import io
from PIL import Image
import fourier_math_cpp
import pickle

def check_similarity_values():
    print("Checking actual similarity values...")
    
    # Load a small sample of logos
    with open('logo_extraction_results.pkl', 'rb') as f:
        data = pickle.load(f)
    
    successful_logos = data['logo_results']
    
    # Take first 10 logos for testing
    sample_logos = []
    sample_domains = []
    
    for i, logo_data in enumerate(successful_logos[:10]):
        if isinstance(logo_data, dict):
            domain = logo_data.get('domain', logo_data.get('website', ''))
            logo_bytes = logo_data.get('logo_data')
            
            if logo_bytes and domain:
                try:
                    # Convert to grayscale array
                    image = Image.open(io.BytesIO(logo_bytes)).convert('L')
                    img_array = np.array(image, dtype=np.float64)
                    
                    sample_logos.append(img_array)
                    sample_domains.append(domain)
                except Exception as e:
                    continue
                
                if len(sample_logos) >= 10:
                    break
    
    print(f"Testing with {len(sample_logos)} sample logos:")
    for i, domain in enumerate(sample_domains):
        print(f"  {i}: {domain}")
    
    # Compute similarity matrix
    similarity_matrix = fourier_math_cpp.compute_similarity_matrix(sample_logos, 0.0)
    
    print(f"\nSimilarity matrix shape: {similarity_matrix.shape}")
    print(f"Matrix diagonal (should be ~1.0): {np.diag(similarity_matrix)}")
    
    # Find max off-diagonal similarities
    mask = np.ones(similarity_matrix.shape, dtype=bool)
    np.fill_diagonal(mask, False)
    
    max_sim = np.max(similarity_matrix[mask])
    min_sim = np.min(similarity_matrix[mask])
    mean_sim = np.mean(similarity_matrix[mask])
    
    print(f"\nOff-diagonal similarities:")
    print(f"  Maximum: {max_sim:.6f}")
    print(f"  Minimum: {min_sim:.6f}")  
    print(f"  Mean: {mean_sim:.6f}")
    
    # Find top 5 similar pairs
    print(f"\nTop 5 most similar pairs:")
    flat_indices = np.argsort(similarity_matrix[mask])[-5:][::-1]
    
    for rank, flat_idx in enumerate(flat_indices):
        # Convert flat index back to 2D coordinates
        row_idx = 0
        col_idx = 0
        count = 0
        for i in range(similarity_matrix.shape[0]):
            for j in range(i+1, similarity_matrix.shape[1]):
                if count == len(similarity_matrix[mask]) - 1 - flat_idx:
                    row_idx, col_idx = i, j
                    break
                count += 1
            else:
                continue
            break
        
        sim_value = similarity_matrix[row_idx, col_idx]
        print(f"  {rank+1}. {sample_domains[row_idx]} <-> {sample_domains[col_idx]} (sim: {sim_value:.6f})")

if __name__ == "__main__":
    check_similarity_values()
