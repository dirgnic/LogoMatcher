#!/usr/bin/env python3

import numpy as np
import io
from PIL import Image
import fourier_math_cpp
import pickle
import matplotlib.pyplot as plt

def analyze_similarity_distribution():
    print("Analyzing similarity distribution to find optimal threshold...")
    
    # Load sample logos for analysis
    with open('logo_extraction_results.pkl', 'rb') as f:
        data = pickle.load(f)
    
    successful_logos = data['logo_results']
    
    # Take larger sample for better distribution analysis
    sample_logos = []
    sample_domains = []
    
    print("Loading sample logos...")
    for i, logo_data in enumerate(successful_logos[:200]):  # Use 200 logos
        if isinstance(logo_data, dict):
            domain = logo_data.get('domain', logo_data.get('website', ''))
            logo_bytes = logo_data.get('logo_data')
            
            if logo_bytes and domain:
                try:
                    image = Image.open(io.BytesIO(logo_bytes)).convert('L')
                    img_array = np.array(image, dtype=np.float64)
                    
                    sample_logos.append(img_array)
                    sample_domains.append(domain)
                except Exception as e:
                    continue
                
                if len(sample_logos) >= 100:  # 100 logos for analysis
                    break
    
    print(f"Analyzing similarity distribution with {len(sample_logos)} logos...")
    
    # Compute similarity matrix
    similarity_matrix = fourier_math_cpp.compute_similarity_matrix(sample_logos, 0.0)
    
    # Extract upper triangle (excluding diagonal)
    n = similarity_matrix.shape[0]
    similarities = []
    for i in range(n):
        for j in range(i+1, n):
            similarities.append(similarity_matrix[i][j])
    
    similarities = np.array(similarities)
    
    print(f"\nSimilarity Statistics:")
    print(f"  Count: {len(similarities)}")
    print(f"  Mean: {np.mean(similarities):.4f}")
    print(f"  Std: {np.std(similarities):.4f}")
    print(f"  Min: {np.min(similarities):.4f}")
    print(f"  Max: {np.max(similarities):.4f}")
    
    # Show percentiles
    percentiles = [50, 60, 70, 80, 85, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        value = np.percentile(similarities, p)
        print(f"  {p:2d}th percentile: {value:.4f}")
    
    # Test different thresholds
    thresholds = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    
    print(f"\nCluster analysis at different thresholds:")
    
    for threshold in thresholds:
        # Count pairs above threshold
        pairs_above = np.sum(similarities >= threshold)
        
        # Simple clustering simulation
        clusters = simulate_clustering(similarity_matrix, sample_domains, threshold)
        
        print(f"  Threshold {threshold:.2f}: {pairs_above:4d} pairs, {len(clusters)} clusters")
        
        # Show cluster sizes
        cluster_sizes = [len(cluster) for cluster in clusters if len(cluster) >= 2]
        if cluster_sizes:
            print(f"    Cluster sizes: {sorted(cluster_sizes, reverse=True)[:5]}")

def simulate_clustering(similarity_matrix, domains, threshold):
    """Simulate the clustering algorithm"""
    similar_pairs = []
    n = min(len(domains), similarity_matrix.shape[0])  # Handle size mismatch
    
    # Extract pairs above threshold
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i][j] >= threshold:
                similar_pairs.append((domains[i], domains[j]))
    
    # Simple clustering
    domain_groups = {}
    group_id = 0
    
    for d1, d2 in similar_pairs:
        g1 = domain_groups.get(d1)
        g2 = domain_groups.get(d2)
        
        if g1 is None and g2 is None:
            domain_groups[d1] = group_id
            domain_groups[d2] = group_id
            group_id += 1
        elif g1 is not None and g2 is None:
            domain_groups[d2] = g1
        elif g1 is None and g2 is not None:
            domain_groups[d1] = g2
        elif g1 != g2:
            # Merge groups
            old_group = g2
            new_group = g1
            for domain, gid in list(domain_groups.items()):
                if gid == old_group:
                    domain_groups[domain] = new_group
    
    # Convert to cluster list
    group_to_domains = {}
    for domain, gid in domain_groups.items():
        if gid not in group_to_domains:
            group_to_domains[gid] = []
        group_to_domains[gid].append(domain)
    
    return [domains for domains in group_to_domains.values() if len(domains) >= 2]

if __name__ == "__main__":
    analyze_similarity_distribution()
