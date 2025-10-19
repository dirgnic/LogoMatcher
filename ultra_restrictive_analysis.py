#!/usr/bin/env python3

import pickle
import time
import numpy as np
import cv2
from python_scraping_class import LogoAnalysisPipeline

def run_enhanced_ultra_restrictive_clustering():
    """Run clustering with enhanced C++ comprehensive metrics to find truly similar logos"""
    
    print("🎯 ENHANCED ULTRA-RESTRICTIVE CLUSTERING ANALYSIS")
    print("=" * 60)
    
    # Try to import enhanced C++ module
    try:
        import enhanced_fourier_math_cpp as enhanced_cpp
        cpp_available = True
        print("🔥 Enhanced C++ comprehensive metrics module loaded")
        analyzer = enhanced_cpp.EnhancedFourierAnalyzer(128)
        similarity_computer = enhanced_cpp.EnhancedSimilarityComputer()
    except ImportError:
        print("⚠️  Enhanced C++ module not available - falling back to basic analysis")
        print("💡 To enable enhanced metrics, build the module:")
        print("   cmake -B build_enhanced -S . -f CMakeLists_enhanced.txt")
        print("   cd build_enhanced && make")
        cpp_available = False
        pipeline = LogoAnalysisPipeline()
    
    # Load enhanced logo data
    print("📋 Loading enhanced logo data...")
    with open('comprehensive_logo_extraction_fast_results.pkl', 'rb') as f:
        enhanced_data = pickle.load(f)
    
    successful_logos = enhanced_data.get('successful_logos', [])[:1000]  # Test with subset for speed
    print(f"✅ Testing with {len(successful_logos)} logos (subset for speed)")
    
    # Remove duplicates by domain
    seen_domains = set()
    unique_logos = []
    for logo in successful_logos:
        if logo['domain'] not in seen_domains:
            seen_domains.add(logo['domain'])
            unique_logos.append(logo)
    
    print(f"🔄 Unique domains: {len(unique_logos)}")
    
    if cpp_available:
        # Use enhanced C++ comprehensive metrics
        return run_enhanced_cpp_analysis(unique_logos, analyzer, similarity_computer)
    else:
        # Fallback to basic analysis
        return run_basic_analysis(unique_logos, pipeline)

def run_enhanced_cpp_analysis(unique_logos, analyzer, similarity_computer):
    """Run analysis using enhanced C++ comprehensive metrics"""
    
    print("🔥 Using Enhanced C++ Comprehensive Metrics Analysis")
    print("-" * 50)
    
    # Extract enhanced features for all logos
    all_features = []
    valid_logos = []
    
    print("📊 Extracting comprehensive features...")
    for i, logo in enumerate(unique_logos):
        if i % 100 == 0:
            print(f"   Processing logo {i+1}/{len(unique_logos)}")
        
        try:
            # Convert binary logo data to numpy array
            img_array = np.frombuffer(logo['logo_data'], dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            
            if img is not None:
                # Resize to standard size
                img_resized = cv2.resize(img, (128, 128)).astype(np.float64)
                
                # Extract comprehensive features
                features = analyzer.extract_all_features(img_resized)
                
                if features.valid:
                    all_features.append(features)
                    valid_logos.append(logo)
                    
        except Exception as e:
            print(f"⚠️  Failed to process {logo['domain']}: {e}")
    
    print(f"✅ Extracted features for {len(valid_logos)} valid logos")
    
    # Try different calibrated similarity thresholds for ultra-restrictive clustering
    thresholds = [0.95, 0.90, 0.85, 0.80]
    
    for threshold in thresholds:
        print(f"\n🔍 Testing enhanced calibrated threshold: {threshold}")
        start_time = time.time()
        
        # Compute pairwise comprehensive similarities
        similar_pairs = []
        
        for i in range(len(all_features)):
            for j in range(i+1, len(all_features)):
                # Compute comprehensive similarity metrics
                metrics = similarity_computer.compute_comprehensive_similarity(
                    all_features[i], all_features[j]
                )
                
                # Use calibrated similarity as primary metric
                if metrics.calibrated_similarity >= threshold:
                    similar_pairs.append({
                        'domain1': valid_logos[i]['domain'],
                        'domain2': valid_logos[j]['domain'],
                        'similarity': metrics.calibrated_similarity,
                        'confidence': metrics.confidence_score,
                        'comprehensive_metrics': {
                            'phash_similar': metrics.phash_similar,
                            'fft_similar': metrics.fft_similar,
                            'fmt_similar': metrics.fmt_similar,
                            'hu_similar': metrics.hu_similar,
                            'zernike_similar': metrics.zernike_similar,
                            'overall_similar': metrics.overall_similar
                        }
                    })
        
        analysis_time = time.time() - start_time
        print(f"   📊 Found {len(similar_pairs)} similar pairs in {analysis_time:.1f}s")
        
        # Show similarity distribution
        if similar_pairs:
            similarities = [pair['similarity'] for pair in similar_pairs]
            confidences = [pair['confidence'] for pair in similar_pairs]
            print(f"   📈 Similarity range: {min(similarities):.3f} - {max(similarities):.3f}")
            print(f"   🎯 Average similarity: {np.mean(similarities):.3f}")
            print(f"   🔮 Average confidence: {np.mean(confidences):.3f}")
            
            # Show some examples with comprehensive metrics
            print(f"   💫 Example enhanced similar pairs:")
            for i, pair in enumerate(similar_pairs[:5]):
                metrics = pair['comprehensive_metrics']
                active_metrics = [k for k, v in metrics.items() if v and k != 'overall_similar']
                print(f"      {pair['domain1']} ↔ {pair['domain2']}")
                print(f"         Similarity: {pair['similarity']:.3f}, Confidence: {pair['confidence']:.3f}")
                print(f"         Active metrics: {', '.join(active_metrics)}")
        
        # If we get a reasonable number, proceed with clustering
        if 0 < len(similar_pairs) <= 5000:
            print(f"✅ Threshold {threshold} gives {len(similar_pairs)} pairs - creating enhanced clusters...")
            
            clusters = create_enhanced_clusters(similar_pairs, max_cluster_size=20)
            
            cluster_sizes = [len(cluster) for cluster in clusters]
            cluster_sizes.sort(reverse=True)
            
            print(f"\n🎊 ENHANCED CLUSTERING RESULTS (Threshold: {threshold})")
            print(f"=" * 50)
            print(f"📊 Total clusters: {len(clusters)}")
            print(f"📈 Cluster sizes: {cluster_sizes}")
            
            # Show meaningful clusters (size > 2)
            meaningful_clusters = [c for c in clusters if len(c) > 2]
            print(f"\n🎯 ENHANCED BRAND CLUSTERS (3+ domains):")
            for i, cluster in enumerate(meaningful_clusters):
                print(f"Enhanced Brand Cluster {i+1} ({len(cluster)} domains): {cluster}")
            
            return len(clusters), cluster_sizes, threshold
    
    print(f"\n❌ No suitable threshold found even with enhanced comprehensive metrics")
    return 0, [], None

def run_basic_analysis(unique_logos, pipeline):
    """Fallback to basic analysis when enhanced C++ not available"""
    
    print("🐍 Using Basic Python Analysis (Fallback)")
    print("-" * 50)
    
    # Prepare logos for basic analysis
    pipeline_logos = []
    for logo in unique_logos:
        pipeline_logos.append({
            'domain': logo['domain'],
            'logo_data': logo['logo_data'],
            'size_bytes': logo['size_bytes'],
            'source': logo.get('source', 'unknown')
        })
    
    # Try basic thresholds
    thresholds = [0.99, 0.98, 0.97, 0.96]
    
    for threshold in thresholds:
        print(f"\n🔍 Testing basic threshold: {threshold}")
        start_time = time.time()
        
        # Run basic similarity analysis
        if hasattr(pipeline, 'fourier_math_cpp') and pipeline.fourier_math_cpp is not None:
            similarity_results = pipeline._cpp_similarity_analysis(pipeline_logos, threshold)
        else:
            similarity_results = pipeline._python_similarity_analysis(pipeline_logos, threshold)
        
        # Extract results
        valid_logos_basic = similarity_results.get('valid_logos', [])
        similarity_matrix = similarity_results.get('similarity_matrix', [])
        
        # Convert similarity matrix to pairs format for natural clustering
        similar_pairs = []
        
        if similarity_matrix is not None and len(similarity_matrix) > 0 and len(valid_logos_basic) > 1:
            for i in range(len(similarity_matrix)):
                for j in range(i+1, len(similarity_matrix[i])):
                    if len(similarity_matrix[i]) > j and similarity_matrix[i][j] >= threshold:
                        similar_pairs.append({
                            'domain1': valid_logos_basic[i]['domain'],
                            'domain2': valid_logos_basic[j]['domain'],
                            'similarity': similarity_matrix[i][j]
                        })
        
        analysis_time = time.time() - start_time
        print(f"   📊 Found {len(similar_pairs)} similar pairs in {analysis_time:.1f}s")
        
        # Show basic examples
        if similar_pairs:
            similarities = [pair['similarity'] for pair in similar_pairs]
            print(f"   📈 Similarity range: {min(similarities):.3f} - {max(similarities):.3f}")
            print(f"   🎯 Average similarity: {np.mean(similarities):.3f}")
            
            print(f"   💫 Example basic similar pairs:")
            for i, pair in enumerate(similar_pairs[:5]):
                print(f"      {pair['domain1']} ↔ {pair['domain2']} (sim: {pair['similarity']:.3f})")
        
        # If we get a reasonable number, proceed with clustering
        if 0 < len(similar_pairs) <= 5000:
            print(f"✅ Threshold {threshold} gives {len(similar_pairs)} pairs - clustering...")
            
            clusters = pipeline._create_natural_similarity_clusters(similar_pairs, max_cluster_size=20)
            
            cluster_sizes = [len(cluster) for cluster in clusters]
            cluster_sizes.sort(reverse=True)
            
            print(f"\n🎊 BASIC CLUSTERING RESULTS (Threshold: {threshold})")
            print(f"=" * 50)
            print(f"📊 Total clusters: {len(clusters)}")
            print(f"📈 Cluster sizes: {cluster_sizes}")
            
            # Show meaningful clusters
            meaningful_clusters = [c for c in clusters if len(c) > 2]
            print(f"\n🎯 BASIC BRAND CLUSTERS (3+ domains):")
            for i, cluster in enumerate(meaningful_clusters):
                print(f"Basic Brand Cluster {i+1} ({len(cluster)} domains): {cluster}")
            
            return len(clusters), cluster_sizes, threshold
    
    print(f"\n❌ No suitable threshold found with basic analysis")
    return 0, [], None

def create_enhanced_clusters(similar_pairs, max_cluster_size=20):
    """Create enhanced clusters from similarity pairs"""
    from collections import defaultdict, deque
    
    # Build similarity graph
    graph = defaultdict(list)
    all_domains = set()
    
    for pair in similar_pairs:
        d1, d2 = pair['domain1'], pair['domain2']
        graph[d1].append(d2)
        graph[d2].append(d1)
        all_domains.add(d1)
        all_domains.add(d2)
    
    print(f"Building enhanced clusters from {len(all_domains)} domains with similarities...")
    
    # Find connected components using BFS
    visited = set()
    clusters = []
    
    for domain in all_domains:
        if domain not in visited:
            # BFS to find connected component
            component = []
            queue = deque([domain])
            visited.add(domain)
            
            while queue:
                current = queue.popleft()
                component.append(current)
                
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # Only keep clusters with multiple domains
            if len(component) >= 2:
                clusters.append(component)
    
    print(f"Found {len(clusters)} natural connected components")
    
    # Apply size constraints
    final_clusters = []
    
    # Sort clusters by size for analysis
    clusters.sort(key=len, reverse=True)
    cluster_sizes = [len(c) for c in clusters]
    print(f"Enhanced cluster sizes: {cluster_sizes[:15]}{'...' if len(cluster_sizes) > 15 else ''}")
    
    for cluster in clusters:
        if len(cluster) <= max_cluster_size:
            final_clusters.append(cluster)
        else:
            # Split large clusters (simplified split)
            print(f"Splitting large enhanced cluster of size {len(cluster)} (max: {max_cluster_size})")
            while cluster:
                chunk = cluster[:max_cluster_size]
                cluster = cluster[max_cluster_size:]
                if len(chunk) >= 2:
                    final_clusters.append(chunk)
    
    return final_clusters

if __name__ == "__main__":
    cluster_count, sizes, best_threshold = run_enhanced_ultra_restrictive_clustering()
    
    if best_threshold:
        print(f"\n🎉 ENHANCED SUCCESS: Found {cluster_count} enhanced clusters using threshold {best_threshold}")
    else:
        print(f"\n💔 No suitable clustering threshold found with enhanced analysis")
