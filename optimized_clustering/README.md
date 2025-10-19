# Optimized Logo Clustering Module

A modular, maintainable implementation of the logo clustering pipeline.

## Architecture

The codebase is organized into clean, focused modules:

```
optimized_clustering/
 __init__.py              # Package initialization
 brand_intelligence.py    # Brand family & industry classification
 visual_analyzer.py       # Color palette & composition analysis
 hashing.py              # Perceptual hashing (pHash, DCT, FFT)
 feature_extractor.py    # Feature extraction orchestration
 clustering_engine.py    # Clustering algorithms & similarity
 clusterer.py            # Main pipeline orchestrator
 run_clustering.py       # Entry point script
 README.md              # This file
```

## Module Responsibilities

### 1. `brand_intelligence.py`
- Extracts brand family from domain names
- Classifies logos by industry (tech, finance, ecommerce, etc.)
- Provides brand pattern matching for major brands (Google, Microsoft, etc.)

### 2. `visual_analyzer.py`
- Fast color palette extraction using HSV histograms
- Logo composition analysis (text vs symbol, layout)
- Edge density computation for layout classification

### 3. `hashing.py`
- pHash computation with bucketing for fast candidate pruning
- DCT-based hashing for frequency domain analysis
- FFT-based hashing for pattern recognition

### 4. `feature_extractor.py`
- Orchestrates all feature extraction
- Combines visual, perceptual, and brand features
- Manages feature caching for performance
- Extracts: pHash, color histograms, ORB descriptors, composition, DCT/FFT hashes

### 5. `clustering_engine.py`
- Similarity computation between logo pairs
- Hierarchical clustering with candidate pruning
- Aggressive singleton merging
- pHash bucket-based optimization

### 6. `clusterer.py`
- Main pipeline orchestrator
- Loads JPEG files
- Coordinates parallel feature extraction
- Executes clustering
- Analyzes cluster quality
- Saves results (CSV + pickle)

### 7. `run_clustering.py`
- Command-line entry point
- Configurable thresholds
- Simple usage interface

## Usage

### Basic Usage

```python
from optimized_clustering import OptimizedLogoClusterer

# Initialize with JPEG folder
clusterer = OptimizedLogoClusterer('path/to/logos')

# Run clustering
results = clusterer.run_clustering()

# Access results
print(f"Clusters: {results['quality_metrics']['total_clusters']}")
print(f"Singleton rate: {results['quality_metrics']['singleton_rate']:.1f}%")
```

### Command Line

```bash
# Run with default folder
python -m optimized_clustering.run_clustering

# Run with custom folder
python -m optimized_clustering.run_clustering /path/to/logos
```

### Custom Thresholds

```python
custom_thresholds = {
    'phash': 20,   # Stricter (fewer bits difference allowed)
    'orb': 10,     # Stricter (more good matches required)
    'color': 0.60  # Stricter (higher similarity required)
}

clusterer = OptimizedLogoClusterer('path/to/logos', thresholds=custom_thresholds)
```

## Output

The clustering produces:

1. **CSV file** (`optimized_logo_clusters_TIMESTAMP.csv`)
   - cluster_id, domain, cluster_size

2. **Pickle file** (`optimized_logo_clustering_results_TIMESTAMP.pkl`)
   - Complete results including features, clusters, similarity matrix

## Features

- **Multi-scale feature extraction**: pHash, color, ORB, DCT, FFT
- **Brand intelligence**: Family and industry classification
- **Optimized clustering**: pHash bucketing for O(N) candidate pruning
- **Parallel processing**: Multi-threaded feature extraction
- **Aggressive merging**: Reduces singleton clusters
- **Quality metrics**: Singleton rate, brand coherence, size distribution

## Performance

- **Feature extraction**: ~10-20 files/sec (parallel)
- **Clustering**: ~1000+ comparisons/sec (optimized)
- **Memory**: Cached features for large datasets
- **Scalability**: pHash bucketing reduces O(N²) to ~O(N)

## Thresholds

Default ultra-relaxed thresholds for maximum merging:

- **pHash**: 60/64 bits (very permissive)
- **ORB**: 2 good matches (very permissive)
- **Color**: 0.10 similarity (very permissive)

Adjust thresholds based on your quality/quantity trade-off.

## Development

### Adding New Features

1. Add extraction logic to `feature_extractor.py`
2. Update similarity calculation in `clustering_engine.py`
3. Adjust weights in `calculate_similarity()`

### Testing Individual Modules

```python
# Test brand intelligence
from optimized_clustering.brand_intelligence import BrandIntelligence
bi = BrandIntelligence()
print(bi.extract_brand_family('google.com'))
print(bi.classify_industry('bank.com'))

# Test visual analyzer
from optimized_clustering.visual_analyzer import OptimizedVisualAnalyzer
import cv2
va = OptimizedVisualAnalyzer()
image = cv2.imread('logo.jpg')
palette = va.extract_color_palette(image)
composition = va.analyze_logo_composition(image)

# Test hashing
from optimized_clustering.hashing import OptimizedMultiScaleHasher
hasher = OptimizedMultiScaleHasher()
phash_data = hasher.compute_phash_with_bucketing(image)
```

## Advantages Over Monolithic Design

 **Separation of concerns** - Each module has a single responsibility  
 **Testability** - Easy to unit test individual components  
 **Maintainability** - Changes are localized to specific modules  
 **Reusability** - Components can be used independently  
 **Readability** - Smaller files are easier to understand  
 **Extensibility** - Easy to add new features or swap implementations  

## License

Part of the LogoMatcher project.
