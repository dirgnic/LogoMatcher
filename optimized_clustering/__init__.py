"""
Optimized Logo Clustering Module

A modular, maintainable implementation of the logo clustering pipeline.
"""

from .brand_intelligence import BrandIntelligence
from .visual_analyzer import OptimizedVisualAnalyzer
from .hashing import OptimizedMultiScaleHasher
from .feature_extractor import FeatureExtractor
from .clustering_engine import ClusteringEngine
from .clusterer import OptimizedLogoClusterer

__all__ = [
    'BrandIntelligence',
    'OptimizedVisualAnalyzer',
    'OptimizedMultiScaleHasher',
    'FeatureExtractor',
    'ClusteringEngine',
    'OptimizedLogoClusterer'
]

__version__ = '1.0.0'
