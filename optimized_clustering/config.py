"""
Configuration for optimized logo clustering

Adjust these settings to control clustering behavior.
"""

# CLUSTERING THRESHOLDS 

# Ultra-relaxed thresholds (maximum merging, lower quality)
ULTRA_RELAXED_THRESHOLDS = {
    'phash': 60,   # Out of 64 bits - very permissive
    'orb': 2,      # Minimum good ORB matches - very permissive
    'color': 0.10  # Color histogram similarity - very permissive
}

# Relaxed thresholds (balanced merging)
RELAXED_THRESHOLDS = {
    'phash': 40,   # Out of 64 bits - permissive
    'orb': 5,      # Minimum good ORB matches - permissive
    'color': 0.30  # Color histogram similarity - permissive
}

# Moderate thresholds (quality over quantity)
MODERATE_THRESHOLDS = {
    'phash': 20,   # Out of 64 bits - moderate
    'orb': 10,     # Minimum good ORB matches - moderate
    'color': 0.60  # Color histogram similarity - moderate
}

# Strict thresholds (high quality, more singletons)
STRICT_THRESHOLDS = {
    'phash': 10,   # Out of 64 bits - strict
    'orb': 15,     # Minimum good ORB matches - strict
    'color': 0.75  # Color histogram similarity - strict
}

# Default thresholds to use
DEFAULT_THRESHOLDS = ULTRA_RELAXED_THRESHOLDS


# === FEATURE EXTRACTION SETTINGS ===

# Number of parallel workers for feature extraction
MAX_WORKERS = 4

# ORB detector parameters
ORB_CONFIG = {
    'nfeatures': 100,      # Number of keypoints to detect
    'scaleFactor': 1.2,    # Pyramid decimation ratio
    'nlevels': 8,          # Number of pyramid levels
    'edgeThreshold': 15,   # Border where features are not detected
    'patchSize': 31        # Size of patch used by oriented BRIEF
}

# Color histogram bins
COLOR_HIST_BINS = {
    'hue': 10,         # Hue bins (0-180)
    'saturation': 10,  # Saturation bins (0-256)
    'value': 10        # Value bins (0-256)
}

# pHash settings
PHASH_HASH_SIZE = 16  # 16x16 = 256-bit hash (reduced to 64-bit in practice)


# === CLUSTERING SETTINGS ===

# Hierarchical clustering method
LINKAGE_METHOD = 'single'  # Options: 'single', 'complete', 'average', 'ward'

# Distance threshold for clustering
CLUSTERING_THRESHOLD = 0.85  # Lower = more aggressive merging

# Maximum cluster size for singleton merging
MAX_CLUSTER_SIZE = 20


# PERFORMANCE SETTINGS 

# Batch size for processing
BATCH_SIZE = 50

# Enable feature caching
ENABLE_CACHING = True

# Bucket similarity tolerance (bits difference)
BUCKET_TOLERANCE = 2  # Check buckets within 2-bit hamming distance


# OUTPUT SETTINGS 

# Output filename suffix
OUTPUT_SUFFIX = "_modular"

# Save detailed similarity matrix
SAVE_SIMILARITY_MATRIX = True

# Verbose logging
VERBOSE = True


# === SIMILARITY WEIGHTS ===

# Feature weights for similarity calculation
FEATURE_WEIGHTS = {
    'phash': 0.5,   # Perceptual hash (primary)
    'color': 0.25,  # Color histogram (secondary)
    'orb': 0.15,    # ORB descriptors (tertiary)
    'dct': 0.1      # DCT hash (quaternary)
}

# Brand family bonus
BRAND_BONUS = 0.1

# Two-channel penalty multiplier
TWO_CHANNEL_PENALTY = 0.5  # Applied when no thresholds are met
