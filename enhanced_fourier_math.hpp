#pragma once

#include <vector>
#include <string>
#include <complex>
#include <unordered_map>
#include <memory>

namespace FourierMath {

// Type definitions
using Complex = std::complex<double>;
using Matrix2D = std::vector<std::vector<double>>;
using ComplexMatrix2D = std::vector<std::vector<Complex>>;
using FeatureVector = std::vector<double>;

// Comprehensive similarity metrics structure matching Python logic
struct SimilarityMetrics {
    // Deep hashing metrics (NEW)
    double calibrated_similarity;
    bool calibrated_similar;
    double fused_hash_distance;
    bool fused_hash_similar;
    double confidence_score;
    
    // Traditional metrics (EXISTING)
    double phash_distance;
    bool phash_similar;
    double fft_similarity;
    bool fft_similar;
    double fmt_similarity;
    bool fmt_similar;
    double color_fmt_similarity;
    bool color_fmt_similar;
    double saliency_fft_similarity;
    bool saliency_fft_similar;
    double hu_similarity;
    bool hu_similar;
    double zernike_similarity;
    bool zernike_similar;
    double sift_similarity;
    bool sift_similar;
    double orb_similarity;
    bool orb_similar;
    bool overall_similar;
};

// Structure for comprehensive feature extraction results
struct ComprehensiveFeatures {
    std::string phash;
    FeatureVector fft_features;
    FeatureVector fmt_signature;
    FeatureVector color_fmt_features;
    FeatureVector saliency_fft_features;
    FeatureVector hu_moments;
    FeatureVector zernike_moments;
    FeatureVector sift_signature;
    FeatureVector orb_signature;
    FeatureVector texture_features;
    FeatureVector color_features;
    bool valid;
};

// Enhanced Fourier analyzer with all 2025 research features
class EnhancedFourierAnalyzer {
public:
    EnhancedFourierAnalyzer(int image_size = 128);
    
    // Comprehensive feature extraction
    ComprehensiveFeatures extract_all_features(const Matrix2D& image);
    
    // Individual feature extractors
    std::string compute_phash(const Matrix2D& image);
    FeatureVector compute_fft_features(const Matrix2D& image);
    FeatureVector compute_fourier_mellin_signature(const Matrix2D& image);
    FeatureVector compute_color_aware_fmt(const Matrix2D& image);
    FeatureVector compute_saliency_weighted_fft(const Matrix2D& image);
    FeatureVector compute_hu_moments(const Matrix2D& image);
    FeatureVector compute_zernike_moments(const Matrix2D& image, int max_order = 8);
    FeatureVector compute_sift_signature(const Matrix2D& image);
    FeatureVector compute_orb_signature(const Matrix2D& image);
    FeatureVector compute_texture_features(const Matrix2D& image);
    FeatureVector compute_color_features(const Matrix2D& image);
    
private:
    int image_size_;
    std::vector<Complex> twiddle_factors_;
    
    void precompute_twiddle_factors(int size);
    ComplexMatrix2D fft2d(const Matrix2D& input);
    Matrix2D compute_saliency_map(const Matrix2D& image);
    Matrix2D apply_gaussian_blur(const Matrix2D& image, double sigma);
    Matrix2D detect_edges(const Matrix2D& image);
};

// Enhanced similarity computer with comprehensive metrics
class EnhancedSimilarityComputer {
public:
    EnhancedSimilarityComputer();
    
    // Main comparison function matching Python logic
    SimilarityMetrics compute_comprehensive_similarity(
        const ComprehensiveFeatures& features1,
        const ComprehensiveFeatures& features2
    );
    
    // Individual similarity functions
    double hamming_distance(const std::string& hash1, const std::string& hash2);
    double cosine_similarity(const FeatureVector& a, const FeatureVector& b);
    double fourier_mellin_similarity(const FeatureVector& sig1, const FeatureVector& sig2);
    double hu_moments_similarity(const FeatureVector& hu1, const FeatureVector& hu2);
    double zernike_similarity(const FeatureVector& z1, const FeatureVector& z2);
    double sift_matching_score(const FeatureVector& sift1, const FeatureVector& sift2);
    double orb_matching_score(const FeatureVector& orb1, const FeatureVector& orb2);
    
    // Calibrated similarity with confidence scoring
    std::pair<double, double> compute_calibrated_similarity(
        const ComprehensiveFeatures& features1,
        const ComprehensiveFeatures& features2
    );
    
    // Fused hash distance combining multiple hash types
    double compute_fused_hash_distance(
        const ComprehensiveFeatures& features1,
        const ComprehensiveFeatures& features2
    );
    
private:
    // Similarity thresholds matching Python configuration
    struct Thresholds {
        double phash_threshold = 6.0;
        double fft_threshold = 0.985;
        double fmt_threshold = 0.995;
        double hu_threshold = 0.95;
        double zernike_threshold = 0.92;
        double sift_threshold = 0.3;
        double orb_threshold = 0.4;
        double overall_threshold = 0.7;
    } thresholds_;
    
    // Calibration parameters for confidence scoring
    struct CalibrationParams {
        double weight_phash = 0.15;
        double weight_fft = 0.20;
        double weight_fmt = 0.25;
        double weight_color_fmt = 0.15;
        double weight_saliency = 0.10;
        double weight_hu = 0.05;
        double weight_zernike = 0.05;
        double weight_sift = 0.03;
        double weight_orb = 0.02;
    } calibration_;
};

// Enhanced logo analysis pipeline with comprehensive metrics
class EnhancedLogoAnalysisPipeline {
public:
    EnhancedLogoAnalysisPipeline(int image_size = 128);
    
    // Comprehensive analysis results
    struct ComprehensiveAnalysisResults {
        std::vector<ComprehensiveFeatures> features;
        std::vector<std::vector<SimilarityMetrics>> similarity_metrics;
        Matrix2D similarity_matrix;
        std::vector<std::vector<int>> clusters;
        std::vector<double> cluster_scores;
        double analysis_time_ms;
        int valid_images;
    };
    
    // Main analysis functions
    ComprehensiveAnalysisResults analyze_logo_batch_comprehensive(
        const std::vector<Matrix2D>& images,
        double threshold = 0.7
    );
    
    std::vector<std::vector<SimilarityMetrics>> compute_pairwise_metrics(
        const std::vector<ComprehensiveFeatures>& features
    );
    
    Matrix2D extract_similarity_matrix(
        const std::vector<std::vector<SimilarityMetrics>>& metrics
    );
    
private:
    std::unique_ptr<EnhancedFourierAnalyzer> analyzer_;
    std::unique_ptr<EnhancedSimilarityComputer> similarity_computer_;
    int image_size_;
};

// Utility functions
namespace Utils {
    Matrix2D normalize_image(const Matrix2D& image);
    bool is_valid_logo_image(const Matrix2D& image);
    Matrix2D resize_image(const Matrix2D& image, int target_size);
    Matrix2D convert_to_grayscale(const Matrix2D& image);
    std::vector<double> extract_statistical_moments(const Matrix2D& image);
    
    // Color space conversions
    std::vector<Matrix2D> split_color_channels(const Matrix2D& image);
    Matrix2D rgb_to_hsv(const Matrix2D& rgb_image);
    Matrix2D apply_otsu_threshold(const Matrix2D& grayscale);
    
    // Mathematical utilities
    double compute_entropy(const std::vector<double>& histogram);
    double compute_correlation(const FeatureVector& a, const FeatureVector& b);
    std::vector<double> compute_histogram(const Matrix2D& image, int bins = 256);
}

} // namespace FourierMath
