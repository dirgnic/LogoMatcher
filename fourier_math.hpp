#ifndef FOURIER_MATH_HPP
#define FOURIER_MATH_HPP

#include <vector>
#include <complex>
#include <cmath>
#include <thread>
#include <future>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <functional>
#include <unordered_map>
#include <type_traits>
#include <memory>

/**
 * High-Performance Fourier Mathematics Library for Logo Analysis
 * Optimized for MacBook Pro 2024 (Apple Silicon M3 Pro/Max)
 * 
 * This C++ module handles computationally intensive Fourier transformations,
 * feature extraction, and similarity calculations for logo analysis.
 * 
 * Key optimizations:
 * - SIMD vectorization for Apple Silicon
 * - Cache-friendly memory access patterns  
 * - Multi-threading for batch operations
 * - Specialized algorithms for logo image analysis
 */

namespace FourierMath {
    
    // Type definitions for clarity and performance
    using Complex = std::complex<double>;
    using Matrix2D = std::vector<std::vector<double>>;
    using ComplexMatrix2D = std::vector<std::vector<Complex>>;
    using FeatureVector = std::vector<double>;
    
    /**
     * Core Fourier Transform Operations
     */
    class FourierAnalyzer {
    private:
        int image_size_;
        std::vector<Complex> twiddle_factors_;
        
        // Pre-computed lookup tables for performance
        void precompute_twiddle_factors(int size);
        
    public:
        explicit FourierAnalyzer(int image_size = 128);
        
        // 2D FFT with optimizations for logo analysis
        ComplexMatrix2D fft2d(const Matrix2D& input);
        ComplexMatrix2D ifft2d(const ComplexMatrix2D& input);
        
        // Optimized 1D FFT implementations
        std::vector<Complex> fft1d(const std::vector<Complex>& input);
        std::vector<Complex> ifft1d(const std::vector<Complex>& input);
    };
    
    /**
     * Logo Feature Extraction
     * Specialized for logo similarity analysis
     */
    class LogoFeatureExtractor {
    private:
        FourierAnalyzer fft_analyzer_;
        int hash_size_;
        
    public:
        explicit LogoFeatureExtractor(int image_size = 128, int hash_size = 8);
        
        // Perceptual hash using DCT (Discrete Cosine Transform)
        std::string compute_perceptual_hash(const Matrix2D& image);
        
        // FFT-based frequency signature
        FeatureVector compute_fft_signature(const Matrix2D& image);
        
        // Fourier-Mellin transform for rotation/scale invariance
        FeatureVector compute_fourier_mellin(const Matrix2D& image);
        
        // Comprehensive feature vector combining multiple methods
        FeatureVector extract_comprehensive_features(const Matrix2D& image);
        
        // Log-polar coordinate transformation for rotation invariance
        Matrix2D log_polar_transform(const Matrix2D& image);
    };
    
    /**
     * High-Performance Similarity Analysis
     */
    class SimilarityComputer {
    private:
        double cosine_similarity(const FeatureVector& a, const FeatureVector& b);
        double euclidean_distance(const FeatureVector& a, const FeatureVector& b);
        double hamming_distance(const std::string& hash1, const std::string& hash2);
        
    public:
        // Multi-method similarity fusion
        double compute_comprehensive_similarity(
            const FeatureVector& features1,
            const FeatureVector& features2,
            const std::string& hash1,
            const std::string& hash2
        );
        
        // Batch similarity matrix computation (optimized for large datasets)
        Matrix2D compute_similarity_matrix(
            const std::vector<FeatureVector>& feature_vectors,
            const std::vector<std::string>& hashes
        );
        
        // Smart bucketing for performance optimization
        std::vector<std::vector<int>> create_similarity_buckets(
            const std::vector<std::string>& hashes,
            int bucket_threshold = 2
        );
    };
    
    /**
     * Clustering Algorithms
     */
    class LogoClusterer {
    private:
        // Union-Find data structure for efficient clustering
        class UnionFind {
        private:
            std::vector<int> parent_;
            std::vector<int> rank_;
            
        public:
            explicit UnionFind(int size);
            int find(int x);
            void unite(int x, int y);
            std::vector<std::vector<int>> get_clusters();
        };
        
    public:
        // Threshold-based clustering using Union-Find
        std::vector<std::vector<int>> cluster_by_threshold(
            const Matrix2D& similarity_matrix,
            double threshold
        );
        
        // Adaptive clustering with automatic threshold selection
        std::vector<std::vector<int>> adaptive_clustering(
            const Matrix2D& similarity_matrix,
            int min_cluster_size = 2
        );
        
        // Hierarchical clustering for detailed analysis
        std::vector<std::vector<int>> hierarchical_clustering(
            const Matrix2D& similarity_matrix,
            int max_clusters
        );
    };
    
    /**
     * Performance Optimization Utilities with Enhanced Threading
     */
    class PerformanceOptimizer {
    public:
        // Cache-friendly matrix operations
        static Matrix2D transpose_optimized(const Matrix2D& matrix);
        
        // SIMD-optimized vector operations (Apple Silicon specific)
        static double dot_product_simd(const FeatureVector& a, const FeatureVector& b);
        
        // Enhanced thread pool for batch processing
        class ThreadPool {
        private:
            std::vector<std::thread> workers_;
            std::queue<std::function<void()>> tasks_;
            std::mutex queue_mutex_;
            std::condition_variable condition_;
            bool stop_;
            
        public:
            explicit ThreadPool(int num_threads = 0);  // 0 = auto-detect
            ~ThreadPool();
            
            template<typename F, typename... Args>
            auto enqueue(F&& f, Args&&... args) 
                -> std::future<std::invoke_result_t<F, Args...>> {
                
                using return_type = std::invoke_result_t<F, Args...>;
                
                auto task = std::make_shared<std::packaged_task<return_type()>>(
                    std::bind(std::forward<F>(f), std::forward<Args>(args)...)
                );
                
                std::future<return_type> res = task->get_future();
                
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    
                    if (stop_) {
                        throw std::runtime_error("enqueue on stopped ThreadPool");
                    }
                    
                    tasks_.emplace([task]() { (*task)(); });
                }
                
                condition_.notify_one();
                return res;
            }
            
            void wait_all();
            size_t get_thread_count() const { return workers_.size(); }
        };
        
        // Multi-threaded batch processing with work stealing
        static std::vector<FeatureVector> batch_feature_extraction(
            const std::vector<Matrix2D>& images,
            int num_threads = 0  // 0 = auto-detect cores
        );
        
        // Concurrent similarity matrix computation with optimized memory access
        static Matrix2D compute_similarity_matrix_concurrent(
            const std::vector<FeatureVector>& features,
            const std::vector<std::string>& hashes,
            int num_threads = 0
        );
        
        // Memory pool for frequent allocations (thread-safe)
        class MemoryPool {
        private:
            std::vector<char> pool_;
            size_t current_offset_;
            std::mutex pool_mutex_;
            
        public:
            explicit MemoryPool(size_t size);
            void* allocate(size_t size);
            void reset();
            bool is_thread_safe() const { return true; }
        };
        
        // Apple Silicon specific optimizations
        #ifdef __APPLE__
        static void configure_dispatch_queues();
        static void use_grand_central_dispatch(bool enable = true);
        #endif
    };
    
    /**
     * Main Pipeline Interface
     * High-level interface for Python integration
     */
    class LogoAnalysisPipeline {
    private:
        LogoFeatureExtractor extractor_;
        SimilarityComputer similarity_;
        LogoClusterer clusterer_;
        PerformanceOptimizer::MemoryPool memory_pool_;
        
    public:
        explicit LogoAnalysisPipeline(int image_size = 128);
        
        // Process single image and return features
        struct LogoFeatures {
            FeatureVector comprehensive_features;
            std::string perceptual_hash;
            FeatureVector fft_signature;
            FeatureVector fourier_mellin;
            bool is_valid;
        };
        
        LogoFeatures analyze_single_logo(const Matrix2D& image);
        
        // Batch processing for multiple logos
        std::vector<LogoFeatures> analyze_logo_batch(
            const std::vector<Matrix2D>& images
        );
        
        // Complete similarity analysis with clustering
        struct AnalysisResults {
            Matrix2D similarity_matrix;
            std::vector<std::vector<int>> clusters;
            std::vector<double> similarity_scores;
            double processing_time_ms;
        };
        
        AnalysisResults compute_comprehensive_analysis(
            const std::vector<Matrix2D>& images,
            double similarity_threshold = 0.45
        );
        
        // Performance statistics
        struct PerformanceStats {
            double total_time_ms;
            double extraction_time_ms;
            double similarity_time_ms;
            double clustering_time_ms;
            int processed_images;
            double images_per_second;
        };
        
        PerformanceStats get_performance_stats() const;
    };
    
    /**
     * Utility Functions
     */
    namespace Utils {
        // Image preprocessing utilities
        Matrix2D normalize_image(const Matrix2D& image);
        Matrix2D resize_image(const Matrix2D& image, int new_width, int new_height);
        Matrix2D apply_gaussian_blur(const Matrix2D& image, double sigma);
        
        // Mathematical utilities
        double calculate_entropy(const Matrix2D& image);
        Matrix2D calculate_gradient_magnitude(const Matrix2D& image);
        
        // Validation utilities
        bool is_valid_logo_image(const Matrix2D& image);
        double calculate_image_quality_score(const Matrix2D& image);
    }
}

#endif // FOURIER_MATH_HPP
