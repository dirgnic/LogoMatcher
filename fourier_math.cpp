#include "fourier_math.hpp"
#include <algorithm>
#include <thread>
#include <chrono>
#include <cstring>
#include <numeric>
#include <future>
#include <mutex>
#ifdef __APPLE__
#include <dispatch/dispatch.h>  // Apple's Grand Central Dispatch
#endif

/**
 * High-Performance Fourier Mathematics Implementation
 * Optimized for Apple Silicon M3 Pro/Max processors
 */

namespace FourierMath {

// =============================================================================
// FourierAnalyzer Implementation
// =============================================================================

FourierAnalyzer::FourierAnalyzer(int image_size) : image_size_(image_size) {
    precompute_twiddle_factors(image_size);
}

void FourierAnalyzer::precompute_twiddle_factors(int size) {
    twiddle_factors_.resize(size);
    for (int i = 0; i < size; ++i) {
        double angle = -2.0 * M_PI * i / size;
        twiddle_factors_[i] = Complex(cos(angle), sin(angle));
    }
}

ComplexMatrix2D FourierAnalyzer::fft2d(const Matrix2D& input) {
    int rows = input.size();
    int cols = input[0].size();
    
    // Initialize complex matrix
    ComplexMatrix2D result(rows, std::vector<Complex>(cols));
    
    // Convert input to complex
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = Complex(input[i][j], 0.0);
        }
    }
    
    // Apply FFT to rows
    for (int i = 0; i < rows; ++i) {
        result[i] = fft1d(result[i]);
    }
    
    // Transpose for column processing
    ComplexMatrix2D transposed(cols, std::vector<Complex>(rows));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            transposed[j][i] = result[i][j];
        }
    }
    
    // Apply FFT to columns (now rows after transpose)
    for (int i = 0; i < cols; ++i) {
        transposed[i] = fft1d(transposed[i]);
    }
    
    // Transpose back
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = transposed[j][i];
        }
    }
    
    return result;
}

std::vector<Complex> FourierAnalyzer::fft1d(const std::vector<Complex>& input) {
    int n = input.size();
    if (n <= 1) return input;
    
    // Cooley-Tukey FFT algorithm
    if (n % 2 != 0) {
        // Fallback to DFT for non-power-of-2 sizes
        std::vector<Complex> output(n);
        for (int k = 0; k < n; ++k) {
            output[k] = 0;
            for (int j = 0; j < n; ++j) {
                double angle = -2.0 * M_PI * k * j / n;
                Complex twiddle = Complex(cos(angle), sin(angle));
                output[k] += input[j] * twiddle;
            }
        }
        return output;
    }
    
    // Divide
    std::vector<Complex> even, odd;
    for (int i = 0; i < n; i += 2) {
        even.push_back(input[i]);
        odd.push_back(input[i + 1]);
    }
    
    // Conquer
    auto fft_even = fft1d(even);
    auto fft_odd = fft1d(odd);
    
    // Combine
    std::vector<Complex> output(n);
    for (int k = 0; k < n/2; ++k) {
        double angle = -2.0 * M_PI * k / n;
        Complex twiddle = Complex(cos(angle), sin(angle));
        Complex t = twiddle * fft_odd[k];
        
        output[k] = fft_even[k] + t;
        output[k + n/2] = fft_even[k] - t;
    }
    
    return output;
}

// =============================================================================
// LogoFeatureExtractor Implementation  
// =============================================================================

LogoFeatureExtractor::LogoFeatureExtractor(int image_size, int hash_size) 
    : fft_analyzer_(image_size), hash_size_(hash_size) {}

std::string LogoFeatureExtractor::compute_perceptual_hash(const Matrix2D& image) {
    int rows = image.size();
    int cols = image[0].size();
    
    // Resize to hash_size x hash_size
    Matrix2D resized(hash_size_, std::vector<double>(hash_size_));
    for (int i = 0; i < hash_size_; ++i) {
        for (int j = 0; j < hash_size_; ++j) {
            int src_i = (i * rows) / hash_size_;
            int src_j = (j * cols) / hash_size_;
            resized[i][j] = image[src_i][src_j];
        }
    }
    
    // Compute DCT (simplified implementation)
    Matrix2D dct(hash_size_, std::vector<double>(hash_size_));
    for (int u = 0; u < hash_size_; ++u) {
        for (int v = 0; v < hash_size_; ++v) {
            double sum = 0.0;
            for (int i = 0; i < hash_size_; ++i) {
                for (int j = 0; j < hash_size_; ++j) {
                    double cos_u = cos((2*i + 1) * u * M_PI / (2 * hash_size_));
                    double cos_v = cos((2*j + 1) * v * M_PI / (2 * hash_size_));
                    sum += resized[i][j] * cos_u * cos_v;
                }
            }
            dct[u][v] = sum;
        }
    }
    
    // Extract top-left coefficients (low frequencies)
    std::vector<double> low_freq;
    for (int i = 0; i < hash_size_-1; ++i) {
        for (int j = 0; j < hash_size_-1; ++j) {
            low_freq.push_back(dct[i][j]);
        }
    }
    
    // Calculate median
    std::vector<double> sorted_freq = low_freq;
    std::sort(sorted_freq.begin(), sorted_freq.end());
    double median = sorted_freq[sorted_freq.size() / 2];
    
    // Generate binary hash
    std::string hash;
    for (double val : low_freq) {
        hash += (val > median) ? '1' : '0';
    }
    
    return hash;
}

FeatureVector LogoFeatureExtractor::compute_fft_signature(const Matrix2D& image) {
    auto fft_result = fft_analyzer_.fft2d(image);
    
    int rows = fft_result.size();
    int cols = fft_result[0].size();
    
    // Compute magnitude spectrum
    Matrix2D magnitude(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            magnitude[i][j] = std::abs(fft_result[i][j]);
        }
    }
    
    // Extract low frequency components (center region)
    int center_i = rows / 2;
    int center_j = cols / 2;
    int radius = std::min(16, std::min(center_i, center_j));
    
    FeatureVector features;
    for (int i = center_i - radius; i < center_i + radius; ++i) {
        for (int j = center_j - radius; j < center_j + radius; ++j) {
            if (i >= 0 && i < rows && j >= 0 && j < cols) {
                features.push_back(magnitude[i][j]);
            }
        }
    }
    
    // Limit to 256 features
    if (features.size() > 256) {
        features.resize(256);
    }
    
    // Normalize
    double mean = std::accumulate(features.begin(), features.end(), 0.0) / features.size();
    double variance = 0.0;
    for (double val : features) {
        variance += (val - mean) * (val - mean);
    }
    double std_dev = sqrt(variance / features.size()) + 1e-8;
    
    for (double& val : features) {
        val = (val - mean) / std_dev;
    }
    
    return features;
}

FeatureVector LogoFeatureExtractor::compute_fourier_mellin(const Matrix2D& image) {
    // Apply log-polar transform
    auto log_polar = log_polar_transform(image);
    
    // Apply FFT to log-polar image
    auto fft_result = fft_analyzer_.fft2d(log_polar);
    
    int rows = fft_result.size();
    int cols = fft_result[0].size();
    
    // Extract magnitude features
    FeatureVector features;
    for (int i = 0; i < std::min(8, rows); i += 2) {
        for (int j = 0; j < std::min(8, cols); j += 2) {
            features.push_back(std::abs(fft_result[i][j]));
        }
    }
    
    // Pad or truncate to 64 features
    features.resize(64, 0.0);
    
    return features;
}

Matrix2D LogoFeatureExtractor::log_polar_transform(const Matrix2D& image) {
    int rows = image.size();
    int cols = image[0].size();
    
    int center_x = cols / 2;
    int center_y = rows / 2;
    double max_radius = std::min(center_x, center_y);
    
    Matrix2D log_polar(rows, std::vector<double>(cols, 0.0));
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Convert to polar coordinates
            double x = j - center_x;
            double y = i - center_y;
            double radius = sqrt(x*x + y*y);
            double angle = atan2(y, x);
            
            if (radius > 0 && radius < max_radius) {
                // Map to log-polar space
                double log_r = log(radius / max_radius + 1.0);
                double theta_norm = (angle + M_PI) / (2 * M_PI);
                
                // Map back to image coordinates
                int src_i = static_cast<int>(log_r * rows);
                int src_j = static_cast<int>(theta_norm * cols);
                
                if (src_i >= 0 && src_i < rows && src_j >= 0 && src_j < cols) {
                    log_polar[i][j] = image[src_i][src_j];
                }
            }
        }
    }
    
    return log_polar;
}

FeatureVector LogoFeatureExtractor::extract_comprehensive_features(const Matrix2D& image) {
    auto fft_features = compute_fft_signature(image);
    auto mellin_features = compute_fourier_mellin(image);
    
    // Combine features
    FeatureVector combined;
    combined.reserve(fft_features.size() + mellin_features.size());
    combined.insert(combined.end(), fft_features.begin(), fft_features.end());
    combined.insert(combined.end(), mellin_features.begin(), mellin_features.end());
    
    return combined;
}

// =============================================================================
// SimilarityComputer Implementation
// =============================================================================

double SimilarityComputer::cosine_similarity(const FeatureVector& a, const FeatureVector& b) {
    if (a.size() != b.size()) return 0.0;
    
    double dot_product = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;
    
    for (size_t i = 0; i < a.size(); ++i) {
        dot_product += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    double denominator = sqrt(norm_a) * sqrt(norm_b);
    return (denominator > 1e-8) ? dot_product / denominator : 0.0;
}

double SimilarityComputer::hamming_distance(const std::string& hash1, const std::string& hash2) {
    if (hash1.size() != hash2.size()) return 1.0;
    
    int different_bits = 0;
    for (size_t i = 0; i < hash1.size(); ++i) {
        if (hash1[i] != hash2[i]) {
            different_bits++;
        }
    }
    
    // Convert to similarity score (0 = identical, 1 = completely different)
    return 1.0 - static_cast<double>(different_bits) / hash1.size();
}

double SimilarityComputer::compute_comprehensive_similarity(
    const FeatureVector& features1,
    const FeatureVector& features2,
    const std::string& hash1,
    const std::string& hash2) {
    
    // Compute individual similarity scores
    double cosine_sim = cosine_similarity(features1, features2);
    double hash_sim = hamming_distance(hash1, hash2);
    
    // Weighted combination (adjust weights based on empirical performance)
    double combined_similarity = 0.7 * cosine_sim + 0.3 * hash_sim;
    
    return std::max(0.0, std::min(1.0, combined_similarity));
}

Matrix2D SimilarityComputer::compute_similarity_matrix(
    const std::vector<FeatureVector>& feature_vectors,
    const std::vector<std::string>& hashes) {
    
    int n = feature_vectors.size();
    Matrix2D similarity_matrix(n, std::vector<double>(n, 0.0));
    
    // Use thread pool for parallel processing
    const int num_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    std::vector<std::future<void>> futures;
    std::mutex matrix_mutex;
    
    // Calculate work distribution
    int work_per_thread = std::max(1, n / num_threads);
    
    for (int thread_id = 0; thread_id < num_threads; ++thread_id) {
        int start_i = thread_id * work_per_thread;
        int end_i = (thread_id == num_threads - 1) ? n : std::min(n, start_i + work_per_thread);
        
        if (start_i >= n) break;
        
        futures.emplace_back(std::async(std::launch::async, [&, start_i, end_i]() {
            // Each thread works on a range of rows
            for (int i = start_i; i < end_i; ++i) {
                for (int j = i; j < n; ++j) {
                    double sim;
                    if (i == j) {
                        sim = 1.0;
                    } else {
                        sim = compute_comprehensive_similarity(
                            feature_vectors[i], feature_vectors[j],
                            hashes[i], hashes[j]
                        );
                    }
                    
                    // Thread-safe matrix update
                    std::lock_guard<std::mutex> lock(matrix_mutex);
                    similarity_matrix[i][j] = sim;
                    if (i != j) {
                        similarity_matrix[j][i] = sim;  // Matrix is symmetric
                    }
                }
            }
        }));
    }
    
    // Wait for all threads to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    return similarity_matrix;
}

// =============================================================================
// LogoClusterer Implementation
// =============================================================================

LogoClusterer::UnionFind::UnionFind(int size) : parent_(size), rank_(size, 0) {
    std::iota(parent_.begin(), parent_.end(), 0);
}

int LogoClusterer::UnionFind::find(int x) {
    if (parent_[x] != x) {
        parent_[x] = find(parent_[x]);  // Path compression
    }
    return parent_[x];
}

void LogoClusterer::UnionFind::unite(int x, int y) {
    int root_x = find(x);
    int root_y = find(y);
    
    if (root_x != root_y) {
        // Union by rank
        if (rank_[root_x] < rank_[root_y]) {
            parent_[root_x] = root_y;
        } else if (rank_[root_x] > rank_[root_y]) {
            parent_[root_y] = root_x;
        } else {
            parent_[root_y] = root_x;
            rank_[root_x]++;
        }
    }
}

std::vector<std::vector<int>> LogoClusterer::UnionFind::get_clusters() {
    std::unordered_map<int, std::vector<int>> clusters_map;
    
    for (int i = 0; i < parent_.size(); ++i) {
        int root = find(i);
        clusters_map[root].push_back(i);
    }
    
    std::vector<std::vector<int>> clusters;
    for (const auto& pair : clusters_map) {
        if (pair.second.size() > 1) {  // Only clusters with multiple elements
            clusters.push_back(pair.second);
        }
    }
    
    return clusters;
}

std::vector<std::vector<int>> LogoClusterer::cluster_by_threshold(
    const Matrix2D& similarity_matrix,
    double threshold) {
    
    int n = similarity_matrix.size();
    UnionFind uf(n);
    
    // Connect similar pairs
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (similarity_matrix[i][j] >= threshold) {
                uf.unite(i, j);
            }
        }
    }
    
    return uf.get_clusters();
}

// =============================================================================
// LogoAnalysisPipeline Implementation
// =============================================================================

LogoAnalysisPipeline::LogoAnalysisPipeline(int image_size) 
    : extractor_(image_size), memory_pool_(1024 * 1024) {}  // 1MB pool

LogoAnalysisPipeline::LogoFeatures 
LogoAnalysisPipeline::analyze_single_logo(const Matrix2D& image) {
    LogoFeatures features;
    
    try {
        features.perceptual_hash = extractor_.compute_perceptual_hash(image);
        features.comprehensive_features = extractor_.extract_comprehensive_features(image);
        features.fft_signature = extractor_.compute_fft_signature(image);
        features.fourier_mellin = extractor_.compute_fourier_mellin(image);
        features.is_valid = !features.perceptual_hash.empty() && 
                          !features.comprehensive_features.empty();
    } catch (...) {
        features.is_valid = false;
    }
    
    return features;
}

std::vector<LogoAnalysisPipeline::LogoFeatures>
LogoAnalysisPipeline::analyze_logo_batch(const std::vector<Matrix2D>& images) {
    std::vector<LogoFeatures> results(images.size());
    
    // Enhanced parallel processing with thread pool
    const int num_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    std::vector<std::future<void>> futures;
    
    // Calculate optimal batch size per thread
    size_t batch_size = std::max(size_t(1), images.size() / num_threads);
    
    for (size_t thread_id = 0; thread_id < num_threads && thread_id * batch_size < images.size(); ++thread_id) {
        size_t start_idx = thread_id * batch_size;
        size_t end_idx = std::min(images.size(), start_idx + batch_size);
        
        futures.emplace_back(std::async(std::launch::async, [&, start_idx, end_idx]() {
            for (size_t i = start_idx; i < end_idx; ++i) {
                results[i] = analyze_single_logo(images[i]);
            }
        }));
    }
    
    // Wait for all threads to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    return results;
}

LogoAnalysisPipeline::AnalysisResults
LogoAnalysisPipeline::compute_comprehensive_analysis(
    const std::vector<Matrix2D>& images,
    double similarity_threshold) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Extract features
    auto features_list = analyze_logo_batch(images);
    
    // Filter valid features
    std::vector<FeatureVector> valid_features;
    std::vector<std::string> valid_hashes;
    
    for (const auto& features : features_list) {
        if (features.is_valid) {
            valid_features.push_back(features.comprehensive_features);
            valid_hashes.push_back(features.perceptual_hash);
        }
    }
    
    AnalysisResults results;
    
    if (valid_features.size() >= 2) {
        // Compute similarity matrix
        results.similarity_matrix = similarity_.compute_similarity_matrix(
            valid_features, valid_hashes
        );
        
        // Perform clustering
        results.clusters = clusterer_.cluster_by_threshold(
            results.similarity_matrix, similarity_threshold
        );
        
        // Extract similarity scores
        for (int i = 0; i < results.similarity_matrix.size(); ++i) {
            for (int j = i + 1; j < results.similarity_matrix[i].size(); ++j) {
                if (results.similarity_matrix[i][j] >= similarity_threshold) {
                    results.similarity_scores.push_back(results.similarity_matrix[i][j]);
                }
            }
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    results.processing_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time
    ).count();
    results.final_threshold_used = similarity_threshold;
    results.threshold_was_adjusted = false;
    
    return results;
}

LogoAnalysisPipeline::AnalysisResults
LogoAnalysisPipeline::compute_adaptive_threshold_analysis(
    const std::vector<Matrix2D>& images,
    const std::vector<double>& initial_thresholds,
    double sample_percentage,
    int min_sample_size) {
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Extract features for all images
    auto features_list = analyze_logo_batch(images);
    
    // Filter valid features
    std::vector<FeatureVector> valid_features;
    std::vector<std::string> valid_hashes;
    
    for (const auto& features : features_list) {
        if (features.is_valid) {
            valid_features.push_back(features.comprehensive_features);
            valid_hashes.push_back(features.perceptual_hash);
        }
    }
    
    AnalysisResults results;
    results.threshold_was_adjusted = false;
    
    if (valid_features.size() < 2) {
        results.processing_time_ms = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start_time
        ).count();
        return results;
    }
    
    // Determine sample size - at least min_sample_size, at most sample_percentage of total
    int sample_size = std::max(min_sample_size, 
                              static_cast<int>(valid_features.size() * sample_percentage));
    sample_size = std::min(sample_size, static_cast<int>(valid_features.size()));
    
    // Try each initial threshold
    for (double threshold : initial_thresholds) {
        // Test with sample first
        int sample_pairs_found = 0;
        int total_sample_comparisons = 0;
        
        // Sample the first sample_size features
        for (int i = 0; i < sample_size; ++i) {
            for (int j = i + 1; j < sample_size; ++j) {
                double similarity = similarity_.compute_comprehensive_similarity(
                    valid_features[i], valid_features[j],
                    valid_hashes[i], valid_hashes[j]
                );
                
                total_sample_comparisons++;
                if (similarity >= threshold) {
                    sample_pairs_found++;
                }
            }
        }
        
        // If no pairs found with highest threshold, try progressively lower ones
        if (sample_pairs_found == 0 && threshold == initial_thresholds[0]) {
            std::vector<double> fallback_thresholds = {0.80, 0.75, 0.70, 0.65, 0.60};
            
            for (double test_threshold : fallback_thresholds) {
                int test_pairs = 0;
                
                // Test sample again with lower threshold
                for (int i = 0; i < sample_size; ++i) {
                    for (int j = i + 1; j < sample_size; ++j) {
                        double similarity = similarity_.compute_comprehensive_similarity(
                            valid_features[i], valid_features[j],
                            valid_hashes[i], valid_hashes[j]
                        );
                        
                        if (similarity >= test_threshold) {
                            test_pairs++;
                        }
                    }
                }
                
                if (test_pairs > 0) {
                    threshold = test_threshold;
                    results.threshold_was_adjusted = true;
                    break;
                }
            }
        }
        
        // Now run full analysis with the (possibly adjusted) threshold
        results.similarity_matrix = similarity_.compute_similarity_matrix(
            valid_features, valid_hashes
        );
        
        // Perform clustering
        results.clusters = clusterer_.cluster_by_threshold(
            results.similarity_matrix, threshold
        );
        
        // Extract similarity scores above threshold
        for (int i = 0; i < results.similarity_matrix.size(); ++i) {
            for (int j = i + 1; j < results.similarity_matrix[i].size(); ++j) {
                if (results.similarity_matrix[i][j] >= threshold) {
                    results.similarity_scores.push_back(results.similarity_matrix[i][j]);
                }
            }
        }
        
        results.final_threshold_used = threshold;
        
        // If we found reasonable results, stop trying higher thresholds
        if (!results.clusters.empty()) {
            break;
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    results.processing_time_ms = std::chrono::duration<double, std::milli>(
        end_time - start_time
    ).count();
    
    return results;
}

// =============================================================================
// Enhanced ThreadPool Implementation
// =============================================================================

PerformanceOptimizer::ThreadPool::ThreadPool(int num_threads) : stop_(false) {
    if (num_threads <= 0) {
        num_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }
    
    // Initialize worker threads
    for (int i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            for (;;) {
                std::function<void()> task;
                
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    
                    if (stop_ && tasks_.empty()) {
                        return;
                    }
                    
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                
                task();
            }
        });
    }
}

PerformanceOptimizer::ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    
    condition_.notify_all();
    
    for (std::thread& worker : workers_) {
        worker.join();
    }
}



void PerformanceOptimizer::ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    condition_.wait(lock, [this] { return tasks_.empty(); });
}

// =============================================================================
// Enhanced Performance Optimizer Functions  
// =============================================================================

std::vector<FeatureVector> PerformanceOptimizer::batch_feature_extraction(
    const std::vector<Matrix2D>& images,
    int num_threads) {
    
    std::vector<FeatureVector> results(images.size());
    
    if (num_threads <= 0) {
        num_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }
    
    ThreadPool pool(num_threads);
    std::vector<std::future<void>> futures;
    
    // Calculate work distribution
    size_t chunk_size = std::max(size_t(1), images.size() / num_threads);
    
    for (size_t i = 0; i < images.size(); i += chunk_size) {
        size_t end = std::min(i + chunk_size, images.size());
        
        futures.emplace_back(pool.enqueue([&images, &results, i, end]() {
            LogoFeatureExtractor extractor;
            
            for (size_t j = i; j < end; ++j) {
                results[j] = extractor.extract_comprehensive_features(images[j]);
            }
        }));
    }
    
    // Wait for all chunks to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    return results;
}

Matrix2D PerformanceOptimizer::compute_similarity_matrix_concurrent(
    const std::vector<FeatureVector>& features,
    const std::vector<std::string>& hashes,
    int num_threads) {
    
    int n = features.size();
    Matrix2D similarity_matrix(n, std::vector<double>(n, 0.0));
    
    if (num_threads <= 0) {
        num_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    }
    
    ThreadPool pool(num_threads);
    std::vector<std::future<void>> futures;
    std::mutex matrix_mutex;
    
    // Work-stealing approach: each thread processes a block of the upper triangle
    int block_size = std::max(1, n / (num_threads * 2));
    
    for (int start_i = 0; start_i < n; start_i += block_size) {
        int end_i = std::min(n, start_i + block_size);
        
        futures.emplace_back(pool.enqueue([&, start_i, end_i]() {
            SimilarityComputer computer;
            
            for (int i = start_i; i < end_i; ++i) {
                for (int j = i; j < n; ++j) {
                    double sim;
                    if (i == j) {
                        sim = 1.0;
                    } else {
                        sim = computer.compute_comprehensive_similarity(
                            features[i], features[j], hashes[i], hashes[j]
                        );
                    }
                    
                    // Update matrix (thread-safe for different rows)
                    similarity_matrix[i][j] = sim;
                    if (i != j) {
                        similarity_matrix[j][i] = sim;
                    }
                }
            }
        }));
    }
    
    // Wait for all blocks to complete
    for (auto& future : futures) {
        future.wait();
    }
    
    return similarity_matrix;
}

#ifdef __APPLE__
void PerformanceOptimizer::configure_dispatch_queues() {
    // Configure Grand Central Dispatch for optimal performance on Apple Silicon
    dispatch_queue_attr_t attr = dispatch_queue_attr_make_with_qos_class(
        DISPATCH_QUEUE_CONCURRENT, QOS_CLASS_USER_INITIATED, 0
    );
    
    // Create high-priority concurrent queue for compute tasks
    static dispatch_queue_t compute_queue = dispatch_queue_create(
        "com.logoanalysis.compute", attr
    );
}

void PerformanceOptimizer::use_grand_central_dispatch(bool enable) {
    // This would be used to switch between std::thread and GCD
    static bool gcd_enabled = enable;
}
#endif

// =============================================================================
// Thread-Safe Memory Pool Implementation
// =============================================================================

PerformanceOptimizer::MemoryPool::MemoryPool(size_t size) 
    : pool_(size), current_offset_(0) {}

void* PerformanceOptimizer::MemoryPool::allocate(size_t size) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    
    if (current_offset_ + size > pool_.size()) {
        // Pool exhausted, could expand or throw
        throw std::bad_alloc();
    }
    
    void* ptr = &pool_[current_offset_];
    current_offset_ += size;
    
    return ptr;
}

void PerformanceOptimizer::MemoryPool::reset() {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    current_offset_ = 0;
}

// =============================================================================
// Utility Functions
// =============================================================================

namespace Utils {
    
Matrix2D normalize_image(const Matrix2D& image) {
    int rows = image.size();
    int cols = image[0].size();
    
    // Find min and max values
    double min_val = image[0][0];
    double max_val = image[0][0];
    
    for (const auto& row : image) {
        for (double val : row) {
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
    }
    
    // Normalize to [0, 1]
    Matrix2D normalized(rows, std::vector<double>(cols));
    double range = max_val - min_val;
    
    if (range > 1e-8) {
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                normalized[i][j] = (image[i][j] - min_val) / range;
            }
        }
    } else {
        // Constant image
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                normalized[i][j] = 0.5;
            }
        }
    }
    
    return normalized;
}

bool is_valid_logo_image(const Matrix2D& image) {
    if (image.empty() || image[0].empty()) return false;
    if (image.size() < 8 || image[0].size() < 8) return false;
    
    // Check for reasonable variance (not blank or constant)
    double sum = 0.0;
    double sum_sq = 0.0;
    int count = 0;
    
    for (const auto& row : image) {
        for (double val : row) {
            sum += val;
            sum_sq += val * val;
            count++;
        }
    }
    
    double mean = sum / count;
    double variance = (sum_sq / count) - (mean * mean);
    
    return variance > 1e-6;  // Must have some variation
}

}  // namespace Utils

}  // namespace FourierMath
