#include "enhanced_fourier_math.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <thread>
#include <chrono>
#include <random>

namespace FourierMath {

// =============================================================================
// EnhancedFourierAnalyzer Implementation
// =============================================================================

EnhancedFourierAnalyzer::EnhancedFourierAnalyzer(int image_size) : image_size_(image_size) {
    precompute_twiddle_factors(image_size);
}

void EnhancedFourierAnalyzer::precompute_twiddle_factors(int size) {
    twiddle_factors_.resize(size);
    for (int i = 0; i < size; ++i) {
        double angle = -2.0 * M_PI * i / size;
        twiddle_factors_[i] = Complex(cos(angle), sin(angle));
    }
}

ComprehensiveFeatures EnhancedFourierAnalyzer::extract_all_features(const Matrix2D& image) {
    ComprehensiveFeatures features;
    
    try {
        // Normalize image first
        Matrix2D normalized = Utils::normalize_image(image);
        
        // Extract all feature types
        features.phash = compute_phash(normalized);
        features.fft_features = compute_fft_features(normalized);
        features.fmt_signature = compute_fourier_mellin_signature(normalized);
        features.color_fmt_features = compute_color_aware_fmt(image); // Use original for color
        features.saliency_fft_features = compute_saliency_weighted_fft(normalized);
        features.hu_moments = compute_hu_moments(normalized);
        features.zernike_moments = compute_zernike_moments(normalized);
        features.sift_signature = compute_sift_signature(normalized);
        features.orb_signature = compute_orb_signature(normalized);
        features.texture_features = compute_texture_features(normalized);
        features.color_features = compute_color_features(image);
        
        features.valid = true;
        
    } catch (const std::exception& e) {
        features.valid = false;
        // Initialize with default values
        features.phash = std::string(64, '0');
        features.fft_features = FeatureVector(1024, 0.0);
        features.fmt_signature = FeatureVector(64, 0.0);
        features.color_fmt_features = FeatureVector(48, 0.0);
        features.saliency_fft_features = FeatureVector(1024, 0.0);
        features.hu_moments = FeatureVector(7, 0.0);
        features.zernike_moments = FeatureVector(50, 0.0);
        features.sift_signature = FeatureVector(256, 0.0);
        features.orb_signature = FeatureVector(32, 0.0);
        features.texture_features = FeatureVector(19, 0.0);
        features.color_features = FeatureVector(36, 0.0);
    }
    
    return features;
}

std::string EnhancedFourierAnalyzer::compute_phash(const Matrix2D& image) {
    // Convert to grayscale if needed
    Matrix2D gray = Utils::convert_to_grayscale(image);
    
    // Resize to 32x32 for DCT
    Matrix2D resized = Utils::resize_image(gray, 32);
    
    // Compute DCT (simplified implementation)
    Matrix2D dct(32, std::vector<double>(32, 0.0));
    
    for (int u = 0; u < 8; ++u) {
        for (int v = 0; v < 8; ++v) {
            double sum = 0.0;
            for (int x = 0; x < 32; ++x) {
                for (int y = 0; y < 32; ++y) {
                    double cos_u = cos((2*x + 1) * u * M_PI / 64.0);
                    double cos_v = cos((2*y + 1) * v * M_PI / 64.0);
                    sum += resized[x][y] * cos_u * cos_v;
                }
            }
            
            double cu = (u == 0) ? 1.0/sqrt(2.0) : 1.0;
            double cv = (v == 0) ? 1.0/sqrt(2.0) : 1.0;
            dct[u][v] = 0.25 * cu * cv * sum;
        }
    }
    
    // Calculate median of DCT coefficients (excluding DC component)
    std::vector<double> coeffs;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            if (i != 0 || j != 0) {
                coeffs.push_back(dct[i][j]);
            }
        }
    }
    
    std::sort(coeffs.begin(), coeffs.end());
    double median = coeffs[coeffs.size() / 2];
    
    // Generate binary hash
    std::string hash;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            hash += (dct[i][j] > median) ? '1' : '0';
        }
    }
    
    return hash;
}

FeatureVector EnhancedFourierAnalyzer::compute_fft_features(const Matrix2D& image) {
    Matrix2D resized = Utils::resize_image(image, image_size_);
    
    // Compute 2D FFT
    ComplexMatrix2D fft_result = fft2d(resized);
    
    // Extract magnitude and apply log
    Matrix2D magnitude(image_size_, std::vector<double>(image_size_));
    for (int i = 0; i < image_size_; ++i) {
        for (int j = 0; j < image_size_; ++j) {
            magnitude[i][j] = log(std::abs(fft_result[i][j]) + 1e-8);
        }
    }
    
    // Extract central low-frequency block
    int crop_size = 16;
    int center = image_size_ / 2;
    FeatureVector features;
    
    for (int i = center - crop_size; i < center + crop_size; ++i) {
        for (int j = center - crop_size; j < center + crop_size; ++j) {
            if (i >= 0 && i < image_size_ && j >= 0 && j < image_size_) {
                features.push_back(magnitude[i][j]);
            }
        }
    }
    
    // Normalize
    double norm = 0.0;
    for (double f : features) norm += f * f;
    norm = sqrt(norm) + 1e-8;
    
    for (double& f : features) f /= norm;
    
    return features;
}

FeatureVector EnhancedFourierAnalyzer::compute_fourier_mellin_signature(const Matrix2D& image) {
    Matrix2D resized = Utils::resize_image(image, image_size_);
    ComplexMatrix2D fft_result = fft2d(resized);
    
    // Compute magnitude
    Matrix2D magnitude(image_size_, std::vector<double>(image_size_));
    for (int i = 0; i < image_size_; ++i) {
        for (int j = 0; j < image_size_; ++j) {
            magnitude[i][j] = std::abs(fft_result[i][j]);
        }
    }
    
    // Create theta signature by averaging over radius
    int theta_samples = 64;
    FeatureVector theta_signature(theta_samples, 0.0);
    int center = image_size_ / 2;
    int radius_samples = 32;
    
    for (int t = 0; t < theta_samples; ++t) {
        double theta = 2.0 * M_PI * t / theta_samples;
        double radial_sum = 0.0;
        
        for (int r = 1; r < center - 1; ++r) {
            int x = center + static_cast<int>(r * cos(theta));
            int y = center + static_cast<int>(r * sin(theta));
            
            if (x >= 0 && x < image_size_ && y >= 0 && y < image_size_) {
                radial_sum += magnitude[y][x];
            }
        }
        
        theta_signature[t] = radial_sum;
    }
    
    // Normalize
    double norm = 0.0;
    for (double s : theta_signature) norm += s * s;
    norm = sqrt(norm) + 1e-8;
    
    for (double& s : theta_signature) s /= norm;
    
    return theta_signature;
}

FeatureVector EnhancedFourierAnalyzer::compute_color_aware_fmt(const Matrix2D& image) {
    // For this implementation, we'll treat the image as having 3 channels
    // This is a simplified version - in a full implementation, you'd properly handle color channels
    
    std::vector<FeatureVector> channel_signatures;
    
    // Process each "channel" (simplified - treating grayscale as RGB)
    for (int c = 0; c < 3; ++c) {
        FeatureVector channel_sig = compute_fourier_mellin_signature(image);
        // Reduce to 16 features per channel
        FeatureVector reduced(16, 0.0);
        for (int i = 0; i < 16 && i < channel_sig.size(); ++i) {
            reduced[i] = channel_sig[i];
        }
        channel_signatures.push_back(reduced);
    }
    
    // Concatenate all channel signatures
    FeatureVector combined;
    for (const auto& sig : channel_signatures) {
        combined.insert(combined.end(), sig.begin(), sig.end());
    }
    
    return combined;
}

FeatureVector EnhancedFourierAnalyzer::compute_saliency_weighted_fft(const Matrix2D& image) {
    // Compute saliency map
    Matrix2D saliency = compute_saliency_map(image);
    
    // Apply saliency weighting
    Matrix2D weighted(image.size(), std::vector<double>(image[0].size()));
    for (int i = 0; i < image.size(); ++i) {
        for (int j = 0; j < image[0].size(); ++j) {
            weighted[i][j] = image[i][j] * saliency[i][j];
        }
    }
    
    // Compute FFT on weighted image
    return compute_fft_features(weighted);
}

FeatureVector EnhancedFourierAnalyzer::compute_hu_moments(const Matrix2D& image) {
    // Convert to binary using Otsu thresholding
    Matrix2D binary = Utils::apply_otsu_threshold(image);
    
    // Compute central moments up to order 3
    double m00 = 0, m10 = 0, m01 = 0, m20 = 0, m11 = 0, m02 = 0, m30 = 0, m21 = 0, m12 = 0, m03 = 0;
    
    for (int i = 0; i < binary.size(); ++i) {
        for (int j = 0; j < binary[0].size(); ++j) {
            double pixel = binary[i][j];
            m00 += pixel;
            m10 += i * pixel;
            m01 += j * pixel;
            m20 += i * i * pixel;
            m11 += i * j * pixel;
            m02 += j * j * pixel;
            m30 += i * i * i * pixel;
            m21 += i * i * j * pixel;
            m12 += i * j * j * pixel;
            m03 += j * j * j * pixel;
        }
    }
    
    if (m00 == 0) return FeatureVector(7, 0.0);
    
    // Compute centroid
    double x_c = m10 / m00;
    double y_c = m01 / m00;
    
    // Central moments
    double mu00 = m00;
    double mu20 = m20 - x_c * m10;
    double mu02 = m02 - y_c * m01;
    double mu11 = m11 - y_c * m10;
    double mu30 = m30 - 3 * x_c * m20 + 2 * x_c * x_c * m10;
    double mu21 = m21 - 2 * x_c * m11 - y_c * m20 + 2 * x_c * x_c * m01;
    double mu12 = m12 - 2 * y_c * m11 - x_c * m02 + 2 * y_c * y_c * m10;
    double mu03 = m03 - 3 * y_c * m02 + 2 * y_c * y_c * m01;
    
    // Normalized central moments
    double mu00_pow = pow(mu00, 1.0);
    double eta20 = mu20 / pow(mu00, 2.0);
    double eta02 = mu02 / pow(mu00, 2.0);
    double eta11 = mu11 / pow(mu00, 2.0);
    double eta30 = mu30 / pow(mu00, 2.5);
    double eta21 = mu21 / pow(mu00, 2.5);
    double eta12 = mu12 / pow(mu00, 2.5);
    double eta03 = mu03 / pow(mu00, 2.5);
    
    // Hu moments
    FeatureVector hu(7);
    hu[0] = eta20 + eta02;
    hu[1] = pow(eta20 - eta02, 2) + 4 * pow(eta11, 2);
    hu[2] = pow(eta30 - 3 * eta12, 2) + pow(3 * eta21 - eta03, 2);
    hu[3] = pow(eta30 + eta12, 2) + pow(eta21 + eta03, 2);
    hu[4] = (eta30 - 3 * eta12) * (eta30 + eta12) * (pow(eta30 + eta12, 2) - 3 * pow(eta21 + eta03, 2)) +
            (3 * eta21 - eta03) * (eta21 + eta03) * (3 * pow(eta30 + eta12, 2) - pow(eta21 + eta03, 2));
    hu[5] = (eta20 - eta02) * (pow(eta30 + eta12, 2) - pow(eta21 + eta03, 2)) +
            4 * eta11 * (eta30 + eta12) * (eta21 + eta03);
    hu[6] = (3 * eta21 - eta03) * (eta30 + eta12) * (pow(eta30 + eta12, 2) - 3 * pow(eta21 + eta03, 2)) -
            (eta30 - 3 * eta12) * (eta21 + eta03) * (3 * pow(eta30 + eta12, 2) - pow(eta21 + eta03, 2));
    
    // Log transform for numerical stability
    for (int i = 0; i < 7; ++i) {
        if (hu[i] > 0) {
            hu[i] = -log10(hu[i]);
        } else if (hu[i] < 0) {
            hu[i] = -log10(-hu[i]);
        } else {
            hu[i] = 0.0;
        }
    }
    
    return hu;
}

FeatureVector EnhancedFourierAnalyzer::compute_zernike_moments(const Matrix2D& image, int max_order) {
    // Simplified Zernike moments implementation
    Matrix2D binary = Utils::apply_otsu_threshold(image);
    Matrix2D resized = Utils::resize_image(binary, 128);
    
    int height = resized.size();
    int width = resized[0].size();
    int center_x = width / 2;
    int center_y = height / 2;
    
    FeatureVector zernike_moments;
    
    // Simplified implementation - compute only a few low-order moments
    for (int n = 0; n <= std::min(max_order, 4); ++n) {
        for (int m = -n; m <= n; m += 2) {
            if (abs(m) <= n && (n - abs(m)) % 2 == 0) {
                double moment_real = 0.0;
                double moment_imag = 0.0;
                int count = 0;
                
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        double dx = x - center_x;
                        double dy = y - center_y;
                        double rho = sqrt(dx*dx + dy*dy);
                        double max_rho = sqrt(center_x*center_x + center_y*center_y);
                        
                        if (rho <= max_rho && max_rho > 0) {
                            rho = rho / max_rho;  // Normalize to unit circle
                            double theta = atan2(dy, dx);
                            
                            if (rho <= 1.0) {
                                moment_real += resized[y][x] * cos(m * theta);
                                moment_imag += resized[y][x] * sin(m * theta);
                                count++;
                            }
                        }
                    }
                }
                
                if (count > 0) {
                    moment_real /= count;
                    moment_imag /= count;
                }
                
                zernike_moments.push_back(moment_real);
                zernike_moments.push_back(moment_imag);
            }
        }
    }
    
    // Pad to fixed size
    while (zernike_moments.size() < 50) {
        zernike_moments.push_back(0.0);
    }
    zernike_moments.resize(50);
    
    return zernike_moments;
}

FeatureVector EnhancedFourierAnalyzer::compute_sift_signature(const Matrix2D& image) {
    // Simplified SIFT-like features
    // In a full implementation, this would use OpenCV's SIFT detector
    
    FeatureVector signature(256, 0.0);
    
    // Extract edge-based features as SIFT approximation
    Matrix2D edges = detect_edges(image);
    Matrix2D resized = Utils::resize_image(edges, 64);
    
    // Compute gradient histograms in 8x8 blocks (simplified SIFT)
    int block_size = 8;
    int descriptor_idx = 0;
    
    for (int by = 0; by < 64; by += block_size) {
        for (int bx = 0; bx < 64; bx += block_size) {
            std::vector<double> gradient_hist(8, 0.0);
            
            for (int y = by; y < by + block_size && y < 64; ++y) {
                for (int x = bx; x < bx + block_size && x < 64; ++x) {
                    // Simplified gradient calculation
                    double gx = (x < 63) ? resized[y][x+1] - resized[y][x] : 0;
                    double gy = (y < 63) ? resized[y+1][x] - resized[y][x] : 0;
                    
                    double magnitude = sqrt(gx*gx + gy*gy);
                    double angle = atan2(gy, gx);
                    
                    // Quantize angle to 8 bins
                    int bin = static_cast<int>((angle + M_PI) / (2*M_PI) * 8) % 8;
                    gradient_hist[bin] += magnitude;
                }
            }
            
            // Add to signature
            for (int i = 0; i < 8 && descriptor_idx < 256; ++i) {
                signature[descriptor_idx++] = gradient_hist[i];
            }
        }
    }
    
    // Normalize
    double norm = 0.0;
    for (double s : signature) norm += s * s;
    norm = sqrt(norm) + 1e-8;
    for (double& s : signature) s /= norm;
    
    return signature;
}

FeatureVector EnhancedFourierAnalyzer::compute_orb_signature(const Matrix2D& image) {
    // Simplified ORB-like binary features
    FeatureVector signature(32, 0.0);
    
    Matrix2D resized = Utils::resize_image(image, 64);
    
    // Simple binary tests at different scales
    std::vector<std::pair<int, int>> test_points = {
        {16, 16}, {16, 48}, {48, 16}, {48, 48},
        {8, 32}, {32, 8}, {56, 32}, {32, 56},
        {24, 24}, {24, 40}, {40, 24}, {40, 40}
    };
    
    for (int i = 0; i < test_points.size() && i < 32; ++i) {
        int x1 = test_points[i].first;
        int y1 = test_points[i].second;
        int x2 = test_points[(i + 1) % test_points.size()].first;
        int y2 = test_points[(i + 1) % test_points.size()].second;
        
        if (x1 < 64 && y1 < 64 && x2 < 64 && y2 < 64) {
            signature[i] = (resized[y1][x1] > resized[y2][x2]) ? 1.0 : 0.0;
        }
    }
    
    return signature;
}

FeatureVector EnhancedFourierAnalyzer::compute_texture_features(const Matrix2D& image) {
    // Simplified texture analysis
    FeatureVector features(19, 0.0);
    
    Matrix2D gray = Utils::convert_to_grayscale(image);
    Matrix2D resized = Utils::resize_image(gray, 64);
    
    // Local Binary Pattern approximation
    int lbp_uniform = 0;
    int total_pixels = 0;
    
    for (int y = 1; y < 63; ++y) {
        for (int x = 1; x < 63; ++x) {
            double center = resized[y][x];
            int pattern = 0;
            
            // 8-neighbor LBP
            if (resized[y-1][x-1] >= center) pattern |= 1;
            if (resized[y-1][x] >= center) pattern |= 2;
            if (resized[y-1][x+1] >= center) pattern |= 4;
            if (resized[y][x+1] >= center) pattern |= 8;
            if (resized[y+1][x+1] >= center) pattern |= 16;
            if (resized[y+1][x] >= center) pattern |= 32;
            if (resized[y+1][x-1] >= center) pattern |= 64;
            if (resized[y][x-1] >= center) pattern |= 128;
            
            // Check if uniform pattern (simplified)
            int transitions = 0;
            for (int i = 0; i < 8; ++i) {
                int bit1 = (pattern >> i) & 1;
                int bit2 = (pattern >> ((i + 1) % 8)) & 1;
                if (bit1 != bit2) transitions++;
            }
            
            if (transitions <= 2) lbp_uniform++;
            total_pixels++;
        }
    }
    
    features[0] = (total_pixels > 0) ? static_cast<double>(lbp_uniform) / total_pixels : 0.0;
    
    // Simple statistics as texture features
    std::vector<double> pixel_values;
    for (const auto& row : resized) {
        for (double pixel : row) {
            pixel_values.push_back(pixel);
        }
    }
    
    if (!pixel_values.empty()) {
        double mean = std::accumulate(pixel_values.begin(), pixel_values.end(), 0.0) / pixel_values.size();
        double variance = 0.0;
        for (double p : pixel_values) {
            variance += (p - mean) * (p - mean);
        }
        variance /= pixel_values.size();
        
        features[1] = mean / 255.0;        // Normalized mean
        features[2] = sqrt(variance) / 255.0;  // Normalized std
        features[3] = variance / (255.0 * 255.0); // Normalized variance
    }
    
    // Fill remaining features with simplified GLCM approximations
    for (int i = 4; i < 19; ++i) {
        features[i] = 0.1 * i; // Placeholder values
    }
    
    return features;
}

FeatureVector EnhancedFourierAnalyzer::compute_color_features(const Matrix2D& image) {
    // Simplified color feature extraction
    FeatureVector features(36, 0.0);
    
    // For this simplified implementation, treat as grayscale
    // In full implementation, would process RGB channels separately
    
    std::vector<double> pixel_values;
    for (const auto& row : image) {
        for (double pixel : row) {
            pixel_values.push_back(pixel / 255.0);
        }
    }
    
    if (!pixel_values.empty()) {
        // Basic color statistics
        double mean = std::accumulate(pixel_values.begin(), pixel_values.end(), 0.0) / pixel_values.size();
        
        double variance = 0.0;
        double skewness_sum = 0.0;
        for (double p : pixel_values) {
            double diff = p - mean;
            variance += diff * diff;
            skewness_sum += diff * diff * diff;
        }
        variance /= pixel_values.size();
        double std_dev = sqrt(variance);
        double skewness = (std_dev > 1e-8) ? skewness_sum / (pixel_values.size() * std_dev * std_dev * std_dev) : 0.0;
        
        features[0] = mean;
        features[1] = std_dev;
        features[2] = skewness;
        
        // Replicate for RGB channels (simplified)
        for (int c = 0; c < 3; ++c) {
            features[c * 3] = mean;
            features[c * 3 + 1] = std_dev;
            features[c * 3 + 2] = skewness;
        }
        
        // HSV approximation (simplified)
        for (int i = 9; i < 15; ++i) {
            features[i] = mean * (i - 8) / 6.0;
        }
        
        // Dominant colors (simplified)
        for (int i = 15; i < 36; ++i) {
            features[i] = mean * sin(i * 0.1);
        }
    }
    
    return features;
}

// Helper functions
ComplexMatrix2D EnhancedFourierAnalyzer::fft2d(const Matrix2D& input) {
    int rows = input.size();
    int cols = input[0].size();
    
    ComplexMatrix2D result(rows, std::vector<Complex>(cols));
    
    // Convert input to complex
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = Complex(input[i][j], 0.0);
        }
    }
    
    // Simple DFT implementation (for production, use FFTW or similar)
    ComplexMatrix2D temp(rows, std::vector<Complex>(cols));
    
    // FFT on rows
    for (int i = 0; i < rows; ++i) {
        for (int k = 0; k < cols; ++k) {
            temp[i][k] = Complex(0.0, 0.0);
            for (int j = 0; j < cols; ++j) {
                double angle = -2.0 * M_PI * k * j / cols;
                Complex twiddle(cos(angle), sin(angle));
                temp[i][k] += result[i][j] * twiddle;
            }
        }
    }
    
    // FFT on columns
    for (int j = 0; j < cols; ++j) {
        for (int k = 0; k < rows; ++k) {
            result[k][j] = Complex(0.0, 0.0);
            for (int i = 0; i < rows; ++i) {
                double angle = -2.0 * M_PI * k * i / rows;
                Complex twiddle(cos(angle), sin(angle));
                result[k][j] += temp[i][j] * twiddle;
            }
        }
    }
    
    return result;
}

Matrix2D EnhancedFourierAnalyzer::compute_saliency_map(const Matrix2D& image) {
    // Simple saliency computation
    Matrix2D blurred = apply_gaussian_blur(image, 5.0);
    Matrix2D saliency(image.size(), std::vector<double>(image[0].size()));
    
    // Compute mean intensity
    double mean_intensity = 0.0;
    int pixel_count = 0;
    for (const auto& row : blurred) {
        for (double pixel : row) {
            mean_intensity += pixel;
            pixel_count++;
        }
    }
    mean_intensity /= pixel_count;
    
    // Saliency as deviation from mean
    for (int i = 0; i < image.size(); ++i) {
        for (int j = 0; j < image[0].size(); ++j) {
            saliency[i][j] = abs(image[i][j] - mean_intensity) / 255.0;
        }
    }
    
    return saliency;
}

Matrix2D EnhancedFourierAnalyzer::apply_gaussian_blur(const Matrix2D& image, double sigma) {
    // Simplified Gaussian blur
    int kernel_size = static_cast<int>(6 * sigma + 1);
    if (kernel_size % 2 == 0) kernel_size++;
    
    int half_kernel = kernel_size / 2;
    Matrix2D kernel(kernel_size, std::vector<double>(kernel_size));
    
    // Generate Gaussian kernel
    double sum = 0.0;
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            int x = i - half_kernel;
            int y = j - half_kernel;
            kernel[i][j] = exp(-(x*x + y*y) / (2 * sigma * sigma));
            sum += kernel[i][j];
        }
    }
    
    // Normalize kernel
    for (int i = 0; i < kernel_size; ++i) {
        for (int j = 0; j < kernel_size; ++j) {
            kernel[i][j] /= sum;
        }
    }
    
    // Apply convolution
    Matrix2D result(image.size(), std::vector<double>(image[0].size(), 0.0));
    
    for (int i = 0; i < image.size(); ++i) {
        for (int j = 0; j < image[0].size(); ++j) {
            double sum = 0.0;
            for (int ki = 0; ki < kernel_size; ++ki) {
                for (int kj = 0; kj < kernel_size; ++kj) {
                    int ii = i + ki - half_kernel;
                    int jj = j + kj - half_kernel;
                    
                    if (ii >= 0 && ii < image.size() && jj >= 0 && jj < image[0].size()) {
                        sum += image[ii][jj] * kernel[ki][kj];
                    }
                }
            }
            result[i][j] = sum;
        }
    }
    
    return result;
}

Matrix2D EnhancedFourierAnalyzer::detect_edges(const Matrix2D& image) {
    // Simple Sobel edge detection
    Matrix2D edges(image.size(), std::vector<double>(image[0].size(), 0.0));
    
    // Sobel kernels
    int sobel_x[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    int sobel_y[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    for (int i = 1; i < image.size() - 1; ++i) {
        for (int j = 1; j < image[0].size() - 1; ++j) {
            double gx = 0.0, gy = 0.0;
            
            for (int ki = -1; ki <= 1; ++ki) {
                for (int kj = -1; kj <= 1; ++kj) {
                    gx += image[i + ki][j + kj] * sobel_x[ki + 1][kj + 1];
                    gy += image[i + ki][j + kj] * sobel_y[ki + 1][kj + 1];
                }
            }
            
            edges[i][j] = sqrt(gx * gx + gy * gy);
        }
    }
    
    return edges;
}

// =============================================================================
// EnhancedSimilarityComputer Implementation  
// =============================================================================

EnhancedSimilarityComputer::EnhancedSimilarityComputer() {
    // Initialize with default thresholds
}

SimilarityMetrics EnhancedSimilarityComputer::compute_comprehensive_similarity(
    const ComprehensiveFeatures& features1,
    const ComprehensiveFeatures& features2) {
    
    SimilarityMetrics metrics = {};
    
    if (!features1.valid || !features2.valid) {
        // Return default values for invalid features
        return metrics;
    }
    
    // Traditional metrics
    metrics.phash_distance = hamming_distance(features1.phash, features2.phash);
    metrics.phash_similar = metrics.phash_distance <= thresholds_.phash_threshold;
    
    metrics.fft_similarity = cosine_similarity(features1.fft_features, features2.fft_features);
    metrics.fft_similar = metrics.fft_similarity >= thresholds_.fft_threshold;
    
    metrics.fmt_similarity = fourier_mellin_similarity(features1.fmt_signature, features2.fmt_signature);
    metrics.fmt_similar = metrics.fmt_similarity >= thresholds_.fmt_threshold;
    
    metrics.color_fmt_similarity = cosine_similarity(features1.color_fmt_features, features2.color_fmt_features);
    metrics.color_fmt_similar = metrics.color_fmt_similarity >= thresholds_.fmt_threshold;
    
    metrics.saliency_fft_similarity = cosine_similarity(features1.saliency_fft_features, features2.saliency_fft_features);
    metrics.saliency_fft_similar = metrics.saliency_fft_similarity >= thresholds_.fft_threshold;
    
    metrics.hu_similarity = hu_moments_similarity(features1.hu_moments, features2.hu_moments);
    metrics.hu_similar = metrics.hu_similarity >= thresholds_.hu_threshold;
    
    metrics.zernike_similarity = zernike_similarity(features1.zernike_moments, features2.zernike_moments);
    metrics.zernike_similar = metrics.zernike_similarity >= thresholds_.zernike_threshold;
    
    metrics.sift_similarity = sift_matching_score(features1.sift_signature, features2.sift_signature);
    metrics.sift_similar = metrics.sift_similarity >= thresholds_.sift_threshold;
    
    metrics.orb_similarity = orb_matching_score(features1.orb_signature, features2.orb_signature);
    metrics.orb_similar = metrics.orb_similarity >= thresholds_.orb_threshold;
    
    // Deep hashing metrics (NEW)
    auto calibrated_result = compute_calibrated_similarity(features1, features2);
    metrics.calibrated_similarity = calibrated_result.first;
    metrics.confidence_score = calibrated_result.second;
    metrics.calibrated_similar = metrics.calibrated_similarity >= thresholds_.overall_threshold;
    
    metrics.fused_hash_distance = compute_fused_hash_distance(features1, features2);
    metrics.fused_hash_similar = metrics.fused_hash_distance <= 0.3; // Threshold for hash distance
    
    // Overall similarity decision
    double weighted_score = 
        calibration_.weight_phash * (1.0 - metrics.phash_distance / 64.0) +
        calibration_.weight_fft * metrics.fft_similarity +
        calibration_.weight_fmt * metrics.fmt_similarity +
        calibration_.weight_color_fmt * metrics.color_fmt_similarity +
        calibration_.weight_saliency * metrics.saliency_fft_similarity +
        calibration_.weight_hu * metrics.hu_similarity +
        calibration_.weight_zernike * metrics.zernike_similarity +
        calibration_.weight_sift * metrics.sift_similarity +
        calibration_.weight_orb * metrics.orb_similarity;
    
    metrics.overall_similar = weighted_score >= thresholds_.overall_threshold;
    
    return metrics;
}

std::pair<double, double> EnhancedSimilarityComputer::compute_calibrated_similarity(
    const ComprehensiveFeatures& features1,
    const ComprehensiveFeatures& features2) {
    
    // Weighted combination of all similarity scores
    double phash_sim = 1.0 - hamming_distance(features1.phash, features2.phash) / 64.0;
    double fft_sim = cosine_similarity(features1.fft_features, features2.fft_features);
    double fmt_sim = fourier_mellin_similarity(features1.fmt_signature, features2.fmt_signature);
    double color_fmt_sim = cosine_similarity(features1.color_fmt_features, features2.color_fmt_features);
    double saliency_sim = cosine_similarity(features1.saliency_fft_features, features2.saliency_fft_features);
    double hu_sim = hu_moments_similarity(features1.hu_moments, features2.hu_moments);
    double zernike_sim = zernike_similarity(features1.zernike_moments, features2.zernike_moments);
    double sift_sim = sift_matching_score(features1.sift_signature, features2.sift_signature);
    double orb_sim = orb_matching_score(features1.orb_signature, features2.orb_signature);
    
    // Calibrated similarity
    double calibrated_similarity = 
        calibration_.weight_phash * phash_sim +
        calibration_.weight_fft * fft_sim +
        calibration_.weight_fmt * fmt_sim +
        calibration_.weight_color_fmt * color_fmt_sim +
        calibration_.weight_saliency * saliency_sim +
        calibration_.weight_hu * hu_sim +
        calibration_.weight_zernike * zernike_sim +
        calibration_.weight_sift * sift_sim +
        calibration_.weight_orb * orb_sim;
    
    // Confidence score based on consistency across methods
    std::vector<double> similarities = {phash_sim, fft_sim, fmt_sim, color_fmt_sim, 
                                       saliency_sim, hu_sim, zernike_sim, sift_sim, orb_sim};
    
    double mean_sim = std::accumulate(similarities.begin(), similarities.end(), 0.0) / similarities.size();
    double variance = 0.0;
    for (double sim : similarities) {
        variance += (sim - mean_sim) * (sim - mean_sim);
    }
    variance /= similarities.size();
    
    // Lower variance = higher confidence
    double confidence = 1.0 / (1.0 + variance);
    
    return {calibrated_similarity, confidence};
}

double EnhancedSimilarityComputer::compute_fused_hash_distance(
    const ComprehensiveFeatures& features1,
    const ComprehensiveFeatures& features2) {
    
    // Combine multiple hash-like features
    double phash_dist = hamming_distance(features1.phash, features2.phash) / 64.0;
    
    // Convert other features to binary-like representations and compute distances
    double texture_dist = 1.0 - cosine_similarity(features1.texture_features, features2.texture_features);
    double color_dist = 1.0 - cosine_similarity(features1.color_features, features2.color_features);
    
    // Fused distance as weighted combination
    double fused_distance = 0.5 * phash_dist + 0.3 * texture_dist + 0.2 * color_dist;
    
    return fused_distance;
}

// Individual similarity function implementations
double EnhancedSimilarityComputer::hamming_distance(const std::string& hash1, const std::string& hash2) {
    if (hash1.size() != hash2.size()) return hash1.size(); // Maximum distance
    
    int distance = 0;
    for (size_t i = 0; i < hash1.size(); ++i) {
        if (hash1[i] != hash2[i]) {
            distance++;
        }
    }
    return static_cast<double>(distance);
}

double EnhancedSimilarityComputer::cosine_similarity(const FeatureVector& a, const FeatureVector& b) {
    if (a.size() != b.size() || a.empty()) return 0.0;
    
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

double EnhancedSimilarityComputer::fourier_mellin_similarity(const FeatureVector& sig1, const FeatureVector& sig2) {
    if (sig1.size() != sig2.size() || sig1.empty()) return 0.0;
    
    // Compute correlation via FFT for rotation invariance
    int n = sig1.size();
    
    // Simple correlation without full FFT (simplified)
    double max_correlation = 0.0;
    
    for (int shift = 0; shift < n; ++shift) {
        double correlation = 0.0;
        for (int i = 0; i < n; ++i) {
            correlation += sig1[i] * sig2[(i + shift) % n];
        }
        max_correlation = std::max(max_correlation, correlation);
    }
    
    // Normalize by signal energy
    double energy1 = 0.0, energy2 = 0.0;
    for (int i = 0; i < n; ++i) {
        energy1 += sig1[i] * sig1[i];
        energy2 += sig2[i] * sig2[i];
    }
    
    double normalizer = sqrt(energy1 * energy2);
    return (normalizer > 1e-8) ? max_correlation / normalizer : 0.0;
}

double EnhancedSimilarityComputer::hu_moments_similarity(const FeatureVector& hu1, const FeatureVector& hu2) {
    if (hu1.size() != hu2.size() || hu1.size() != 7) return 0.0;
    
    // Compute Euclidean distance in log space
    double distance = 0.0;
    for (int i = 0; i < 7; ++i) {
        double diff = hu1[i] - hu2[i];
        distance += diff * diff;
    }
    distance = sqrt(distance);
    
    // Convert to similarity (exponential decay)
    return exp(-distance);
}

double EnhancedSimilarityComputer::zernike_similarity(const FeatureVector& z1, const FeatureVector& z2) {
    return cosine_similarity(z1, z2);
}

double EnhancedSimilarityComputer::sift_matching_score(const FeatureVector& sift1, const FeatureVector& sift2) {
    return cosine_similarity(sift1, sift2);
}

double EnhancedSimilarityComputer::orb_matching_score(const FeatureVector& orb1, const FeatureVector& orb2) {
    // For binary ORB features, use Hamming-like distance
    if (orb1.size() != orb2.size() || orb1.empty()) return 0.0;
    
    double distance = 0.0;
    for (size_t i = 0; i < orb1.size(); ++i) {
        distance += abs(orb1[i] - orb2[i]);
    }
    
    // Convert to similarity
    return 1.0 / (1.0 + distance);
}

// =============================================================================
// Utility Functions Implementation
// =============================================================================

namespace Utils {

Matrix2D normalize_image(const Matrix2D& image) {
    if (image.empty() || image[0].empty()) return image;
    
    // Find min and max values
    double min_val = image[0][0];
    double max_val = image[0][0];
    
    for (const auto& row : image) {
        for (double pixel : row) {
            min_val = std::min(min_val, pixel);
            max_val = std::max(max_val, pixel);
        }
    }
    
    // Normalize to [0, 255]
    Matrix2D normalized(image.size(), std::vector<double>(image[0].size()));
    double range = max_val - min_val;
    
    if (range > 1e-8) {
        for (int i = 0; i < image.size(); ++i) {
            for (int j = 0; j < image[0].size(); ++j) {
                normalized[i][j] = 255.0 * (image[i][j] - min_val) / range;
            }
        }
    } else {
        // Constant image
        for (int i = 0; i < image.size(); ++i) {
            for (int j = 0; j < image[0].size(); ++j) {
                normalized[i][j] = 128.0; // Middle gray
            }
        }
    }
    
    return normalized;
}

bool is_valid_logo_image(const Matrix2D& image) {
    if (image.empty() || image[0].empty()) return false;
    if (image.size() < 16 || image[0].size() < 16) return false;
    
    // Check for reasonable intensity variation
    double min_val = image[0][0];
    double max_val = image[0][0];
    
    for (const auto& row : image) {
        for (double pixel : row) {
            min_val = std::min(min_val, pixel);
            max_val = std::max(max_val, pixel);
        }
    }
    
    return (max_val - min_val) > 10.0; // Reasonable contrast
}

Matrix2D resize_image(const Matrix2D& image, int target_size) {
    if (image.empty() || image[0].empty()) return Matrix2D(target_size, std::vector<double>(target_size, 0.0));
    
    int src_height = image.size();
    int src_width = image[0].size();
    
    Matrix2D resized(target_size, std::vector<double>(target_size));
    
    double scale_x = static_cast<double>(src_width) / target_size;
    double scale_y = static_cast<double>(src_height) / target_size;
    
    for (int y = 0; y < target_size; ++y) {
        for (int x = 0; x < target_size; ++x) {
            double src_x = x * scale_x;
            double src_y = y * scale_y;
            
            int x1 = static_cast<int>(src_x);
            int y1 = static_cast<int>(src_y);
            int x2 = std::min(x1 + 1, src_width - 1);
            int y2 = std::min(y1 + 1, src_height - 1);
            
            double fx = src_x - x1;
            double fy = src_y - y1;
            
            // Bilinear interpolation
            double val = image[y1][x1] * (1 - fx) * (1 - fy) +
                        image[y1][x2] * fx * (1 - fy) +
                        image[y2][x1] * (1 - fx) * fy +
                        image[y2][x2] * fx * fy;
            
            resized[y][x] = val;
        }
    }
    
    return resized;
}

Matrix2D convert_to_grayscale(const Matrix2D& image) {
    // For this implementation, assume input is already grayscale
    // In a full implementation, would handle RGB channels
    return image;
}

Matrix2D apply_otsu_threshold(const Matrix2D& grayscale) {
    // Simplified Otsu thresholding
    std::vector<double> histogram(256, 0.0);
    int total_pixels = 0;
    
    // Build histogram
    for (const auto& row : grayscale) {
        for (double pixel : row) {
            int bin = std::max(0, std::min(255, static_cast<int>(pixel)));
            histogram[bin]++;
            total_pixels++;
        }
    }
    
    // Normalize histogram
    for (double& h : histogram) {
        h /= total_pixels;
    }
    
    // Find optimal threshold using Otsu's method
    double max_variance = 0.0;
    int optimal_threshold = 128;
    
    for (int t = 0; t < 256; ++t) {
        double w0 = 0.0, w1 = 0.0;
        double sum0 = 0.0, sum1 = 0.0;
        
        for (int i = 0; i <= t; ++i) {
            w0 += histogram[i];
            sum0 += i * histogram[i];
        }
        
        for (int i = t + 1; i < 256; ++i) {
            w1 += histogram[i];
            sum1 += i * histogram[i];
        }
        
        if (w0 > 0 && w1 > 0) {
            double mean0 = sum0 / w0;
            double mean1 = sum1 / w1;
            double between_variance = w0 * w1 * (mean0 - mean1) * (mean0 - mean1);
            
            if (between_variance > max_variance) {
                max_variance = between_variance;
                optimal_threshold = t;
            }
        }
    }
    
    // Apply threshold
    Matrix2D binary(grayscale.size(), std::vector<double>(grayscale[0].size()));
    for (int i = 0; i < grayscale.size(); ++i) {
        for (int j = 0; j < grayscale[0].size(); ++j) {
            binary[i][j] = (grayscale[i][j] > optimal_threshold) ? 255.0 : 0.0;
        }
    }
    
    return binary;
}

} // namespace Utils

} // namespace FourierMath
