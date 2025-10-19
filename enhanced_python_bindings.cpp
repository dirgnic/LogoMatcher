#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include "enhanced_fourier_math.hpp"

namespace py = pybind11;
using namespace FourierMath;

// Helper function to convert numpy array to C++ Matrix2D
Matrix2D numpy_to_matrix2d(py::array_t<double> input) {
    py::buffer_info buf_info = input.request();
    
    if (buf_info.ndim != 2) {
        throw std::runtime_error("Input array must be 2-dimensional");
    }
    
    int rows = buf_info.shape[0];
    int cols = buf_info.shape[1];
    double* ptr = static_cast<double*>(buf_info.ptr);
    
    Matrix2D matrix(rows, std::vector<double>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i][j] = ptr[i * cols + j];
        }
    }
    
    return matrix;
}

// Helper function to convert C++ Matrix2D to numpy array
py::array_t<double> matrix2d_to_numpy(const Matrix2D& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    
    auto result = py::array_t<double>(
        {rows, cols},
        {sizeof(double) * cols, sizeof(double)},
        nullptr
    );
    
    py::buffer_info buf_info = result.request();
    double* ptr = static_cast<double*>(buf_info.ptr);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            ptr[i * cols + j] = matrix[i][j];
        }
    }
    
    return result;
}

// Convert list of numpy arrays to vector of Matrix2D
std::vector<Matrix2D> numpy_list_to_matrix_vector(py::list images) {
    std::vector<Matrix2D> matrices;
    matrices.reserve(images.size());
    
    for (auto item : images) {
        py::array_t<double> img = item.cast<py::array_t<double>>();
        matrices.push_back(numpy_to_matrix2d(img));
    }
    
    return matrices;
}

PYBIND11_MODULE(enhanced_fourier_math_cpp, m) {
    m.doc() = "Enhanced C++ Fourier Mathematics for Logo Analysis with Comprehensive Metrics";
    
    // Bind SimilarityMetrics structure
    py::class_<SimilarityMetrics>(m, "SimilarityMetrics")
        .def(py::init<>())
        .def_readwrite("calibrated_similarity", &SimilarityMetrics::calibrated_similarity)
        .def_readwrite("calibrated_similar", &SimilarityMetrics::calibrated_similar)
        .def_readwrite("fused_hash_distance", &SimilarityMetrics::fused_hash_distance)
        .def_readwrite("fused_hash_similar", &SimilarityMetrics::fused_hash_similar)
        .def_readwrite("confidence_score", &SimilarityMetrics::confidence_score)
        .def_readwrite("phash_distance", &SimilarityMetrics::phash_distance)
        .def_readwrite("phash_similar", &SimilarityMetrics::phash_similar)
        .def_readwrite("fft_similarity", &SimilarityMetrics::fft_similarity)
        .def_readwrite("fft_similar", &SimilarityMetrics::fft_similar)
        .def_readwrite("fmt_similarity", &SimilarityMetrics::fmt_similarity)
        .def_readwrite("fmt_similar", &SimilarityMetrics::fmt_similar)
        .def_readwrite("color_fmt_similarity", &SimilarityMetrics::color_fmt_similarity)
        .def_readwrite("color_fmt_similar", &SimilarityMetrics::color_fmt_similar)
        .def_readwrite("saliency_fft_similarity", &SimilarityMetrics::saliency_fft_similarity)
        .def_readwrite("saliency_fft_similar", &SimilarityMetrics::saliency_fft_similar)
        .def_readwrite("hu_similarity", &SimilarityMetrics::hu_similarity)
        .def_readwrite("hu_similar", &SimilarityMetrics::hu_similar)
        .def_readwrite("zernike_similarity", &SimilarityMetrics::zernike_similarity)
        .def_readwrite("zernike_similar", &SimilarityMetrics::zernike_similar)
        .def_readwrite("sift_similarity", &SimilarityMetrics::sift_similarity)
        .def_readwrite("sift_similar", &SimilarityMetrics::sift_similar)
        .def_readwrite("orb_similarity", &SimilarityMetrics::orb_similarity)
        .def_readwrite("orb_similar", &SimilarityMetrics::orb_similar)
        .def_readwrite("overall_similar", &SimilarityMetrics::overall_similar)
        .def("__repr__", [](const SimilarityMetrics& m) {
            return "<SimilarityMetrics calibrated=" + std::to_string(m.calibrated_similarity) + 
                   " overall=" + (m.overall_similar ? "True" : "False") + ">";
        });
    
    // Bind ComprehensiveFeatures structure
    py::class_<ComprehensiveFeatures>(m, "ComprehensiveFeatures")
        .def(py::init<>())
        .def_readwrite("phash", &ComprehensiveFeatures::phash)
        .def_readwrite("fft_features", &ComprehensiveFeatures::fft_features)
        .def_readwrite("fmt_signature", &ComprehensiveFeatures::fmt_signature)
        .def_readwrite("color_fmt_features", &ComprehensiveFeatures::color_fmt_features)
        .def_readwrite("saliency_fft_features", &ComprehensiveFeatures::saliency_fft_features)
        .def_readwrite("hu_moments", &ComprehensiveFeatures::hu_moments)
        .def_readwrite("zernike_moments", &ComprehensiveFeatures::zernike_moments)
        .def_readwrite("sift_signature", &ComprehensiveFeatures::sift_signature)
        .def_readwrite("orb_signature", &ComprehensiveFeatures::orb_signature)
        .def_readwrite("texture_features", &ComprehensiveFeatures::texture_features)
        .def_readwrite("color_features", &ComprehensiveFeatures::color_features)
        .def_readwrite("valid", &ComprehensiveFeatures::valid);
    
    // Bind EnhancedFourierAnalyzer
    py::class_<EnhancedFourierAnalyzer>(m, "EnhancedFourierAnalyzer")
        .def(py::init<int>(), py::arg("image_size") = 128)
        .def("extract_all_features", 
             [](EnhancedFourierAnalyzer& self, py::array_t<double> image) {
                 Matrix2D matrix = numpy_to_matrix2d(image);
                 return self.extract_all_features(matrix);
             },
             "Extract all comprehensive features from image")
        .def("compute_phash",
             [](EnhancedFourierAnalyzer& self, py::array_t<double> image) {
                 Matrix2D matrix = numpy_to_matrix2d(image);
                 return self.compute_phash(matrix);
             },
             "Compute perceptual hash")
        .def("compute_fft_features",
             [](EnhancedFourierAnalyzer& self, py::array_t<double> image) {
                 Matrix2D matrix = numpy_to_matrix2d(image);
                 return self.compute_fft_features(matrix);
             },
             "Compute FFT features")
        .def("compute_fourier_mellin_signature",
             [](EnhancedFourierAnalyzer& self, py::array_t<double> image) {
                 Matrix2D matrix = numpy_to_matrix2d(image);
                 return self.compute_fourier_mellin_signature(matrix);
             },
             "Compute Fourier-Mellin signature")
        .def("compute_hu_moments",
             [](EnhancedFourierAnalyzer& self, py::array_t<double> image) {
                 Matrix2D matrix = numpy_to_matrix2d(image);
                 return self.compute_hu_moments(matrix);
             },
             "Compute Hu invariant moments")
        .def("compute_zernike_moments",
             [](EnhancedFourierAnalyzer& self, py::array_t<double> image, int max_order) {
                 Matrix2D matrix = numpy_to_matrix2d(image);
                 return self.compute_zernike_moments(matrix, max_order);
             },
             "Compute Zernike moments",
             py::arg("image"), py::arg("max_order") = 8);
    
    // Bind EnhancedSimilarityComputer
    py::class_<EnhancedSimilarityComputer>(m, "EnhancedSimilarityComputer")
        .def(py::init<>())
        .def("compute_comprehensive_similarity",
             &EnhancedSimilarityComputer::compute_comprehensive_similarity,
             "Compute comprehensive similarity metrics between two feature sets")
        .def("hamming_distance",
             &EnhancedSimilarityComputer::hamming_distance,
             "Compute Hamming distance between two hash strings")
        .def("cosine_similarity",
             &EnhancedSimilarityComputer::cosine_similarity,
             "Compute cosine similarity between two feature vectors")
        .def("fourier_mellin_similarity",
             &EnhancedSimilarityComputer::fourier_mellin_similarity,
             "Compute Fourier-Mellin similarity with rotation invariance")
        .def("compute_calibrated_similarity",
             &EnhancedSimilarityComputer::compute_calibrated_similarity,
             "Compute calibrated similarity with confidence score")
        .def("compute_fused_hash_distance",
             &EnhancedSimilarityComputer::compute_fused_hash_distance,
             "Compute fused hash distance combining multiple hash types");
    
    // Bind EnhancedLogoAnalysisPipeline and results structure
    py::class_<EnhancedLogoAnalysisPipeline::ComprehensiveAnalysisResults>(m, "ComprehensiveAnalysisResults")
        .def_readwrite("features", &EnhancedLogoAnalysisPipeline::ComprehensiveAnalysisResults::features)
        .def_readwrite("similarity_metrics", &EnhancedLogoAnalysisPipeline::ComprehensiveAnalysisResults::similarity_metrics)
        .def_readwrite("similarity_matrix", &EnhancedLogoAnalysisPipeline::ComprehensiveAnalysisResults::similarity_matrix)
        .def_readwrite("clusters", &EnhancedLogoAnalysisPipeline::ComprehensiveAnalysisResults::clusters)
        .def_readwrite("cluster_scores", &EnhancedLogoAnalysisPipeline::ComprehensiveAnalysisResults::cluster_scores)
        .def_readwrite("analysis_time_ms", &EnhancedLogoAnalysisPipeline::ComprehensiveAnalysisResults::analysis_time_ms)
        .def_readwrite("valid_images", &EnhancedLogoAnalysisPipeline::ComprehensiveAnalysisResults::valid_images);
    
    py::class_<EnhancedLogoAnalysisPipeline>(m, "EnhancedLogoAnalysisPipeline")
        .def(py::init<int>(), "Constructor", py::arg("image_size") = 128)
        .def("analyze_logo_batch_comprehensive", 
             [](EnhancedLogoAnalysisPipeline& self, py::list images, double threshold) {
                 // Convert Python list to C++ vector of Matrix2D
                 std::vector<Matrix2D> image_matrices;
                 for (int i = 0; i < py::len(images); ++i) {
                     py::array_t<double> img = images[i].cast<py::array_t<double>>();
                     image_matrices.push_back(numpy_to_matrix2d(img));
                 }
                 return self.analyze_logo_batch_comprehensive(image_matrices, threshold);
             },
             "Run comprehensive logo batch analysis",
             py::arg("images"), py::arg("threshold") = 0.7);
    
    // Utility functions
    m.def("normalize_image", 
          [](py::array_t<double> image) {
              Matrix2D matrix = numpy_to_matrix2d(image);
              Matrix2D normalized = Utils::normalize_image(matrix);
              return matrix2d_to_numpy(normalized);
          },
          "Normalize image to [0, 255] range");
    
    m.def("is_valid_logo_image",
          [](py::array_t<double> image) {
              Matrix2D matrix = numpy_to_matrix2d(image);
              return Utils::is_valid_logo_image(matrix);
          },
          "Check if image is valid for logo analysis");
    
    m.def("resize_image",
          [](py::array_t<double> image, int target_size) {
              Matrix2D matrix = numpy_to_matrix2d(image);
              Matrix2D resized = Utils::resize_image(matrix, target_size);
              return matrix2d_to_numpy(resized);
          },
          "Resize image to target size with bilinear interpolation");
    
    // High-level convenience functions matching your Python logic
    m.def("compute_comprehensive_similarity_metrics",
          [](py::array_t<double> image1, py::array_t<double> image2) {
              Matrix2D matrix1 = numpy_to_matrix2d(image1);
              Matrix2D matrix2 = numpy_to_matrix2d(image2);
              
              EnhancedFourierAnalyzer analyzer(128);
              EnhancedSimilarityComputer computer;
              
              ComprehensiveFeatures features1 = analyzer.extract_all_features(matrix1);
              ComprehensiveFeatures features2 = analyzer.extract_all_features(matrix2);
              
              return computer.compute_comprehensive_similarity(features1, features2);
          },
          "High-level function to compute all similarity metrics between two images",
          py::arg("image1"), py::arg("image2"));
    
    m.def("extract_all_features_from_image",
          [](py::array_t<double> image) {
              Matrix2D matrix = numpy_to_matrix2d(image);
              EnhancedFourierAnalyzer analyzer(128);
              return analyzer.extract_all_features(matrix);
          },
          "Extract all comprehensive features from a single image");
    
    m.def("batch_feature_extraction",
          [](py::list images) {
              std::vector<Matrix2D> matrices = numpy_list_to_matrix_vector(images);
              EnhancedFourierAnalyzer analyzer(128);
              
              std::vector<ComprehensiveFeatures> all_features;
              all_features.reserve(matrices.size());
              
              for (const auto& matrix : matrices) {
                  all_features.push_back(analyzer.extract_all_features(matrix));
              }
              
              return all_features;
          },
          "Extract features from multiple images in batch");
    
    m.def("compute_similarity_matrix_comprehensive",
          [](py::list images, double threshold) {
              std::vector<Matrix2D> matrices = numpy_list_to_matrix_vector(images);
              EnhancedLogoAnalysisPipeline pipeline(128);
              
              auto results = pipeline.analyze_logo_batch_comprehensive(matrices, threshold);
              return matrix2d_to_numpy(results.similarity_matrix);
          },
          "Compute comprehensive similarity matrix from images",
          py::arg("images"), py::arg("threshold") = 0.7);
    
    // Performance benchmarking
    m.def("benchmark_comprehensive_analysis",
          [](py::list images, int iterations) {
              std::vector<Matrix2D> matrices = numpy_list_to_matrix_vector(images);
              
              auto start = std::chrono::high_resolution_clock::now();
              
              EnhancedLogoAnalysisPipeline pipeline(128);
              for (int i = 0; i < iterations; ++i) {
                  auto results = pipeline.analyze_logo_batch_comprehensive(matrices, 0.7);
              }
              
              auto end = std::chrono::high_resolution_clock::now();
              double total_time = std::chrono::duration<double, std::milli>(end - start).count();
              
              py::dict benchmark_results;
              benchmark_results["total_time_ms"] = total_time;
              benchmark_results["iterations"] = iterations;
              benchmark_results["images_per_iteration"] = matrices.size();
              benchmark_results["avg_time_per_iteration_ms"] = total_time / iterations;
              benchmark_results["features_per_second"] = (matrices.size() * iterations) / (total_time / 1000.0);
              
              return benchmark_results;
          },
          "Benchmark comprehensive analysis performance",
          py::arg("images"), py::arg("iterations") = 10);
}
