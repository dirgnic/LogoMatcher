#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include "fourier_math.hpp"

/**
 * Python Bindings for High-Performance C++ Fourier Mathematics
 * Creates 'fourier_math_cpp' Python module for integration with logo analysis pipeline
 */

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
std::vector<Matrix2D> numpy_list_to_matrix_vector(py::list input_list) {
    std::vector<Matrix2D> matrices;
    matrices.reserve(input_list.size());
    
    for (py::handle item : input_list) {
        py::array_t<double> array = item.cast<py::array_t<double>>();
        matrices.push_back(numpy_to_matrix2d(array));
    }
    
    return matrices;
}

PYBIND11_MODULE(fourier_math_cpp, m) {
    m.doc() = "High-Performance C++ Fourier Mathematics for Logo Analysis";
    
    // Main Pipeline Class
    py::class_<LogoAnalysisPipeline>(m, "LogoAnalysisPipeline")
        .def(py::init<int>(), "Initialize pipeline", py::arg("image_size") = 128)
        
        .def("analyze_single_logo", 
             [](LogoAnalysisPipeline& self, py::array_t<double> image) {
                 Matrix2D matrix = numpy_to_matrix2d(image);
                 return self.analyze_single_logo(matrix);
             },
             "Analyze single logo image",
             py::arg("image"))
        
        .def("analyze_logo_batch",
             [](LogoAnalysisPipeline& self, py::list images) {
                 std::vector<Matrix2D> matrices = numpy_list_to_matrix_vector(images);
                 return self.analyze_logo_batch(matrices);
             },
             "Analyze batch of logo images",
             py::arg("images"))
        
        .def("compute_comprehensive_analysis",
             [](LogoAnalysisPipeline& self, py::list images, double threshold) {
                 std::vector<Matrix2D> matrices = numpy_list_to_matrix_vector(images);
                 auto results = self.compute_comprehensive_analysis(matrices, threshold);
                 
                 // Convert similarity matrix back to numpy for Python
                 py::dict py_results;
                 py_results["similarity_matrix"] = matrix2d_to_numpy(results.similarity_matrix);
                 py_results["clusters"] = results.clusters;
                 py_results["similarity_scores"] = results.similarity_scores;
                 py_results["processing_time_ms"] = results.processing_time_ms;
                 
                 return py_results;
             },
             "Run complete analysis with clustering",
             py::arg("images"), py::arg("threshold") = 0.45);
    
    // LogoFeatures structure
    py::class_<LogoAnalysisPipeline::LogoFeatures>(m, "LogoFeatures")
        .def_readwrite("comprehensive_features", &LogoAnalysisPipeline::LogoFeatures::comprehensive_features)
        .def_readwrite("perceptual_hash", &LogoAnalysisPipeline::LogoFeatures::perceptual_hash)
        .def_readwrite("fft_signature", &LogoAnalysisPipeline::LogoFeatures::fft_signature)
        .def_readwrite("fourier_mellin", &LogoAnalysisPipeline::LogoFeatures::fourier_mellin)
        .def_readwrite("is_valid", &LogoAnalysisPipeline::LogoFeatures::is_valid);
    
    // Feature Extractor
    py::class_<LogoFeatureExtractor>(m, "LogoFeatureExtractor")
        .def(py::init<int, int>(), py::arg("image_size") = 128, py::arg("hash_size") = 8)
        
        .def("compute_perceptual_hash",
             [](LogoFeatureExtractor& self, py::array_t<double> image) {
                 Matrix2D matrix = numpy_to_matrix2d(image);
                 return self.compute_perceptual_hash(matrix);
             })
        
        .def("compute_fft_signature",
             [](LogoFeatureExtractor& self, py::array_t<double> image) {
                 Matrix2D matrix = numpy_to_matrix2d(image);
                 return self.compute_fft_signature(matrix);
             })
        
        .def("compute_fourier_mellin",
             [](LogoFeatureExtractor& self, py::array_t<double> image) {
                 Matrix2D matrix = numpy_to_matrix2d(image);
                 return self.compute_fourier_mellin(matrix);
             })
        
        .def("extract_comprehensive_features",
             [](LogoFeatureExtractor& self, py::array_t<double> image) {
                 Matrix2D matrix = numpy_to_matrix2d(image);
                 return self.extract_comprehensive_features(matrix);
             });
    
    // Similarity Computer
    py::class_<SimilarityComputer>(m, "SimilarityComputer")
        .def(py::init<>())
        
        .def("compute_similarity_matrix",
             [](SimilarityComputer& self, 
                py::list feature_vectors, 
                py::list hashes) {
                 
                 std::vector<FeatureVector> features;
                 std::vector<std::string> hash_strings;
                 
                 for (py::handle item : feature_vectors) {
                     features.push_back(item.cast<FeatureVector>());
                 }
                 
                 for (py::handle item : hashes) {
                     hash_strings.push_back(item.cast<std::string>());
                 }
                 
                 Matrix2D result = self.compute_similarity_matrix(features, hash_strings);
                 return matrix2d_to_numpy(result);
             })
        
        .def("compute_comprehensive_similarity",
             &SimilarityComputer::compute_comprehensive_similarity,
             "Compute similarity between two feature vectors",
             py::arg("features1"), py::arg("features2"), 
             py::arg("hash1"), py::arg("hash2"));
    
    // Clustering
    py::class_<LogoClusterer>(m, "LogoClusterer")
        .def(py::init<>())
        
        .def("cluster_by_threshold",
             [](LogoClusterer& self, py::array_t<double> similarity_matrix, double threshold) {
                 Matrix2D matrix = numpy_to_matrix2d(similarity_matrix);
                 return self.cluster_by_threshold(matrix, threshold);
             },
             "Cluster logos by similarity threshold",
             py::arg("similarity_matrix"), py::arg("threshold"));
    
    // Utility functions
    m.def("normalize_image", 
          [](py::array_t<double> image) {
              Matrix2D matrix = numpy_to_matrix2d(image);
              Matrix2D result = Utils::normalize_image(matrix);
              return matrix2d_to_numpy(result);
          },
          "Normalize image to [0,1] range");
    
    m.def("is_valid_logo_image",
          [](py::array_t<double> image) {
              Matrix2D matrix = numpy_to_matrix2d(image);
              return Utils::is_valid_logo_image(matrix);
          },
          "Check if image is valid for logo analysis");
    
    // High-level convenience functions for Python integration
    m.def("compute_similarity_matrix",
          [](py::list images, double threshold) {
              // Convert images
              std::vector<Matrix2D> matrices = numpy_list_to_matrix_vector(images);
              
              // Create pipeline and analyze
              LogoAnalysisPipeline pipeline(128);
              auto results = pipeline.compute_comprehensive_analysis(matrices, threshold);
              
              return matrix2d_to_numpy(results.similarity_matrix);
          },
          "High-level function to compute similarity matrix from images",
          py::arg("images"), py::arg("threshold") = 0.45);
    
    m.def("find_clusters",
          [](py::array_t<double> similarity_matrix, double threshold) {
              Matrix2D matrix = numpy_to_matrix2d(similarity_matrix);
              LogoClusterer clusterer;
              return clusterer.cluster_by_threshold(matrix, threshold);
          },
          "High-level function to find clusters from similarity matrix",
          py::arg("similarity_matrix"), py::arg("threshold") = 0.45);
    
    // Performance benchmarking
    m.def("benchmark_analysis",
          [](py::list images, int iterations) {
              std::vector<Matrix2D> matrices = numpy_list_to_matrix_vector(images);
              
              auto start = std::chrono::high_resolution_clock::now();
              
              LogoAnalysisPipeline pipeline(128);
              for (int i = 0; i < iterations; ++i) {
                  auto results = pipeline.analyze_logo_batch(matrices);
              }
              
              auto end = std::chrono::high_resolution_clock::now();
              double total_time = std::chrono::duration<double, std::milli>(end - start).count();
              
              py::dict benchmark_results;
              benchmark_results["total_time_ms"] = total_time;
              benchmark_results["iterations"] = iterations;
              benchmark_results["images_per_iteration"] = matrices.size();
              benchmark_results["avg_time_per_iteration_ms"] = total_time / iterations;
              benchmark_results["images_per_second"] = (matrices.size() * iterations) / (total_time / 1000.0);
              
              return benchmark_results;
          },
          "Benchmark analysis performance",
          py::arg("images"), py::arg("iterations") = 10);
    
    // Version info
    m.attr("__version__") = "1.0.0";
    m.attr("__author__") = "Logo Analysis Pipeline";
    m.attr("__description__") = "High-Performance C++ Fourier Mathematics for MacBook Pro 2024";
}
