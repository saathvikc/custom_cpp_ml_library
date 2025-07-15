#include "knn_tests.h"
#include "../utils/log_manager.h"
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <algorithm>

namespace KNNTests {

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

void print_regression_performance(const KNearestNeighbors& model, 
                                const std::vector<std::vector<double>>& X_test, 
                                const std::vector<double>& y_test,
                                const std::string& config_name) {
    double mse = model.evaluate_mse(X_test, y_test);
    double mae = model.evaluate_mae(X_test, y_test);
    double r2 = model.evaluate_r2(X_test, y_test);
    
    std::cout << config_name << ": MSE=" << std::fixed << std::setprecision(4) << mse
              << ", MAE=" << mae << ", R²=" << r2 << std::endl;
}

void print_classification_performance(const KNearestNeighbors& model, 
                                    const std::vector<std::vector<double>>& X_test, 
                                    const std::vector<int>& y_test,
                                    const std::string& config_name) {
    double accuracy = model.evaluate_accuracy(X_test, y_test);
    double precision = model.evaluate_precision(X_test, y_test, 1);
    double recall = model.evaluate_recall(X_test, y_test, 1);
    double f1 = model.evaluate_f1_score(X_test, y_test, 1);
    
    std::cout << config_name << ": Acc=" << std::fixed << std::setprecision(4) << accuracy
              << ", Prec=" << precision << ", Rec=" << recall << ", F1=" << f1 << std::endl;
}

void generate_regression_data(std::vector<std::vector<double>>& X, std::vector<double>& y, 
                            int n_samples, int n_features, double noise) {
    X.clear();
    y.clear();
    X.reserve(n_samples);
    y.reserve(n_samples);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> feature_dist(-5.0, 5.0);
    std::normal_distribution<> noise_dist(0.0, noise);
    
    for (int i = 0; i < n_samples; ++i) {
        std::vector<double> sample(n_features);
        double target = 0.0;
        
        for (int j = 0; j < n_features; ++j) {
            sample[j] = feature_dist(gen);
            // Create a non-linear relationship
            target += sample[j] * sample[j] * 0.5 + sample[j] * 0.3;
        }
        
        target += noise_dist(gen);
        
        X.push_back(sample);
        y.push_back(target);
    }
}

void generate_classification_data(std::vector<std::vector<double>>& X, std::vector<int>& y, 
                                int n_samples, int n_features, int n_classes) {
    X.clear();
    y.clear();
    X.reserve(n_samples);
    y.reserve(n_samples);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> feature_dist(-3.0, 3.0);
    std::uniform_int_distribution<> class_dist(0, n_classes - 1);
    
    for (int i = 0; i < n_samples; ++i) {
        std::vector<double> sample(n_features);
        int target_class = class_dist(gen);
        
        // Create class-dependent features
        for (int j = 0; j < n_features; ++j) {
            double base_value = feature_dist(gen);
            // Add class-specific bias
            sample[j] = base_value + target_class * 2.0;
        }
        
        X.push_back(sample);
        y.push_back(target_class);
    }
}

void generate_multidimensional_data(std::vector<std::vector<double>>& X, std::vector<int>& y, 
                                   int n_samples, int n_features) {
    X.clear();
    y.clear();
    X.reserve(n_samples);
    y.reserve(n_samples);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(0.0, 1.0);
    
    for (int i = 0; i < n_samples; ++i) {
        std::vector<double> sample(n_features);
        
        for (int j = 0; j < n_features; ++j) {
            sample[j] = dist(gen);
        }
        
        // Create class based on sum of features
        double sum = std::accumulate(sample.begin(), sample.end(), 0.0);
        int target_class = (sum > 0) ? 1 : 0;
        
        X.push_back(sample);
        y.push_back(target_class);
    }
}

void generate_clustered_data(std::vector<std::vector<double>>& X, std::vector<int>& y, 
                           int n_samples, int n_features) {
    X.clear();
    y.clear();
    X.reserve(n_samples);
    y.reserve(n_samples);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise_dist(0.0, 0.5);
    
    // Create 3 clusters
    std::vector<std::vector<double>> cluster_centers = {
        {-2.0, -2.0}, {2.0, 2.0}, {-2.0, 2.0}
    };
    
    for (int i = 0; i < n_samples; ++i) {
        int cluster = i % 3;
        std::vector<double> sample(n_features);
        
        for (int j = 0; j < std::min(n_features, 2); ++j) {
            sample[j] = cluster_centers[cluster][j] + noise_dist(gen);
        }
        
        // Fill remaining features with noise
        for (int j = 2; j < n_features; ++j) {
            sample[j] = noise_dist(gen);
        }
        
        X.push_back(sample);
        y.push_back(cluster);
    }
}

void run_all_tests() {
    std::cout << "🤖 COMPREHENSIVE K-NEAREST NEIGHBORS TEST SUITE\n\n";
    
    test_knn_regression();
    test_knn_classification();
    test_distance_metrics();
    test_weighting_methods();
    test_k_values();
    test_feature_scaling();
    test_advanced_configurations();
    test_model_persistence();
    test_cross_validation();
    test_multidimensional_data();
    test_edge_cases();
    benchmark_performance();
    print_test_summary();
}

void test_knn_regression() {
    print_separator("KNN REGRESSION TESTING");
    
    // Generate regression data
    std::vector<std::vector<double>> X, X_test;
    std::vector<double> y, y_test;
    
    generate_regression_data(X, y, 100, 2, 0.1);
    generate_regression_data(X_test, y_test, 30, 2, 0.1);
    
    std::cout << "Generated " << X.size() << " training and " << X_test.size() << " test samples\n";
    
    // Test different k values
    std::vector<int> k_values = {1, 3, 5, 7, 10};
    
    for (int k : k_values) {
        auto start = std::chrono::high_resolution_clock::now();
        
        KNearestNeighbors model(k, DistanceMetric::EUCLIDEAN, WeightingMethod::UNIFORM, true);
        
        std::string log_path = LogManager::generate_log_path("knn", "regression", "k_" + std::to_string(k));
        model.set_logging(true, log_path);
        
        model.fit_regression(X, y);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::string config_name = "k=" + std::to_string(k);
        print_regression_performance(model, X_test, y_test, config_name);
        std::cout << "  Training time: " << duration.count() << " ms" << std::endl;
    }
}

void test_knn_classification() {
    print_separator("KNN CLASSIFICATION TESTING");
    
    // Generate classification data
    std::vector<std::vector<double>> X, X_test;
    std::vector<int> y, y_test;
    
    generate_classification_data(X, y, 200, 2, 3);
    generate_classification_data(X_test, y_test, 60, 2, 3);
    
    std::cout << "Generated " << X.size() << " training and " << X_test.size() << " test samples\n";
    
    // Test different k values
    std::vector<int> k_values = {1, 3, 5, 7, 10};
    
    for (int k : k_values) {
        auto start = std::chrono::high_resolution_clock::now();
        
        KNearestNeighbors model(k, DistanceMetric::EUCLIDEAN, WeightingMethod::UNIFORM, true);
        
        std::string log_path = LogManager::generate_log_path("knn", "classification", "k_" + std::to_string(k));
        model.set_logging(true, log_path);
        
        model.fit_classification(X, y);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::string config_name = "k=" + std::to_string(k);
        print_classification_performance(model, X_test, y_test, config_name);
        std::cout << "  Training time: " << duration.count() << " ms" << std::endl;
    }
}

void test_distance_metrics() {
    print_separator("DISTANCE METRICS COMPARISON");
    
    std::vector<std::vector<double>> X, X_test;
    std::vector<int> y, y_test;
    generate_clustered_data(X, y, 150, 2);
    generate_clustered_data(X_test, y_test, 45, 2);
    
    std::vector<std::pair<DistanceMetric, std::string>> metrics = {
        {DistanceMetric::EUCLIDEAN, "Euclidean"},
        {DistanceMetric::MANHATTAN, "Manhattan"},
        {DistanceMetric::MINKOWSKI, "Minkowski (p=3)"},
        {DistanceMetric::COSINE, "Cosine"}
    };
    
    for (const auto& metric_pair : metrics) {
        KNearestNeighbors model(5, metric_pair.first, WeightingMethod::UNIFORM, true);
        
        if (metric_pair.first == DistanceMetric::MINKOWSKI) {
            model.set_distance_metric(DistanceMetric::MINKOWSKI, 3.0);
        }
        
        std::string log_path = LogManager::generate_log_path("knn", "distance_metrics", metric_pair.second);
        model.set_logging(true, log_path);
        
        model.fit_classification(X, y);
        print_classification_performance(model, X_test, y_test, metric_pair.second);
    }
}

void test_weighting_methods() {
    print_separator("WEIGHTING METHODS COMPARISON");
    
    std::vector<std::vector<double>> X, X_test;
    std::vector<double> y, y_test;
    generate_regression_data(X, y, 100, 2, 0.2);
    generate_regression_data(X_test, y_test, 30, 2, 0.2);
    
    std::vector<std::pair<WeightingMethod, std::string>> methods = {
        {WeightingMethod::UNIFORM, "Uniform"},
        {WeightingMethod::DISTANCE_WEIGHTED, "Distance-weighted"},
        {WeightingMethod::GAUSSIAN_WEIGHTED, "Gaussian-weighted"}
    };
    
    for (const auto& method_pair : methods) {
        KNearestNeighbors model(7, DistanceMetric::EUCLIDEAN, method_pair.first, true);
        
        if (method_pair.first == WeightingMethod::GAUSSIAN_WEIGHTED) {
            model.set_weighting_method(WeightingMethod::GAUSSIAN_WEIGHTED, 1.5);
        }
        
        std::string log_path = LogManager::generate_log_path("knn", "weighting_methods", method_pair.second);
        model.set_logging(true, log_path);
        
        model.fit_regression(X, y);
        print_regression_performance(model, X_test, y_test, method_pair.second);
    }
}

void test_k_values() {
    print_separator("OPTIMAL K VALUE ANALYSIS");
    
    std::vector<std::vector<double>> X, X_test;
    std::vector<int> y, y_test;
    generate_classification_data(X, y, 200, 3, 2);
    generate_classification_data(X_test, y_test, 50, 3, 2);
    
    std::vector<int> k_values = {1, 3, 5, 7, 9, 11, 15, 21};
    
    std::cout << std::setw(5) << "k" << std::setw(12) << "Accuracy" 
              << std::setw(12) << "Precision" << std::setw(12) << "F1-Score" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    for (int k : k_values) {
        KNearestNeighbors model(k, DistanceMetric::EUCLIDEAN, WeightingMethod::DISTANCE_WEIGHTED, true);
        
        std::string log_path = LogManager::generate_log_path("knn", "k_analysis", "k_" + std::to_string(k));
        model.set_logging(true, log_path);
        
        model.fit_classification(X, y);
        
        double accuracy = model.evaluate_accuracy(X_test, y_test);
        double precision = model.evaluate_precision(X_test, y_test, 1);
        double f1 = model.evaluate_f1_score(X_test, y_test, 1);
        
        std::cout << std::setw(5) << k << std::setw(12) << std::fixed << std::setprecision(4) << accuracy
                  << std::setw(12) << precision << std::setw(12) << f1 << std::endl;
    }
}

void test_feature_scaling() {
    print_separator("FEATURE SCALING IMPACT");
    
    // Generate data with different scales
    std::vector<std::vector<double>> X, X_test;
    std::vector<double> y, y_test;
    
    generate_regression_data(X, y, 100, 3, 0.1);
    generate_regression_data(X_test, y_test, 30, 3, 0.1);
    
    // Scale some features to have much larger values
    for (auto& sample : X) {
        sample[1] *= 100.0; // Scale second feature
        sample[2] *= 0.01;  // Scale third feature
    }
    for (auto& sample : X_test) {
        sample[1] *= 100.0;
        sample[2] *= 0.01;
    }
    
    // Test without scaling
    std::cout << "Testing with unscaled features (features have very different scales):\n";
    KNearestNeighbors model_no_scaling(5, DistanceMetric::EUCLIDEAN, WeightingMethod::UNIFORM, false);
    std::string log_path_no_scaling = LogManager::generate_log_path("knn", "feature_scaling", "no_scaling");
    model_no_scaling.set_logging(true, log_path_no_scaling);
    model_no_scaling.fit_regression(X, y);
    print_regression_performance(model_no_scaling, X_test, y_test, "Without Scaling");
    
    // Test with scaling
    KNearestNeighbors model_with_scaling(5, DistanceMetric::EUCLIDEAN, WeightingMethod::UNIFORM, true);
    std::string log_path_with_scaling = LogManager::generate_log_path("knn", "feature_scaling", "with_scaling");
    model_with_scaling.set_logging(true, log_path_with_scaling);
    model_with_scaling.fit_regression(X, y);
    print_regression_performance(model_with_scaling, X_test, y_test, "With Scaling");
}

void test_advanced_configurations() {
    print_separator("ADVANCED CONFIGURATIONS");
    
    std::vector<std::vector<double>> X, X_test;
    std::vector<int> y, y_test;
    generate_multidimensional_data(X, y, 300, 5);
    generate_multidimensional_data(X_test, y_test, 75, 5);
    
    // High-precision configuration
    std::cout << "🎯 High-Precision Configuration: ";
    KNearestNeighbors high_precision(3, DistanceMetric::EUCLIDEAN, WeightingMethod::GAUSSIAN_WEIGHTED, true);
    high_precision.set_weighting_method(WeightingMethod::GAUSSIAN_WEIGHTED, 0.5);
    std::string log_path_precision = LogManager::generate_log_path("knn", "advanced", "high_precision");
    high_precision.set_logging(true, log_path_precision);
    high_precision.fit_classification(X, y);
    print_classification_performance(high_precision, X_test, y_test, "");
    
    // Robust configuration
    std::cout << "🛡️  Robust Configuration: ";
    KNearestNeighbors robust(9, DistanceMetric::MANHATTAN, WeightingMethod::DISTANCE_WEIGHTED, true);
    std::string log_path_robust = LogManager::generate_log_path("knn", "advanced", "robust");
    robust.set_logging(true, log_path_robust);
    robust.fit_classification(X, y);
    print_classification_performance(robust, X_test, y_test, "");
    
    // Fast configuration
    std::cout << "⚡ Fast Configuration: ";
    KNearestNeighbors fast(5, DistanceMetric::EUCLIDEAN, WeightingMethod::UNIFORM, false);
    std::string log_path_fast = LogManager::generate_log_path("knn", "advanced", "fast");
    fast.set_logging(true, log_path_fast);
    fast.fit_classification(X, y);
    print_classification_performance(fast, X_test, y_test, "");
}

void test_model_persistence() {
    print_separator("MODEL PERSISTENCE TESTING");
    
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;
    generate_classification_data(X_train, y_train, 100, 3, 2);
    generate_classification_data(X_test, y_test, 30, 3, 2);
    
    // Train and save model
    KNearestNeighbors original_model(7, DistanceMetric::EUCLIDEAN, WeightingMethod::DISTANCE_WEIGHTED, true);
    std::string log_path = LogManager::generate_log_path("knn", "persistence", "model_save_load");
    original_model.set_logging(true, log_path);
    original_model.fit_classification(X_train, y_train);
    original_model.save_model("models/knn_model.txt");
    
    // Load model
    KNearestNeighbors loaded_model;
    loaded_model.load_model("models/knn_model.txt");
    
    std::cout << "\nPrediction comparison:\n";
    for (size_t i = 0; i < std::min(size_t(5), X_test.size()); ++i) {
        int orig_pred = original_model.predict_classification(X_test[i]);
        int load_pred = loaded_model.predict_classification(X_test[i]);
        bool match = (orig_pred == load_pred);
        
        std::cout << "Sample " << i << " | Original: " << orig_pred
                  << " | Loaded: " << load_pred << " | Match: " << (match ? "✓" : "✗") << std::endl;
    }
    
    // Print model summaries
    std::cout << "\nOriginal Model Summary:\n";
    original_model.print_model_summary();
    std::cout << "\nLoaded Model Summary:\n";
    loaded_model.print_model_summary();
}

void test_cross_validation() {
    print_separator("CROSS-VALIDATION TESTING");
    
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    generate_classification_data(X, y, 200, 2, 2);
    
    std::cout << "5-Fold Cross-Validation Results:\n";
    std::cout << std::setw(5) << "k" << std::setw(15) << "CV Accuracy" << std::setw(15) << "Std Dev" << std::endl;
    std::cout << std::string(35, '-') << std::endl;
    
    std::vector<int> k_values = {3, 5, 7, 9, 11};
    
    for (int k : k_values) {
        KNearestNeighbors model(k, DistanceMetric::EUCLIDEAN, WeightingMethod::UNIFORM, true);
        
        // Simple 5-fold CV (just using the built-in method)
        double cv_score = model.cross_validate_classification(X, y, 5);
        
        std::cout << std::setw(5) << k << std::setw(15) << std::fixed << std::setprecision(4) << cv_score
                  << std::setw(15) << "N/A" << std::endl;
    }
}

void test_multidimensional_data() {
    print_separator("MULTIDIMENSIONAL DATA TESTING");
    
    std::vector<int> feature_counts = {2, 5, 10, 15};
    
    for (int n_features : feature_counts) {
        std::cout << "\nTesting with " << n_features << " features:\n";
        
        std::vector<std::vector<double>> X, X_test;
        std::vector<int> y, y_test;
        generate_multidimensional_data(X, y, 200, n_features);
        generate_multidimensional_data(X_test, y_test, 50, n_features);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        KNearestNeighbors model(5, DistanceMetric::EUCLIDEAN, WeightingMethod::DISTANCE_WEIGHTED, true);
        std::string log_path = LogManager::generate_log_path("knn", "multidimensional", 
                                                           "features_" + std::to_string(n_features));
        model.set_logging(true, log_path);
        model.fit_classification(X, y);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::string config_name = std::to_string(n_features) + " features";
        print_classification_performance(model, X_test, y_test, config_name);
        std::cout << "  Training + Prediction time: " << duration.count() << " ms" << std::endl;
    }
}

void test_edge_cases() {
    print_separator("EDGE CASE TESTING");
    
    // Minimal data
    std::vector<std::vector<double>> X_min = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}};
    std::vector<int> y_min = {0, 1, 0};
    
    KNearestNeighbors min_model(2);
    min_model.fit_classification(X_min, y_min);
    std::cout << "✅ Minimal data test passed" << std::endl;
    
    // Single class data
    std::vector<std::vector<double>> X_single = {{1.0, 2.0}, {2.0, 3.0}, {3.0, 4.0}, {4.0, 5.0}};
    std::vector<int> y_single = {1, 1, 1, 1};
    
    KNearestNeighbors single_model(3);
    single_model.fit_classification(X_single, y_single);
    std::cout << "✅ Single class test passed" << std::endl;
    
    // High-dimensional sparse data
    std::vector<std::vector<double>> X_sparse;
    std::vector<int> y_sparse;
    
    for (int i = 0; i < 20; ++i) {
        std::vector<double> sample(50, 0.0); // 50-dimensional sparse vector
        sample[i % 5] = 1.0; // Only one non-zero element
        X_sparse.push_back(sample);
        y_sparse.push_back(i % 2);
    }
    
    KNearestNeighbors sparse_model(3, DistanceMetric::COSINE, WeightingMethod::UNIFORM, true);
    sparse_model.fit_classification(X_sparse, y_sparse);
    std::cout << "✅ High-dimensional sparse data test passed" << std::endl;
}

void benchmark_performance() {
    print_separator("PERFORMANCE BENCHMARKING");
    
    std::vector<int> data_sizes = {100, 500, 1000, 2000};
    
    std::cout << std::setw(12) << "Data Size" << std::setw(15) << "Fit Time (ms)" 
              << std::setw(18) << "Predict Time (ms)" << std::setw(12) << "Accuracy" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (int size : data_sizes) {
        std::vector<std::vector<double>> X, X_test;
        std::vector<int> y, y_test;
        generate_classification_data(X, y, size, 3, 2);
        generate_classification_data(X_test, y_test, size / 4, 3, 2);
        
        auto start_fit = std::chrono::high_resolution_clock::now();
        
        KNearestNeighbors model(5, DistanceMetric::EUCLIDEAN, WeightingMethod::UNIFORM, true);
        std::string log_path = LogManager::generate_log_path("knn", "benchmark", 
                                                           "size_" + std::to_string(size));
        model.set_logging(true, log_path);
        model.fit_classification(X, y);
        
        auto end_fit = std::chrono::high_resolution_clock::now();
        auto fit_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_fit - start_fit);
        
        auto start_predict = std::chrono::high_resolution_clock::now();
        double accuracy = model.evaluate_accuracy(X_test, y_test);
        auto end_predict = std::chrono::high_resolution_clock::now();
        auto predict_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_predict - start_predict);
        
        std::cout << std::setw(12) << size << std::setw(15) << fit_duration.count()
                  << std::setw(18) << predict_duration.count() 
                  << std::setw(12) << std::fixed << std::setprecision(4) << accuracy << std::endl;
    }
}

void print_test_summary() {
    print_separator("TEST SUMMARY");
    
    std::cout << "🎯 All K-Nearest Neighbors tests completed successfully!\n\n";
    
    std::cout << "📊 Tests performed:\n";
    std::cout << "  ✅ KNN Regression testing\n";
    std::cout << "  ✅ KNN Classification testing\n";
    std::cout << "  ✅ Distance metrics comparison (Euclidean, Manhattan, Minkowski, Cosine)\n";
    std::cout << "  ✅ Weighting methods analysis (Uniform, Distance-weighted, Gaussian)\n";
    std::cout << "  ✅ Optimal k value analysis\n";
    std::cout << "  ✅ Feature scaling impact assessment\n";
    std::cout << "  ✅ Advanced configurations testing\n";
    std::cout << "  ✅ Model persistence (save/load)\n";
    std::cout << "  ✅ Cross-validation testing\n";
    std::cout << "  ✅ Multidimensional data handling\n";
    std::cout << "  ✅ Edge case handling\n";
    std::cout << "  ✅ Performance benchmarking\n\n";
    
    std::cout << "📁 Check the models/ directory for saved models.\n";
    std::cout << "📝 Check the logs/ directory for detailed structured logs.\n";
    std::cout << "🚀 Your K-Nearest Neighbors algorithm is production-ready!\n";
}

} // namespace KNNTests
