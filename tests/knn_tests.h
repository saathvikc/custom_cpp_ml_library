#pragma once
#include "../classification/knn.h"
#include <vector>
#include <string>

namespace KNNTests {
    // Test suite functions
    void run_all_tests();
    void test_knn_regression();
    void test_knn_classification();
    void test_distance_metrics();
    void test_weighting_methods();
    void test_k_values();
    void test_feature_scaling();
    void test_advanced_configurations();
    void test_model_persistence();
    void test_cross_validation();
    void test_edge_cases();
    void benchmark_performance();
    void test_multidimensional_data();
    
    // Utility functions
    void print_separator(const std::string& title);
    void print_regression_performance(const KNearestNeighbors& model, 
                                    const std::vector<std::vector<double>>& X_test, 
                                    const std::vector<double>& y_test,
                                    const std::string& config_name);
    void print_classification_performance(const KNearestNeighbors& model, 
                                        const std::vector<std::vector<double>>& X_test, 
                                        const std::vector<int>& y_test,
                                        const std::string& config_name);
    void print_test_summary();
    
    // Data generation utilities
    void generate_regression_data(std::vector<std::vector<double>>& X, std::vector<double>& y, 
                                int n_samples = 100, int n_features = 2, double noise = 0.1);
    void generate_classification_data(std::vector<std::vector<double>>& X, std::vector<int>& y, 
                                    int n_samples = 200, int n_features = 2, int n_classes = 2);
    void generate_multidimensional_data(std::vector<std::vector<double>>& X, std::vector<int>& y, 
                                      int n_samples = 300, int n_features = 5);
    void generate_clustered_data(std::vector<std::vector<double>>& X, std::vector<int>& y, 
                               int n_samples = 150, int n_features = 2);
}
