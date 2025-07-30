#pragma once
#include "../classification/naive_bayes.h"
#include "../utils/log_manager.h"
#include <vector>
#include <string>

namespace NaiveBayesTests {
    // Test suite functions
    void run_all_tests();
    void test_naive_bayes_types();
    void test_smoothing_methods();
    void test_feature_scaling();
    void test_cross_validation();
    void test_model_persistence();
    void test_classification_metrics();
    void test_probability_predictions();
    void test_feature_importance();
    void test_edge_cases();
    void benchmark_performance();
    
    // Data generation utilities
    std::pair<std::vector<std::vector<double>>, std::vector<int>> generate_gaussian_data(int n_samples = 300, int n_features = 4, int n_classes = 3);
    std::pair<std::vector<std::vector<double>>, std::vector<int>> generate_binary_data(int n_samples = 200, int n_features = 5, int n_classes = 2);
    std::pair<std::vector<std::vector<double>>, std::vector<int>> generate_multinomial_data(int n_samples = 250, int n_features = 6, int n_classes = 3);
    std::pair<std::vector<std::vector<double>>, std::vector<int>> generate_iris_like_data(int n_samples = 150);
    
    // Test utilities
    void print_separator(const std::string& title);
    void print_classification_results(const NaiveBayes& model, 
                                    const std::vector<std::vector<double>>& X_test, 
                                    const std::vector<int>& y_test,
                                    const std::string& test_name);
    void compare_naive_bayes_types(const std::vector<std::vector<double>>& X, 
                                  const std::vector<int>& y);
    void demonstrate_probability_predictions(const NaiveBayes& model,
                                           const std::vector<std::vector<double>>& X_test,
                                           const std::vector<int>& y_test);
};
