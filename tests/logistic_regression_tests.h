#pragma once
#include "../regression/logistic_regression.h"
#include <vector>
#include <string>

namespace LogisticRegressionTests {
    // Test suite functions
    void run_all_tests(const std::vector<double>& x, const std::vector<int>& y);
    void test_optimizers(const std::vector<double>& x, const std::vector<int>& y);
    void test_regularization(const std::vector<double>& x, const std::vector<int>& y);
    void test_learning_rates(const std::vector<double>& x, const std::vector<int>& y);
    void test_feature_scaling(const std::vector<double>& x, const std::vector<int>& y);
    void test_advanced_configurations(const std::vector<double>& x, const std::vector<int>& y);
    void test_model_persistence();
    void demonstrate_real_time_training(const std::vector<double>& x, const std::vector<int>& y);
    void test_edge_cases();
    void benchmark_performance(const std::vector<double>& x, const std::vector<int>& y);
    void test_classification_metrics(const std::vector<double>& x, const std::vector<int>& y);
    
    // Utility functions
    void print_separator(const std::string& title);
    void print_model_performance(const LogisticRegression& model, 
                               const std::vector<double>& x_test, 
                               const std::vector<int>& y_test,
                               const std::string& config_name);
    void print_test_summary();
    
    // Data generation utilities
    void generate_classification_data(std::vector<double>& x, std::vector<int>& y, 
                                    int n_samples = 100, double noise = 0.1);
}
