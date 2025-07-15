#pragma once
#include "../regression/polynomial_regression.h"
#include <vector>
#include <string>

namespace PolynomialRegressionTests {
    // Test suite functions
    void run_all_tests(const std::vector<double>& x, const std::vector<double>& y);
    void test_polynomial_degrees(const std::vector<double>& x, const std::vector<double>& y);
    void test_optimizers(const std::vector<double>& x, const std::vector<double>& y);
    void test_regularization(const std::vector<double>& x, const std::vector<double>& y);
    void test_learning_rates(const std::vector<double>& x, const std::vector<double>& y);
    void test_feature_scaling(const std::vector<double>& x, const std::vector<double>& y);
    void test_advanced_configurations(const std::vector<double>& x, const std::vector<double>& y);
    void test_model_persistence();
    void demonstrate_overfitting_prevention(const std::vector<double>& x, const std::vector<double>& y);
    void test_edge_cases();
    void benchmark_performance(const std::vector<double>& x, const std::vector<double>& y);
    void test_polynomial_interpretability(const std::vector<double>& x, const std::vector<double>& y);
    
    // Utility functions
    void print_separator(const std::string& title);
    void print_model_performance(const PolynomialRegression& model, 
                               const std::vector<double>& x_test, 
                               const std::vector<double>& y_test,
                               const std::string& config_name);
    void print_test_summary();
    
    // Data generation utilities
    void generate_polynomial_data(std::vector<double>& x, std::vector<double>& y, 
                                int degree = 3, int n_samples = 100, double noise = 0.1);
    void generate_complex_polynomial_data(std::vector<double>& x, std::vector<double>& y, 
                                        int n_samples = 200, double noise = 0.2);
}
