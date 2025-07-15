#pragma once
#include "../regression/linear_regression.h"
#include "../utils/log_manager.h"
#include <vector>
#include <string>

namespace LinearRegressionTests {
    // Test suite functions
    void run_all_tests(const std::vector<double>& x, const std::vector<double>& y);
    void test_optimizers(const std::vector<double>& x, const std::vector<double>& y);
    void test_regularization(const std::vector<double>& x, const std::vector<double>& y);
    void test_learning_rates(const std::vector<double>& x, const std::vector<double>& y);
    void test_feature_scaling(const std::vector<double>& x, const std::vector<double>& y);
    void test_advanced_configurations(const std::vector<double>& x, const std::vector<double>& y);
    void test_model_persistence();
    void demonstrate_real_time_training(const std::vector<double>& x, const std::vector<double>& y);
    void test_edge_cases();
    void benchmark_performance(const std::vector<double>& x, const std::vector<double>& y);
    
    // Utility functions
    void print_separator(const std::string& title);
    void print_model_performance(const LinearRegression& model, 
                               const std::vector<double>& x_test, 
                               const std::vector<double>& y_test,
                               const std::string& config_name);
    void print_test_summary();
}
