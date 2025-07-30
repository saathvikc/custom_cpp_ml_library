#pragma once

#include "../neural_networks/neural_network.h"
#include <vector>
#include <string>

namespace NeuralNetworkTests {

// Test data generation
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> generate_regression_data(int n_samples, int n_features, int n_outputs = 1, double noise = 0.1);
std::pair<std::vector<std::vector<double>>, std::vector<int>> generate_classification_data(int n_samples, int n_features, int n_classes = 2, double noise = 0.1);
std::pair<std::vector<std::vector<double>>, std::vector<int>> generate_xor_data(int n_samples = 1000);
std::pair<std::vector<std::vector<double>>, std::vector<int>> generate_spiral_data(int n_samples = 300, int n_classes = 3);
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> generate_sine_wave_data(int n_samples = 500, int n_features = 1);

// Utility functions
void print_separator(const std::string& title);
void print_regression_results(const NeuralNetwork& model, 
                             const std::vector<std::vector<double>>& X_test, 
                             const std::vector<std::vector<double>>& y_test,
                             const std::string& test_name);
void print_classification_results(const NeuralNetwork& model, 
                                const std::vector<std::vector<double>>& X_test, 
                                const std::vector<int>& y_test,
                                const std::string& test_name);

// Core test functions
void test_basic_regression();
void test_basic_classification();
void test_multiclass_classification();
void test_activation_functions();
void test_loss_functions();
void test_optimizers();
void test_regularization();
void test_network_architectures();
void test_learning_rate_scheduling();
void test_early_stopping();
void test_model_persistence();
void test_batch_training();
void test_xor_problem();
void test_spiral_classification();
void test_sine_wave_regression();
void test_overfitting_prevention();
void test_convergence_analysis();
void test_gradient_checking();
void test_edge_cases();
void benchmark_performance();

// Main test runner
void run_all_tests();

}
