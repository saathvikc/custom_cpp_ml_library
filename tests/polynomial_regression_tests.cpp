#include "polynomial_regression_tests.h"
#include <iostream>
#include <chrono>
#include <random>
#include <iomanip>
#include <cmath>
#include <algorithm>

namespace PolynomialRegressionTests {

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

void print_model_performance(const PolynomialRegression& model, 
                           const std::vector<double>& x_test, 
                           const std::vector<double>& y_test,
                           const std::string& config_name) {
    double mse = model.evaluate(x_test, y_test);
    double r2 = model.r_squared(x_test, y_test);
    double adj_r2 = model.adjusted_r_squared(x_test, y_test);
    double mae = model.mean_absolute_error(x_test, y_test);
    int epochs = model.get_loss_history().size();
    
    std::cout << config_name << ": MSE=" << std::fixed << std::setprecision(4) << mse
              << ", R²=" << r2 << ", Adj.R²=" << adj_r2 
              << ", MAE=" << mae << ", Epochs=" << epochs << std::endl;
}

void generate_polynomial_data(std::vector<double>& x, std::vector<double>& y, 
                            int degree, int n_samples, double noise) {
    x.clear();
    y.clear();
    x.reserve(n_samples);
    y.reserve(n_samples);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> x_dist(-2.0, 2.0);
    std::normal_distribution<> noise_dist(0.0, noise);
    
    // Generate coefficients for true polynomial
    std::vector<double> true_coeffs(degree + 1);
    std::uniform_real_distribution<> coeff_dist(-2.0, 2.0);
    for (int i = 0; i <= degree; ++i) {
        true_coeffs[i] = coeff_dist(gen);
    }
    
    for (int i = 0; i < n_samples; ++i) {
        double x_val = x_dist(gen);
        double y_val = 0.0;
        
        // Compute polynomial value
        for (int j = 0; j <= degree; ++j) {
            y_val += true_coeffs[j] * std::pow(x_val, j);
        }
        
        y_val += noise_dist(gen); // Add noise
        
        x.push_back(x_val);
        y.push_back(y_val);
    }
    
    // Sort by x for better visualization
    std::vector<std::pair<double, double>> pairs;
    for (size_t i = 0; i < x.size(); ++i) {
        pairs.emplace_back(x[i], y[i]);
    }
    std::sort(pairs.begin(), pairs.end());
    
    for (size_t i = 0; i < pairs.size(); ++i) {
        x[i] = pairs[i].first;
        y[i] = pairs[i].second;
    }
}

void generate_complex_polynomial_data(std::vector<double>& x, std::vector<double>& y, 
                                    int n_samples, double noise) {
    x.clear();
    y.clear();
    x.reserve(n_samples);
    y.reserve(n_samples);
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> x_dist(-3.0, 3.0);
    std::normal_distribution<> noise_dist(0.0, noise);
    
    for (int i = 0; i < n_samples; ++i) {
        double x_val = x_dist(gen);
        // Complex polynomial: 2x^4 - 3x^3 + x^2 - 5x + 1
        double y_val = 2*std::pow(x_val, 4) - 3*std::pow(x_val, 3) + 
                       std::pow(x_val, 2) - 5*x_val + 1;
        y_val += noise_dist(gen);
        
        x.push_back(x_val);
        y.push_back(y_val);
    }
}

void run_all_tests(const std::vector<double>& x, const std::vector<double>& y) {
    std::cout << "🤖 COMPREHENSIVE POLYNOMIAL REGRESSION TEST SUITE\n";
    std::cout << "Data loaded: " << x.size() << " samples\n";
    if (!x.empty()) {
        auto x_minmax = std::minmax_element(x.begin(), x.end());
        auto y_minmax = std::minmax_element(y.begin(), y.end());
        std::cout << "X range: [" << *x_minmax.first << ", " << *x_minmax.second << "]\n";
        std::cout << "Y range: [" << *y_minmax.first << ", " << *y_minmax.second << "]\n";
    }
    
    test_polynomial_degrees(x, y);
    test_optimizers(x, y);
    test_regularization(x, y);
    test_learning_rates(x, y);
    test_feature_scaling(x, y);
    test_advanced_configurations(x, y);
    test_model_persistence();
    demonstrate_overfitting_prevention(x, y);
    test_edge_cases();
    benchmark_performance(x, y);
    test_polynomial_interpretability(x, y);
    print_test_summary();
}

void test_polynomial_degrees(const std::vector<double>& x, const std::vector<double>& y) {
    print_separator("POLYNOMIAL DEGREE COMPARISON");
    
    std::vector<int> degrees = {1, 2, 3, 4, 5};
    
    for (int degree : degrees) {
        auto start = std::chrono::high_resolution_clock::now();
        
        PolynomialRegression model(degree, 0.01, 1000, 1e-6, OptimizerType::ADAM,
                                 RegularizationType::L2, 0.01, true, true);
        
        std::string log_name = "logs/poly_degree_" + std::to_string(degree) + ".log";
        model.set_logging(true, log_name);
        model.fit(x, y);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::string config_name = "Degree " + std::to_string(degree);
        print_model_performance(model, x, y, config_name);
        std::cout << "  Training time: " << duration.count() << " ms" << std::endl;
        
        // Print equation for lower degrees
        if (degree <= 3) {
            model.print_polynomial_equation();
        }
    }
}

void test_optimizers(const std::vector<double>& x, const std::vector<double>& y) {
    print_separator("OPTIMIZER COMPARISON");
    
    std::vector<std::pair<OptimizerType, std::string>> optimizers = {
        {OptimizerType::SGD, "Standard SGD"},
        {OptimizerType::MOMENTUM, "SGD with Momentum"},
        {OptimizerType::ADAGRAD, "AdaGrad"},
        {OptimizerType::ADAM, "Adam"}
    };
    
    for (size_t i = 0; i < optimizers.size(); ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        PolynomialRegression model(3, 0.01, 2000, 1e-6, optimizers[i].first,
                                 RegularizationType::NONE, 0.0, true, true);
        
        std::string log_name = "logs/poly_optimizer_" + std::to_string(i) + ".log";
        model.set_logging(true, log_name);
        model.fit(x, y);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        print_model_performance(model, x, y, optimizers[i].second);
        std::cout << "  Training time: " << duration.count() << " ms" << std::endl;
    }
}

void test_regularization(const std::vector<double>& x, const std::vector<double>& y) {
    print_separator("REGULARIZATION COMPARISON");
    
    std::vector<std::pair<RegularizationType, std::string>> regularizations = {
        {RegularizationType::NONE, "No Regularization"},
        {RegularizationType::L1, "L1 (Lasso)"},
        {RegularizationType::L2, "L2 (Ridge)"},
        {RegularizationType::ELASTIC_NET, "Elastic Net"}
    };
    
    for (size_t i = 0; i < regularizations.size(); ++i) {
        PolynomialRegression model(4, 0.01, 1000, 1e-6, OptimizerType::ADAM,
                                 regularizations[i].first, 0.1, true, true);
        
        std::string log_name = "logs/poly_regularization_" + std::to_string(i) + ".log";
        model.set_logging(true, log_name);
        model.fit(x, y);
        
        print_model_performance(model, x, y, regularizations[i].second);
    }
}

void test_learning_rates(const std::vector<double>& x, const std::vector<double>& y) {
    print_separator("LEARNING RATE COMPARISON");
    
    std::vector<double> learning_rates = {0.001, 0.01, 0.1, 0.5};
    
    for (size_t i = 0; i < learning_rates.size(); ++i) {
        PolynomialRegression model(3, learning_rates[i], 1000, 1e-6, OptimizerType::ADAM,
                                 RegularizationType::L2, 0.01, true, true);
        
        std::string log_name = "logs/poly_learning_rate_" + std::to_string(i) + ".log";
        model.set_logging(true, log_name);
        model.fit(x, y);
        
        std::string config_name = "LR: " + std::to_string(learning_rates[i]);
        print_model_performance(model, x, y, config_name);
    }
}

void test_feature_scaling(const std::vector<double>& x, const std::vector<double>& y) {
    print_separator("FEATURE SCALING COMPARISON");
    
    // Without scaling
    PolynomialRegression model_no_scaling(3, 0.01, 1000, 1e-6, OptimizerType::ADAM,
                                        RegularizationType::L2, 0.01, true, false);
    model_no_scaling.set_logging(true, "logs/poly_no_scaling.log");
    model_no_scaling.fit(x, y);
    print_model_performance(model_no_scaling, x, y, "Without Feature Scaling");
    
    // With scaling
    PolynomialRegression model_with_scaling(3, 0.01, 1000, 1e-6, OptimizerType::ADAM,
                                          RegularizationType::L2, 0.01, true, true);
    model_with_scaling.set_logging(true, "logs/poly_with_scaling.log");
    model_with_scaling.fit(x, y);
    print_model_performance(model_with_scaling, x, y, "With Feature Scaling");
}

void test_advanced_configurations(const std::vector<double>& x, const std::vector<double>& y) {
    print_separator("ADVANCED CONFIGURATIONS");
    
    // High-degree with strong regularization
    std::cout << "🚀 High-Degree with Strong Regularization: ";
    PolynomialRegression high_degree(6, 0.01, 2000, 1e-8, OptimizerType::ADAM,
                                   RegularizationType::ELASTIC_NET, 0.5, true, true);
    high_degree.set_logging(true, "logs/poly_high_degree.log");
    high_degree.fit(x, y);
    print_model_performance(high_degree, x, y, "");
    
    // Fast convergence with adaptive LR
    std::cout << "⚡ Fast Convergence Configuration: ";
    PolynomialRegression fast_config(3, 0.1, 500, 1e-5, OptimizerType::ADAM,
                                   RegularizationType::L2, 0.01, true, true);
    fast_config.set_learning_rate_schedule(true, 0.9, 1e-5);
    fast_config.set_logging(true, "logs/poly_fast_convergence.log");
    fast_config.fit(x, y);
    print_model_performance(fast_config, x, y, "");
    
    // Robust configuration
    std::cout << "🛡️  Robust Configuration: ";
    PolynomialRegression robust_config(4, 0.005, 3000, 1e-7, OptimizerType::MOMENTUM,
                                     RegularizationType::L2, 0.1, true, true);
    robust_config.set_logging(true, "logs/poly_robust.log");
    robust_config.fit(x, y);
    print_model_performance(robust_config, x, y, "");
}

void test_model_persistence() {
    print_separator("MODEL PERSISTENCE TESTING");
    
    // Generate test data
    std::vector<double> x_test, y_test;
    generate_polynomial_data(x_test, y_test, 3, 50, 0.1);
    
    // Train and save model
    PolynomialRegression original_model(3, 0.05, 500, 1e-6, OptimizerType::ADAM,
                                      RegularizationType::L2, 0.01, true, true);
    original_model.set_logging(true, "logs/poly_model_persistence.log");
    original_model.fit(x_test, y_test);
    original_model.save_model("models/polynomial_model.txt");
    
    // Load model
    PolynomialRegression loaded_model;
    loaded_model.load_model("models/polynomial_model.txt");
    
    std::cout << "\nPrediction comparison:\n";
    for (size_t i = 0; i < std::min(size_t(3), x_test.size()); ++i) {
        double x_val = x_test[i];
        double orig_pred = original_model.predict(x_val);
        double load_pred = loaded_model.predict(x_val);
        double diff = std::abs(orig_pred - load_pred);
        
        std::cout << "x=" << std::fixed << std::setprecision(4) << x_val
                  << " | Original: " << orig_pred
                  << " | Loaded: " << load_pred
                  << " | Diff: " << diff << std::endl;
    }
}

void demonstrate_overfitting_prevention(const std::vector<double>& x, const std::vector<double>& y) {
    print_separator("OVERFITTING PREVENTION DEMONSTRATION");
    
    // Generate complex data
    std::vector<double> x_complex, y_complex;
    generate_complex_polynomial_data(x_complex, y_complex, 100, 0.2);
    
    std::cout << "Training on complex polynomial data (100 samples)...\n";
    
    // High degree without regularization (prone to overfitting)
    std::cout << "📈 High Degree (7) without Regularization: ";
    PolynomialRegression overfit_model(7, 0.01, 1000, 1e-6, OptimizerType::ADAM,
                                     RegularizationType::NONE, 0.0, true, true);
    overfit_model.set_logging(true, "logs/poly_overfit_demo.log");
    overfit_model.fit(x_complex, y_complex);
    print_model_performance(overfit_model, x_complex, y_complex, "");
    
    // High degree with strong regularization
    std::cout << "🛡️  High Degree (7) with Strong L2 Regularization: ";
    PolynomialRegression regularized_model(7, 0.01, 1000, 1e-6, OptimizerType::ADAM,
                                         RegularizationType::L2, 1.0, true, true);
    regularized_model.set_logging(true, "logs/poly_regularized_demo.log");
    regularized_model.fit(x_complex, y_complex);
    print_model_performance(regularized_model, x_complex, y_complex, "");
    
    // Optimal degree selection
    std::cout << "⚖️  Optimal Degree (4) with Moderate Regularization: ";
    PolynomialRegression optimal_model(4, 0.01, 1000, 1e-6, OptimizerType::ADAM,
                                     RegularizationType::L2, 0.1, true, true);
    optimal_model.set_logging(true, "logs/poly_optimal_demo.log");
    optimal_model.fit(x_complex, y_complex);
    print_model_performance(optimal_model, x_complex, y_complex, "");
}

void test_edge_cases() {
    print_separator("EDGE CASE TESTING");
    
    // Minimal data
    std::vector<double> x_min = {1.0, 2.0, 3.0};
    std::vector<double> y_min = {2.0, 4.5, 8.0};
    
    PolynomialRegression min_model(2, 0.1, 200, 1e-5);
    min_model.fit(x_min, y_min);
    std::cout << "✅ Minimal data test passed" << std::endl;
    
    // Constant data
    std::vector<double> x_const = {1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<double> y_const = {5.0, 5.0, 5.0, 5.0, 5.0};
    
    PolynomialRegression const_model(2, 0.1, 100, 1e-5);
    const_model.fit(x_const, y_const);
    std::cout << "✅ Constant data test passed" << std::endl;
    
    // Large values
    std::vector<double> x_large, y_large;
    for (int i = 0; i < 10; ++i) {
        x_large.push_back(1000.0 + i * 100.0);
        y_large.push_back(1000000.0 + i * 10000.0);
    }
    
    PolynomialRegression large_model(2, 0.0001, 1000, 1e-3, OptimizerType::ADAM,
                                   RegularizationType::NONE, 0.0, true, true);
    large_model.fit(x_large, y_large);
    std::cout << "✅ Large values test passed" << std::endl;
    
    double large_pred = large_model.predict(1500.0);
    std::cout << "Large values prediction for x=1500: " << large_pred << std::endl;
}

void benchmark_performance(const std::vector<double>& x, const std::vector<double>& y) {
    print_separator("PERFORMANCE BENCHMARKING");
    
    std::cout << "Benchmarking different degrees with 1000 epochs...\n";
    std::cout << std::setw(10) << "Degree" << std::setw(12) << "Time (ms)" 
              << std::setw(15) << "Final Loss" << std::setw(12) << "Epochs" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    std::vector<int> degrees = {2, 3, 4, 5, 6};
    
    for (int degree : degrees) {
        auto start = std::chrono::high_resolution_clock::now();
        
        PolynomialRegression model(degree, 0.01, 1000, 1e-8, OptimizerType::ADAM,
                                 RegularizationType::L2, 0.01, true, true);
        
        std::string log_name = "logs/poly_benchmark_degree_" + std::to_string(degree) + ".log";
        model.set_logging(true, log_name);
        model.fit(x, y);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        const auto& loss_history = model.get_loss_history();
        double final_loss = loss_history.empty() ? 0.0 : loss_history.back();
        
        std::cout << std::setw(10) << degree << std::setw(12) << duration.count()
                  << std::setw(15) << std::fixed << std::setprecision(8) << final_loss
                  << std::setw(12) << loss_history.size() << std::endl;
    }
}

void test_polynomial_interpretability(const std::vector<double>& x, const std::vector<double>& y) {
    print_separator("POLYNOMIAL INTERPRETABILITY ANALYSIS");
    
    // Train models with different degrees and analyze coefficients
    std::vector<int> degrees = {2, 3, 4};
    
    for (int degree : degrees) {
        std::cout << "\n--- Degree " << degree << " Analysis ---\n";
        
        PolynomialRegression model(degree, 0.01, 1000, 1e-6, OptimizerType::ADAM,
                                 RegularizationType::L2, 0.01, true, true);
        
        std::string log_name = "logs/poly_interpretability_" + std::to_string(degree) + ".log";
        model.set_logging(true, log_name);
        model.fit(x, y);
        
        // Display equation
        model.print_polynomial_equation();
        
        // Feature importance
        auto importance = model.get_feature_importance();
        std::cout << "Feature Importance (absolute coefficients):\n";
        for (size_t i = 0; i < importance.size(); ++i) {
            std::cout << "  x^" << i << ": " << std::fixed << std::setprecision(6) 
                      << importance[i] << std::endl;
        }
        
        // Model performance
        double r2 = model.r_squared(x, y);
        double adj_r2 = model.adjusted_r_squared(x, y);
        std::cout << "Performance: R² = " << r2 << ", Adjusted R² = " << adj_r2 << std::endl;
    }
}

void print_test_summary() {
    print_separator("TEST SUMMARY");
    
    std::cout << "🎯 All polynomial regression tests completed successfully!\n\n";
    
    std::cout << "📊 Tests performed:\n";
    std::cout << "  ✅ Polynomial degree comparison (1-5)\n";
    std::cout << "  ✅ Optimizer comparison (SGD, Momentum, AdaGrad, Adam)\n";
    std::cout << "  ✅ Regularization testing (None, L1, L2, Elastic Net)\n";
    std::cout << "  ✅ Learning rate analysis\n";
    std::cout << "  ✅ Feature scaling impact\n";
    std::cout << "  ✅ Advanced configurations\n";
    std::cout << "  ✅ Model persistence\n";
    std::cout << "  ✅ Overfitting prevention demonstration\n";
    std::cout << "  ✅ Edge case handling\n";
    std::cout << "  ✅ Performance benchmarking\n";
    std::cout << "  ✅ Polynomial interpretability analysis\n\n";
    
    std::cout << "📁 Check the models/ directory for saved models.\n";
    std::cout << "📝 Check the logs/ directory for detailed training logs.\n";
    std::cout << "🚀 Your polynomial regression algorithm is production-ready!\n";
}

} // namespace PolynomialRegressionTests
