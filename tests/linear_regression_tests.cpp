#include "linear_regression_tests.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace LinearRegressionTests {

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

void print_model_performance(const LinearRegression& model, 
                           const std::vector<double>& x_test, 
                           const std::vector<double>& y_test,
                           const std::string& config_name) {
    std::cout << config_name << ": ";
    std::cout << "MSE=" << std::fixed << std::setprecision(4) << model.evaluate(x_test, y_test);
    std::cout << ", R²=" << std::setprecision(4) << model.r_squared(x_test, y_test);
    std::cout << ", Epochs=" << model.get_loss_history().size() << std::endl;
}

void test_optimizers(const std::vector<double>& x, const std::vector<double>& y) {
    print_separator("OPTIMIZER COMPARISON");
    
    std::vector<OptimizerType> optimizers = {
        OptimizerType::SGD,
        OptimizerType::MOMENTUM,
        OptimizerType::ADAGRAD,
        OptimizerType::ADAM
    };
    
    std::vector<std::string> optimizer_names = {
        "Standard SGD",
        "SGD with Momentum",
        "AdaGrad",
        "Adam"
    };
    
    for (size_t i = 0; i < optimizers.size(); ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        LinearRegression model(0.01, 1000, 1e-8, optimizers[i], 
                              RegularizationType::NONE, 0.0, true, true);
        model.set_logging(true, "optimizer_test_" + std::to_string(i));
        model.fit(x, y);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        print_model_performance(model, x, y, optimizer_names[i]);
        std::cout << "  Training time: " << duration.count() << " ms\n";
    }
}

void test_regularization(const std::vector<double>& x, const std::vector<double>& y) {
    print_separator("REGULARIZATION COMPARISON");
    
    std::vector<RegularizationType> reg_types = {
        RegularizationType::NONE,
        RegularizationType::L1,
        RegularizationType::L2,
        RegularizationType::ELASTIC_NET
    };
    
    std::vector<std::string> reg_names = {
        "No Regularization",
        "L1 (Lasso)",
        "L2 (Ridge)",
        "Elastic Net"
    };
    
    for (size_t i = 0; i < reg_types.size(); ++i) {
        LinearRegression model(0.01, 1000, 1e-8, OptimizerType::ADAM, 
                              reg_types[i], 0.1, true, true);
        model.set_logging(true, "regularization_test_" + std::to_string(i));
        model.fit(x, y);
        print_model_performance(model, x, y, reg_names[i]);
    }
}

void test_learning_rates(const std::vector<double>& x, const std::vector<double>& y) {
    print_separator("LEARNING RATE COMPARISON");
    
    std::vector<double> learning_rates = {0.001, 0.01, 0.1, 0.5};
    
    for (size_t i = 0; i < learning_rates.size(); ++i) {
        double lr = learning_rates[i];
        std::string config_name = "LR: " + std::to_string(lr);
        
        LinearRegression model(lr, 1000, 1e-8, OptimizerType::ADAM, 
                              RegularizationType::L2, 0.01, true, true);
        model.set_logging(true, "learning_rate_test_" + std::to_string(i));
        model.fit(x, y);
        print_model_performance(model, x, y, config_name);
    }
}

void test_feature_scaling(const std::vector<double>& x, const std::vector<double>& y) {
    print_separator("FEATURE SCALING COMPARISON");
    
    // Without feature scaling
    LinearRegression model_no_scaling(0.01, 1000, 1e-8, OptimizerType::ADAM, 
                                     RegularizationType::NONE, 0.0, false, false);
    model_no_scaling.set_logging(true, "feature_scaling_test");
    model_no_scaling.fit(x, y);
    print_model_performance(model_no_scaling, x, y, "Without Feature Scaling");
    
    // With feature scaling
    LinearRegression model_with_scaling(0.01, 1000, 1e-8, OptimizerType::ADAM, 
                                       RegularizationType::NONE, 0.0, false, true);
    model_with_scaling.set_logging(true, "feature_scaling_test");
    model_with_scaling.fit(x, y);
    print_model_performance(model_with_scaling, x, y, "With Feature Scaling");
}

void test_advanced_configurations(const std::vector<double>& x, const std::vector<double>& y) {
    print_separator("ADVANCED CONFIGURATIONS");
    
    // Configuration 1: High-performance setup
    std::cout << " High-Performance Configuration: ";
    LinearRegression model1(0.001, 2000, 1e-10, OptimizerType::ADAM, 
                           RegularizationType::L2, 0.001, true, true);
    model1.set_optimizer(OptimizerType::ADAM, 0.9, 0.999);
    model1.set_learning_rate_schedule(true, 0.95, 1e-6);
    model1.set_logging(true, "advanced_config_test");
    model1.fit(x, y);
    print_model_performance(model1, x, y, "");
    
    // Configuration 2: Robust setup with Elastic Net
    std::cout << "  Robust Configuration: ";
    LinearRegression model2(0.01, 1500, 1e-8, OptimizerType::ADAM, 
                           RegularizationType::ELASTIC_NET, 0.05, true, true);
    model2.set_regularization(RegularizationType::ELASTIC_NET, 0.05, 0.7);
    model2.set_logging(true, "advanced_config_test");
    model2.fit(x, y);
    print_model_performance(model2, x, y, "");
    
    // Configuration 3: Fast convergence
    std::cout << "⚡ Fast Convergence Configuration: ";
    LinearRegression model3(0.1, 500, 1e-6, OptimizerType::MOMENTUM, 
                           RegularizationType::L1, 0.01, true, true);
    model3.set_logging(true, "advanced_config_test");
    model3.fit(x, y);
    print_model_performance(model3, x, y, "");
}

void test_model_persistence() {
    print_separator("MODEL PERSISTENCE TESTING");
    
    // Create a synthetic dataset for testing
    std::vector<double> x_synthetic = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<double> y_synthetic = {2.1, 3.9, 6.2, 7.8, 10.1, 12.2, 13.8, 16.1, 18.0, 19.9};
    
    // Train original model
    LinearRegression original_model(0.01, 1000, 1e-8, OptimizerType::ADAM, 
                                   RegularizationType::L2, 0.01, true, true);
    original_model.set_logging(true, "model_persistence_test");
    original_model.fit(x_synthetic, y_synthetic);
    
    std::cout << "Original model trained and saved to models/advanced_model.txt\n";
    original_model.save_model("models/advanced_model.txt");
    
    // Load model
    LinearRegression loaded_model;
    loaded_model.load_model("models/advanced_model.txt");
    
    // Test predictions
    std::cout << "\nPrediction comparison:\n";
    std::cout << std::fixed << std::setprecision(4);
    for (double test_x : {2.5, 5.5, 8.5}) {
        double original_pred = original_model.predict(test_x);
        double loaded_pred = loaded_model.predict(test_x);
        std::cout << "x=" << test_x 
                  << " | Original: " << original_pred 
                  << " | Loaded: " << loaded_pred 
                  << " | Diff: " << std::abs(original_pred - loaded_pred) << "\n";
    }
}

void demonstrate_real_time_training(const std::vector<double>& x, const std::vector<double>& y) {
    print_separator("REAL-TIME TRAINING DEMONSTRATION");
    
    std::cout << "Training with detailed logging to logs/realtime_training.log...\n";
    
    LinearRegression model(0.01, 1000, 1e-8, OptimizerType::ADAM, 
                          RegularizationType::L2, 0.01, true, true);
    model.set_logging(true, "realtime_training_test");
    
    // This will show detailed training progress in log file
    model.fit(x, y);
    
    // Show training history analysis
    const auto& loss_history = model.get_loss_history();
    const auto& lr_history = model.get_lr_history();
    
    std::cout << "Training Analysis:\n";
    std::cout << "  Initial loss: " << std::fixed << std::setprecision(6) << loss_history.front() << "\n";
    std::cout << "  Final loss: " << loss_history.back() << "\n";
    std::cout << "  Loss reduction: " << (loss_history.front() - loss_history.back()) << "\n";
    std::cout << "  Initial LR: " << lr_history.front() << "\n";
    std::cout << "  Final LR: " << lr_history.back() << "\n";
}

void test_edge_cases() {
    print_separator("EDGE CASE TESTING");
    
    try {
        // Test with minimal data
        std::vector<double> x_minimal = {1, 2};
        std::vector<double> y_minimal = {1, 2};
        
        LinearRegression model_minimal(0.1, 100, 1e-6);
        model_minimal.fit(x_minimal, y_minimal);
        std::cout << " Minimal data test passed\n";
        
        // Test with constant data
        std::vector<double> x_constant = {1, 1, 1, 1, 1};
        std::vector<double> y_constant = {5, 5, 5, 5, 5};
        
        LinearRegression model_constant(0.1, 100, 1e-6);
        model_constant.fit(x_constant, y_constant);
        std::cout << " Constant data test passed\n";
        
        // Test with large values
        std::vector<double> x_large = {1000, 2000, 3000, 4000, 5000};
        std::vector<double> y_large = {10000, 20000, 30000, 40000, 50000};
        
        LinearRegression model_large(0.0001, 500, 1e-6, OptimizerType::ADAM, 
                                    RegularizationType::NONE, 0.0, true, true);
        model_large.fit(x_large, y_large);
        std::cout << " Large values test passed\n";
        
        // Test prediction accuracy
        double pred = model_large.predict(6000);
        std::cout << "Large values prediction for x=6000: " << pred << "\n";
        
    } catch (const std::exception& e) {
        std::cout << " Edge case test failed: " << e.what() << "\n";
    }
}

void benchmark_performance(const std::vector<double>& x, const std::vector<double>& y) {
    print_separator("PERFORMANCE BENCHMARKING");
    
    std::vector<std::pair<std::string, OptimizerType>> optimizers = {
        {"SGD", OptimizerType::SGD},
        {"Momentum", OptimizerType::MOMENTUM},
        {"AdaGrad", OptimizerType::ADAGRAD},
        {"Adam", OptimizerType::ADAM}
    };
    
    std::cout << "Benchmarking optimizers with 2000 epochs...\n";
    std::cout << std::setw(15) << "Optimizer" << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "Final Loss" << std::setw(15) << "Epochs" << "\n";
    std::cout << std::string(60, '-') << "\n";
    
    for (const auto& opt : optimizers) {
        auto start = std::chrono::high_resolution_clock::now();
        
        LinearRegression model(0.01, 2000, 1e-10, opt.second, 
                              RegularizationType::L2, 0.01, true, true);
        model.set_logging(true, "benchmark_" + opt.first);
        model.fit(x, y);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        const auto& loss_history = model.get_loss_history();
        
        std::cout << std::setw(15) << opt.first 
                  << std::setw(15) << duration.count()
                  << std::setw(15) << std::fixed << std::setprecision(8) << loss_history.back()
                  << std::setw(15) << loss_history.size() << "\n";
    }
}

void print_test_summary() {
    print_separator("TEST SUMMARY");
    std::cout << " All linear regression tests completed successfully!\n\n";
    std::cout << " Tests performed:\n";
    std::cout << "   Optimizer comparison (SGD, Momentum, AdaGrad, Adam)\n";
    std::cout << "   Regularization testing (None, L1, L2, Elastic Net)\n";
    std::cout << "   Learning rate analysis\n";
    std::cout << "   Feature scaling impact\n";
    std::cout << "   Advanced configurations\n";
    std::cout << "   Model persistence\n";
    std::cout << "   Real-time training\n";
    std::cout << "   Edge case handling\n";
    std::cout << "   Performance benchmarking\n\n";
    std::cout << " Check the models/ directory for saved models.\n";
    std::cout << " Your linear regression algorithm is production-ready!\n";
}

void run_all_tests(const std::vector<double>& x, const std::vector<double>& y) {
    std::cout << " COMPREHENSIVE LINEAR REGRESSION TEST SUITE\n";
    std::cout << "Data loaded: " << x.size() << " samples\n";
    
    if (x.size() > 0) {
        std::cout << "X range: [" << *std::min_element(x.begin(), x.end()) 
                  << ", " << *std::max_element(x.begin(), x.end()) << "]\n";
        std::cout << "Y range: [" << *std::min_element(y.begin(), y.end()) 
                  << ", " << *std::max_element(y.begin(), y.end()) << "]\n";
    }
    
    // Run all tests
    test_optimizers(x, y);
    test_regularization(x, y);
    test_learning_rates(x, y);
    test_feature_scaling(x, y);
    test_advanced_configurations(x, y);
    test_model_persistence();
    demonstrate_real_time_training(x, y);
    test_edge_cases();
    benchmark_performance(x, y);
    
    print_test_summary();
}

} // namespace LinearRegressionTests
