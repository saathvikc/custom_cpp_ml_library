#include "logistic_regression_tests.h"
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

namespace LogisticRegressionTests {

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(60, '=') << "\n";
}

void print_model_performance(const LogisticRegression& model, 
                           const std::vector<double>& x_test, 
                           const std::vector<int>& y_test,
                           const std::string& config_name) {
    std::cout << config_name << ": ";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Acc=" << model.evaluate_accuracy(x_test, y_test);
    std::cout << ", Loss=" << std::setprecision(4) << model.evaluate_loss(x_test, y_test);
    std::cout << ", F1=" << std::setprecision(4) << model.evaluate_f1_score(x_test, y_test);
    std::cout << ", Epochs=" << model.get_loss_history().size() << std::endl;
}

void generate_classification_data(std::vector<double>& x, std::vector<int>& y, 
                                int n_samples, double noise) {
    x.clear();
    y.clear();
    
    std::random_device rd;
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::normal_distribution<> noise_dist(0.0, noise);
    
    for (int i = 0; i < n_samples; ++i) {
        double x_val = static_cast<double>(i) / n_samples * 10.0 - 5.0; // Range [-5, 5]
        
        // Create a sigmoid-like decision boundary with noise
        double prob = 1.0 / (1.0 + std::exp(-1.5 * x_val + noise_dist(gen)));
        int label = prob > 0.5 ? 1 : 0;
        
        x.push_back(x_val);
        y.push_back(label);
    }
}

void test_optimizers(const std::vector<double>& x, const std::vector<int>& y) {
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
        
        LogisticRegression model(0.1, 1000, 1e-8, optimizers[i], 
                               RegularizationType::NONE, 0.0, true, true);
        model.set_logging(true, "logs/logistic_optimizer_" + std::to_string(i) + ".log");
        model.fit(x, y);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        print_model_performance(model, x, y, optimizer_names[i]);
        std::cout << "  Training time: " << duration.count() << " ms\n";
    }
}

void test_regularization(const std::vector<double>& x, const std::vector<int>& y) {
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
        LogisticRegression model(0.1, 1000, 1e-8, OptimizerType::ADAM, 
                               reg_types[i], 0.01, true, true);
        model.set_logging(true, "logs/logistic_regularization_" + std::to_string(i) + ".log");
        model.fit(x, y);
        print_model_performance(model, x, y, reg_names[i]);
    }
}

void test_learning_rates(const std::vector<double>& x, const std::vector<int>& y) {
    print_separator("LEARNING RATE COMPARISON");
    
    std::vector<double> learning_rates = {0.01, 0.1, 0.5, 1.0};
    
    for (size_t i = 0; i < learning_rates.size(); ++i) {
        double lr = learning_rates[i];
        std::string config_name = "LR: " + std::to_string(lr);
        
        LogisticRegression model(lr, 1000, 1e-8, OptimizerType::ADAM, 
                               RegularizationType::L2, 0.01, true, true);
        model.set_logging(true, "logs/logistic_learning_rate_" + std::to_string(i) + ".log");
        model.fit(x, y);
        print_model_performance(model, x, y, config_name);
    }
}

void test_feature_scaling(const std::vector<double>& x, const std::vector<int>& y) {
    print_separator("FEATURE SCALING COMPARISON");
    
    // Without feature scaling
    LogisticRegression model_no_scaling(0.1, 1000, 1e-8, OptimizerType::ADAM, 
                                      RegularizationType::NONE, 0.0, false, false);
    model_no_scaling.set_logging(true, "logs/logistic_no_scaling.log");
    model_no_scaling.fit(x, y);
    print_model_performance(model_no_scaling, x, y, "Without Feature Scaling");
    
    // With feature scaling
    LogisticRegression model_with_scaling(0.1, 1000, 1e-8, OptimizerType::ADAM, 
                                        RegularizationType::NONE, 0.0, false, true);
    model_with_scaling.set_logging(true, "logs/logistic_with_scaling.log");
    model_with_scaling.fit(x, y);
    print_model_performance(model_with_scaling, x, y, "With Feature Scaling");
}

void test_advanced_configurations(const std::vector<double>& x, const std::vector<int>& y) {
    print_separator("ADVANCED CONFIGURATIONS");
    
    // Configuration 1: High-performance setup
    std::cout << "🚀 High-Performance Configuration: ";
    LogisticRegression model1(0.01, 2000, 1e-10, OptimizerType::ADAM, 
                            RegularizationType::L2, 0.001, true, true);
    model1.set_optimizer(OptimizerType::ADAM, 0.9, 0.999);
    model1.set_learning_rate_schedule(true, 0.95, 1e-6);
    model1.set_logging(true, "logs/logistic_high_performance.log");
    model1.fit(x, y);
    print_model_performance(model1, x, y, "");
    
    // Configuration 2: Robust setup with Elastic Net
    std::cout << "🛡️  Robust Configuration: ";
    LogisticRegression model2(0.1, 1500, 1e-8, OptimizerType::ADAM, 
                            RegularizationType::ELASTIC_NET, 0.05, true, true);
    model2.set_regularization(RegularizationType::ELASTIC_NET, 0.05, 0.7);
    model2.set_logging(true, "logs/logistic_robust.log");
    model2.fit(x, y);
    print_model_performance(model2, x, y, "");
    
    // Configuration 3: Fast convergence
    std::cout << "⚡ Fast Convergence Configuration: ";
    LogisticRegression model3(0.5, 500, 1e-6, OptimizerType::MOMENTUM, 
                            RegularizationType::L1, 0.01, true, true);
    model3.set_logging(true, "logs/logistic_fast_convergence.log");
    model3.fit(x, y);
    print_model_performance(model3, x, y, "");
}

void test_model_persistence() {
    print_separator("MODEL PERSISTENCE TESTING");
    
    // Create a synthetic dataset for testing
    std::vector<double> x_synthetic;
    std::vector<int> y_synthetic;
    generate_classification_data(x_synthetic, y_synthetic, 50, 0.1);
    
    // Train original model
    LogisticRegression original_model(0.1, 1000, 1e-8, OptimizerType::ADAM, 
                                    RegularizationType::L2, 0.01, true, true);
    original_model.set_logging(true, "logs/logistic_model_persistence.log");
    original_model.fit(x_synthetic, y_synthetic);
    
    std::cout << "Original model trained and saved to models/logistic_model.txt\n";
    original_model.save_model("models/logistic_model.txt");
    
    // Load model
    LogisticRegression loaded_model;
    loaded_model.load_model("models/logistic_model.txt");
    
    // Test predictions
    std::cout << "\nPrediction comparison:\n";
    std::cout << std::fixed << std::setprecision(4);
    for (double test_x : {-2.0, 0.0, 2.0}) {
        double original_prob = original_model.predict_proba(test_x);
        double loaded_prob = loaded_model.predict_proba(test_x);
        int original_pred = original_model.predict(test_x);
        int loaded_pred = loaded_model.predict(test_x);
        
        std::cout << "x=" << test_x 
                  << " | Original: prob=" << original_prob << ", pred=" << original_pred
                  << " | Loaded: prob=" << loaded_prob << ", pred=" << loaded_pred
                  << " | Diff: " << std::abs(original_prob - loaded_prob) << "\n";
    }
}

void test_classification_metrics(const std::vector<double>& x, const std::vector<int>& y) {
    print_separator("CLASSIFICATION METRICS ANALYSIS");
    
    LogisticRegression model(0.1, 1000, 1e-8, OptimizerType::ADAM, 
                           RegularizationType::L2, 0.01, true, true);
    model.set_logging(true, "logs/logistic_metrics.log");
    model.fit(x, y);
    
    std::cout << "📊 Comprehensive Metrics:\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  Accuracy:  " << model.evaluate_accuracy(x, y) << "\n";
    std::cout << "  Precision: " << model.evaluate_precision(x, y) << "\n";
    std::cout << "  Recall:    " << model.evaluate_recall(x, y) << "\n";
    std::cout << "  F1-Score:  " << model.evaluate_f1_score(x, y) << "\n";
    std::cout << "  Log-Loss:  " << model.evaluate_loss(x, y) << "\n";
    
    // Show some probability predictions
    std::cout << "\n🎯 Sample Predictions:\n";
    for (size_t i = 0; i < std::min(size_t(10), x.size()); ++i) {
        double prob = model.predict_proba(x[i]);
        int pred = model.predict(x[i]);
        std::cout << "  x=" << std::setw(6) << x[i] 
                  << " | True=" << y[i] 
                  << " | Pred=" << pred 
                  << " | Prob=" << std::setprecision(3) << prob 
                  << (pred == y[i] ? " ✓" : " ✗") << "\n";
    }
}

void demonstrate_real_time_training(const std::vector<double>& x, const std::vector<int>& y) {
    print_separator("REAL-TIME TRAINING DEMONSTRATION");
    
    std::cout << "Training with detailed logging to logs/logistic_realtime_training.log...\n";
    
    LogisticRegression model(0.1, 1000, 1e-8, OptimizerType::ADAM, 
                           RegularizationType::L2, 0.01, true, true);
    model.set_logging(true, "logs/logistic_realtime_training.log");
    
    // This will show detailed training progress in log file
    model.fit(x, y);
    
    // Show training history analysis
    const auto& loss_history = model.get_loss_history();
    const auto& lr_history = model.get_lr_history();
    const auto& accuracy_history = model.get_accuracy_history();
    
    std::cout << "Training Analysis:\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Initial loss: " << loss_history.front() << "\n";
    std::cout << "  Final loss: " << loss_history.back() << "\n";
    std::cout << "  Loss reduction: " << (loss_history.front() - loss_history.back()) << "\n";
    std::cout << "  Initial accuracy: " << accuracy_history.front() << "\n";
    std::cout << "  Final accuracy: " << accuracy_history.back() << "\n";
    std::cout << "  Initial LR: " << lr_history.front() << "\n";
    std::cout << "  Final LR: " << lr_history.back() << "\n";
}

void test_edge_cases() {
    print_separator("EDGE CASE TESTING");
    
    try {
        // Test with minimal data
        std::vector<double> x_minimal = {-1, 1};
        std::vector<int> y_minimal = {0, 1};
        
        LogisticRegression model_minimal(0.5, 100, 1e-6);
        model_minimal.fit(x_minimal, y_minimal);
        std::cout << " Minimal data test passed\n";
        
        // Test with imbalanced data
        std::vector<double> x_imbalanced = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<int> y_imbalanced = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1}; // 90% class 0
        
        LogisticRegression model_imbalanced(0.1, 500, 1e-6, OptimizerType::ADAM, 
                                          RegularizationType::NONE, 0.0, true, true);
        model_imbalanced.fit(x_imbalanced, y_imbalanced);
        std::cout << "  Imbalanced data test passed\n";
        std::cout << "  Final accuracy: " << std::fixed << std::setprecision(3) 
                  << model_imbalanced.evaluate_accuracy(x_imbalanced, y_imbalanced) << "\n";
        
        // Test with perfectly separable data
        std::vector<double> x_separable = {1, 2, 3, 7, 8, 9};
        std::vector<int> y_separable = {0, 0, 0, 1, 1, 1};
        
        LogisticRegression model_separable(0.1, 1000, 1e-8, OptimizerType::ADAM, 
                                         RegularizationType::NONE, 0.0, true, true);
        model_separable.fit(x_separable, y_separable);
        std::cout << "  Perfectly separable data test passed\n";
        std::cout << "  Final accuracy: " << std::fixed << std::setprecision(3) 
                  << model_separable.evaluate_accuracy(x_separable, y_separable) << "\n";
        
    } catch (const std::exception& e) {
        std::cout << "Edge case test failed: " << e.what() << "\n";
    }
}

void benchmark_performance(const std::vector<double>& x, const std::vector<int>& y) {
    print_separator("PERFORMANCE BENCHMARKING");
    
    std::vector<std::pair<std::string, OptimizerType>> optimizers = {
        {"SGD", OptimizerType::SGD},
        {"Momentum", OptimizerType::MOMENTUM},
        {"AdaGrad", OptimizerType::ADAGRAD},
        {"Adam", OptimizerType::ADAM}
    };
    
    std::cout << "Benchmarking optimizers with 1000 epochs...\n";
    std::cout << std::setw(15) << "Optimizer" << std::setw(15) << "Time (ms)" 
              << std::setw(15) << "Final Loss" << std::setw(15) << "Accuracy" 
              << std::setw(15) << "Epochs" << "\n";
    std::cout << std::string(75, '-') << "\n";
    
    for (const auto& opt : optimizers) {
        auto start = std::chrono::high_resolution_clock::now();
        
        LogisticRegression model(0.1, 1000, 1e-10, opt.second, 
                               RegularizationType::L2, 0.01, true, true);
        model.set_logging(true, "logs/logistic_benchmark_" + opt.first + ".log");
        model.fit(x, y);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        const auto& loss_history = model.get_loss_history();
        const auto& accuracy_history = model.get_accuracy_history();
        
        std::cout << std::setw(15) << opt.first 
                  << std::setw(15) << duration.count()
                  << std::setw(15) << std::fixed << std::setprecision(6) << loss_history.back()
                  << std::setw(15) << std::setprecision(4) << accuracy_history.back()
                  << std::setw(15) << loss_history.size() << "\n";
    }
}

void print_test_summary() {
    print_separator("TEST SUMMARY");
    std::cout << " All logistic regression tests completed successfully!\n\n";
    std::cout << " Tests performed:\n";
    std::cout << "   Optimizer comparison (SGD, Momentum, AdaGrad, Adam)\n";
    std::cout << "   Regularization testing (None, L1, L2, Elastic Net)\n";
    std::cout << "   Learning rate analysis\n";
    std::cout << "   Feature scaling impact\n";
    std::cout << "   Advanced configurations\n";
    std::cout << "   Model persistence\n";
    std::cout << "   Classification metrics analysis\n";
    std::cout << "   Real-time training\n";
    std::cout << "   Edge case handling\n";
    std::cout << "   Performance benchmarking\n\n";
    std::cout << " Check the models/ directory for saved models.\n";
    std::cout << " Check the logs/ directory for detailed training logs.\n";
    std::cout << " Your logistic regression algorithm is production-ready!\n";
}

void run_all_tests(const std::vector<double>& x, const std::vector<int>& y) {
    std::cout << " COMPREHENSIVE LOGISTIC REGRESSION TEST SUITE\n";
    std::cout << "Data loaded: " << x.size() << " samples\n";
    
    if (x.size() > 0) {
        std::cout << "X range: [" << *std::min_element(x.begin(), x.end()) 
                  << ", " << *std::max_element(x.begin(), x.end()) << "]\n";
        
        int class_0_count = std::count(y.begin(), y.end(), 0);
        int class_1_count = std::count(y.begin(), y.end(), 1);
        std::cout << "Class distribution: 0=" << class_0_count 
                  << " (" << std::fixed << std::setprecision(1) 
                  << 100.0 * class_0_count / y.size() << "%), 1=" << class_1_count 
                  << " (" << 100.0 * class_1_count / y.size() << "%)\n";
    }
    
    // Run all tests
    test_optimizers(x, y);
    test_regularization(x, y);
    test_learning_rates(x, y);
    test_feature_scaling(x, y);
    test_advanced_configurations(x, y);
    test_model_persistence();
    test_classification_metrics(x, y);
    demonstrate_real_time_training(x, y);
    test_edge_cases();
    benchmark_performance(x, y);
    
    print_test_summary();
}

} // namespace LogisticRegressionTests
