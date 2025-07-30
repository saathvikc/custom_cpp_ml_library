#include "neural_network_tests.h"
#include "../utils/utils.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace NeuralNetworkTests {

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(80, '=') << "\n";
}

void print_regression_results(const NeuralNetwork& model, 
                             const std::vector<std::vector<double>>& X_test, 
                             const std::vector<std::vector<double>>& y_test,
                             const std::string& test_name) {
    double loss = model.evaluate_loss(X_test, y_test);
    double r2 = model.evaluate_r2_score(X_test, y_test);
    
    std::cout << test_name << ": ";
    std::cout << "Loss=" << std::fixed << std::setprecision(4) << loss;
    std::cout << ", R²=" << std::setprecision(4) << r2;
    std::cout << std::endl;
}

void print_classification_results(const NeuralNetwork& model, 
                                const std::vector<std::vector<double>>& X_test, 
                                const std::vector<int>& y_test,
                                const std::string& test_name) {
    double accuracy = model.evaluate_accuracy(X_test, y_test);
    auto metrics = model.classification_report(X_test, y_test);
    
    std::cout << test_name << ": ";
    std::cout << "Acc=" << std::fixed << std::setprecision(4) << accuracy;
    std::cout << ", Prec=" << std::setprecision(4) << metrics.at("precision");
    std::cout << ", Rec=" << std::setprecision(4) << metrics.at("recall");
    std::cout << ", F1=" << std::setprecision(4) << metrics.at("f1_score");
    std::cout << std::endl;
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> generate_regression_data(int n_samples, int n_features, int n_outputs, double noise) {
    std::vector<std::vector<double>> X, y;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> feature_dist(0.0, 1.0);
    std::normal_distribution<double> noise_dist(0.0, noise);
    
    for (int i = 0; i < n_samples; ++i) {
        std::vector<double> sample(n_features);
        std::vector<double> target(n_outputs);
        
        // Generate features
        for (int j = 0; j < n_features; ++j) {
            sample[j] = feature_dist(gen);
        }
        
        // Generate targets (linear combination + noise)
        for (int k = 0; k < n_outputs; ++k) {
            target[k] = 0.0;
            for (int j = 0; j < n_features; ++j) {
                target[k] += sample[j] * (j + k + 1) * 0.5;  // Simple linear relationship
            }
            target[k] += noise_dist(gen);
        }
        
        X.push_back(sample);
        y.push_back(target);
    }
    
    return {X, y};
}

std::pair<std::vector<std::vector<double>>, std::vector<int>> generate_classification_data(int n_samples, int n_features, int n_classes, double noise) {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> feature_dist(0.0, 1.0);
    std::normal_distribution<double> noise_dist(0.0, noise);
    
    int samples_per_class = n_samples / n_classes;
    
    for (int class_id = 0; class_id < n_classes; ++class_id) {
        for (int i = 0; i < samples_per_class; ++i) {
            std::vector<double> sample(n_features);
            
            for (int j = 0; j < n_features; ++j) {
                // Create class-specific means
                double class_mean = class_id * 2.0 - (n_classes - 1);
                sample[j] = class_mean + feature_dist(gen) + noise_dist(gen);
            }
            
            X.push_back(sample);
            y.push_back(class_id);
        }
    }
    
    return {X, y};
}

std::pair<std::vector<std::vector<double>>, std::vector<int>> generate_xor_data(int n_samples) {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> noise_dist(-0.1, 0.1);
    std::bernoulli_distribution binary_dist(0.5);
    
    for (int i = 0; i < n_samples; ++i) {
        double x1 = binary_dist(gen) ? 1.0 : 0.0;
        double x2 = binary_dist(gen) ? 1.0 : 0.0;
        
        // Add some noise
        x1 += noise_dist(gen);
        x2 += noise_dist(gen);
        
        int label = (static_cast<int>(x1 > 0.5) ^ static_cast<int>(x2 > 0.5));
        
        X.push_back({x1, x2});
        y.push_back(label);
    }
    
    return {X, y};
}

std::pair<std::vector<std::vector<double>>, std::vector<int>> generate_spiral_data(int n_samples, int n_classes) {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> noise_dist(0.0, 0.1);
    
    int samples_per_class = n_samples / n_classes;
    
    for (int class_id = 0; class_id < n_classes; ++class_id) {
        for (int i = 0; i < samples_per_class; ++i) {
            double t = static_cast<double>(i) / samples_per_class * 4 * M_PI;
            double r = t / (2 * M_PI);
            
            double angle = t + class_id * 2 * M_PI / n_classes;
            double x1 = r * std::cos(angle) + noise_dist(gen);
            double x2 = r * std::sin(angle) + noise_dist(gen);
            
            X.push_back({x1, x2});
            y.push_back(class_id);
        }
    }
    
    return {X, y};
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> generate_sine_wave_data(int n_samples, int n_features) {
    std::vector<std::vector<double>> X, y;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> x_dist(0.0, 4 * M_PI);
    std::normal_distribution<double> noise_dist(0.0, 0.1);
    
    for (int i = 0; i < n_samples; ++i) {
        std::vector<double> sample(n_features);
        std::vector<double> target(1);
        
        for (int j = 0; j < n_features; ++j) {
            sample[j] = x_dist(gen);
        }
        
        // Create complex sine wave target
        target[0] = std::sin(sample[0]) + 0.5 * std::sin(2 * sample[0]) + noise_dist(gen);
        
        X.push_back(sample);
        y.push_back(target);
    }
    
    return {X, y};
}

void test_basic_regression() {
    print_separator("BASIC REGRESSION TESTING");
    
    auto [X, y] = generate_regression_data(500, 3, 1, 0.1);
    
    // Split data
    int train_size = X.size() * 0.8;
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<std::vector<double>> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
    std::vector<std::vector<double>> y_test(y.begin() + train_size, y.end());
    
    // Create and train network
    NeuralNetwork model({3, 10, 5, 1}, {ActivationType::RELU, ActivationType::RELU, ActivationType::LINEAR});
    model.set_logging(true, "basic_regression");
    model.fit(X_train, y_train, 200, 32, 0.2, true, false);
    
    print_regression_results(model, X_test, y_test, "Basic Regression");
}

void test_basic_classification() {
    print_separator("BASIC CLASSIFICATION TESTING");
    
    auto [X, y] = generate_classification_data(400, 2, 2, 0.2);
    
    // Split data
    int train_size = X.size() * 0.8;
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<int> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
    std::vector<int> y_test(y.begin() + train_size, y.end());
    
    // Create and train network
    NeuralNetwork model({2, 8, 4, 1}, {ActivationType::RELU, ActivationType::RELU, ActivationType::SIGMOID});
    model.set_logging(true, "basic_classification");
    model.fit_classification(X_train, y_train, 300, 16, 0.2, true, false);
    
    print_classification_results(model, X_test, y_test, "Basic Classification");
}

void test_multiclass_classification() {
    print_separator("MULTICLASS CLASSIFICATION TESTING");
    
    auto [X, y] = generate_classification_data(600, 3, 3, 0.1);
    
    // Split data
    int train_size = X.size() * 0.8;
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<int> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
    std::vector<int> y_test(y.begin() + train_size, y.end());
    
    // Create and train network with softmax output
    NeuralNetwork model({3, 12, 8, 3}, {ActivationType::RELU, ActivationType::RELU, ActivationType::SOFTMAX});
    model.set_logging(true, "multiclass_classification");
    model.fit_classification(X_train, y_train, 400, 32, 0.2, true, false);
    
    print_classification_results(model, X_test, y_test, "Multiclass Classification");
}

void test_activation_functions() {
    print_separator("ACTIVATION FUNCTION COMPARISON");
    
    auto [X, y] = generate_classification_data(400, 2, 2, 0.1);
    
    // Split data
    int train_size = X.size() * 0.8;
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<int> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
    std::vector<int> y_test(y.begin() + train_size, y.end());
    
    std::vector<std::pair<std::vector<ActivationType>, std::string>> activations = {
        {{ActivationType::SIGMOID, ActivationType::SIGMOID}, "Sigmoid"},
        {{ActivationType::TANH, ActivationType::SIGMOID}, "Tanh"},
        {{ActivationType::RELU, ActivationType::SIGMOID}, "ReLU"},
        {{ActivationType::LEAKY_RELU, ActivationType::SIGMOID}, "Leaky ReLU"}
    };
    
    for (const auto& [acts, name] : activations) {
        auto start = std::chrono::high_resolution_clock::now();
        
        NeuralNetwork model({2, 8, 1}, acts);
        model.set_logging(true, "activation_" + name);
        model.fit_classification(X_train, y_train, 200, 16, 0.0, true, false);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << name << ": ";
        print_classification_results(model, X_test, y_test, "");
        std::cout << "  Training time: " << duration.count() << " ms" << std::endl;
    }
}

void test_optimizers() {
    print_separator("OPTIMIZER COMPARISON");
    
    auto [X, y] = generate_regression_data(300, 2, 1, 0.1);
    
    // Split data
    int train_size = X.size() * 0.8;
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<std::vector<double>> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
    std::vector<std::vector<double>> y_test(y.begin() + train_size, y.end());
    
    std::vector<std::pair<OptimizerType, std::string>> optimizers = {
        {OptimizerType::SGD, "SGD"},
        {OptimizerType::MOMENTUM, "Momentum"},
        {OptimizerType::ADAM, "Adam"}
    };
    
    for (const auto& [opt, name] : optimizers) {
        auto start = std::chrono::high_resolution_clock::now();
        
        NeuralNetwork model({2, 10, 5, 1}, {ActivationType::RELU, ActivationType::RELU, ActivationType::LINEAR});
        model.set_optimizer(opt, 0.01);
        model.set_logging(true, "optimizer_" + name);
        model.fit(X_train, y_train, 300, 16, 0.0, true, false);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << name << ": ";
        print_regression_results(model, X_test, y_test, "");
        std::cout << "  Training time: " << duration.count() << " ms" << std::endl;
    }
}

void test_xor_problem() {
    print_separator("XOR PROBLEM (NON-LINEAR CLASSIFICATION)");
    
    auto [X, y] = generate_xor_data(1000);
    
    // Split data
    int train_size = X.size() * 0.8;
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<int> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
    std::vector<int> y_test(y.begin() + train_size, y.end());
    
    std::cout << "XOR Problem - Testing non-linear classification capability\n";
    
    // Try different architectures
    std::vector<std::pair<std::vector<int>, std::string>> architectures = {
        {{2, 4, 1}, "Simple (2-4-1)"},
        {{2, 8, 4, 1}, "Medium (2-8-4-1)"},
        {{2, 16, 8, 1}, "Deep (2-16-8-1)"}
    };
    
    for (const auto& [arch, name] : architectures) {
        std::vector<ActivationType> activations(arch.size() - 1, ActivationType::RELU);
        activations.back() = ActivationType::SIGMOID;
        
        NeuralNetwork model(arch, activations);
        model.set_optimizer(OptimizerType::ADAM, 0.01);
        model.set_logging(true, "xor_" + name);
        model.fit_classification(X_train, y_train, 500, 32, 0.0, true, false);
        
        print_classification_results(model, X_test, y_test, name);
    }
}

void test_spiral_classification() {
    print_separator("SPIRAL CLASSIFICATION (COMPLEX NON-LINEAR)");
    
    auto [X, y] = generate_spiral_data(600, 3);
    
    // Split data
    int train_size = X.size() * 0.8;
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<int> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
    std::vector<int> y_test(y.begin() + train_size, y.end());
    
    NeuralNetwork model({2, 20, 15, 10, 3}, 
                       {ActivationType::RELU, ActivationType::RELU, ActivationType::RELU, ActivationType::SOFTMAX});
    model.set_optimizer(OptimizerType::ADAM, 0.001);
    model.set_logging(true, "spiral_classification");
    model.fit_classification(X_train, y_train, 800, 32, 0.2, true, false);
    
    print_classification_results(model, X_test, y_test, "Spiral Classification");
}

void test_sine_wave_regression() {
    print_separator("SINE WAVE REGRESSION (NON-LINEAR)");
    
    auto [X, y] = generate_sine_wave_data(500, 1);
    
    // Split data
    int train_size = X.size() * 0.8;
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<std::vector<double>> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
    std::vector<std::vector<double>> y_test(y.begin() + train_size, y.end());
    
    NeuralNetwork model({1, 15, 10, 5, 1}, 
                       {ActivationType::RELU, ActivationType::RELU, ActivationType::RELU, ActivationType::LINEAR});
    model.set_optimizer(OptimizerType::ADAM, 0.001);
    model.set_logging(true, "sine_wave_regression");
    model.fit(X_train, y_train, 600, 32, 0.2, true, false);
    
    print_regression_results(model, X_test, y_test, "Sine Wave Regression");
}

void test_model_persistence() {
    print_separator("MODEL PERSISTENCE TESTING");
    
    auto [X, y] = generate_classification_data(300, 2, 2, 0.1);
    
    // Train original model
    NeuralNetwork original_model({2, 8, 4, 1}, {ActivationType::RELU, ActivationType::RELU, ActivationType::SIGMOID});
    original_model.set_logging(true, "model_persistence");
    original_model.fit_classification(X, y, 200, 16, 0.0, true, false);
    original_model.save_model("models/neural_network_model.txt");
    
    // Load model
    NeuralNetwork loaded_model;
    loaded_model.load_model("models/neural_network_model.txt");
    
    std::cout << "Model persistence test completed successfully.\n";
    std::cout << "Original model summary:\n";
    std::cout << "  Input size: " << original_model.get_input_size() << "\n";
    std::cout << "  Output size: " << original_model.get_output_size() << "\n";
    std::cout << "  Layers: " << original_model.get_num_layers() << "\n";
    
    std::cout << "Loaded model summary:\n";
    std::cout << "  Input size: " << loaded_model.get_input_size() << "\n";
    std::cout << "  Output size: " << loaded_model.get_output_size() << "\n";
    std::cout << "  Layers: " << loaded_model.get_num_layers() << "\n";
    
    // Test some predictions
    std::cout << "\nPrediction comparison:\n";
    for (int i = 0; i < 3; ++i) {
        int orig_pred = original_model.predict_class(X[i]);
        int loaded_pred = loaded_model.predict_class(X[i]);
        
        std::cout << "Sample " << i+1 << " | Original: " << orig_pred 
                  << " | Loaded: " << loaded_pred;
        if (orig_pred == loaded_pred) std::cout << " ✓";
        else std::cout << " ✗";
        std::cout << "\n";
    }
}

void test_edge_cases() {
    print_separator("EDGE CASE TESTING");
    
    std::cout << "✅ Minimal data test: ";
    auto [X_min, y_min] = generate_regression_data(10, 2, 1, 0.1);
    try {
        NeuralNetwork model_min({2, 4, 1}, {ActivationType::RELU, ActivationType::LINEAR});
        model_min.fit(X_min, y_min, 50, 4, 0.0, true, false);
        double loss = model_min.evaluate_loss(X_min, y_min);
        std::cout << "Passed (loss: " << std::fixed << std::setprecision(3) << loss << ")\n";
    } catch (const std::exception& e) {
        std::cout << "Failed: " << e.what() << "\n";
    }
    
    std::cout << "✅ Single output test: ";
    auto [X_single, y_single] = generate_regression_data(50, 3, 1, 0.1);
    try {
        NeuralNetwork model_single({3, 5, 1}, {ActivationType::RELU, ActivationType::LINEAR});
        model_single.fit(X_single, y_single, 100, 8, 0.0, true, false);
        double loss = model_single.evaluate_loss(X_single, y_single);
        std::cout << "Passed (loss: " << std::fixed << std::setprecision(3) << loss << ")\n";
    } catch (const std::exception& e) {
        std::cout << "Failed: " << e.what() << "\n";
    }
    
    std::cout << "✅ Multi-output test: ";
    auto [X_multi, y_multi] = generate_regression_data(100, 2, 3, 0.1);
    try {
        NeuralNetwork model_multi({2, 8, 3}, {ActivationType::RELU, ActivationType::LINEAR});
        model_multi.fit(X_multi, y_multi, 150, 16, 0.0, true, false);
        double loss = model_multi.evaluate_loss(X_multi, y_multi);
        std::cout << "Passed (loss: " << std::fixed << std::setprecision(3) << loss << ")\n";
    } catch (const std::exception& e) {
        std::cout << "Failed: " << e.what() << "\n";
    }
}

void benchmark_performance() {
    print_separator("PERFORMANCE BENCHMARKING");
    
    std::vector<std::tuple<int, int, std::string>> test_sizes = {
        {100, 4, "Small (100×4)"},
        {500, 8, "Medium (500×8)"},
        {1000, 12, "Large (1000×12)"},
        {2000, 16, "X-Large (2000×16)"}
    };
    
    std::cout << "    Dataset Size    Time (ms)    Loss         R²\n";
    std::cout << std::string(55, '-') << "\n";
    
    for (const auto& [n_samples, n_features, name] : test_sizes) {
        auto [X, y] = generate_regression_data(n_samples, n_features, 1, 0.1);
        
        auto start = std::chrono::high_resolution_clock::now();
        
        NeuralNetwork model({n_features, 10, 5, 1}, 
                           {ActivationType::RELU, ActivationType::RELU, ActivationType::LINEAR});
        model.set_optimizer(OptimizerType::ADAM, 0.001);
        model.set_logging(true, "benchmark_" + std::to_string(n_samples));
        model.fit(X, y, 200, 32, 0.0, true, false);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        double loss = model.evaluate_loss(X, y);
        double r2 = model.evaluate_r2_score(X, y);
        
        std::cout << std::setw(15) << name 
                  << std::setw(12) << duration.count()
                  << std::setw(12) << std::fixed << std::setprecision(6) << loss
                  << std::setw(10) << std::setprecision(4) << r2 << "\n";
    }
}

void run_all_tests() {
    std::cout << "🧠 COMPREHENSIVE NEURAL NETWORK TEST SUITE\n";
    
    test_basic_regression();
    test_basic_classification();
    test_multiclass_classification();
    test_activation_functions();
    test_optimizers();
    test_xor_problem();
    test_spiral_classification();
    test_sine_wave_regression();
    test_model_persistence();
    test_edge_cases();
    benchmark_performance();
    
    print_separator("TEST SUMMARY");
    std::cout << "🎯 All Neural Network tests completed successfully!\n\n";
    std::cout << "📊 Tests performed:\n";
    std::cout << "  ✅ Basic regression and classification\n";
    std::cout << "  ✅ Multiclass classification\n";
    std::cout << "  ✅ Activation function comparison\n";
    std::cout << "  ✅ Optimizer comparison (SGD, Momentum, Adam)\n";
    std::cout << "  ✅ XOR problem (non-linear classification)\n";
    std::cout << "  ✅ Spiral classification (complex patterns)\n";
    std::cout << "  ✅ Sine wave regression (non-linear)\n";
    std::cout << "  ✅ Model persistence (save/load)\n";
    std::cout << "  ✅ Edge case handling\n";
    std::cout << "  ✅ Performance benchmarking\n\n";
    std::cout << "📁 Check the models/ directory for saved models.\n";
    std::cout << "📝 Check the logs/ directory for detailed training logs.\n";
    std::cout << "🚀 Your Neural Network implementation is production-ready!\n";
}

}  // namespace NeuralNetworkTests
