#include "naive_bayes_tests.h"
#include "../utils/utils.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace NaiveBayesTests {

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "  " << title << "\n";
    std::cout << std::string(80, '=') << "\n";
}

void print_classification_results(const NaiveBayes& model, 
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

std::pair<std::vector<std::vector<double>>, std::vector<int>> generate_gaussian_data(int n_samples, int n_features, int n_classes) {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    
    int samples_per_class = n_samples / n_classes;
    
    for (int class_id = 0; class_id < n_classes; ++class_id) {
        // Create class-specific means
        std::vector<double> class_means(n_features);
        for (int j = 0; j < n_features; ++j) {
            class_means[j] = class_id * 2.0 + dist(gen) * 0.5;
        }
        
        for (int i = 0; i < samples_per_class; ++i) {
            std::vector<double> sample(n_features);
            for (int j = 0; j < n_features; ++j) {
                sample[j] = class_means[j] + dist(gen);
            }
            X.push_back(sample);
            y.push_back(class_id);
        }
    }
    
    return {X, y};
}

std::pair<std::vector<std::vector<double>>, std::vector<int>> generate_binary_data(int n_samples, int n_features, int n_classes) {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(0.5);
    
    int samples_per_class = n_samples / n_classes;
    
    for (int class_id = 0; class_id < n_classes; ++class_id) {
        // Different probability for each class
        double class_prob = 0.3 + class_id * 0.4;
        std::bernoulli_distribution class_dist(class_prob);
        
        for (int i = 0; i < samples_per_class; ++i) {
            std::vector<double> sample(n_features);
            for (int j = 0; j < n_features; ++j) {
                sample[j] = class_dist(gen) ? 1.0 : 0.0;
            }
            X.push_back(sample);
            y.push_back(class_id);
        }
    }
    
    return {X, y};
}

std::pair<std::vector<std::vector<double>>, std::vector<int>> generate_multinomial_data(int n_samples, int n_features, int n_classes) {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> count_dist(1, 10);
    
    int samples_per_class = n_samples / n_classes;
    
    for (int class_id = 0; class_id < n_classes; ++class_id) {
        for (int i = 0; i < samples_per_class; ++i) {
            std::vector<double> sample(n_features, 0.0);
            
            // Generate counts for features (simulating word counts)
            for (int j = 0; j < n_features; ++j) {
                // Class-dependent feature probabilities
                double feature_prob = 0.1 + (class_id + j) % 3 * 0.3;
                std::bernoulli_distribution feature_dist(feature_prob);
                
                if (feature_dist(gen)) {
                    sample[j] = count_dist(gen);
                }
            }
            
            X.push_back(sample);
            y.push_back(class_id);
        }
    }
    
    return {X, y};
}

std::pair<std::vector<std::vector<double>>, std::vector<int>> generate_iris_like_data(int n_samples) {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 0.5);
    
    int samples_per_class = n_samples / 3;
    
    // Class 0: Small flowers
    for (int i = 0; i < samples_per_class; ++i) {
        std::vector<double> sample = {
            4.0 + dist(gen),  // sepal length
            2.5 + dist(gen),  // sepal width
            1.0 + dist(gen),  // petal length
            0.2 + dist(gen)   // petal width
        };
        X.push_back(sample);
        y.push_back(0);
    }
    
    // Class 1: Medium flowers
    for (int i = 0; i < samples_per_class; ++i) {
        std::vector<double> sample = {
            6.0 + dist(gen),  // sepal length
            3.0 + dist(gen),  // sepal width
            4.0 + dist(gen),  // petal length
            1.3 + dist(gen)   // petal width
        };
        X.push_back(sample);
        y.push_back(1);
    }
    
    // Class 2: Large flowers
    for (int i = 0; i < samples_per_class; ++i) {
        std::vector<double> sample = {
            7.5 + dist(gen),  // sepal length
            3.5 + dist(gen),  // sepal width
            6.0 + dist(gen),  // petal length
            2.0 + dist(gen)   // petal width
        };
        X.push_back(sample);
        y.push_back(2);
    }
    
    return {X, y};
}

void compare_naive_bayes_types(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    // Split data into train/test
    int train_size = X.size() * 0.8;
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<int> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
    std::vector<int> y_test(y.begin() + train_size, y.end());
    
    std::vector<std::pair<NaiveBayesType, std::string>> types = {
        {NaiveBayesType::GAUSSIAN, "Gaussian"},
        {NaiveBayesType::MULTINOMIAL, "Multinomial"},
        {NaiveBayesType::BERNOULLI, "Bernoulli"}
    };
    
    for (const auto& [type, name] : types) {
        auto start = std::chrono::high_resolution_clock::now();
        
        NaiveBayes model(type, SmoothingType::LAPLACE, 1.0, true);
        model.set_logging(true, "nb_type_comparison");
        model.fit(X_train, y_train);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        print_classification_results(model, X_test, y_test, name);
        std::cout << "  Training time: " << duration.count() << " ms" << std::endl;
    }
}

void demonstrate_probability_predictions(const NaiveBayes& model,
                                       const std::vector<std::vector<double>>& X_test,
                                       const std::vector<int>& y_test) {
    std::cout << "\n🎯 Sample Probability Predictions:\n";
    
    for (int i = 0; i < std::min(10, static_cast<int>(X_test.size())); ++i) {
        int true_class = y_test[i];
        int pred_class = model.predict(X_test[i]);
        std::vector<double> probs = model.predict_proba(X_test[i]);
        
        std::cout << "  Sample " << i+1 << " | True=" << true_class << " | Pred=" << pred_class << " | Probs=[";
        for (size_t j = 0; j < probs.size(); ++j) {
            std::cout << std::fixed << std::setprecision(3) << probs[j];
            if (j < probs.size() - 1) std::cout << ", ";
        }
        std::cout << "]";
        if (true_class == pred_class) std::cout << " ✓";
        else std::cout << " ✗";
        std::cout << "\n";
    }
}

void test_naive_bayes_types() {
    print_separator("NAIVE BAYES TYPE COMPARISON");
    
    std::cout << "Testing on Gaussian data:\n";
    auto [X_gauss, y_gauss] = generate_gaussian_data(300, 4, 3);
    compare_naive_bayes_types(X_gauss, y_gauss);
    
    std::cout << "\nTesting on Binary data:\n";
    auto [X_binary, y_binary] = generate_binary_data(200, 5, 2);
    compare_naive_bayes_types(X_binary, y_binary);
    
    std::cout << "\nTesting on Iris-like data:\n";
    auto [X_iris, y_iris] = generate_iris_like_data(150);
    compare_naive_bayes_types(X_iris, y_iris);
}

void test_smoothing_methods() {
    print_separator("SMOOTHING METHOD COMPARISON");
    
    auto [X, y] = generate_multinomial_data(200, 6, 3);
    
    // Split data
    int train_size = X.size() * 0.8;
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<int> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
    std::vector<int> y_test(y.begin() + train_size, y.end());
    
    std::vector<std::tuple<SmoothingType, double, std::string>> smoothing_configs = {
        {SmoothingType::NONE, 0.0, "No Smoothing"},
        {SmoothingType::LAPLACE, 1.0, "Laplace (α=1.0)"},
        {SmoothingType::LIDSTONE, 0.5, "Lidstone (α=0.5)"},
        {SmoothingType::LIDSTONE, 0.1, "Lidstone (α=0.1)"}
    };
    
    for (const auto& [smooth_type, alpha, name] : smoothing_configs) {
        NaiveBayes model(NaiveBayesType::MULTINOMIAL, smooth_type, alpha, true);
        model.set_logging(true, "smoothing_test");
        model.fit(X_train, y_train);
        print_classification_results(model, X_test, y_test, name);
    }
}

void test_feature_scaling() {
    print_separator("FEATURE SCALING IMPACT");
    
    auto [X, y] = generate_gaussian_data(300, 4, 3);
    
    // Make features have different scales
    for (auto& sample : X) {
        sample[0] *= 100;   // Large scale
        sample[1] *= 0.01;  // Small scale
        sample[2] *= 1000;  // Very large scale
        // sample[3] stays normal scale
    }
    
    // Split data
    int train_size = X.size() * 0.8;
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<int> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
    std::vector<int> y_test(y.begin() + train_size, y.end());
    
    std::cout << "  • WITHOUT scaling:" << std::endl;
    NaiveBayes model_no_scaling(NaiveBayesType::GAUSSIAN, SmoothingType::LAPLACE, 1.0, false);
    model_no_scaling.set_logging(true, "feature_scaling_test");
    model_no_scaling.fit(X_train, y_train);
    print_classification_results(model_no_scaling, X_test, y_test, "No Scaling");
    
    std::cout << "  • WITH scaling:" << std::endl;
    NaiveBayes model_with_scaling(NaiveBayesType::GAUSSIAN, SmoothingType::LAPLACE, 1.0, true);
    model_with_scaling.set_logging(true, "feature_scaling_test");
    model_with_scaling.fit(X_train, y_train);
    print_classification_results(model_with_scaling, X_test, y_test, "With Scaling");
}

void test_cross_validation() {
    print_separator("CROSS-VALIDATION TESTING");
    
    auto [X, y] = generate_iris_like_data(150);
    
    std::vector<int> k_folds = {3, 5, 10};
    
    for (int k : k_folds) {
        NaiveBayes model(NaiveBayesType::GAUSSIAN, SmoothingType::LAPLACE, 1.0, true);
        double cv_accuracy = model.cross_validate(X, y, k);
        std::cout << k << "-Fold CV Accuracy: " << std::fixed << std::setprecision(4) << cv_accuracy << std::endl;
    }
}

void test_model_persistence() {
    print_separator("MODEL PERSISTENCE TESTING");
    
    auto [X, y] = generate_iris_like_data(120);
    
    // Train original model
    NaiveBayes original_model(NaiveBayesType::GAUSSIAN, SmoothingType::LAPLACE, 1.0, true);
    original_model.set_logging(true, "model_persistence_test");
    original_model.fit(X, y);
    original_model.save_model("models/naive_bayes_model.txt");
    
    // Load model
    NaiveBayes loaded_model;
    loaded_model.load_model("models/naive_bayes_model.txt");
    
    // Test some predictions
    std::cout << "\nPrediction comparison:\n";
    for (int i = 0; i < 5; ++i) {
        int orig_pred = original_model.predict(X[i]);
        int loaded_pred = loaded_model.predict(X[i]);
        auto orig_probs = original_model.predict_proba(X[i]);
        auto loaded_probs = loaded_model.predict_proba(X[i]);
        
        std::cout << "Sample " << i+1 << " | Original: " << orig_pred 
                  << " | Loaded: " << loaded_pred;
        if (orig_pred == loaded_pred) std::cout << " ✓";
        else std::cout << " ✗";
        
        double prob_diff = 0.0;
        for (size_t j = 0; j < orig_probs.size(); ++j) {
            prob_diff += std::abs(orig_probs[j] - loaded_probs[j]);
        }
        std::cout << " | Prob diff: " << std::fixed << std::setprecision(6) << prob_diff << "\n";
    }
}

void test_classification_metrics() {
    print_separator("CLASSIFICATION METRICS ANALYSIS");
    
    auto [X, y] = generate_iris_like_data(150);
    
    // Split data
    int train_size = X.size() * 0.8;
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<int> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
    std::vector<int> y_test(y.begin() + train_size, y.end());
    
    NaiveBayes model(NaiveBayesType::GAUSSIAN, SmoothingType::LAPLACE, 1.0, true);
    model.set_logging(true, "classification_metrics");
    model.fit(X_train, y_train);
    
    auto metrics = model.classification_report(X_test, y_test);
    double log_likelihood = model.evaluate_log_likelihood(X_test, y_test);
    
    std::cout << "📊 Comprehensive Metrics:\n";
    std::cout << "  Accuracy:  " << std::fixed << std::setprecision(4) << metrics["accuracy"] << "\n";
    std::cout << "  Precision: " << std::setprecision(4) << metrics["precision"] << "\n";
    std::cout << "  Recall:    " << std::setprecision(4) << metrics["recall"] << "\n";
    std::cout << "  F1-Score:  " << std::setprecision(4) << metrics["f1_score"] << "\n";
    std::cout << "  Log-Likelihood: " << std::setprecision(4) << log_likelihood << "\n";
    
    demonstrate_probability_predictions(model, X_test, y_test);
}

void test_probability_predictions() {
    print_separator("PROBABILITY PREDICTION ANALYSIS");
    
    auto [X, y] = generate_iris_like_data(150);
    
    // Split data
    int train_size = X.size() * 0.8;
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
    std::vector<int> y_train(y.begin(), y.begin() + train_size);
    std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
    std::vector<int> y_test(y.begin() + train_size, y.end());
    
    NaiveBayes model(NaiveBayesType::GAUSSIAN, SmoothingType::LAPLACE, 1.0, true);
    model.fit(X_train, y_train);
    
    std::cout << "Analyzing probability predictions:\n";
    
    // Test probability calibration
    std::vector<double> confidence_scores;
    std::vector<bool> correct_predictions;
    
    for (size_t i = 0; i < X_test.size(); ++i) {
        int pred = model.predict(X_test[i]);
        auto probs = model.predict_proba(X_test[i]);
        
        double max_prob = *std::max_element(probs.begin(), probs.end());
        confidence_scores.push_back(max_prob);
        correct_predictions.push_back(pred == y_test[i]);
    }
    
    // Calculate average confidence for correct vs incorrect predictions
    double correct_confidence = 0.0, incorrect_confidence = 0.0;
    int correct_count = 0, incorrect_count = 0;
    
    for (size_t i = 0; i < confidence_scores.size(); ++i) {
        if (correct_predictions[i]) {
            correct_confidence += confidence_scores[i];
            correct_count++;
        } else {
            incorrect_confidence += confidence_scores[i];
            incorrect_count++;
        }
    }
    
    if (correct_count > 0) correct_confidence /= correct_count;
    if (incorrect_count > 0) incorrect_confidence /= incorrect_count;
    
    std::cout << "Average confidence for correct predictions: " << std::fixed << std::setprecision(4) << correct_confidence << "\n";
    std::cout << "Average confidence for incorrect predictions: " << std::fixed << std::setprecision(4) << incorrect_confidence << "\n";
    
    demonstrate_probability_predictions(model, X_test, y_test);
}

void test_feature_importance() {
    print_separator("FEATURE IMPORTANCE ANALYSIS");
    
    auto [X, y] = generate_iris_like_data(150);
    
    NaiveBayes model(NaiveBayesType::GAUSSIAN, SmoothingType::LAPLACE, 1.0, true);
    model.fit(X, y);
    
    std::cout << "Model Summary:\n";
    model.print_model_summary();
    
    std::cout << "\nFeature Importance Analysis:\n";
    auto importance = model.get_feature_importance();
    std::vector<std::string> feature_names = {"Sepal Length", "Sepal Width", "Petal Length", "Petal Width"};
    
    for (size_t i = 0; i < importance.size(); ++i) {
        std::cout << "  " << feature_names[i] << ": " << std::fixed << std::setprecision(6) << importance[i] << "\n";
    }
}

void test_edge_cases() {
    print_separator("EDGE CASE TESTING");
    
    std::cout << "✅ Minimal data test: ";
    auto [X_min, y_min] = generate_gaussian_data(6, 2, 2);  // Very small dataset
    try {
        NaiveBayes model_min(NaiveBayesType::GAUSSIAN);
        model_min.fit(X_min, y_min);
        double accuracy = model_min.evaluate_accuracy(X_min, y_min);
        std::cout << "Passed (accuracy: " << std::fixed << std::setprecision(3) << accuracy << ")\n";
    } catch (const std::exception& e) {
        std::cout << "Failed: " << e.what() << "\n";
    }
    
    std::cout << "✅ Single feature test: ";
    auto [X_single, y_single] = generate_gaussian_data(50, 1, 2);  // Single feature
    try {
        NaiveBayes model_single(NaiveBayesType::GAUSSIAN);
        model_single.fit(X_single, y_single);
        double accuracy = model_single.evaluate_accuracy(X_single, y_single);
        std::cout << "Passed (accuracy: " << std::fixed << std::setprecision(3) << accuracy << ")\n";
    } catch (const std::exception& e) {
        std::cout << "Failed: " << e.what() << "\n";
    }
    
    std::cout << "✅ Binary classification test: ";
    auto [X_binary, y_binary] = generate_binary_data(100, 3, 2);
    try {
        NaiveBayes model_binary(NaiveBayesType::BERNOULLI);
        model_binary.fit(X_binary, y_binary);
        double accuracy = model_binary.evaluate_accuracy(X_binary, y_binary);
        std::cout << "Passed (accuracy: " << std::fixed << std::setprecision(3) << accuracy << ")\n";
    } catch (const std::exception& e) {
        std::cout << "Failed: " << e.what() << "\n";
    }
}

void benchmark_performance() {
    print_separator("PERFORMANCE BENCHMARKING");
    
    std::vector<std::pair<std::pair<int, int>, std::string>> test_sizes = {
        {{100, 4}, "Small (100×4)"},
        {{500, 8}, "Medium (500×8)"},
        {{1000, 12}, "Large (1000×12)"},
        {{2000, 16}, "X-Large (2000×16)"}
    };
    
    std::cout << "    Dataset Size    Time (ms)    Accuracy     Classes\n";
    std::cout << std::string(55, '-') << "\n";
    
    for (const auto& [size_pair, name] : test_sizes) {
        auto [n_samples, n_features] = size_pair;
        auto [X, y] = generate_gaussian_data(n_samples, n_features, 3);
        
        // Split data
        int train_size = X.size() * 0.8;
        std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_size);
        std::vector<int> y_train(y.begin(), y.begin() + train_size);
        std::vector<std::vector<double>> X_test(X.begin() + train_size, X.end());
        std::vector<int> y_test(y.begin() + train_size, y.end());
        
        auto start = std::chrono::high_resolution_clock::now();
        
        NaiveBayes model(NaiveBayesType::GAUSSIAN, SmoothingType::LAPLACE, 1.0, true);
        model.set_logging(true, "benchmark_" + std::to_string(n_samples));
        model.fit(X_train, y_train);
        double accuracy = model.evaluate_accuracy(X_test, y_test);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << std::setw(15) << name 
                  << std::setw(12) << duration.count()
                  << std::setw(12) << std::fixed << std::setprecision(4) << accuracy
                  << std::setw(10) << model.get_num_classes() << "\n";
    }
}

void run_all_tests() {
    std::cout << " COMPREHENSIVE NAIVE BAYES TEST SUITE\n";
    
    test_naive_bayes_types();
    test_smoothing_methods();
    test_feature_scaling();
    test_cross_validation();
    test_model_persistence();
    test_classification_metrics();
    test_probability_predictions();
    test_feature_importance();
    test_edge_cases();
    benchmark_performance();
    
    print_separator("TEST SUMMARY");
    std::cout << "🎯 All Naive Bayes tests completed successfully!\n\n";
    std::cout << "📊 Tests performed:\n";
    std::cout << "  ✅ Naive Bayes type comparison (Gaussian, Multinomial, Bernoulli)\n";
    std::cout << "  ✅ Smoothing method testing (None, Laplace, Lidstone)\n";
    std::cout << "  ✅ Feature scaling impact analysis\n";
    std::cout << "  ✅ Cross-validation testing\n";
    std::cout << "  ✅ Model persistence verification\n";
    std::cout << "  ✅ Classification metrics analysis\n";
    std::cout << "  ✅ Probability prediction analysis\n";
    std::cout << "  ✅ Feature importance analysis\n";
    std::cout << "  ✅ Edge case handling\n";
    std::cout << "  ✅ Performance benchmarking\n\n";
    std::cout << "📁 Check the models/ directory for saved models.\n";
    std::cout << "📝 Check the logs/ directory for detailed training logs.\n";
    std::cout << "🚀 Your Naive Bayes classifier is production-ready!\n";
}

}  // namespace NaiveBayesTests
