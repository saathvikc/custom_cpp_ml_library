#include "regression/linear_regression.h"
#include "regression/logistic_regression.h"
#include "regression/polynomial_regression.h"
#include "classification/knn.h"
#include "classification/naive_bayes.h"
#include "neural_networks/neural_network.h"
#include "utils/utils.h"
#include "utils/data_loader.h"
#include "tests/linear_regression_tests.h"
#include "tests/logistic_regression_tests.h"
#include "tests/polynomial_regression_tests.h"
#include "tests/knn_tests.h"
#include "tests/naive_bayes_tests.h"
#include "tests/neural_network_tests.h"
#include <iostream>
#include <vector>

int main() {
    try {
        std::cout << "CUSTOM C++ ML LIBRARY - ADVANCED MACHINE LEARNING SUITE\n";
        std::cout << std::string(60, '=') << "\n\n";
        
        // Load data for linear regression
        std::cout << "Loading linear regression data from data/regression_sample.csv...\n";
        auto [X_reg, y_reg] = DataLoader::load_regression_data("data/regression_sample.csv");
        
        if (X_reg.empty()) {
            std::cout << "No regression data loaded. Please check data/regression_sample.csv\n";
            return 1;
        }
        
        // Convert multi-feature data to single feature for basic demo
        std::vector<double> x, y;
        for (size_t i = 0; i < X_reg.size(); ++i) {
            x.push_back(X_reg[i][0]);  // Use first feature for basic demo
            y.push_back(y_reg[i]);
        }
        
        // LINEAR REGRESSION DEMO
        std::cout << "LINEAR REGRESSION DEMO\n";
        std::cout << std::string(40, '-') << "\n";
        
        LinearRegression demo_model(0.01, 500, 1e-6, OptimizerType::ADAM, 
                                   RegularizationType::L2, 0.01, true, true);
        demo_model.set_logging(true, "logs/demo_linear_training.log");
        demo_model.fit(x, y);
        
        std::cout << "\nLinear Regression Demo Results:\n";
        std::cout << "Final model: y = " << demo_model.get_weight() 
                  << " * x + " << demo_model.get_bias() << "\n";
        std::cout << "R² Score: " << demo_model.r_squared(x, y) << "\n";
        std::cout << "MSE: " << demo_model.evaluate(x, y) << "\n";
        
        // Save demo model
        demo_model.save_model("models/demo_linear_model.txt");
        std::cout << "Linear demo model saved to models/demo_linear_model.txt\n";
        
        // Quick prediction test
        if (x.size() > 2) {
            double test_x = x[x.size()/2]; // Middle value
            double prediction = demo_model.predict(test_x);
            std::cout << "Prediction for x=" << test_x << ": " << prediction << "\n";
        }
        
        // POLYNOMIAL REGRESSION DEMO
        std::cout << "\n" << std::string(40, '-') << "\n";
        std::cout << "POLYNOMIAL REGRESSION DEMO\n";
        std::cout << std::string(40, '-') << "\n";
        
        // Load polynomial data from file
        auto [x_poly, y_poly] = DataLoader::load_polynomial_data("data/polynomial_sample.csv");
        
        PolynomialRegression poly_demo(3, 0.01, 1000, 1e-6, OptimizerType::ADAM, 
                                      RegularizationType::L2, 0.01, true, true);
        poly_demo.set_logging(true, "logs/demo_polynomial_training.log");
        poly_demo.fit(x_poly, y_poly);
        
        std::cout << "\nPolynomial Regression Demo Results:\n";
        poly_demo.print_polynomial_equation();
        std::cout << "R² Score: " << poly_demo.r_squared(x_poly, y_poly) << "\n";
        std::cout << "Adjusted R² Score: " << poly_demo.adjusted_r_squared(x_poly, y_poly) << "\n";
        std::cout << "MSE: " << poly_demo.evaluate(x_poly, y_poly) << "\n";
        
        // Save polynomial demo model
        poly_demo.save_model("models/demo_polynomial_model.txt");
        std::cout << "Polynomial demo model saved to models/demo_polynomial_model.txt\n";
        
        // Quick polynomial prediction test
        if (x_poly.size() > 2) {
            double test_x = x_poly[x_poly.size()/2];
            double prediction = poly_demo.predict(test_x);
            std::cout << "Polynomial prediction for x=" << test_x << ": " << prediction << "\n";
        }
        
        // LOGISTIC REGRESSION DEMO
        std::cout << "\n" << std::string(40, '-') << "\n";
        std::cout << "LOGISTIC REGRESSION DEMO\n";
        std::cout << std::string(40, '-') << "\n";
        
        // Load binary classification data from file
        auto [X_class, y_class] = DataLoader::load_classification_data("data/classification_binary.csv");
        
        // Convert multi-feature data to single feature for basic demo
        std::vector<double> x_class;
        for (const auto& features : X_class) {
            x_class.push_back(features[0]);  // Use first feature for basic demo
        }
        
        LogisticRegression logistic_demo(0.1, 1000, 1e-6, OptimizerType::ADAM, 
                                        RegularizationType::L2, 0.01, true, true);
        logistic_demo.set_logging(true, "logs/demo_logistic_training.log");
        logistic_demo.fit(x_class, y_class);
        
        std::cout << "\nLogistic Regression Demo Results:\n";
        std::cout << "Final model weights: w = " << logistic_demo.get_weight() 
                  << ", b = " << logistic_demo.get_bias() << "\n";
        std::cout << "Accuracy: " << logistic_demo.evaluate_accuracy(x_class, y_class) << "\n";
        std::cout << "Precision: " << logistic_demo.evaluate_precision(x_class, y_class) << "\n";
        std::cout << "Recall: " << logistic_demo.evaluate_recall(x_class, y_class) << "\n";
        std::cout << "F1-Score: " << logistic_demo.evaluate_f1_score(x_class, y_class) << "\n";
        
        // Save logistic demo model
        logistic_demo.save_model("models/demo_logistic_model.txt");
        std::cout << "Logistic demo model saved to models/demo_logistic_model.txt\n";
        
        // Quick classification test
        if (x_class.size() > 2) {
            double test_x = x_class[x_class.size()/2];
            double prob = logistic_demo.predict_proba(test_x);
            int prediction = logistic_demo.predict(test_x);
            std::cout << "Prediction for x=" << test_x << ": class=" << prediction 
                      << " (probability=" << prob << ")\n";
        }
        
        // NAIVE BAYES DEMO
        std::cout << "\n" << std::string(40, '-') << "\n";
        std::cout << "NAIVE BAYES DEMO\n";
        std::cout << std::string(40, '-') << "\n";
        
        // Load multi-class classification data from file
        auto [X_nb, y_nb] = DataLoader::load_classification_data("data/classification_multiclass.csv");
        
        // Split data for training and testing
        int train_size = X_nb.size() * 0.8;
        std::vector<std::vector<double>> X_train(X_nb.begin(), X_nb.begin() + train_size);
        std::vector<int> y_train(y_nb.begin(), y_nb.begin() + train_size);
        std::vector<std::vector<double>> X_test(X_nb.begin() + train_size, X_nb.end());
        std::vector<int> y_test(y_nb.begin() + train_size, y_nb.end());
        
        NaiveBayes nb_demo(NaiveBayesType::GAUSSIAN, SmoothingType::LAPLACE, 1.0, true);
        nb_demo.set_logging(true, "demo_naive_bayes");
        nb_demo.fit(X_train, y_train);
        
        std::cout << "\nNaive Bayes Demo Results:\n";
        std::cout << "Number of classes: " << nb_demo.get_num_classes() << "\n";
        std::cout << "Number of features: " << nb_demo.get_num_features() << "\n";
        std::cout << "Training accuracy: " << nb_demo.evaluate_accuracy(X_train, y_train) << "\n";
        std::cout << "Test accuracy: " << nb_demo.evaluate_accuracy(X_test, y_test) << "\n";
        
        auto metrics = nb_demo.classification_report(X_test, y_test);
        std::cout << "Test precision: " << metrics.at("precision") << "\n";
        std::cout << "Test recall: " << metrics.at("recall") << "\n";
        std::cout << "Test F1-score: " << metrics.at("f1_score") << "\n";
        
        // Save naive bayes demo model
        nb_demo.save_model("models/demo_naive_bayes_model.txt");
        std::cout << "Naive Bayes demo model saved to models/demo_naive_bayes_model.txt\n";
        
        // Quick prediction test
        if (X_test.size() > 0) {
            auto test_sample = X_test[0];
            int prediction = nb_demo.predict(test_sample);
            auto probabilities = nb_demo.predict_proba(test_sample);
            std::cout << "Sample prediction: class=" << prediction << ", probabilities=[";
            for (size_t i = 0; i < probabilities.size(); ++i) {
                std::cout << probabilities[i];
                if (i < probabilities.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
        
        // NEURAL NETWORK DEMO
        std::cout << "\n" << std::string(40, '-') << "\n";
        std::cout << "NEURAL NETWORK DEMO\n";
        std::cout << std::string(40, '-') << "\n";
        
        // Load neural network regression data from file
        auto [X_nn_reg, y_nn_reg] = DataLoader::load_neural_regression_data("data/neural_network_regression.csv");
        
        // Split data for training and testing
        int nn_train_size = X_nn_reg.size() * 0.8;
        std::vector<std::vector<double>> X_nn_train(X_nn_reg.begin(), X_nn_reg.begin() + nn_train_size);
        std::vector<std::vector<double>> y_nn_train(y_nn_reg.begin(), y_nn_reg.begin() + nn_train_size);
        std::vector<std::vector<double>> X_nn_test(X_nn_reg.begin() + nn_train_size, X_nn_reg.end());
        std::vector<std::vector<double>> y_nn_test(y_nn_reg.begin() + nn_train_size, y_nn_reg.end());
        
        // Create and train neural network for regression
        NeuralNetwork nn_reg_demo({3, 10, 5, 1}, {ActivationType::RELU, ActivationType::RELU, ActivationType::LINEAR});
        nn_reg_demo.set_optimizer(OptimizerType::ADAM, 0.001);
        nn_reg_demo.set_logging(true, "demo_neural_network_regression");
        nn_reg_demo.fit(X_nn_train, y_nn_train, 200, 32, 0.2, true, false);
        
        std::cout << "\nNeural Network Regression Demo Results:\n";
        double nn_loss = nn_reg_demo.evaluate_loss(X_nn_test, y_nn_test);
        double nn_r2 = nn_reg_demo.evaluate_r2_score(X_nn_test, y_nn_test);
        std::cout << "Test loss: " << nn_loss << "\n";
        std::cout << "Test R² score: " << nn_r2 << "\n";
        
        // Neural Network Classification Demo
        auto [X_nn_cls, y_nn_cls] = DataLoader::load_classification_data("data/neural_network_classification.csv");
        
        // Split classification data
        int nn_cls_train_size = X_nn_cls.size() * 0.8;
        std::vector<std::vector<double>> X_nn_cls_train(X_nn_cls.begin(), X_nn_cls.begin() + nn_cls_train_size);
        std::vector<int> y_nn_cls_train(y_nn_cls.begin(), y_nn_cls.begin() + nn_cls_train_size);
        std::vector<std::vector<double>> X_nn_cls_test(X_nn_cls.begin() + nn_cls_train_size, X_nn_cls.end());
        std::vector<int> y_nn_cls_test(y_nn_cls.begin() + nn_cls_train_size, y_nn_cls.end());
        
        // Create and train neural network for classification
        NeuralNetwork nn_cls_demo({2, 8, 6, 2}, {ActivationType::RELU, ActivationType::RELU, ActivationType::SIGMOID});
        nn_cls_demo.set_optimizer(OptimizerType::ADAM, 0.001);
        nn_cls_demo.set_logging(true, "demo_neural_network_classification");
        nn_cls_demo.fit_classification(X_nn_cls_train, y_nn_cls_train, 300, 32, 0.2, true, false);
        
        std::cout << "\nNeural Network Classification Demo Results:\n";
        double nn_accuracy = nn_cls_demo.evaluate_accuracy(X_nn_cls_test, y_nn_cls_test);
        auto nn_metrics = nn_cls_demo.classification_report(X_nn_cls_test, y_nn_cls_test);
        std::cout << "Test accuracy: " << nn_accuracy << "\n";
        std::cout << "Test precision: " << nn_metrics.at("precision") << "\n";
        std::cout << "Test recall: " << nn_metrics.at("recall") << "\n";
        std::cout << "Test F1-score: " << nn_metrics.at("f1_score") << "\n";
        
        // Save neural network demo models
        nn_reg_demo.save_model("models/demo_neural_network_regression.txt");
        nn_cls_demo.save_model("models/demo_neural_network_classification.txt");
        std::cout << "Neural Network demo models saved to models/ directory\n";
        
        // Quick prediction test
        if (!X_nn_cls_test.empty()) {
            auto test_sample = X_nn_cls_test[0];
            int prediction = nn_cls_demo.predict_class(test_sample);
            auto probabilities = nn_cls_demo.predict_proba(test_sample);
            std::cout << "Sample prediction: class=" << prediction << ", probabilities=[";
            for (size_t i = 0; i < probabilities.size(); ++i) {
                std::cout << probabilities[i];
                if (i < probabilities.size() - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        }
        
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "Running comprehensive test suites...\n";
        std::cout << "Press Enter to continue or Ctrl+C to exit: ";
        std::cin.get();
        
        // Run comprehensive tests
        std::cout << "\nRunning Linear Regression Test Suite...\n";
        LinearRegressionTests::run_all_tests(x, y);
        
        std::cout << "\nRunning Polynomial Regression Test Suite...\n";
        PolynomialRegressionTests::run_all_tests(x_poly, y_poly);
        
        std::cout << "\nRunning Logistic Regression Test Suite...\n";
        LogisticRegressionTests::run_all_tests(x_class, y_class);
        
        std::cout << "\nRunning KNN Test Suite...\n";
        KNNTests::run_all_tests();
        
        std::cout << "\nRunning Naive Bayes Test Suite...\n";
        NaiveBayesTests::run_all_tests();
        
        std::cout << "\nRunning Neural Network Test Suite...\n";
        NeuralNetworkTests::run_all_tests();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
