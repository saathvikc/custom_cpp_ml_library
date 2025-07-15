#include "regression/linear_regression.h"
#include "regression/logistic_regression.h"
#include "regression/polynomial_regression.h"
#include "classification/knn.h"
#include "utils/utils.h"
#include "tests/linear_regression_tests.h"
#include "tests/logistic_regression_tests.h"
#include "tests/polynomial_regression_tests.h"
#include "tests/knn_tests.h"
#include <iostream>
#include <vector>

int main() {
    try {
        std::cout << "CUSTOM C++ ML LIBRARY - ADVANCED MACHINE LEARNING SUITE\n";
        std::cout << std::string(60, '=') << "\n\n";
        
        // Load data for linear regression
        std::cout << "Loading linear regression data from data/sample.csv...\n";
        std::vector<double> x, y;
        load_csv("data/sample.csv", x, y);
        
        if (x.empty()) {
            std::cout << "No data loaded. Please check data/sample.csv\n";
            return 1;
        }
        
        std::cout << "Linear regression data loaded successfully!\n\n";
        
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
        
        // Generate polynomial data for demo
        std::vector<double> x_poly, y_poly;
        PolynomialRegressionTests::generate_polynomial_data(x_poly, y_poly, 3, 50, 0.1);
        
        std::cout << "Generated " << x_poly.size() << " samples for polynomial demo\n";
        
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
        
        // Generate synthetic classification data
        std::vector<double> x_class;
        std::vector<int> y_class;
        LogisticRegressionTests::generate_classification_data(x_class, y_class, 200, 0.1);
        
        std::cout << "Generated " << x_class.size() << " samples for classification demo\n";
        
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
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
