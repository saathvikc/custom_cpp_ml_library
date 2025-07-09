#include "regression/linear_regression.h"
#include "utils/utils.h"
#include "tests/linear_regression_tests.h"
#include <iostream>
#include <vector>

int main() {
    try {
        std::cout << "CUSTOM C++ ML LIBRARY - LINEAR REGRESSION\n";
        std::cout << std::string(50, '=') << "\n\n";
        
        // Load data
        std::cout << "Loading data from data/sample.csv...\n";
        std::vector<double> x, y;
        load_csv("data/sample.csv", x, y);
        
        if (x.empty()) {
            std::cout << "No data loaded. Please check data/sample.csv\n";
            return 1;
        }
        
        std::cout << "Data loaded successfully!\n\n";
        
        // Simple demo first
        std::cout << "QUICK DEMO - Basic Linear Regression\n";
        std::cout << std::string(40, '-') << "\n";
        
        LinearRegression demo_model(0.01, 500, 1e-6, OptimizerType::ADAM, 
                                   RegularizationType::L2, 0.01, true, true);
        demo_model.set_logging(true, "logs/demo_training.log");
        demo_model.fit(x, y);
        
        std::cout << "\nDemo Results:\n";
        std::cout << "Final model: y = " << demo_model.get_weight() 
                  << " * x + " << demo_model.get_bias() << "\n";
        std::cout << "R² Score: " << demo_model.r_squared(x, y) << "\n";
        std::cout << "MSE: " << demo_model.evaluate(x, y) << "\n";
        
        // Save demo model
        demo_model.save_model("models/demo_model.txt");
        std::cout << "Demo model saved to models/demo_model.txt\n";
        
        // Quick prediction test
        if (x.size() > 2) {
            double test_x = x[x.size()/2]; // Middle value
            double prediction = demo_model.predict(test_x);
            std::cout << "Prediction for x=" << test_x << ": " << prediction << "\n";
        }
        
        std::cout << "\n" << std::string(50, '=') << "\n";
        std::cout << "Running comprehensive test suite...\n";
        std::cout << "Press Enter to continue or Ctrl+C to exit: ";
        std::cin.get();
        
        // Run comprehensive tests
        LinearRegressionTests::run_all_tests(x, y);
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
