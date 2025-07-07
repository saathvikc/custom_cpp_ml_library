#include "regression/linear_regression.h"
#include "utils/utils.h"
#include <iostream>

int main() {
    std::vector<double> x, y;
    load_csv("data/sample.csv", x, y);

    LinearRegression model(0.01, 1000, 1e-6);
    model.fit(x, y);

    std::cout << "\nFinal model: y = " << model.get_weight() << " * x + " << model.get_bias() << "\n";

    // Save model
    model.save_model("models/linear_model.txt");

    // Load model
    LinearRegression loaded_model;
    loaded_model.load_model("models/linear_model.txt");

    std::cout << "Prediction for x = 6.0: " << loaded_model.predict(6.0) << std::endl;
    return 0;
}
