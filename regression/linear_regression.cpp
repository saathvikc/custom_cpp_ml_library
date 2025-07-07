#include "linear_regression.h"
#include <iostream>
#include <cmath>

LinearRegression::LinearRegression(double learning_rate, int max_epochs)
    : w(0.0), b(0.0), alpha(learning_rate), epochs(max_epochs) {}

double LinearRegression::predict(double x_val) const {
    return w * x_val + b;
}

void LinearRegression::fit(const std::vector<double>& x, const std::vector<double>& y) {
    int n = x.size();
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double dw = 0.0, db = 0.0;
        for (int i = 0; i < n; ++i) {
            double error = predict(x[i]) - y[i];
            dw += error * x[i];
            db += error;
        }
        dw /= n;
        db /= n;

        w -= alpha * dw;
        b -= alpha * db;

        if (epoch % 100 == 0 || epoch == 1)
            std::cout << "Epoch " << epoch << ", Loss: " << evaluate(x, y)
                      << ", w: " << w << ", b: " << b << std::endl;
    }
}

double LinearRegression::evaluate(const std::vector<double>& x, const std::vector<double>& y) const {
    double loss = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double err = predict(x[i]) - y[i];
        loss += err * err;
    }
    return loss / x.size();
}
