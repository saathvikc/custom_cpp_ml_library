#include "linear_regression.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>

LinearRegression::LinearRegression(double learning_rate, int max_epochs, double tol)
    : w(0.0), b(0.0), alpha(learning_rate), epochs(max_epochs), tolerance(tol) {}

double LinearRegression::predict(double x_val) const {
    return w * x_val + b;
}

std::vector<double> LinearRegression::predict(const std::vector<double>& x_vals) const {
    std::vector<double> results;
    for (double x : x_vals)
        results.push_back(predict(x));
    return results;
}

double LinearRegression::evaluate(const std::vector<double>& x, const std::vector<double>& y) const {
    if (x.size() != y.size()) throw std::invalid_argument("x and y size mismatch");
    double loss = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double err = predict(x[i]) - y[i];
        loss += err * err;
    }
    return loss / x.size();
}

void LinearRegression::fit(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size()) throw std::invalid_argument("x and y size mismatch");
    int n = x.size();
    double prev_loss = 1e10;

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

        double loss = evaluate(x, y);
        if (epoch % 100 == 0 || epoch == 1)
            std::cout << "Epoch " << epoch << ", Loss: " << loss
                      << ", w: " << w << ", b: " << b << std::endl;

        if (std::abs(prev_loss - loss) < tolerance)
            break;

        prev_loss = loss;
    }
}

double LinearRegression::get_weight() const { return w; }
double LinearRegression::get_bias() const { return b; }

void LinearRegression::save_model(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Failed to open file to save model.");
    out << w << " " << b << std::endl;
    out.close();
}

void LinearRegression::load_model(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Failed to open file to load model.");
    in >> w >> b;
    in.close();
}
