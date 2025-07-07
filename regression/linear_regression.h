#pragma once
#include <vector>
#include <string>

class LinearRegression {
private:
    double w, b;
    double alpha;
    int epochs;
    double tolerance;

public:
    LinearRegression(double learning_rate = 0.01, int max_epochs = 1000, double tol = 1e-6);

    void fit(const std::vector<double>& x, const std::vector<double>& y);
    double predict(double x_val) const;
    std::vector<double> predict(const std::vector<double>& x_vals) const;

    double evaluate(const std::vector<double>& x, const std::vector<double>& y) const;

    double get_weight() const;
    double get_bias() const;

    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);
};
