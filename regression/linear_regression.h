#pragma once
#include <vector>
#include <string>

class LinearRegression {
private:
    double w, b;
    double alpha;
    int epochs;

public:
    LinearRegression(double learning_rate = 0.01, int max_epochs = 1000);
    
    void fit(const std::vector<double>& x, const std::vector<double>& y);
    double predict(double x_val) const;
    double evaluate(const std::vector<double>& x, const std::vector<double>& y) const;
};
