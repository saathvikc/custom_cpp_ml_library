#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

// Training data
vector<double> x;
vector<double> y;

// Hyperparameters
const double alpha = 0.01;
const int epochs = 1000;

// Model parameters
double w = 0.0, b = 0.0;

void data_initialization() {
    // Initialize training data
    x.push_back(1.0);
    x.push_back(2.0);
    x.push_back(3.0);
    x.push_back(4.0);
    x.push_back(5.0);
    
    y.push_back(2.0);
    y.push_back(4.1);
    y.push_back(6.0);
    y.push_back(8.1);
    y.push_back(10.2);
}

// Predict y for a given x
double predict(double x_val) {
    return w * x_val + b;
}

// Compute mean squared error
double compute_loss() {
    double loss = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double error = predict(x[i]) - y[i];
        loss += error * error;
    }
    return loss / x.size();
}

// Perform one step of gradient descent
void update_weights() {
    double dw = 0.0, db = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double error = predict(x[i]) - y[i];
        dw += error * x[i];
        db += error;
    }
    dw /= x.size();
    db /= x.size();

    w -= alpha * dw;
    b -= alpha * db;
}

void train() {
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        update_weights();
        if (epoch % 100 == 0 || epoch == 1) {
            cout << "Epoch " << epoch
                 << " | Loss: " << compute_loss()
                 << " | w: " << w << " | b: " << b << endl;
        }
    }
}

int main() {
    data_initialization();
    train();

    cout << "\nFinal model: y = " << w << " * x + " << b << endl;

    // Test prediction
    double test_x = 6.0;
    cout << "Prediction for x = " << test_x << ": " << predict(test_x) << endl;

    return 0;
}
