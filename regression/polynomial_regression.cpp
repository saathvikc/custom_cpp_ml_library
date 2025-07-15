#include "polynomial_regression.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iomanip>

PolynomialRegression::PolynomialRegression(int poly_degree, double learning_rate, int max_epochs, double tol,
                                         OptimizerType opt, RegularizationType reg, double reg_strength,
                                         bool adaptive_lr, bool feature_scaling)
    : degree(poly_degree), alpha(learning_rate), epochs(max_epochs), tolerance(tol),
      optimizer(opt), momentum_beta(0.9), adam_beta1(0.9), adam_beta2(0.999), epsilon(1e-8),
      reg_type(reg), lambda_reg(reg_strength), l1_ratio(0.5),
      use_adaptive_lr(adaptive_lr), lr_decay(0.95), min_lr(1e-6),
      use_feature_scaling(feature_scaling), x_mean(0.0), x_std(1.0), y_mean(0.0), y_std(1.0),
      enable_logging(false), log_filename("polynomial_training.log") {
    
    // Initialize weights for polynomial of given degree (degree + 1 coefficients)
    weights.resize(degree + 1, 0.0);
    initialize_optimizer_state();
}

void PolynomialRegression::initialize_optimizer_state() {
    v_weights.assign(weights.size(), 0.0);
    s_weights.assign(weights.size(), 0.0);
    g_weights_sum.assign(weights.size(), 0.0);
    loss_history.clear();
    lr_history.clear();
}

void PolynomialRegression::scale_features(std::vector<double>& x, std::vector<double>& y) {
    if (!use_feature_scaling) return;
    
    // Compute statistics for x
    x_mean = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    double x_var = 0.0;
    for (double val : x) {
        x_var += (val - x_mean) * (val - x_mean);
    }
    x_std = std::sqrt(x_var / x.size());
    if (x_std < 1e-10) x_std = 1.0;
    
    // Compute statistics for y
    y_mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    double y_var = 0.0;
    for (double val : y) {
        y_var += (val - y_mean) * (val - y_mean);
    }
    y_std = std::sqrt(y_var / y.size());
    if (y_std < 1e-10) y_std = 1.0;
    
    // Scale features
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = (x[i] - x_mean) / x_std;
        y[i] = (y[i] - y_mean) / y_std;
    }
}

double PolynomialRegression::unscale_prediction(double prediction) const {
    if (!use_feature_scaling) return prediction;
    return prediction * y_std + y_mean;
}

std::vector<double> PolynomialRegression::generate_polynomial_features(double x) const {
    std::vector<double> features(degree + 1);
    features[0] = 1.0; // bias term (x^0)
    for (int i = 1; i <= degree; ++i) {
        features[i] = std::pow(x, i);
    }
    return features;
}

std::vector<std::vector<double>> PolynomialRegression::generate_polynomial_features(const std::vector<double>& x) const {
    std::vector<std::vector<double>> feature_matrix;
    feature_matrix.reserve(x.size());
    
    for (double x_val : x) {
        feature_matrix.push_back(generate_polynomial_features(x_val));
    }
    
    return feature_matrix;
}

double PolynomialRegression::predict(double x_val) const {
    double scaled_x = use_feature_scaling ? (x_val - x_mean) / x_std : x_val;
    std::vector<double> features = generate_polynomial_features(scaled_x);
    
    double prediction = 0.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        prediction += weights[i] * features[i];
    }
    
    return unscale_prediction(prediction);
}

std::vector<double> PolynomialRegression::predict(const std::vector<double>& x_vals) const {
    std::vector<double> results;
    results.reserve(x_vals.size());
    
    for (double x : x_vals) {
        results.push_back(predict(x));
    }
    
    return results;
}

double PolynomialRegression::compute_regularization_loss() const {
    double reg_loss = 0.0;
    
    // Skip bias term (index 0) for regularization
    for (size_t i = 1; i < weights.size(); ++i) {
        switch (reg_type) {
            case RegularizationType::L1:
                reg_loss += lambda_reg * std::abs(weights[i]);
                break;
            case RegularizationType::L2:
                reg_loss += lambda_reg * weights[i] * weights[i];
                break;
            case RegularizationType::ELASTIC_NET:
                reg_loss += lambda_reg * (l1_ratio * std::abs(weights[i]) + (1 - l1_ratio) * weights[i] * weights[i]);
                break;
            default:
                break;
        }
    }
    
    return reg_loss;
}

void PolynomialRegression::update_parameters(const std::vector<double>& gradients, int iteration) {
    // Add regularization to gradients (skip bias term)
    std::vector<double> reg_gradients = gradients;
    for (size_t i = 1; i < weights.size(); ++i) {
        switch (reg_type) {
            case RegularizationType::L1:
                reg_gradients[i] += lambda_reg * (weights[i] > 0 ? 1.0 : -1.0);
                break;
            case RegularizationType::L2:
                reg_gradients[i] += 2.0 * lambda_reg * weights[i];
                break;
            case RegularizationType::ELASTIC_NET:
                reg_gradients[i] += lambda_reg * (l1_ratio * (weights[i] > 0 ? 1.0 : -1.0) + 2.0 * (1 - l1_ratio) * weights[i]);
                break;
            default:
                break;
        }
    }
    
    double current_lr = alpha;
    
    // Adaptive learning rate
    if (use_adaptive_lr) {
        current_lr = std::max(alpha * std::pow(lr_decay, iteration / 100), min_lr);
    }
    
    switch (optimizer) {
        case OptimizerType::SGD:
            for (size_t i = 0; i < weights.size(); ++i) {
                weights[i] -= current_lr * reg_gradients[i];
            }
            break;
            
        case OptimizerType::MOMENTUM:
            for (size_t i = 0; i < weights.size(); ++i) {
                v_weights[i] = momentum_beta * v_weights[i] + (1 - momentum_beta) * reg_gradients[i];
                weights[i] -= current_lr * v_weights[i];
            }
            break;
            
        case OptimizerType::ADAGRAD:
            for (size_t i = 0; i < weights.size(); ++i) {
                g_weights_sum[i] += reg_gradients[i] * reg_gradients[i];
                weights[i] -= current_lr * reg_gradients[i] / (std::sqrt(g_weights_sum[i]) + epsilon);
            }
            break;
            
        case OptimizerType::ADAM:
            for (size_t i = 0; i < weights.size(); ++i) {
                // Update biased first moment estimates
                v_weights[i] = adam_beta1 * v_weights[i] + (1 - adam_beta1) * reg_gradients[i];
                
                // Update biased second moment estimates
                s_weights[i] = adam_beta2 * s_weights[i] + (1 - adam_beta2) * reg_gradients[i] * reg_gradients[i];
                
                // Bias correction
                double v_corrected = v_weights[i] / (1 - std::pow(adam_beta1, iteration));
                double s_corrected = s_weights[i] / (1 - std::pow(adam_beta2, iteration));
                
                // Update parameters
                weights[i] -= current_lr * v_corrected / (std::sqrt(s_corrected) + epsilon);
            }
            break;
    }
    
    lr_history.push_back(current_lr);
}

void PolynomialRegression::fit(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("x and y size mismatch");
    }
    
    // Create local copies for scaling
    std::vector<double> x_scaled = x;
    std::vector<double> y_scaled = y;
    
    // Scale features if enabled
    scale_features(x_scaled, y_scaled);
    
    // Generate polynomial features
    auto feature_matrix = generate_polynomial_features(x_scaled);
    
    // Initialize optimizer state
    initialize_optimizer_state();
    
    int n = x_scaled.size();
    double prev_loss = 1e10;
    int no_improvement_count = 0;
    const int early_stopping_patience = 100;
    
    // Open log file if logging is enabled
    std::ofstream log_file;
    if (enable_logging) {
        log_file.open(log_filename, std::ios::app);
        if (log_file.is_open()) {
            log_file << "\n" << std::string(80, '=') << "\n";
            log_file << "NEW POLYNOMIAL REGRESSION TRAINING SESSION - " << n << " samples, degree " << degree << "\n";
            log_file << "Optimizer: ";
            switch (optimizer) {
                case OptimizerType::SGD: log_file << "SGD"; break;
                case OptimizerType::MOMENTUM: log_file << "Momentum"; break;
                case OptimizerType::ADAGRAD: log_file << "AdaGrad"; break;
                case OptimizerType::ADAM: log_file << "Adam"; break;
            }
            log_file << ", Regularization: ";
            switch (reg_type) {
                case RegularizationType::NONE: log_file << "None"; break;
                case RegularizationType::L1: log_file << "L1"; break;
                case RegularizationType::L2: log_file << "L2"; break;
                case RegularizationType::ELASTIC_NET: log_file << "Elastic Net"; break;
            }
            log_file << "\n" << std::string(80, '-') << "\n";
        }
    }
    
    std::cout << "Training Polynomial Regression (degree " << degree << ") with " << n << " samples" << std::endl;
    
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::vector<double> gradients(weights.size(), 0.0);
        
        // Compute gradients
        for (int i = 0; i < n; ++i) {
            const auto& features = feature_matrix[i];
            
            // Forward pass
            double prediction = 0.0;
            for (size_t j = 0; j < weights.size(); ++j) {
                prediction += weights[j] * features[j];
            }
            
            double error = prediction - y_scaled[i];
            
            // Backward pass
            for (size_t j = 0; j < weights.size(); ++j) {
                gradients[j] += error * features[j];
            }
        }
        
        // Average gradients
        for (double& grad : gradients) {
            grad /= n;
        }
        
        // Update parameters using selected optimizer
        update_parameters(gradients, epoch);
        
        // Compute loss with regularization
        double mse_loss = 0.0;
        for (int i = 0; i < n; ++i) {
            const auto& features = feature_matrix[i];
            double prediction = 0.0;
            for (size_t j = 0; j < weights.size(); ++j) {
                prediction += weights[j] * features[j];
            }
            double error = prediction - y_scaled[i];
            mse_loss += error * error;
        }
        mse_loss /= n;
        
        double total_loss = mse_loss + compute_regularization_loss();
        loss_history.push_back(total_loss);
        
        // Logging
        if (epoch % 100 == 0 || epoch == 1 || epoch == epochs) {
            std::string log_entry = "Epoch " + std::to_string(epoch) + 
                                  " | Loss: " + std::to_string(total_loss) +
                                  " | MSE: " + std::to_string(mse_loss) +
                                  " | LR: " + std::to_string(lr_history.back());
            
            if (enable_logging && log_file.is_open()) {
                log_file << log_entry << std::endl;
            } else {
                std::cout << log_entry << std::endl;
            }
        }
        
        // Convergence check
        if (has_converged(total_loss, prev_loss)) {
            std::string conv_msg = "Converged at epoch " + std::to_string(epoch) + " (loss change < " + std::to_string(tolerance) + ")";
            if (enable_logging && log_file.is_open()) {
                log_file << conv_msg << std::endl;
            } else {
                std::cout << conv_msg << std::endl;
            }
            break;
        }
        
        // Early stopping
        if (total_loss >= prev_loss) {
            no_improvement_count++;
            if (no_improvement_count >= early_stopping_patience) {
                std::string early_msg = "Early stopping at epoch " + std::to_string(epoch);
                if (enable_logging && log_file.is_open()) {
                    log_file << early_msg << std::endl;
                } else {
                    std::cout << early_msg << std::endl;
                }
                break;
            }
        } else {
            no_improvement_count = 0;
        }
        
        prev_loss = total_loss;
    }
    
    // Close log file and print summary
    if (enable_logging && log_file.is_open()) {
        log_file << "\nFinal Results:" << std::endl;
        log_file << "Final Loss: " << (loss_history.empty() ? 0.0 : loss_history.back()) << std::endl;
        log_file << "Total Epochs: " << loss_history.size() << std::endl;
        log_file << "Polynomial Coefficients: ";
        for (size_t i = 0; i < weights.size(); ++i) {
            log_file << "w" << i << "=" << weights[i] << " ";
        }
        log_file << std::endl << std::string(80, '=') << std::endl;
        log_file.close();
    }
    
    std::cout << "✅ Training completed - " << loss_history.size() << " epochs";
    if (enable_logging) {
        std::cout << " (detailed logs in " << log_filename << ")";
    }
    std::cout << std::endl;
}

double PolynomialRegression::evaluate(const std::vector<double>& x, const std::vector<double>& y) const {
    if (x.size() != y.size()) throw std::invalid_argument("x and y size mismatch");
    
    double loss = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double error = predict(x[i]) - y[i];
        loss += error * error;
    }
    return loss / x.size();
}

double PolynomialRegression::r_squared(const std::vector<double>& x, const std::vector<double>& y) const {
    if (x.size() != y.size()) throw std::invalid_argument("x and y size mismatch");
    
    std::vector<double> predictions = predict(x);
    double y_mean = std::accumulate(y.begin(), y.end(), 0.0) / y.size();
    
    double ss_res = 0.0;
    double ss_tot = 0.0;
    
    for (size_t i = 0; i < y.size(); ++i) {
        double residual = y[i] - predictions[i];
        ss_res += residual * residual;
        
        double deviation = y[i] - y_mean;
        ss_tot += deviation * deviation;
    }
    
    return 1.0 - (ss_res / ss_tot);
}

double PolynomialRegression::mean_absolute_error(const std::vector<double>& x, const std::vector<double>& y) const {
    if (x.size() != y.size()) throw std::invalid_argument("x and y size mismatch");
    
    double mae = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        mae += std::abs(predict(x[i]) - y[i]);
    }
    return mae / x.size();
}

double PolynomialRegression::adjusted_r_squared(const std::vector<double>& x, const std::vector<double>& y) const {
    double r2 = r_squared(x, y);
    int n = x.size();
    int p = degree; // number of predictors (excluding intercept)
    
    if (n <= p + 1) return r2; // Avoid division by zero or negative
    
    return 1.0 - ((1.0 - r2) * (n - 1)) / (n - p - 1);
}

bool PolynomialRegression::has_converged(double current_loss, double previous_loss) const {
    return std::abs(previous_loss - current_loss) < tolerance;
}

void PolynomialRegression::print_polynomial_equation() const {
    std::cout << "\nPolynomial Equation:\ny = ";
    
    for (int i = degree; i >= 0; --i) {
        if (i == degree) {
            std::cout << std::fixed << std::setprecision(6) << weights[i];
        } else {
            if (weights[i] >= 0) std::cout << " + " << std::fixed << std::setprecision(6) << weights[i];
            else std::cout << " - " << std::fixed << std::setprecision(6) << std::abs(weights[i]);
        }
        
        if (i > 1) std::cout << "*x^" << i;
        else if (i == 1) std::cout << "*x";
    }
    std::cout << std::endl;
}

// Getters
const std::vector<double>& PolynomialRegression::get_weights() const { return weights; }
int PolynomialRegression::get_degree() const { return degree; }
const std::vector<double>& PolynomialRegression::get_loss_history() const { return loss_history; }
const std::vector<double>& PolynomialRegression::get_lr_history() const { return lr_history; }

// Setters
void PolynomialRegression::set_optimizer(OptimizerType opt, double beta1, double beta2) {
    optimizer = opt;
    adam_beta1 = beta1;
    adam_beta2 = beta2;
}

void PolynomialRegression::set_regularization(RegularizationType reg, double strength, double l1_ratio_val) {
    reg_type = reg;
    lambda_reg = strength;
    l1_ratio = l1_ratio_val;
}

void PolynomialRegression::set_learning_rate_schedule(bool adaptive, double decay, double min_learning_rate) {
    use_adaptive_lr = adaptive;
    lr_decay = decay;
    min_lr = min_learning_rate;
}

void PolynomialRegression::set_feature_scaling(bool enable) {
    use_feature_scaling = enable;
}

void PolynomialRegression::set_logging(bool enable, const std::string& filename) {
    enable_logging = enable;
    log_filename = filename;
}

void PolynomialRegression::set_degree(int new_degree) {
    if (new_degree < 1) throw std::invalid_argument("Degree must be at least 1");
    degree = new_degree;
    weights.resize(degree + 1, 0.0);
    initialize_optimizer_state();
}

void PolynomialRegression::save_model(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Failed to open file to save model.");
    
    // Save degree and weights
    out << degree << "\n";
    for (double w : weights) {
        out << w << " ";
    }
    out << "\n";
    
    // Save scaling parameters
    out << use_feature_scaling << " " << x_mean << " " << x_std << " " << y_mean << " " << y_std << "\n";
    
    // Save hyperparameters
    out << static_cast<int>(optimizer) << " " << static_cast<int>(reg_type) << " " << lambda_reg << "\n";
    
    out.close();
    std::cout << "Polynomial Regression model saved to " << filename << std::endl;
}

void PolynomialRegression::load_model(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Failed to open file to load model.");
    
    // Load degree and weights
    in >> degree;
    weights.resize(degree + 1);
    for (size_t i = 0; i < weights.size(); ++i) {
        in >> weights[i];
    }
    
    // Load scaling parameters
    in >> use_feature_scaling >> x_mean >> x_std >> y_mean >> y_std;
    
    // Load hyperparameters
    int opt_int, reg_int;
    in >> opt_int >> reg_int >> lambda_reg;
    optimizer = static_cast<OptimizerType>(opt_int);
    reg_type = static_cast<RegularizationType>(reg_int);
    
    in.close();
    std::cout << "Polynomial Regression model loaded from " << filename << std::endl;
}

std::vector<double> PolynomialRegression::get_feature_importance() const {
    std::vector<double> importance = weights;
    // Convert to absolute values for importance (excluding bias)
    for (size_t i = 1; i < importance.size(); ++i) {
        importance[i] = std::abs(importance[i]);
    }
    return importance;
}

double PolynomialRegression::compute_gradient_norm() const {
    // Placeholder - would need to store last computed gradients
    return 0.0;
}

void PolynomialRegression::print_training_summary() const {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "POLYNOMIAL REGRESSION TRAINING SUMMARY\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Degree: " << degree << "\n";
    std::cout << "Final Loss: " << (loss_history.empty() ? 0.0 : loss_history.back()) << "\n";
    std::cout << "Total Epochs: " << loss_history.size() << "\n";
    std::cout << "Feature Scaling: " << (use_feature_scaling ? "Enabled" : "Disabled") << "\n";
    if (use_feature_scaling) {
        std::cout << "X: mean=" << x_mean << ", std=" << x_std << "\n";
        std::cout << "Y: mean=" << y_mean << ", std=" << y_std << "\n";
    }
    print_polynomial_equation();
    std::cout << std::string(60, '=') << "\n";
}

double PolynomialRegression::evaluate_polynomial(double x) const {
    return predict(x);
}
