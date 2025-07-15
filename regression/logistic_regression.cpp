#include "logistic_regression.h"
#include "../utils/log_manager.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iomanip>

LogisticRegression::LogisticRegression(double learning_rate, int max_epochs, double tol,
                                     OptimizerType opt, RegularizationType reg, double reg_strength,
                                     bool adaptive_lr, bool feature_scaling)
    : w(0.0), b(0.0), alpha(learning_rate), epochs(max_epochs), tolerance(tol),
      optimizer(opt), momentum_beta(0.9), adam_beta1(0.9), adam_beta2(0.999), epsilon(1e-8),
      vw(0.0), vb(0.0), sw(0.0), sb(0.0), gw_sum(0.0), gb_sum(0.0),
      reg_type(reg), lambda_reg(reg_strength), l1_ratio(0.5),
      use_adaptive_lr(adaptive_lr), lr_decay(0.95), min_lr(1e-6),
      use_feature_scaling(feature_scaling), x_mean(0.0), x_std(1.0),
      enable_logging(false), data_name("default") {
    
    initialize_optimizer_state();
}

void LogisticRegression::initialize_optimizer_state() {
    vw = vb = 0.0;
    sw = sb = 0.0;
    gw_sum = gb_sum = 0.0;
    loss_history.clear();
    lr_history.clear();
    accuracy_history.clear();
}

void LogisticRegression::scale_features(std::vector<double>& x) {
    if (!use_feature_scaling) return;
    
    // Compute statistics for x
    x_mean = std::accumulate(x.begin(), x.end(), 0.0) / x.size();
    double x_var = 0.0;
    for (double val : x) {
        x_var += (val - x_mean) * (val - x_mean);
    }
    x_std = std::sqrt(x_var / x.size());
    if (x_std < 1e-10) x_std = 1.0; // Prevent division by zero
    
    // Scale features
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] = (x[i] - x_mean) / x_std;
    }
}

double LogisticRegression::sigmoid(double z) const {
    // Numerical stability for sigmoid function
    if (z > 500) return 1.0;
    if (z < -500) return 0.0;
    return 1.0 / (1.0 + std::exp(-z));
}

double LogisticRegression::predict_proba(double x_val) const {
    double scaled_x = use_feature_scaling ? (x_val - x_mean) / x_std : x_val;
    double z = w * scaled_x + b;
    return sigmoid(z);
}

int LogisticRegression::predict(double x_val) const {
    return predict_proba(x_val) >= 0.5 ? 1 : 0;
}

std::vector<double> LogisticRegression::predict_proba(const std::vector<double>& x_vals) const {
    std::vector<double> results;
    results.reserve(x_vals.size());
    
    for (double x : x_vals) {
        results.push_back(predict_proba(x));
    }
    
    return results;
}

std::vector<int> LogisticRegression::predict(const std::vector<double>& x_vals) const {
    std::vector<int> results;
    results.reserve(x_vals.size());
    
    for (double x : x_vals) {
        results.push_back(predict(x));
    }
    
    return results;
}

double LogisticRegression::compute_regularization_loss() const {
    switch (reg_type) {
        case RegularizationType::L1:
            return lambda_reg * std::abs(w);
        case RegularizationType::L2:
            return lambda_reg * w * w;
        case RegularizationType::ELASTIC_NET:
            return lambda_reg * (l1_ratio * std::abs(w) + (1 - l1_ratio) * w * w);
        default:
            return 0.0;
    }
}

void LogisticRegression::update_parameters(double dw, double db, int iteration) {
    // Add regularization to weight gradient
    switch (reg_type) {
        case RegularizationType::L1:
            dw += lambda_reg * (w > 0 ? 1.0 : -1.0);
            break;
        case RegularizationType::L2:
            dw += 2.0 * lambda_reg * w;
            break;
        case RegularizationType::ELASTIC_NET:
            dw += lambda_reg * (l1_ratio * (w > 0 ? 1.0 : -1.0) + 2.0 * (1 - l1_ratio) * w);
            break;
        default:
            break;
    }
    
    double current_lr = alpha;
    
    // Adaptive learning rate
    if (use_adaptive_lr) {
        current_lr = std::max(alpha * std::pow(lr_decay, iteration / 100), min_lr);
    }
    
    switch (optimizer) {
        case OptimizerType::SGD:
            w -= current_lr * dw;
            b -= current_lr * db;
            break;
            
        case OptimizerType::MOMENTUM:
            vw = momentum_beta * vw + (1 - momentum_beta) * dw;
            vb = momentum_beta * vb + (1 - momentum_beta) * db;
            w -= current_lr * vw;
            b -= current_lr * vb;
            break;
            
        case OptimizerType::ADAGRAD:
            gw_sum += dw * dw;
            gb_sum += db * db;
            w -= current_lr * dw / (std::sqrt(gw_sum) + epsilon);
            b -= current_lr * db / (std::sqrt(gb_sum) + epsilon);
            break;
            
        case OptimizerType::ADAM:
            // Update biased first moment estimates
            vw = adam_beta1 * vw + (1 - adam_beta1) * dw;
            vb = adam_beta1 * vb + (1 - adam_beta1) * db;
            
            // Update biased second moment estimates
            sw = adam_beta2 * sw + (1 - adam_beta2) * dw * dw;
            sb = adam_beta2 * sb + (1 - adam_beta2) * db * db;
            
            // Bias correction
            double vw_corrected = vw / (1 - std::pow(adam_beta1, iteration));
            double vb_corrected = vb / (1 - std::pow(adam_beta1, iteration));
            double sw_corrected = sw / (1 - std::pow(adam_beta2, iteration));
            double sb_corrected = sb / (1 - std::pow(adam_beta2, iteration));
            
            // Update parameters
            w -= current_lr * vw_corrected / (std::sqrt(sw_corrected) + epsilon);
            b -= current_lr * vb_corrected / (std::sqrt(sb_corrected) + epsilon);
            break;
    }
    
    lr_history.push_back(current_lr);
}

double LogisticRegression::compute_accuracy(const std::vector<double>& x, const std::vector<int>& y) const {
    int correct = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        if (predict(x[i]) == y[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / x.size();
}

void LogisticRegression::fit(const std::vector<double>& x, const std::vector<int>& y) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("x and y size mismatch");
    }
    
    // Validate that y contains only 0 and 1
    for (int label : y) {
        if (label != 0 && label != 1) {
            throw std::invalid_argument("Labels must be 0 or 1 for binary classification");
        }
    }
    
    // Create local copy for scaling
    std::vector<double> x_scaled = x;
    
    // Scale features if enabled
    scale_features(x_scaled);
    
    // Initialize optimizer state
    initialize_optimizer_state();
    
    int n = x_scaled.size();
    double prev_loss = 1e10;
    int no_improvement_count = 0;
    const int early_stopping_patience = 50;
    
    // Open log file if logging is enabled
    std::ofstream log_file;
    std::string log_path;
    if (enable_logging) {
        log_path = LogManager::generate_log_path("logistic_regression", data_name, "training");
        log_file.open(log_path, std::ios::app);
        if (log_file.is_open()) {
            log_file << LogManager::create_session_header("Logistic Regression", 
                                                         "Training on " + std::to_string(n) + " samples");
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
    
    std::cout << "Training Logistic Regression with " << n << " samples" << std::endl;
    
    for (int epoch = 1; epoch <= epochs; ++epoch) {
        double dw = 0.0, db = 0.0;
        double log_loss = 0.0;
        
        // Compute gradients using logistic regression
        for (int i = 0; i < n; ++i) {
            double z = w * x_scaled[i] + b;
            double pred = sigmoid(z);
            
            // Gradient computation
            double error = pred - y[i];
            dw += error * x_scaled[i];
            db += error;
            
            // Log-likelihood loss
            if (y[i] == 1) {
                log_loss -= std::log(pred + epsilon);  // Avoid log(0)
            } else {
                log_loss -= std::log(1 - pred + epsilon);  // Avoid log(0)
            }
        }
        
        dw /= n;
        db /= n;
        log_loss /= n;
        
        // Update parameters using selected optimizer
        update_parameters(dw, db, epoch);
        
        // Total loss with regularization
        double total_loss = log_loss + compute_regularization_loss();
        loss_history.push_back(total_loss);
        
        // Compute accuracy
        double accuracy = compute_accuracy(x_scaled, y);
        accuracy_history.push_back(accuracy);
        
        // Logging
        if (epoch % 100 == 0 || epoch == 1 || epoch == epochs) {
            std::string log_entry = "Epoch " + std::to_string(epoch) + 
                                  " | Loss: " + std::to_string(total_loss) +
                                  " | Accuracy: " + std::to_string(accuracy) +
                                  " | LR: " + std::to_string(lr_history.back()) +
                                  " | w: " + std::to_string(w) + 
                                  " | b: " + std::to_string(b);
            
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
                std::string early_msg = "Early stopping at epoch " + std::to_string(epoch) + " (no improvement for " + 
                                      std::to_string(early_stopping_patience) + " epochs)";
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
        log_file << "Final Accuracy: " << (accuracy_history.empty() ? 0.0 : accuracy_history.back()) << std::endl;
        log_file << "Total Epochs: " << loss_history.size() << std::endl;
        log_file << "Final Weight: " << w << std::endl;
        log_file << "Final Bias: " << b << std::endl;
        log_file << std::string(80, '=') << std::endl;
        log_file.close();
    }
    
    // Console shows only summary
    std::cout << "✅ Training completed - " << loss_history.size() << " epochs";
    if (enable_logging) {
        std::cout << " (detailed logs in " << log_path << ")";
    }
    std::cout << std::endl;
}

double LogisticRegression::evaluate_loss(const std::vector<double>& x, const std::vector<int>& y) const {
    if (x.size() != y.size()) throw std::invalid_argument("x and y size mismatch");
    
    double log_loss = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        double pred = predict_proba(x[i]);
        if (y[i] == 1) {
            log_loss -= std::log(pred + epsilon);
        } else {
            log_loss -= std::log(1 - pred + epsilon);
        }
    }
    return log_loss / x.size();
}

double LogisticRegression::evaluate_accuracy(const std::vector<double>& x, const std::vector<int>& y) const {
    if (x.size() != y.size()) throw std::invalid_argument("x and y size mismatch");
    
    int correct = 0;
    for (size_t i = 0; i < x.size(); ++i) {
        if (predict(x[i]) == y[i]) {
            correct++;
        }
    }
    return static_cast<double>(correct) / x.size();
}

double LogisticRegression::evaluate_precision(const std::vector<double>& x, const std::vector<int>& y) const {
    if (x.size() != y.size()) throw std::invalid_argument("x and y size mismatch");
    
    int true_positives = 0;
    int false_positives = 0;
    
    for (size_t i = 0; i < x.size(); ++i) {
        int pred = predict(x[i]);
        if (pred == 1 && y[i] == 1) true_positives++;
        if (pred == 1 && y[i] == 0) false_positives++;
    }
    
    if (true_positives + false_positives == 0) return 0.0;
    return static_cast<double>(true_positives) / (true_positives + false_positives);
}

double LogisticRegression::evaluate_recall(const std::vector<double>& x, const std::vector<int>& y) const {
    if (x.size() != y.size()) throw std::invalid_argument("x and y size mismatch");
    
    int true_positives = 0;
    int false_negatives = 0;
    
    for (size_t i = 0; i < x.size(); ++i) {
        int pred = predict(x[i]);
        if (pred == 1 && y[i] == 1) true_positives++;
        if (pred == 0 && y[i] == 1) false_negatives++;
    }
    
    if (true_positives + false_negatives == 0) return 0.0;
    return static_cast<double>(true_positives) / (true_positives + false_negatives);
}

double LogisticRegression::evaluate_f1_score(const std::vector<double>& x, const std::vector<int>& y) const {
    double precision = evaluate_precision(x, y);
    double recall = evaluate_recall(x, y);
    
    if (precision + recall == 0.0) return 0.0;
    return 2.0 * precision * recall / (precision + recall);
}

bool LogisticRegression::has_converged(double current_loss, double previous_loss) const {
    return std::abs(previous_loss - current_loss) < tolerance;
}

double LogisticRegression::compute_gradient_norm() const {
    // Placeholder for gradient norm computation
    return 0.0;
}

void LogisticRegression::print_training_summary() const {
    std::cout << "\n" << std::string(50, '=') << "\n";
    std::cout << "LOGISTIC REGRESSION TRAINING SUMMARY\n";
    std::cout << std::string(50, '=') << "\n";
    std::cout << "Final Loss: " << (loss_history.empty() ? 0.0 : loss_history.back()) << "\n";
    std::cout << "Final Accuracy: " << (accuracy_history.empty() ? 0.0 : accuracy_history.back()) << "\n";
    std::cout << "Total Epochs: " << loss_history.size() << "\n";
    std::cout << "Final Weight: " << w << "\n";
    std::cout << "Final Bias: " << b << "\n";
    std::cout << "Feature Scaling: " << (use_feature_scaling ? "Enabled" : "Disabled") << "\n";
    if (use_feature_scaling) {
        std::cout << "X: mean=" << x_mean << ", std=" << x_std << "\n";
    }
    std::cout << std::string(50, '=') << "\n";
}

// Getters
double LogisticRegression::get_weight() const { return w; }
double LogisticRegression::get_bias() const { return b; }
const std::vector<double>& LogisticRegression::get_loss_history() const { return loss_history; }
const std::vector<double>& LogisticRegression::get_lr_history() const { return lr_history; }
const std::vector<double>& LogisticRegression::get_accuracy_history() const { return accuracy_history; }

// Setters
void LogisticRegression::set_optimizer(OptimizerType opt, double beta1, double beta2) {
    optimizer = opt;
    adam_beta1 = beta1;
    adam_beta2 = beta2;
}

void LogisticRegression::set_regularization(RegularizationType reg, double strength, double l1_ratio_val) {
    reg_type = reg;
    lambda_reg = strength;
    l1_ratio = l1_ratio_val;
}

void LogisticRegression::set_learning_rate_schedule(bool adaptive, double decay, double min_learning_rate) {
    use_adaptive_lr = adaptive;
    lr_decay = decay;
    min_lr = min_learning_rate;
}

void LogisticRegression::set_feature_scaling(bool enable) {
    use_feature_scaling = enable;
}

void LogisticRegression::set_logging(bool enable, const std::string& dataset_name) {
    enable_logging = enable;
    data_name = dataset_name;
}

void LogisticRegression::save_model(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Failed to open file to save model.");
    
    // Save model parameters
    out << w << " " << b << "\n";
    
    // Save scaling parameters
    out << use_feature_scaling << " " << x_mean << " " << x_std << "\n";
    
    // Save hyperparameters
    out << static_cast<int>(optimizer) << " " << static_cast<int>(reg_type) << " " << lambda_reg << "\n";
    
    out.close();
    std::cout << "Logistic Regression model saved to " << filename << std::endl;
}

void LogisticRegression::load_model(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Failed to open file to load model.");
    
    // Load model parameters
    in >> w >> b;
    
    // Load scaling parameters
    in >> use_feature_scaling >> x_mean >> x_std;
    
    // Load hyperparameters
    int opt_int, reg_int;
    in >> opt_int >> reg_int >> lambda_reg;
    optimizer = static_cast<OptimizerType>(opt_int);
    reg_type = static_cast<RegularizationType>(reg_int);
    
    in.close();
    std::cout << "Logistic Regression model loaded from " << filename << std::endl;
}
