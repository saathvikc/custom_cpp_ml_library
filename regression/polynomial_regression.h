#pragma once
#include "regression.h"
#include "../utils/log_manager.h"
#include <vector>
#include <string>

class PolynomialRegression {
private:
    // Model parameters
    std::vector<double> weights;     // Polynomial coefficients [w0, w1, w2, ..., w_degree]
    int degree;                      // Polynomial degree
    
    // Training hyperparameters
    double alpha;                    // Learning rate
    int epochs;                      // Maximum epochs
    double tolerance;                // Convergence tolerance
    
    // Advanced optimizer parameters
    OptimizerType optimizer;
    double momentum_beta;            // Momentum parameter
    double adam_beta1, adam_beta2;   // Adam parameters
    double epsilon;                  // Small constant for numerical stability
    
    // Optimizer state variables
    std::vector<double> v_weights;   // Momentum/Adam first moment
    std::vector<double> s_weights;   // Adam second moment
    std::vector<double> g_weights_sum; // AdaGrad accumulated gradients
    
    // Regularization
    RegularizationType reg_type;
    double lambda_reg;               // Regularization strength
    double l1_ratio;                 // Elastic net mixing parameter
    
    // Adaptive learning rate
    bool use_adaptive_lr;
    double lr_decay;
    double min_lr;
    
    // Feature scaling
    bool use_feature_scaling;
    double x_mean, x_std, y_mean, y_std;
    
    // Training history
    std::vector<double> loss_history;
    std::vector<double> lr_history;
    
    // Logging
    bool enable_logging;
    std::string data_name;  // Name of dataset for logging organization
    
    // Internal methods
    void initialize_optimizer_state();
    void update_parameters(const std::vector<double>& gradients, int iteration);
    double compute_regularization_loss() const;
    void scale_features(std::vector<double>& x, std::vector<double>& y);
    double unscale_prediction(double prediction) const;
    std::vector<double> generate_polynomial_features(double x) const;
    std::vector<std::vector<double>> generate_polynomial_features(const std::vector<double>& x) const;

public:
    // Constructor with comprehensive parameters
    PolynomialRegression(int poly_degree = 2,
                        double learning_rate = 0.01, 
                        int max_epochs = 1000, 
                        double tol = 1e-6,
                        OptimizerType opt = OptimizerType::ADAM,
                        RegularizationType reg = RegularizationType::NONE,
                        double reg_strength = 0.01,
                        bool adaptive_lr = true,
                        bool feature_scaling = true);
    
    // Core training and prediction
    void fit(const std::vector<double>& x, const std::vector<double>& y);
    double predict(double x_val) const;
    std::vector<double> predict(const std::vector<double>& x_vals) const;
    
    // Evaluation metrics
    double evaluate(const std::vector<double>& x, const std::vector<double>& y) const;
    double r_squared(const std::vector<double>& x, const std::vector<double>& y) const;
    double mean_absolute_error(const std::vector<double>& x, const std::vector<double>& y) const;
    double adjusted_r_squared(const std::vector<double>& x, const std::vector<double>& y) const;
    
    // Getters for model parameters
    const std::vector<double>& get_weights() const;
    int get_degree() const;
    const std::vector<double>& get_loss_history() const;
    const std::vector<double>& get_lr_history() const;
    
    // Hyperparameter setters
    void set_optimizer(OptimizerType opt, double beta1 = 0.9, double beta2 = 0.999);
    void set_regularization(RegularizationType reg, double strength = 0.01, double l1_ratio = 0.5);
    void set_learning_rate_schedule(bool adaptive, double decay = 0.95, double min_lr = 1e-6);
    void set_feature_scaling(bool enable);
    void set_logging(bool enable, const std::string& dataset_name = "default");
    
    // Model persistence
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);
    
    // Advanced analysis
    void print_training_summary() const;
    void print_polynomial_equation() const;
    double compute_gradient_norm() const;
    bool has_converged(double current_loss, double previous_loss) const;
    
    // Polynomial-specific methods
    double evaluate_polynomial(double x) const;
    std::vector<double> get_feature_importance() const;
    void set_degree(int new_degree);
};
