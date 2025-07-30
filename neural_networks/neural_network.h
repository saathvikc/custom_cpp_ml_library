#pragma once

#include "../regression/regression_common.h"
#include "../regression/regression.h"
#include "../utils/log_manager.h"
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <random>
#include <memory>
#include <limits>

// Activation functions
enum class ActivationType {
    SIGMOID,
    TANH,
    RELU,
    LEAKY_RELU,
    SOFTMAX,
    LINEAR
};

// Loss functions
enum class LossType {
    MEAN_SQUARED_ERROR,
    BINARY_CROSSENTROPY,
    CATEGORICAL_CROSSENTROPY,
    HUBER
};

// Layer types
enum class LayerType {
    DENSE,
    DROPOUT,
    BATCH_NORM
};

// Forward declaration
class Layer;

// Neural Network class
class NeuralNetwork {
public:
    // Constructors
    NeuralNetwork();
    NeuralNetwork(const std::vector<int>& layer_sizes,
                  const std::vector<ActivationType>& activations,
                  LossType loss = LossType::MEAN_SQUARED_ERROR,
                  OptimizerType optimizer = OptimizerType::ADAM,
                  double learning_rate = 0.001,
                  bool use_bias = true,
                  bool use_batch_norm = false);

    // Destructor
    ~NeuralNetwork() = default;

    // Core training methods
    void fit(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y,
             int epochs = 1000, int batch_size = 32, double validation_split = 0.2,
             bool shuffle = true, bool verbose = true);
    
    void fit_classification(const std::vector<std::vector<double>>& X, const std::vector<int>& y,
                           int epochs = 1000, int batch_size = 32, double validation_split = 0.2,
                           bool shuffle = true, bool verbose = true);

    // Prediction methods
    std::vector<double> predict(const std::vector<double>& features) const;
    std::vector<std::vector<double>> predict_batch(const std::vector<std::vector<double>>& X) const;
    
    // Classification prediction methods
    int predict_class(const std::vector<double>& features) const;
    std::vector<int> predict_classes(const std::vector<std::vector<double>>& X) const;
    std::vector<double> predict_proba(const std::vector<double>& features) const;

    // Network architecture methods
    void add_layer(int neurons, ActivationType activation = ActivationType::RELU, bool use_dropout = false, double dropout_rate = 0.5);
    void set_output_layer(int neurons, ActivationType activation = ActivationType::LINEAR);
    void compile(LossType loss, OptimizerType optimizer = OptimizerType::ADAM, double learning_rate = 0.001);

    // Evaluation methods
    double evaluate_loss(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y) const;
    double evaluate_accuracy(const std::vector<std::vector<double>>& X, const std::vector<int>& y) const;
    double evaluate_r2_score(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y) const;
    std::map<std::string, double> classification_report(const std::vector<std::vector<double>>& X, const std::vector<int>& y) const;

    // Model persistence
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);

    // Configuration methods
    void set_optimizer(OptimizerType optimizer, double learning_rate = 0.001);
    void set_regularization(RegularizationType reg_type, double lambda = 0.01);
    void set_early_stopping(bool enable, int patience = 10, double min_delta = 1e-6);
    void set_learning_rate_schedule(bool enable, double decay_rate = 0.95, int decay_steps = 100);

    // Logging and monitoring
    void set_logging(bool enable, const std::string& data_name = "neural_network");
    void print_model_summary() const;
    void plot_training_history() const;

    // Getters
    std::vector<double> get_training_loss_history() const { return training_loss_history; }
    std::vector<double> get_validation_loss_history() const { return validation_loss_history; }
    std::vector<double> get_training_accuracy_history() const { return training_accuracy_history; }
    std::vector<double> get_validation_accuracy_history() const { return validation_accuracy_history; }
    
    // Model info getters
    size_t get_num_layers() const { return layers.size(); }
    int get_input_size() const { return input_size; }
    int get_output_size() const { return output_size; }
    bool is_fitted() const { return fitted; }

private:
    // Network architecture
    std::vector<std::unique_ptr<Layer>> layers;
    int input_size;
    int output_size;
    bool fitted;

    // Training configuration
    LossType loss_function;
    OptimizerType optimizer_type;
    double learning_rate;
    double initial_learning_rate;
    bool use_bias;
    bool use_batch_normalization;

    // Regularization
    RegularizationType regularization_type;
    double regularization_lambda;

    // Early stopping
    bool early_stopping_enabled;
    int early_stopping_patience;
    double early_stopping_min_delta;
    int early_stopping_counter;
    double best_validation_loss;

    // Learning rate scheduling
    bool lr_schedule_enabled;
    double lr_decay_rate;
    int lr_decay_steps;

    // Training history
    std::vector<double> training_loss_history;
    std::vector<double> validation_loss_history;
    std::vector<double> training_accuracy_history;
    std::vector<double> validation_accuracy_history;

    // Logging
    bool enable_logging;
    std::string data_name;
    LogManager log_manager;

    // Random number generator
    mutable std::mt19937 rng;

    // Helper methods
    void initialize_network();
    void forward_pass(const std::vector<double>& input) const;
    void backward_pass(const std::vector<double>& target);
    void update_weights(int batch_size);
    
    std::vector<double> compute_loss_gradient(const std::vector<double>& predictions, const std::vector<double>& targets) const;
    double compute_loss(const std::vector<double>& predictions, const std::vector<double>& targets) const;
    
    void shuffle_data(std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& y) const;
    void shuffle_data_classification(std::vector<std::vector<double>>& X, std::vector<int>& y) const;
    
    std::vector<double> one_hot_encode(int class_label, int num_classes) const;
    std::vector<std::vector<double>> one_hot_encode_batch(const std::vector<int>& labels, int num_classes) const;
    
    void update_learning_rate(int epoch);
    bool check_early_stopping(double validation_loss);
    
    void log_epoch(int epoch, double train_loss, double val_loss, double train_acc, double val_acc) const;
    void validate_input_dimensions(const std::vector<std::vector<double>>& X) const;
};

// Layer base class
class Layer {
public:
    Layer(int input_size, int output_size, ActivationType activation = ActivationType::RELU, bool use_bias = true);
    virtual ~Layer() = default;

    virtual std::vector<double> forward(const std::vector<double>& input) = 0;
    virtual std::vector<double> backward(const std::vector<double>& gradient) = 0;
    virtual void update_weights(OptimizerType optimizer, double learning_rate, RegularizationType reg_type, double reg_lambda) = 0;
    
    virtual void save_layer(std::ofstream& file) const = 0;
    virtual void load_layer(std::ifstream& file) = 0;

    // Getters
    int get_input_size() const { return input_size; }
    int get_output_size() const { return output_size; }
    ActivationType get_activation() const { return activation_type; }
    
    // Public access to outputs for neural network
    std::vector<double> last_input;
    std::vector<double> last_output;
    std::vector<double> last_pre_activation;

protected:
    int input_size;
    int output_size;
    ActivationType activation_type;
    bool use_bias;
};

// Dense (fully connected) layer
class DenseLayer : public Layer {
public:
    DenseLayer(int input_size, int output_size, ActivationType activation = ActivationType::RELU, bool use_bias = true);

    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& gradient) override;
    void update_weights(OptimizerType optimizer, double learning_rate, RegularizationType reg_type, double reg_lambda) override;
    
    void save_layer(std::ofstream& file) const override;
    void load_layer(std::ifstream& file) override;

private:
    std::vector<std::vector<double>> weights;
    std::vector<double> biases;
    std::vector<double> weight_gradients;
    std::vector<double> bias_gradients;
    
    // Optimizer state (for Adam, momentum, etc.)
    std::vector<std::vector<double>> weight_momentum;
    std::vector<std::vector<double>> weight_velocity;
    std::vector<double> bias_momentum;
    std::vector<double> bias_velocity;
    int update_count;

    void initialize_weights();
    std::vector<double> apply_activation(const std::vector<double>& input) const;
    std::vector<double> activation_derivative(const std::vector<double>& input) const;
};

// Activation functions namespace
namespace Activation {
    double sigmoid(double x);
    double sigmoid_derivative(double x);
    double tanh_fn(double x);
    double tanh_derivative(double x);
    double relu(double x);
    double relu_derivative(double x);
    double leaky_relu(double x, double alpha = 0.01);
    double leaky_relu_derivative(double x, double alpha = 0.01);
    std::vector<double> softmax(const std::vector<double>& x);
    std::vector<double> softmax_derivative(const std::vector<double>& x);
}

// Loss functions namespace
namespace Loss {
    double mean_squared_error(const std::vector<double>& predictions, const std::vector<double>& targets);
    std::vector<double> mse_gradient(const std::vector<double>& predictions, const std::vector<double>& targets);
    
    double binary_crossentropy(const std::vector<double>& predictions, const std::vector<double>& targets);
    std::vector<double> binary_crossentropy_gradient(const std::vector<double>& predictions, const std::vector<double>& targets);
    
    double categorical_crossentropy(const std::vector<double>& predictions, const std::vector<double>& targets);
    std::vector<double> categorical_crossentropy_gradient(const std::vector<double>& predictions, const std::vector<double>& targets);
    
    double huber_loss(const std::vector<double>& predictions, const std::vector<double>& targets, double delta = 1.0);
    std::vector<double> huber_gradient(const std::vector<double>& predictions, const std::vector<double>& targets, double delta = 1.0);
}
