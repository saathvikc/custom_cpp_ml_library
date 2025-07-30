#include "neural_network.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <cassert>
#include <iomanip>

// ============================================================================
// NEURAL NETWORK IMPLEMENTATION
// ============================================================================

NeuralNetwork::NeuralNetwork() 
    : input_size(0), output_size(0), fitted(false), loss_function(LossType::MEAN_SQUARED_ERROR),
      optimizer_type(OptimizerType::ADAM), learning_rate(0.001), initial_learning_rate(0.001),
      use_bias(true), use_batch_normalization(false), regularization_type(RegularizationType::NONE),
      regularization_lambda(0.01), early_stopping_enabled(false), early_stopping_patience(10),
      early_stopping_min_delta(1e-6), early_stopping_counter(0), best_validation_loss(std::numeric_limits<double>::max()),
      lr_schedule_enabled(false), lr_decay_rate(0.95), lr_decay_steps(100),
      enable_logging(false), data_name("neural_network"), rng(std::random_device{}()) {
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layer_sizes,
                           const std::vector<ActivationType>& activations,
                           LossType loss, OptimizerType optimizer, double lr,
                           bool bias, bool batch_norm)
    : input_size(layer_sizes.empty() ? 0 : layer_sizes[0]), 
      output_size(layer_sizes.empty() ? 0 : layer_sizes.back()), 
      fitted(false), loss_function(loss), optimizer_type(optimizer), 
      learning_rate(lr), initial_learning_rate(lr), use_bias(bias), 
      use_batch_normalization(batch_norm), regularization_type(RegularizationType::NONE),
      regularization_lambda(0.01), early_stopping_enabled(false), early_stopping_patience(10),
      early_stopping_min_delta(1e-6), early_stopping_counter(0), best_validation_loss(std::numeric_limits<double>::max()),
      lr_schedule_enabled(false), lr_decay_rate(0.95), lr_decay_steps(100),
      enable_logging(false), data_name("neural_network"), rng(std::random_device{}()) {
    
    if (layer_sizes.size() < 2) {
        throw std::invalid_argument("Neural network must have at least input and output layers");
    }
    
    if (activations.size() != layer_sizes.size() - 1) {
        throw std::invalid_argument("Number of activations must equal number of layer transitions");
    }
    
    // Create layers
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        layers.push_back(std::make_unique<DenseLayer>(layer_sizes[i], layer_sizes[i + 1], activations[i], use_bias));
    }
}

void NeuralNetwork::add_layer(int neurons, ActivationType activation, bool use_dropout, double dropout_rate) {
    if (layers.empty()) {
        throw std::runtime_error("Cannot add layer without knowing input size. Use constructor or set input size first.");
    }
    
    int input_size = layers.back()->get_output_size();
    layers.push_back(std::make_unique<DenseLayer>(input_size, neurons, activation, use_bias));
    output_size = neurons;
}

void NeuralNetwork::set_output_layer(int neurons, ActivationType activation) {
    if (layers.empty()) {
        throw std::runtime_error("Cannot set output layer without any hidden layers");
    }
    
    int input_size = layers.back()->get_output_size();
    layers.push_back(std::make_unique<DenseLayer>(input_size, neurons, activation, use_bias));
    output_size = neurons;
}

void NeuralNetwork::compile(LossType loss, OptimizerType optimizer, double lr) {
    loss_function = loss;
    optimizer_type = optimizer;
    learning_rate = lr;
    initial_learning_rate = lr;
}

void NeuralNetwork::fit(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y,
                       int epochs, int batch_size, double validation_split, bool shuffle, bool verbose) {
    if (X.empty() || y.empty() || X.size() != y.size()) {
        throw std::invalid_argument("Invalid training data dimensions");
    }
    
    if (layers.empty()) {
        throw std::runtime_error("Network architecture not defined. Add layers before training.");
    }
    
    input_size = X[0].size();
    if (layers[0]->get_input_size() != input_size) {
        throw std::invalid_argument("Input size mismatch between data and first layer");
    }
    
    // Split data into training and validation sets
    int total_samples = X.size();
    int val_samples = static_cast<int>(total_samples * validation_split);
    int train_samples = total_samples - val_samples;
    
    std::vector<std::vector<double>> X_train(X.begin(), X.begin() + train_samples);
    std::vector<std::vector<double>> y_train(y.begin(), y.begin() + train_samples);
    std::vector<std::vector<double>> X_val(X.begin() + train_samples, X.end());
    std::vector<std::vector<double>> y_val(y.begin() + train_samples, y.end());
    
    // Initialize logging
    if (enable_logging) {
        std::string log_path = log_manager.generate_log_path("neural_networks", data_name);
        std::ofstream log_file(log_path);
        log_file << log_manager.create_session_header("Neural Network Training", "Regression");
        log_file << "Network Architecture:\n";
        log_file << "Input size: " << input_size << "\n";
        log_file << "Output size: " << output_size << "\n";
        log_file << "Hidden layers: " << (layers.size() - 1) << "\n";
        log_file << "Loss function: " << static_cast<int>(loss_function) << "\n";
        log_file << "Optimizer: " << static_cast<int>(optimizer_type) << "\n";
        log_file << "Learning rate: " << learning_rate << "\n";
        log_file << std::string(80, '=') << "\n";
        log_file.close();
    }
    
    // Training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Shuffle training data
        if (shuffle) {
            shuffle_data(X_train, y_train);
        }
        
        double epoch_train_loss = 0.0;
        int num_batches = (train_samples + batch_size - 1) / batch_size;
        
        // Mini-batch training
        for (int batch = 0; batch < num_batches; ++batch) {
            int start_idx = batch * batch_size;
            int end_idx = std::min(start_idx + batch_size, train_samples);
            int current_batch_size = end_idx - start_idx;
            
            double batch_loss = 0.0;
            
            // Process batch
            for (int i = start_idx; i < end_idx; ++i) {
                forward_pass(X_train[i]);
                
                // Get output from last layer
                std::vector<double> predictions = layers.back()->last_output;
                
                // Compute loss
                batch_loss += compute_loss(predictions, y_train[i]);
                
                // Backward pass
                backward_pass(y_train[i]);
            }
            
            // Update weights
            update_weights(current_batch_size);
            epoch_train_loss += batch_loss;
        }
        
        epoch_train_loss /= train_samples;
        
        // Validation
        double val_loss = 0.0;
        if (!X_val.empty()) {
            val_loss = evaluate_loss(X_val, y_val);
        }
        
        // Store history
        training_loss_history.push_back(epoch_train_loss);
        validation_loss_history.push_back(val_loss);
        
        // Update learning rate
        update_learning_rate(epoch);
        
        // Early stopping check
        if (early_stopping_enabled && check_early_stopping(val_loss)) {
            if (verbose) {
                std::cout << "Early stopping at epoch " << epoch + 1 << std::endl;
            }
            break;
        }
        
        // Logging
        if (verbose && (epoch + 1) % 100 == 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs 
                      << " - Loss: " << std::fixed << std::setprecision(6) << epoch_train_loss;
            if (!X_val.empty()) {
                std::cout << " - Val Loss: " << val_loss;
            }
            std::cout << " - LR: " << learning_rate << std::endl;
        }
        
        if (enable_logging) {
            log_epoch(epoch + 1, epoch_train_loss, val_loss, 0.0, 0.0);
        }
    }
    
    fitted = true;
    
    if (verbose) {
        std::string loss_type_str = (loss_function == LossType::MEAN_SQUARED_ERROR) ? "MSE" :
                                   (loss_function == LossType::BINARY_CROSSENTROPY) ? "Binary CE" :
                                   (loss_function == LossType::CATEGORICAL_CROSSENTROPY) ? "Categorical CE" : "Huber";
        
        std::cout << "✅ Neural Network (" << loss_type_str << ") trained with " 
                  << X.size() << " samples, " << input_size << " features, " 
                  << output_size << " outputs" << std::endl;
    }
}

void NeuralNetwork::fit_classification(const std::vector<std::vector<double>>& X, const std::vector<int>& y,
                                      int epochs, int batch_size, double validation_split, bool shuffle, bool verbose) {
    // Convert class labels to one-hot encoding
    int num_classes = *std::max_element(y.begin(), y.end()) + 1;
    auto y_one_hot = one_hot_encode_batch(y, num_classes);
    
    // Set appropriate loss function for classification
    if (num_classes == 2) {
        loss_function = LossType::BINARY_CROSSENTROPY;
    } else {
        loss_function = LossType::CATEGORICAL_CROSSENTROPY;
    }
    
    // Train using regression interface
    fit(X, y_one_hot, epochs, batch_size, validation_split, shuffle, verbose);
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& features) const {
    if (!fitted) {
        throw std::runtime_error("Model must be fitted before making predictions");
    }
    
    validate_input_dimensions({features});
    forward_pass(features);
    return layers.back()->last_output;
}

std::vector<std::vector<double>> NeuralNetwork::predict_batch(const std::vector<std::vector<double>>& X) const {
    std::vector<std::vector<double>> predictions;
    predictions.reserve(X.size());
    
    for (const auto& sample : X) {
        predictions.push_back(predict(sample));
    }
    
    return predictions;
}

int NeuralNetwork::predict_class(const std::vector<double>& features) const {
    auto probabilities = predict(features);
    return std::distance(probabilities.begin(), std::max_element(probabilities.begin(), probabilities.end()));
}

std::vector<int> NeuralNetwork::predict_classes(const std::vector<std::vector<double>>& X) const {
    std::vector<int> predictions;
    predictions.reserve(X.size());
    
    for (const auto& sample : X) {
        predictions.push_back(predict_class(sample));
    }
    
    return predictions;
}

std::vector<double> NeuralNetwork::predict_proba(const std::vector<double>& features) const {
    return predict(features);  // For neural networks, output is already probabilities
}

void NeuralNetwork::forward_pass(const std::vector<double>& input) const {
    std::vector<double> current_input = input;
    
    for (const auto& layer : layers) {
        current_input = layer->forward(current_input);
    }
}

void NeuralNetwork::backward_pass(const std::vector<double>& target) {
    // Get prediction from last layer
    std::vector<double> predictions = layers.back()->last_output;
    
    // Compute output gradient
    std::vector<double> gradient = compute_loss_gradient(predictions, target);
    
    // Backpropagate through layers
    for (int i = layers.size() - 1; i >= 0; --i) {
        gradient = layers[i]->backward(gradient);
    }
}

void NeuralNetwork::update_weights(int batch_size) {
    for (auto& layer : layers) {
        layer->update_weights(optimizer_type, learning_rate / batch_size, regularization_type, regularization_lambda);
    }
}

double NeuralNetwork::compute_loss(const std::vector<double>& predictions, const std::vector<double>& targets) const {
    switch (loss_function) {
        case LossType::MEAN_SQUARED_ERROR:
            return Loss::mean_squared_error(predictions, targets);
        case LossType::BINARY_CROSSENTROPY:
            return Loss::binary_crossentropy(predictions, targets);
        case LossType::CATEGORICAL_CROSSENTROPY:
            return Loss::categorical_crossentropy(predictions, targets);
        case LossType::HUBER:
            return Loss::huber_loss(predictions, targets);
        default:
            return Loss::mean_squared_error(predictions, targets);
    }
}

std::vector<double> NeuralNetwork::compute_loss_gradient(const std::vector<double>& predictions, const std::vector<double>& targets) const {
    switch (loss_function) {
        case LossType::MEAN_SQUARED_ERROR:
            return Loss::mse_gradient(predictions, targets);
        case LossType::BINARY_CROSSENTROPY:
            return Loss::binary_crossentropy_gradient(predictions, targets);
        case LossType::CATEGORICAL_CROSSENTROPY:
            return Loss::categorical_crossentropy_gradient(predictions, targets);
        case LossType::HUBER:
            return Loss::huber_gradient(predictions, targets);
        default:
            return Loss::mse_gradient(predictions, targets);
    }
}

// Additional evaluation methods
double NeuralNetwork::evaluate_loss(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y) const {
    if (!fitted) return 0.0;
    
    double total_loss = 0.0;
    for (size_t i = 0; i < X.size(); ++i) {
        auto predictions = predict(X[i]);
        total_loss += compute_loss(predictions, y[i]);
    }
    
    return total_loss / X.size();
}

double NeuralNetwork::evaluate_accuracy(const std::vector<std::vector<double>>& X, const std::vector<int>& y) const {
    if (!fitted) return 0.0;
    
    int correct = 0;
    for (size_t i = 0; i < X.size(); ++i) {
        int predicted_class = predict_class(X[i]);
        if (predicted_class == y[i]) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / X.size();
}

// Helper methods
std::vector<double> NeuralNetwork::one_hot_encode(int class_label, int num_classes) const {
    std::vector<double> encoded(num_classes, 0.0);
    if (class_label >= 0 && class_label < num_classes) {
        encoded[class_label] = 1.0;
    }
    return encoded;
}

std::vector<std::vector<double>> NeuralNetwork::one_hot_encode_batch(const std::vector<int>& labels, int num_classes) const {
    std::vector<std::vector<double>> encoded;
    encoded.reserve(labels.size());
    
    for (int label : labels) {
        encoded.push_back(one_hot_encode(label, num_classes));
    }
    
    return encoded;
}

void NeuralNetwork::shuffle_data(std::vector<std::vector<double>>& X, std::vector<std::vector<double>>& y) const {
    std::vector<int> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    std::vector<std::vector<double>> X_shuffled, y_shuffled;
    X_shuffled.reserve(X.size());
    y_shuffled.reserve(y.size());
    
    for (int idx : indices) {
        X_shuffled.push_back(X[idx]);
        y_shuffled.push_back(y[idx]);
    }
    
    X = std::move(X_shuffled);
    y = std::move(y_shuffled);
}

void NeuralNetwork::update_learning_rate(int epoch) {
    if (lr_schedule_enabled && (epoch + 1) % lr_decay_steps == 0) {
        learning_rate *= lr_decay_rate;
    }
}

bool NeuralNetwork::check_early_stopping(double validation_loss) {
    if (validation_loss < best_validation_loss - early_stopping_min_delta) {
        best_validation_loss = validation_loss;
        early_stopping_counter = 0;
        return false;
    } else {
        early_stopping_counter++;
        return early_stopping_counter >= early_stopping_patience;
    }
}

void NeuralNetwork::validate_input_dimensions(const std::vector<std::vector<double>>& X) const {
    if (!X.empty() && X[0].size() != static_cast<size_t>(input_size)) {
        throw std::invalid_argument("Input feature size mismatch");
    }
}

void NeuralNetwork::log_epoch(int epoch, double train_loss, double val_loss, double train_acc, double val_acc) const {
    if (!enable_logging) return;
    
    std::string log_path = log_manager.generate_log_path("neural_networks", data_name);
    std::ofstream log_file(log_path, std::ios::app);
    
    log_file << "Epoch " << epoch << " | Train Loss: " << std::fixed << std::setprecision(6) << train_loss
             << " | Val Loss: " << val_loss << " | LR: " << learning_rate << "\n";
}

void NeuralNetwork::set_logging(bool enable, const std::string& name) {
    enable_logging = enable;
    data_name = name;
}

void NeuralNetwork::print_model_summary() const {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "NEURAL NETWORK MODEL SUMMARY\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Input size: " << input_size << "\n";
    std::cout << "Output size: " << output_size << "\n";
    std::cout << "Number of layers: " << layers.size() << "\n";
    
    int total_params = 0;
    for (size_t i = 0; i < layers.size(); ++i) {
        int layer_params = layers[i]->get_input_size() * layers[i]->get_output_size();
        if (use_bias) layer_params += layers[i]->get_output_size();
        total_params += layer_params;
        
        std::cout << "Layer " << i + 1 << ": " << layers[i]->get_input_size() 
                  << " -> " << layers[i]->get_output_size() 
                  << " (params: " << layer_params << ")\n";
    }
    
    std::cout << "Total parameters: " << total_params << "\n";
    std::cout << "Loss function: " << static_cast<int>(loss_function) << "\n";
    std::cout << "Optimizer: " << static_cast<int>(optimizer_type) << "\n";
    std::cout << "Learning rate: " << learning_rate << "\n";
    std::cout << std::string(60, '=') << "\n";
}

// Configuration setters
void NeuralNetwork::set_optimizer(OptimizerType optimizer, double lr) {
    optimizer_type = optimizer;
    learning_rate = lr;
    initial_learning_rate = lr;
}

void NeuralNetwork::set_regularization(RegularizationType reg_type, double lambda) {
    regularization_type = reg_type;
    regularization_lambda = lambda;
}

void NeuralNetwork::set_early_stopping(bool enable, int patience, double min_delta) {
    early_stopping_enabled = enable;
    early_stopping_patience = patience;
    early_stopping_min_delta = min_delta;
}

void NeuralNetwork::set_learning_rate_schedule(bool enable, double decay_rate, int decay_steps) {
    lr_schedule_enabled = enable;
    lr_decay_rate = decay_rate;
    lr_decay_steps = decay_steps;
}

// ============================================================================
// LAYER IMPLEMENTATIONS
// ============================================================================

Layer::Layer(int input_size, int output_size, ActivationType activation, bool use_bias)
    : input_size(input_size), output_size(output_size), activation_type(activation), use_bias(use_bias) {
}

DenseLayer::DenseLayer(int input_size, int output_size, ActivationType activation, bool use_bias)
    : Layer(input_size, output_size, activation, use_bias), update_count(0) {
    initialize_weights();
}

void DenseLayer::initialize_weights() {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    // Xavier/Glorot initialization
    double limit = std::sqrt(6.0 / (input_size + output_size));
    std::uniform_real_distribution<double> dis(-limit, limit);
    
    // Initialize weights
    weights.resize(input_size, std::vector<double>(output_size));
    weight_gradients.resize(input_size * output_size);
    weight_momentum.resize(input_size, std::vector<double>(output_size, 0.0));
    weight_velocity.resize(input_size, std::vector<double>(output_size, 0.0));
    
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            weights[i][j] = dis(gen);
        }
    }
    
    // Initialize biases
    if (use_bias) {
        biases.resize(output_size, 0.0);
        bias_gradients.resize(output_size);
        bias_momentum.resize(output_size, 0.0);
        bias_velocity.resize(output_size, 0.0);
    }
}

std::vector<double> DenseLayer::forward(const std::vector<double>& input) {
    last_input = input;
    
    // Linear transformation: output = input * weights + bias
    std::vector<double> pre_activation(output_size, 0.0);
    
    for (int j = 0; j < output_size; ++j) {
        for (int i = 0; i < input_size; ++i) {
            pre_activation[j] += input[i] * weights[i][j];
        }
        if (use_bias) {
            pre_activation[j] += biases[j];
        }
    }
    
    last_pre_activation = pre_activation;
    last_output = apply_activation(pre_activation);
    
    return last_output;
}

std::vector<double> DenseLayer::backward(const std::vector<double>& gradient) {
    // Compute activation derivative
    std::vector<double> activation_grad = activation_derivative(last_pre_activation);
    
    // Element-wise multiplication with incoming gradient
    std::vector<double> delta(output_size);
    for (int i = 0; i < output_size; ++i) {
        delta[i] = gradient[i] * activation_grad[i];
    }
    
    // Compute weight gradients
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            weight_gradients[i * output_size + j] = last_input[i] * delta[j];
        }
    }
    
    // Compute bias gradients
    if (use_bias) {
        bias_gradients = delta;
    }
    
    // Compute gradient for previous layer
    std::vector<double> prev_gradient(input_size, 0.0);
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            prev_gradient[i] += weights[i][j] * delta[j];
        }
    }
    
    return prev_gradient;
}

void DenseLayer::update_weights(OptimizerType optimizer, double learning_rate, RegularizationType reg_type, double reg_lambda) {
    update_count++;
    
    // Update weights
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            double gradient = weight_gradients[i * output_size + j];
            
            // Add regularization
            if (reg_type == RegularizationType::L1) {
                gradient += reg_lambda * (weights[i][j] > 0 ? 1.0 : -1.0);
            } else if (reg_type == RegularizationType::L2) {
                gradient += reg_lambda * weights[i][j];
            }
            
            // Apply optimizer
            switch (optimizer) {
                case OptimizerType::SGD:
                    weights[i][j] -= learning_rate * gradient;
                    break;
                    
                case OptimizerType::MOMENTUM: {
                    double momentum = 0.9;
                    weight_momentum[i][j] = momentum * weight_momentum[i][j] - learning_rate * gradient;
                    weights[i][j] += weight_momentum[i][j];
                    break;
                }
                
                case OptimizerType::ADAM: {
                    double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
                    
                    weight_momentum[i][j] = beta1 * weight_momentum[i][j] + (1 - beta1) * gradient;
                    weight_velocity[i][j] = beta2 * weight_velocity[i][j] + (1 - beta2) * gradient * gradient;
                    
                    double m_hat = weight_momentum[i][j] / (1 - std::pow(beta1, update_count));
                    double v_hat = weight_velocity[i][j] / (1 - std::pow(beta2, update_count));
                    
                    weights[i][j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
                    break;
                }
                
                default:
                    weights[i][j] -= learning_rate * gradient;
                    break;
            }
        }
    }
    
    // Update biases
    if (use_bias) {
        for (int j = 0; j < output_size; ++j) {
            double gradient = bias_gradients[j];
            
            switch (optimizer) {
                case OptimizerType::SGD:
                    biases[j] -= learning_rate * gradient;
                    break;
                    
                case OptimizerType::MOMENTUM: {
                    double momentum = 0.9;
                    bias_momentum[j] = momentum * bias_momentum[j] - learning_rate * gradient;
                    biases[j] += bias_momentum[j];
                    break;
                }
                
                case OptimizerType::ADAM: {
                    double beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8;
                    
                    bias_momentum[j] = beta1 * bias_momentum[j] + (1 - beta1) * gradient;
                    bias_velocity[j] = beta2 * bias_velocity[j] + (1 - beta2) * gradient * gradient;
                    
                    double m_hat = bias_momentum[j] / (1 - std::pow(beta1, update_count));
                    double v_hat = bias_velocity[j] / (1 - std::pow(beta2, update_count));
                    
                    biases[j] -= learning_rate * m_hat / (std::sqrt(v_hat) + epsilon);
                    break;
                }
                
                default:
                    biases[j] -= learning_rate * gradient;
                    break;
            }
        }
    }
}

std::vector<double> DenseLayer::apply_activation(const std::vector<double>& input) const {
    std::vector<double> output(input.size());
    
    switch (activation_type) {
        case ActivationType::SIGMOID:
            for (size_t i = 0; i < input.size(); ++i) {
                output[i] = Activation::sigmoid(input[i]);
            }
            break;
            
        case ActivationType::TANH:
            for (size_t i = 0; i < input.size(); ++i) {
                output[i] = Activation::tanh_fn(input[i]);
            }
            break;
            
        case ActivationType::RELU:
            for (size_t i = 0; i < input.size(); ++i) {
                output[i] = Activation::relu(input[i]);
            }
            break;
            
        case ActivationType::LEAKY_RELU:
            for (size_t i = 0; i < input.size(); ++i) {
                output[i] = Activation::leaky_relu(input[i]);
            }
            break;
            
        case ActivationType::SOFTMAX:
            output = Activation::softmax(input);
            break;
            
        case ActivationType::LINEAR:
        default:
            output = input;
            break;
    }
    
    return output;
}

std::vector<double> DenseLayer::activation_derivative(const std::vector<double>& input) const {
    std::vector<double> output(input.size());
    
    switch (activation_type) {
        case ActivationType::SIGMOID:
            for (size_t i = 0; i < input.size(); ++i) {
                output[i] = Activation::sigmoid_derivative(input[i]);
            }
            break;
            
        case ActivationType::TANH:
            for (size_t i = 0; i < input.size(); ++i) {
                output[i] = Activation::tanh_derivative(input[i]);
            }
            break;
            
        case ActivationType::RELU:
            for (size_t i = 0; i < input.size(); ++i) {
                output[i] = Activation::relu_derivative(input[i]);
            }
            break;
            
        case ActivationType::LEAKY_RELU:
            for (size_t i = 0; i < input.size(); ++i) {
                output[i] = Activation::leaky_relu_derivative(input[i]);
            }
            break;
            
        case ActivationType::SOFTMAX:
            output = Activation::softmax_derivative(input);
            break;
            
        case ActivationType::LINEAR:
        default:
            std::fill(output.begin(), output.end(), 1.0);
            break;
    }
    
    return output;
}

void DenseLayer::save_layer(std::ofstream& file) const {
    // Save layer configuration
    file << input_size << " " << output_size << " " << static_cast<int>(activation_type) << " " << use_bias << "\n";
    
    // Save weights
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            file << weights[i][j] << " ";
        }
        file << "\n";
    }
    
    // Save biases
    if (use_bias) {
        for (int j = 0; j < output_size; ++j) {
            file << biases[j] << " ";
        }
        file << "\n";
    }
}

void DenseLayer::load_layer(std::ifstream& file) {
    // Load layer configuration
    int activation_int;
    file >> input_size >> output_size >> activation_int >> use_bias;
    activation_type = static_cast<ActivationType>(activation_int);
    
    // Resize containers
    weights.resize(input_size, std::vector<double>(output_size));
    if (use_bias) {
        biases.resize(output_size);
    }
    
    // Load weights
    for (int i = 0; i < input_size; ++i) {
        for (int j = 0; j < output_size; ++j) {
            file >> weights[i][j];
        }
    }
    
    // Load biases
    if (use_bias) {
        for (int j = 0; j < output_size; ++j) {
            file >> biases[j];
        }
    }
}

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

namespace Activation {
    double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-std::max(-500.0, std::min(500.0, x))));
    }
    
    double sigmoid_derivative(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }
    
    double tanh_fn(double x) {
        return std::tanh(std::max(-500.0, std::min(500.0, x)));
    }
    
    double tanh_derivative(double x) {
        double t = tanh_fn(x);
        return 1.0 - t * t;
    }
    
    double relu(double x) {
        return std::max(0.0, x);
    }
    
    double relu_derivative(double x) {
        return x > 0.0 ? 1.0 : 0.0;
    }
    
    double leaky_relu(double x, double alpha) {
        return x > 0.0 ? x : alpha * x;
    }
    
    double leaky_relu_derivative(double x, double alpha) {
        return x > 0.0 ? 1.0 : alpha;
    }
    
    std::vector<double> softmax(const std::vector<double>& x) {
        std::vector<double> result(x.size());
        double max_val = *std::max_element(x.begin(), x.end());
        
        double sum = 0.0;
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::exp(x[i] - max_val);
            sum += result[i];
        }
        
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] /= sum;
        }
        
        return result;
    }
    
    std::vector<double> softmax_derivative(const std::vector<double>& x) {
        auto s = softmax(x);
        std::vector<double> result(x.size());
        
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = s[i] * (1.0 - s[i]);
        }
        
        return result;
    }
}

// ============================================================================
// LOSS FUNCTIONS
// ============================================================================

namespace Loss {
    double mean_squared_error(const std::vector<double>& predictions, const std::vector<double>& targets) {
        double mse = 0.0;
        for (size_t i = 0; i < predictions.size(); ++i) {
            double diff = predictions[i] - targets[i];
            mse += diff * diff;
        }
        return mse / predictions.size();
    }
    
    std::vector<double> mse_gradient(const std::vector<double>& predictions, const std::vector<double>& targets) {
        std::vector<double> gradient(predictions.size());
        for (size_t i = 0; i < predictions.size(); ++i) {
            gradient[i] = 2.0 * (predictions[i] - targets[i]) / predictions.size();
        }
        return gradient;
    }
    
    double binary_crossentropy(const std::vector<double>& predictions, const std::vector<double>& targets) {
        double loss = 0.0;
        const double epsilon = 1e-15;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            double p = std::max(epsilon, std::min(1.0 - epsilon, predictions[i]));
            loss -= targets[i] * std::log(p) + (1.0 - targets[i]) * std::log(1.0 - p);
        }
        
        return loss / predictions.size();
    }
    
    std::vector<double> binary_crossentropy_gradient(const std::vector<double>& predictions, const std::vector<double>& targets) {
        std::vector<double> gradient(predictions.size());
        const double epsilon = 1e-15;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            double p = std::max(epsilon, std::min(1.0 - epsilon, predictions[i]));
            gradient[i] = (p - targets[i]) / (p * (1.0 - p)) / predictions.size();
        }
        
        return gradient;
    }
    
    double categorical_crossentropy(const std::vector<double>& predictions, const std::vector<double>& targets) {
        double loss = 0.0;
        const double epsilon = 1e-15;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            double p = std::max(epsilon, std::min(1.0 - epsilon, predictions[i]));
            loss -= targets[i] * std::log(p);
        }
        
        return loss;
    }
    
    std::vector<double> categorical_crossentropy_gradient(const std::vector<double>& predictions, const std::vector<double>& targets) {
        std::vector<double> gradient(predictions.size());
        const double epsilon = 1e-15;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            double p = std::max(epsilon, std::min(1.0 - epsilon, predictions[i]));
            gradient[i] = (p - targets[i]) / p;
        }
        
        return gradient;
    }
    
    double huber_loss(const std::vector<double>& predictions, const std::vector<double>& targets, double delta) {
        double loss = 0.0;
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            double diff = std::abs(predictions[i] - targets[i]);
            if (diff <= delta) {
                loss += 0.5 * diff * diff;
            } else {
                loss += delta * (diff - 0.5 * delta);
            }
        }
        
        return loss / predictions.size();
    }
    
    std::vector<double> huber_gradient(const std::vector<double>& predictions, const std::vector<double>& targets, double delta) {
        std::vector<double> gradient(predictions.size());
        
        for (size_t i = 0; i < predictions.size(); ++i) {
            double diff = predictions[i] - targets[i];
            if (std::abs(diff) <= delta) {
                gradient[i] = diff / predictions.size();
            } else {
                gradient[i] = delta * (diff > 0 ? 1.0 : -1.0) / predictions.size();
            }
        }
        
        return gradient;
    }
}

// ============================================================================
// MODEL PERSISTENCE
// ============================================================================

void NeuralNetwork::save_model(const std::string& filename) const {
    if (!fitted) {
        throw std::runtime_error("Cannot save unfitted model");
    }
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }
    
    // Save network configuration
    file << "# Neural Network Model\n";
    file << input_size << " " << output_size << " " << layers.size() << "\n";
    file << static_cast<int>(loss_function) << " " << static_cast<int>(optimizer_type) << "\n";
    file << learning_rate << " " << initial_learning_rate << "\n";
    file << use_bias << " " << use_batch_normalization << "\n";
    file << static_cast<int>(regularization_type) << " " << regularization_lambda << "\n";
    
    // Save each layer
    for (const auto& layer : layers) {
        layer->save_layer(file);
    }
    
    file.close();
    std::cout << "Neural Network model saved to " << filename << std::endl;
}

void NeuralNetwork::load_model(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }
    
    std::string header;
    std::getline(file, header);  // Skip header comment
    
    // Load network configuration
    int num_layers;
    file >> input_size >> output_size >> num_layers;
    
    int loss_int, optimizer_int;
    file >> loss_int >> optimizer_int;
    loss_function = static_cast<LossType>(loss_int);
    optimizer_type = static_cast<OptimizerType>(optimizer_int);
    
    file >> learning_rate >> initial_learning_rate;
    file >> use_bias >> use_batch_normalization;
    
    int reg_int;
    file >> reg_int >> regularization_lambda;
    regularization_type = static_cast<RegularizationType>(reg_int);
    
    // Clear existing layers and load new ones
    layers.clear();
    layers.reserve(num_layers);
    
    for (int i = 0; i < num_layers; ++i) {
        auto layer = std::make_unique<DenseLayer>(1, 1);  // Temporary sizing
        layer->load_layer(file);
        layers.push_back(std::move(layer));
    }
    
    fitted = true;
    file.close();
    std::cout << "Neural Network model loaded from " << filename << std::endl;
}

// ============================================================================
// ADDITIONAL EVALUATION METHODS
// ============================================================================

double NeuralNetwork::evaluate_r2_score(const std::vector<std::vector<double>>& X, const std::vector<std::vector<double>>& y) const {
    if (!fitted || X.empty() || y.empty()) return 0.0;
    
    // Calculate mean of targets
    std::vector<double> y_mean(output_size, 0.0);
    for (const auto& target : y) {
        for (int i = 0; i < output_size; ++i) {
            y_mean[i] += target[i];
        }
    }
    for (int i = 0; i < output_size; ++i) {
        y_mean[i] /= y.size();
    }
    
    // Calculate total sum of squares and residual sum of squares
    double ss_tot = 0.0, ss_res = 0.0;
    
    for (size_t i = 0; i < X.size(); ++i) {
        auto predictions = predict(X[i]);
        
        for (int j = 0; j < output_size; ++j) {
            double res = y[i][j] - predictions[j];
            double tot = y[i][j] - y_mean[j];
            
            ss_res += res * res;
            ss_tot += tot * tot;
        }
    }
    
    return 1.0 - (ss_res / ss_tot);
}

std::map<std::string, double> NeuralNetwork::classification_report(const std::vector<std::vector<double>>& X, const std::vector<int>& y) const {
    if (!fitted) return {};
    
    std::map<std::string, double> metrics;
    
    // Calculate accuracy
    double accuracy = evaluate_accuracy(X, y);
    metrics["accuracy"] = accuracy;
    
    // For detailed metrics, we'd need class-wise calculations
    // This is a simplified version
    metrics["precision"] = accuracy;  // Simplified
    metrics["recall"] = accuracy;     // Simplified
    metrics["f1_score"] = accuracy;   // Simplified
    
    return metrics;
}
