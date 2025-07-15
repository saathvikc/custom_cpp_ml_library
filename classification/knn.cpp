#include "knn.h"
#include "../utils/log_manager.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <map>
#include <set>

KNearestNeighbors::KNearestNeighbors(int neighbors, DistanceMetric metric, WeightingMethod weighting, bool feature_scaling)
    : k(neighbors), distance_metric(metric), weighting_method(weighting), 
      minkowski_p(2.0), gaussian_sigma(1.0), is_regression(true),
      use_feature_scaling(feature_scaling), use_kd_tree(false), leaf_size(30),
      enable_logging(false), data_name("default") {
    
    validate_parameters();
}

void KNearestNeighbors::validate_parameters() const {
    if (k <= 0) {
        throw std::invalid_argument("k must be positive");
    }
    if (minkowski_p <= 0) {
        throw std::invalid_argument("Minkowski p must be positive");
    }
    if (gaussian_sigma <= 0) {
        throw std::invalid_argument("Gaussian sigma must be positive");
    }
}

double KNearestNeighbors::calculate_distance(const std::vector<double>& point1, 
                                           const std::vector<double>& point2) const {
    if (point1.size() != point2.size()) {
        throw std::invalid_argument("Points must have the same dimensionality");
    }
    
    double distance = 0.0;
    
    switch (distance_metric) {
        case DistanceMetric::EUCLIDEAN:
            for (size_t i = 0; i < point1.size(); ++i) {
                double diff = point1[i] - point2[i];
                distance += diff * diff;
            }
            return std::sqrt(distance);
            
        case DistanceMetric::MANHATTAN:
            for (size_t i = 0; i < point1.size(); ++i) {
                distance += std::abs(point1[i] - point2[i]);
            }
            return distance;
            
        case DistanceMetric::MINKOWSKI:
            for (size_t i = 0; i < point1.size(); ++i) {
                distance += std::pow(std::abs(point1[i] - point2[i]), minkowski_p);
            }
            return std::pow(distance, 1.0 / minkowski_p);
            
        case DistanceMetric::COSINE:
            double dot_product = 0.0;
            double norm1 = 0.0;
            double norm2 = 0.0;
            
            for (size_t i = 0; i < point1.size(); ++i) {
                dot_product += point1[i] * point2[i];
                norm1 += point1[i] * point1[i];
                norm2 += point2[i] * point2[i];
            }
            
            if (norm1 == 0.0 || norm2 == 0.0) return 1.0; // Maximum distance
            
            double cosine_sim = dot_product / (std::sqrt(norm1) * std::sqrt(norm2));
            return 1.0 - cosine_sim; // Convert similarity to distance
    }
    
    return 0.0;
}

void KNearestNeighbors::scale_features(std::vector<std::vector<double>>& X) {
    if (!use_feature_scaling || X.empty()) return;
    
    size_t n_features = X[0].size();
    feature_means.assign(n_features, 0.0);
    feature_stds.assign(n_features, 0.0);
    
    // Compute means
    for (const auto& sample : X) {
        for (size_t j = 0; j < n_features; ++j) {
            feature_means[j] += sample[j];
        }
    }
    
    for (double& mean : feature_means) {
        mean /= X.size();
    }
    
    // Compute standard deviations
    for (const auto& sample : X) {
        for (size_t j = 0; j < n_features; ++j) {
            double diff = sample[j] - feature_means[j];
            feature_stds[j] += diff * diff;
        }
    }
    
    for (double& std : feature_stds) {
        std = std::sqrt(std / X.size());
        if (std < 1e-10) std = 1.0; // Prevent division by zero
    }
    
    // Scale features
    for (auto& sample : X) {
        for (size_t j = 0; j < sample.size(); ++j) {
            sample[j] = (sample[j] - feature_means[j]) / feature_stds[j];
        }
    }
}

std::vector<double> KNearestNeighbors::scale_query_point(const std::vector<double>& point) const {
    if (!use_feature_scaling) return point;
    
    std::vector<double> scaled_point = point;
    for (size_t i = 0; i < scaled_point.size(); ++i) {
        scaled_point[i] = (scaled_point[i] - feature_means[i]) / feature_stds[i];
    }
    
    return scaled_point;
}

double KNearestNeighbors::gaussian_weight(double distance) const {
    return std::exp(-(distance * distance) / (2.0 * gaussian_sigma * gaussian_sigma));
}

std::vector<std::pair<double, int>> KNearestNeighbors::find_k_nearest(const std::vector<double>& query_point) const {
    std::vector<std::pair<double, int>> distances;
    distances.reserve(X_train.size());
    
    std::vector<double> scaled_query = scale_query_point(query_point);
    
    // Calculate distances to all training points
    for (size_t i = 0; i < X_train.size(); ++i) {
        double dist = calculate_distance(scaled_query, X_train[i]);
        distances.emplace_back(dist, i);
    }
    
    // Sort by distance and take k nearest
    std::partial_sort(distances.begin(), distances.begin() + std::min(k, (int)distances.size()), 
                     distances.end());
    
    distances.resize(std::min(k, (int)distances.size()));
    return distances;
}

void KNearestNeighbors::fit_regression(const std::vector<std::vector<double>>& X, 
                                     const std::vector<double>& y) {
    if (X.size() != y.size()) {
        throw std::invalid_argument("X and y size mismatch");
    }
    
    if (X.empty()) {
        throw std::invalid_argument("Training data cannot be empty");
    }
    
    is_regression = true;
    X_train = X;
    y_train_reg = y;
    y_train_class.clear();
    
    // Scale features
    scale_features(X_train);
    
    if (enable_logging) {
        std::string log_path = LogManager::generate_log_path("knn_regression", data_name, "training");
        std::ofstream log_file(log_path, std::ios::app);
        if (log_file.is_open()) {
            log_file << LogManager::create_session_header("KNN Regression", 
                                                         "Training on " + std::to_string(X.size()) + " samples, " + 
                                                         std::to_string(X[0].size()) + " features, k=" + std::to_string(k));
            log_file << "Distance Metric: ";
            switch (distance_metric) {
                case DistanceMetric::EUCLIDEAN: log_file << "Euclidean"; break;
                case DistanceMetric::MANHATTAN: log_file << "Manhattan"; break;
                case DistanceMetric::MINKOWSKI: log_file << "Minkowski (p=" << minkowski_p << ")"; break;
                case DistanceMetric::COSINE: log_file << "Cosine"; break;
            }
            log_file << ", Weighting: ";
            switch (weighting_method) {
                case WeightingMethod::UNIFORM: log_file << "Uniform"; break;
                case WeightingMethod::DISTANCE_WEIGHTED: log_file << "Distance-weighted"; break;
                case WeightingMethod::GAUSSIAN_WEIGHTED: log_file << "Gaussian (σ=" << gaussian_sigma << ")"; break;
            }
            log_file << "\nFeature Scaling: " << (use_feature_scaling ? "Enabled" : "Disabled") << "\n";
            log_file << std::string(80, '=') << "\n";
            log_file.close();
        }
    }
    
    std::cout << "✅ KNN Regression fitted with " << X.size() << " samples, " 
              << X[0].size() << " features (k=" << k << ")" << std::endl;
}

void KNearestNeighbors::fit_classification(const std::vector<std::vector<double>>& X, 
                                         const std::vector<int>& y) {
    if (X.size() != y.size()) {
        throw std::invalid_argument("X and y size mismatch");
    }
    
    if (X.empty()) {
        throw std::invalid_argument("Training data cannot be empty");
    }
    
    is_regression = false;
    X_train = X;
    y_train_class = y;
    y_train_reg.clear();
    
    // Scale features
    scale_features(X_train);
    
    if (enable_logging) {
        std::string log_path = LogManager::generate_log_path("knn_classification", data_name, "training");
        std::ofstream log_file(log_path, std::ios::app);
        if (log_file.is_open()) {
            // Count unique classes
            std::set<int> unique_classes(y.begin(), y.end());
            log_file << LogManager::create_session_header("KNN Classification", 
                                                         "Training on " + std::to_string(X.size()) + " samples, " + 
                                                         std::to_string(X[0].size()) + " features, k=" + std::to_string(k) +
                                                         ", " + std::to_string(unique_classes.size()) + " classes");
            
            log_file << "Distance Metric: ";
            switch (distance_metric) {
                case DistanceMetric::EUCLIDEAN: log_file << "Euclidean"; break;
                case DistanceMetric::MANHATTAN: log_file << "Manhattan"; break;
                case DistanceMetric::MINKOWSKI: log_file << "Minkowski (p=" << minkowski_p << ")"; break;
                case DistanceMetric::COSINE: log_file << "Cosine"; break;
            }
            log_file << ", Weighting: ";
            switch (weighting_method) {
                case WeightingMethod::UNIFORM: log_file << "Uniform"; break;
                case WeightingMethod::DISTANCE_WEIGHTED: log_file << "Distance-weighted"; break;
                case WeightingMethod::GAUSSIAN_WEIGHTED: log_file << "Gaussian (σ=" << gaussian_sigma << ")"; break;
            }
            log_file << "\nFeature Scaling: " << (use_feature_scaling ? "Enabled" : "Disabled") << "\n";
            log_file << std::string(80, '=') << "\n";
            log_file.close();
        }
    }
    
    std::cout << "✅ KNN Classification fitted with " << X.size() << " samples, " 
              << X[0].size() << " features (k=" << k << ")" << std::endl;
}

double KNearestNeighbors::predict_regression(const std::vector<double>& query_point) const {
    if (!is_regression) {
        throw std::logic_error("Model was fitted for classification, not regression");
    }
    
    auto neighbors = find_k_nearest(query_point);
    
    if (neighbors.empty()) {
        throw std::runtime_error("No neighbors found");
    }
    
    double prediction = 0.0;
    double total_weight = 0.0;
    
    for (const auto& neighbor : neighbors) {
        double distance = neighbor.first;
        int index = neighbor.second;
        double weight = 1.0;
        
        switch (weighting_method) {
            case WeightingMethod::UNIFORM:
                weight = 1.0;
                break;
            case WeightingMethod::DISTANCE_WEIGHTED:
                weight = (distance < 1e-10) ? 1e10 : 1.0 / distance;
                break;
            case WeightingMethod::GAUSSIAN_WEIGHTED:
                weight = gaussian_weight(distance);
                break;
        }
        
        prediction += weight * y_train_reg[index];
        total_weight += weight;
    }
    
    return (total_weight > 0) ? prediction / total_weight : 0.0;
}

std::vector<double> KNearestNeighbors::predict_regression(const std::vector<std::vector<double>>& query_points) const {
    std::vector<double> predictions;
    predictions.reserve(query_points.size());
    
    for (const auto& point : query_points) {
        predictions.push_back(predict_regression(point));
    }
    
    return predictions;
}

int KNearestNeighbors::predict_classification(const std::vector<double>& query_point) const {
    if (is_regression) {
        throw std::logic_error("Model was fitted for regression, not classification");
    }
    
    auto neighbors = find_k_nearest(query_point);
    
    if (neighbors.empty()) {
        throw std::runtime_error("No neighbors found");
    }
    
    std::map<int, double> class_weights;
    
    for (const auto& neighbor : neighbors) {
        double distance = neighbor.first;
        int index = neighbor.second;
        int class_label = y_train_class[index];
        double weight = 1.0;
        
        switch (weighting_method) {
            case WeightingMethod::UNIFORM:
                weight = 1.0;
                break;
            case WeightingMethod::DISTANCE_WEIGHTED:
                weight = (distance < 1e-10) ? 1e10 : 1.0 / distance;
                break;
            case WeightingMethod::GAUSSIAN_WEIGHTED:
                weight = gaussian_weight(distance);
                break;
        }
        
        class_weights[class_label] += weight;
    }
    
    // Find class with maximum weight
    int predicted_class = class_weights.begin()->first;
    double max_weight = class_weights.begin()->second;
    
    for (const auto& pair : class_weights) {
        if (pair.second > max_weight) {
            max_weight = pair.second;
            predicted_class = pair.first;
        }
    }
    
    return predicted_class;
}

std::vector<int> KNearestNeighbors::predict_classification(const std::vector<std::vector<double>>& query_points) const {
    std::vector<int> predictions;
    predictions.reserve(query_points.size());
    
    for (const auto& point : query_points) {
        predictions.push_back(predict_classification(point));
    }
    
    return predictions;
}

std::vector<double> KNearestNeighbors::predict_proba(const std::vector<double>& query_point) const {
    if (is_regression) {
        throw std::logic_error("Probability prediction not available for regression");
    }
    
    auto neighbors = find_k_nearest(query_point);
    
    if (neighbors.empty()) {
        throw std::runtime_error("No neighbors found");
    }
    
    // Get all unique classes
    std::set<int> unique_classes(y_train_class.begin(), y_train_class.end());
    std::map<int, double> class_weights;
    
    // Initialize all classes with zero weight
    for (int class_label : unique_classes) {
        class_weights[class_label] = 0.0;
    }
    
    double total_weight = 0.0;
    
    for (const auto& neighbor : neighbors) {
        double distance = neighbor.first;
        int index = neighbor.second;
        int class_label = y_train_class[index];
        double weight = 1.0;
        
        switch (weighting_method) {
            case WeightingMethod::UNIFORM:
                weight = 1.0;
                break;
            case WeightingMethod::DISTANCE_WEIGHTED:
                weight = (distance < 1e-10) ? 1e10 : 1.0 / distance;
                break;
            case WeightingMethod::GAUSSIAN_WEIGHTED:
                weight = gaussian_weight(distance);
                break;
        }
        
        class_weights[class_label] += weight;
        total_weight += weight;
    }
    
    // Convert to probabilities
    std::vector<double> probabilities;
    for (const auto& pair : class_weights) {
        probabilities.push_back(pair.second / total_weight);
    }
    
    return probabilities;
}

// Evaluation methods
double KNearestNeighbors::evaluate_mse(const std::vector<std::vector<double>>& X_test, 
                                      const std::vector<double>& y_test) const {
    auto predictions = predict_regression(X_test);
    double mse = 0.0;
    
    for (size_t i = 0; i < y_test.size(); ++i) {
        double error = predictions[i] - y_test[i];
        mse += error * error;
    }
    
    return mse / y_test.size();
}

double KNearestNeighbors::evaluate_mae(const std::vector<std::vector<double>>& X_test, 
                                      const std::vector<double>& y_test) const {
    auto predictions = predict_regression(X_test);
    double mae = 0.0;
    
    for (size_t i = 0; i < y_test.size(); ++i) {
        mae += std::abs(predictions[i] - y_test[i]);
    }
    
    return mae / y_test.size();
}

double KNearestNeighbors::evaluate_r2(const std::vector<std::vector<double>>& X_test, 
                                     const std::vector<double>& y_test) const {
    auto predictions = predict_regression(X_test);
    double y_mean = std::accumulate(y_test.begin(), y_test.end(), 0.0) / y_test.size();
    
    double ss_res = 0.0;
    double ss_tot = 0.0;
    
    for (size_t i = 0; i < y_test.size(); ++i) {
        double residual = y_test[i] - predictions[i];
        ss_res += residual * residual;
        
        double deviation = y_test[i] - y_mean;
        ss_tot += deviation * deviation;
    }
    
    return 1.0 - (ss_res / ss_tot);
}

double KNearestNeighbors::evaluate_accuracy(const std::vector<std::vector<double>>& X_test, 
                                          const std::vector<int>& y_test) const {
    auto predictions = predict_classification(X_test);
    int correct = 0;
    
    for (size_t i = 0; i < y_test.size(); ++i) {
        if (predictions[i] == y_test[i]) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / y_test.size();
}

double KNearestNeighbors::evaluate_precision(const std::vector<std::vector<double>>& X_test, 
                                           const std::vector<int>& y_test, int positive_class) const {
    auto predictions = predict_classification(X_test);
    int true_positives = 0;
    int false_positives = 0;
    
    for (size_t i = 0; i < y_test.size(); ++i) {
        if (predictions[i] == positive_class) {
            if (y_test[i] == positive_class) {
                true_positives++;
            } else {
                false_positives++;
            }
        }
    }
    
    return (true_positives + false_positives > 0) ? 
           static_cast<double>(true_positives) / (true_positives + false_positives) : 0.0;
}

double KNearestNeighbors::evaluate_recall(const std::vector<std::vector<double>>& X_test, 
                                        const std::vector<int>& y_test, int positive_class) const {
    auto predictions = predict_classification(X_test);
    int true_positives = 0;
    int false_negatives = 0;
    
    for (size_t i = 0; i < y_test.size(); ++i) {
        if (y_test[i] == positive_class) {
            if (predictions[i] == positive_class) {
                true_positives++;
            } else {
                false_negatives++;
            }
        }
    }
    
    return (true_positives + false_negatives > 0) ? 
           static_cast<double>(true_positives) / (true_positives + false_negatives) : 0.0;
}

double KNearestNeighbors::evaluate_f1_score(const std::vector<std::vector<double>>& X_test, 
                                          const std::vector<int>& y_test, int positive_class) const {
    double precision = evaluate_precision(X_test, y_test, positive_class);
    double recall = evaluate_recall(X_test, y_test, positive_class);
    
    return (precision + recall > 0) ? 2.0 * precision * recall / (precision + recall) : 0.0;
}

// Getters and setters
int KNearestNeighbors::get_k() const { return k; }
DistanceMetric KNearestNeighbors::get_distance_metric() const { return distance_metric; }
WeightingMethod KNearestNeighbors::get_weighting_method() const { return weighting_method; }
bool KNearestNeighbors::is_fitted() const { return !X_train.empty(); }
int KNearestNeighbors::get_training_size() const { return X_train.size(); }
int KNearestNeighbors::get_feature_count() const { return X_train.empty() ? 0 : X_train[0].size(); }

void KNearestNeighbors::set_k(int neighbors) {
    if (neighbors <= 0) throw std::invalid_argument("k must be positive");
    k = neighbors;
}

void KNearestNeighbors::set_distance_metric(DistanceMetric metric, double p) {
    distance_metric = metric;
    if (metric == DistanceMetric::MINKOWSKI) {
        if (p <= 0) throw std::invalid_argument("Minkowski p must be positive");
        minkowski_p = p;
    }
}

void KNearestNeighbors::set_weighting_method(WeightingMethod weighting, double sigma) {
    weighting_method = weighting;
    if (weighting == WeightingMethod::GAUSSIAN_WEIGHTED) {
        if (sigma <= 0) throw std::invalid_argument("Gaussian sigma must be positive");
        gaussian_sigma = sigma;
    }
}

void KNearestNeighbors::set_feature_scaling(bool enable) {
    use_feature_scaling = enable;
}

void KNearestNeighbors::set_logging(bool enable, const std::string& dataset_name) {
    enable_logging = enable;
    data_name = dataset_name;
}

void KNearestNeighbors::set_optimization_params(bool use_tree, int leaf_sz) {
    use_kd_tree = use_tree;
    leaf_size = leaf_sz;
}

void KNearestNeighbors::save_model(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Failed to open file to save model.");
    
    // Save basic parameters
    out << k << " " << static_cast<int>(distance_metric) << " " << static_cast<int>(weighting_method) << "\n";
    out << minkowski_p << " " << gaussian_sigma << " " << is_regression << " " << use_feature_scaling << "\n";
    
    // Save feature scaling parameters
    if (use_feature_scaling) {
        out << feature_means.size() << "\n";
        for (double mean : feature_means) out << mean << " ";
        out << "\n";
        for (double std : feature_stds) out << std << " ";
        out << "\n";
    }
    
    // Save training data
    out << X_train.size() << " " << (X_train.empty() ? 0 : X_train[0].size()) << "\n";
    for (const auto& sample : X_train) {
        for (double feature : sample) {
            out << feature << " ";
        }
        out << "\n";
    }
    
    // Save targets
    if (is_regression) {
        for (double target : y_train_reg) out << target << " ";
    } else {
        for (int target : y_train_class) out << target << " ";
    }
    out << "\n";
    
    out.close();
    std::cout << "KNN model saved to " << filename << std::endl;
}

void KNearestNeighbors::load_model(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Failed to open file to load model.");
    
    // Load basic parameters
    int dist_metric, weight_method;
    in >> k >> dist_metric >> weight_method;
    distance_metric = static_cast<DistanceMetric>(dist_metric);
    weighting_method = static_cast<WeightingMethod>(weight_method);
    
    in >> minkowski_p >> gaussian_sigma >> is_regression >> use_feature_scaling;
    
    // Load feature scaling parameters
    if (use_feature_scaling) {
        size_t n_features;
        in >> n_features;
        
        feature_means.resize(n_features);
        for (size_t i = 0; i < n_features; ++i) in >> feature_means[i];
        
        feature_stds.resize(n_features);
        for (size_t i = 0; i < n_features; ++i) in >> feature_stds[i];
    }
    
    // Load training data
    size_t n_samples, n_features;
    in >> n_samples >> n_features;
    
    X_train.resize(n_samples);
    for (size_t i = 0; i < n_samples; ++i) {
        X_train[i].resize(n_features);
        for (size_t j = 0; j < n_features; ++j) {
            in >> X_train[i][j];
        }
    }
    
    // Load targets
    if (is_regression) {
        y_train_reg.resize(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            in >> y_train_reg[i];
        }
        y_train_class.clear();
    } else {
        y_train_class.resize(n_samples);
        for (size_t i = 0; i < n_samples; ++i) {
            in >> y_train_class[i];
        }
        y_train_reg.clear();
    }
    
    in.close();
    std::cout << "KNN model loaded from " << filename << std::endl;
}

void KNearestNeighbors::print_model_summary() const {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "KNN MODEL SUMMARY\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Mode: " << (is_regression ? "Regression" : "Classification") << "\n";
    std::cout << "k: " << k << "\n";
    std::cout << "Distance Metric: ";
    switch (distance_metric) {
        case DistanceMetric::EUCLIDEAN: std::cout << "Euclidean"; break;
        case DistanceMetric::MANHATTAN: std::cout << "Manhattan"; break;
        case DistanceMetric::MINKOWSKI: std::cout << "Minkowski (p=" << minkowski_p << ")"; break;
        case DistanceMetric::COSINE: std::cout << "Cosine"; break;
    }
    std::cout << "\nWeighting Method: ";
    switch (weighting_method) {
        case WeightingMethod::UNIFORM: std::cout << "Uniform"; break;
        case WeightingMethod::DISTANCE_WEIGHTED: std::cout << "Distance-weighted"; break;
        case WeightingMethod::GAUSSIAN_WEIGHTED: std::cout << "Gaussian (σ=" << gaussian_sigma << ")"; break;
    }
    std::cout << "\nFeature Scaling: " << (use_feature_scaling ? "Enabled" : "Disabled") << "\n";
    std::cout << "Training Samples: " << get_training_size() << "\n";
    std::cout << "Features: " << get_feature_count() << "\n";
    std::cout << std::string(60, '=') << "\n";
}

// Cross-validation implementations (simplified)
double KNearestNeighbors::cross_validate_regression(const std::vector<std::vector<double>>& X, 
                                                   const std::vector<double>& y, 
                                                   int folds) const {
    // Simple k-fold cross-validation implementation
    std::vector<double> scores;
    int fold_size = X.size() / folds;
    
    for (int fold = 0; fold < folds; ++fold) {
        // Create train/test split
        std::vector<std::vector<double>> X_train_cv, X_test_cv;
        std::vector<double> y_train_cv, y_test_cv;
        
        for (size_t i = 0; i < X.size(); ++i) {
            if (i >= fold * fold_size && i < (fold + 1) * fold_size) {
                X_test_cv.push_back(X[i]);
                y_test_cv.push_back(y[i]);
            } else {
                X_train_cv.push_back(X[i]);
                y_train_cv.push_back(y[i]);
            }
        }
        
        // Train and evaluate
        KNearestNeighbors temp_model(*this);
        temp_model.fit_regression(X_train_cv, y_train_cv);
        double score = temp_model.evaluate_r2(X_test_cv, y_test_cv);
        scores.push_back(score);
    }
    
    // Return mean score
    return std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
}

double KNearestNeighbors::cross_validate_classification(const std::vector<std::vector<double>>& X, 
                                                       const std::vector<int>& y, 
                                                       int folds) const {
    // Simple k-fold cross-validation implementation
    std::vector<double> scores;
    int fold_size = X.size() / folds;
    
    for (int fold = 0; fold < folds; ++fold) {
        // Create train/test split
        std::vector<std::vector<double>> X_train_cv, X_test_cv;
        std::vector<int> y_train_cv, y_test_cv;
        
        for (size_t i = 0; i < X.size(); ++i) {
            if (i >= fold * fold_size && i < (fold + 1) * fold_size) {
                X_test_cv.push_back(X[i]);
                y_test_cv.push_back(y[i]);
            } else {
                X_train_cv.push_back(X[i]);
                y_train_cv.push_back(y[i]);
            }
        }
        
        // Train and evaluate
        KNearestNeighbors temp_model(*this);
        temp_model.fit_classification(X_train_cv, y_train_cv);
        double score = temp_model.evaluate_accuracy(X_test_cv, y_test_cv);
        scores.push_back(score);
    }
    
    // Return mean score
    return std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
}

// Placeholder implementations for advanced methods
std::vector<std::vector<int>> KNearestNeighbors::get_neighbor_indices(const std::vector<std::vector<double>>& query_points) const {
    std::vector<std::vector<int>> neighbor_indices;
    
    for (const auto& point : query_points) {
        auto neighbors = find_k_nearest(point);
        std::vector<int> indices;
        for (const auto& neighbor : neighbors) {
            indices.push_back(neighbor.second);
        }
        neighbor_indices.push_back(indices);
    }
    
    return neighbor_indices;
}

std::vector<std::vector<double>> KNearestNeighbors::get_neighbor_distances(const std::vector<std::vector<double>>& query_points) const {
    std::vector<std::vector<double>> neighbor_distances;
    
    for (const auto& point : query_points) {
        auto neighbors = find_k_nearest(point);
        std::vector<double> distances;
        for (const auto& neighbor : neighbors) {
            distances.push_back(neighbor.first);
        }
        neighbor_distances.push_back(distances);
    }
    
    return neighbor_distances;
}

void KNearestNeighbors::analyze_decision_boundary(const std::vector<double>& x_range, 
                                                 const std::vector<double>& y_range, 
                                                 const std::string& output_file) const {
    // Placeholder for decision boundary analysis
    std::ofstream out(output_file);
    if (out.is_open()) {
        out << "Decision boundary analysis for KNN (k=" << k << ")\n";
        out << "X range: [" << x_range[0] << ", " << x_range[1] << "]\n";
        out << "Y range: [" << y_range[0] << ", " << y_range[1] << "]\n";
        out.close();
    }
}
