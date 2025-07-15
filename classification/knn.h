#pragma once
#include "../regression/regression.h"
#include "../utils/log_manager.h"
#include <vector>
#include <string>
#include <utility>

enum class DistanceMetric {
    EUCLIDEAN,
    MANHATTAN,
    MINKOWSKI,
    COSINE
};

enum class WeightingMethod {
    UNIFORM,
    DISTANCE_WEIGHTED,
    GAUSSIAN_WEIGHTED
};

class KNearestNeighbors {
private:
    // Model parameters
    int k;                              // Number of neighbors
    DistanceMetric distance_metric;     // Distance calculation method
    WeightingMethod weighting_method;   // Neighbor weighting method
    double minkowski_p;                 // Parameter for Minkowski distance
    double gaussian_sigma;              // Parameter for Gaussian weighting
    
    // Training data storage
    std::vector<std::vector<double>> X_train;  // Feature vectors
    std::vector<double> y_train_reg;          // Target values for regression
    std::vector<int> y_train_class;           // Target labels for classification
    bool is_regression;                       // Mode: regression or classification
    
    // Feature scaling
    bool use_feature_scaling;
    std::vector<double> feature_means;
    std::vector<double> feature_stds;
    
    // Performance optimization
    bool use_kd_tree;                   // Use KD-tree for fast neighbor search
    int leaf_size;                      // KD-tree leaf size
    
    // Logging
    bool enable_logging;
    std::string data_name;  // Name of dataset for logging organization
    
    // Internal methods
    double calculate_distance(const std::vector<double>& point1, 
                            const std::vector<double>& point2) const;
    std::vector<std::pair<double, int>> find_k_nearest(const std::vector<double>& query_point) const;
    void scale_features(std::vector<std::vector<double>>& X);
    std::vector<double> scale_query_point(const std::vector<double>& point) const;
    double gaussian_weight(double distance) const;
    void validate_parameters() const;

public:
    // Constructors
    KNearestNeighbors(int neighbors = 5,
                     DistanceMetric metric = DistanceMetric::EUCLIDEAN,
                     WeightingMethod weighting = WeightingMethod::UNIFORM,
                     bool feature_scaling = true);
    
    // Training methods
    void fit_regression(const std::vector<std::vector<double>>& X, 
                       const std::vector<double>& y);
    void fit_classification(const std::vector<std::vector<double>>& X, 
                           const std::vector<int>& y);
    
    // Prediction methods - Regression
    double predict_regression(const std::vector<double>& query_point) const;
    std::vector<double> predict_regression(const std::vector<std::vector<double>>& query_points) const;
    
    // Prediction methods - Classification
    int predict_classification(const std::vector<double>& query_point) const;
    std::vector<int> predict_classification(const std::vector<std::vector<double>>& query_points) const;
    std::vector<double> predict_proba(const std::vector<double>& query_point) const;
    
    // Evaluation methods - Regression
    double evaluate_mse(const std::vector<std::vector<double>>& X_test, 
                       const std::vector<double>& y_test) const;
    double evaluate_mae(const std::vector<std::vector<double>>& X_test, 
                       const std::vector<double>& y_test) const;
    double evaluate_r2(const std::vector<std::vector<double>>& X_test, 
                      const std::vector<double>& y_test) const;
    
    // Evaluation methods - Classification
    double evaluate_accuracy(const std::vector<std::vector<double>>& X_test, 
                           const std::vector<int>& y_test) const;
    double evaluate_precision(const std::vector<std::vector<double>>& X_test, 
                            const std::vector<int>& y_test, int positive_class = 1) const;
    double evaluate_recall(const std::vector<std::vector<double>>& X_test, 
                         const std::vector<int>& y_test, int positive_class = 1) const;
    double evaluate_f1_score(const std::vector<std::vector<double>>& X_test, 
                           const std::vector<int>& y_test, int positive_class = 1) const;
    
    // Getters
    int get_k() const;
    DistanceMetric get_distance_metric() const;
    WeightingMethod get_weighting_method() const;
    bool is_fitted() const;
    int get_training_size() const;
    int get_feature_count() const;
    
    // Setters
    void set_k(int neighbors);
    void set_distance_metric(DistanceMetric metric, double p = 2.0);
    void set_weighting_method(WeightingMethod weighting, double sigma = 1.0);
    void set_feature_scaling(bool enable);
    void set_logging(bool enable, const std::string& dataset_name = "default");
    void set_optimization_params(bool use_tree = false, int leaf_sz = 30);
    
    // Model persistence
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);
    
    // Advanced analysis
    void print_model_summary() const;
    std::vector<std::vector<int>> get_neighbor_indices(const std::vector<std::vector<double>>& query_points) const;
    std::vector<std::vector<double>> get_neighbor_distances(const std::vector<std::vector<double>>& query_points) const;
    void analyze_decision_boundary(const std::vector<double>& x_range, 
                                 const std::vector<double>& y_range, 
                                 const std::string& output_file) const;
    
    // Cross-validation support
    double cross_validate_regression(const std::vector<std::vector<double>>& X, 
                                   const std::vector<double>& y, 
                                   int folds = 5) const;
    double cross_validate_classification(const std::vector<std::vector<double>>& X, 
                                       const std::vector<int>& y, 
                                       int folds = 5) const;
};
