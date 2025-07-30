#pragma once
#include "../regression/regression.h"
#include "../utils/log_manager.h"
#include <vector>
#include <string>
#include <map>
#include <utility>

enum class NaiveBayesType {
    GAUSSIAN,        // Gaussian Naive Bayes (continuous features)
    MULTINOMIAL,     // Multinomial Naive Bayes (discrete features)
    BERNOULLI        // Bernoulli Naive Bayes (binary features)
};

enum class SmoothingType {
    NONE,           // No smoothing
    LAPLACE,        // Laplace (add-one) smoothing
    LIDSTONE        // Lidstone smoothing with custom alpha
};

class NaiveBayes {
private:
    // Model configuration
    NaiveBayesType nb_type;
    SmoothingType smoothing;
    double smoothing_alpha;     // Smoothing parameter
    
    // Training data storage
    std::vector<std::vector<double>> X_train;
    std::vector<int> y_train;
    std::vector<int> unique_classes;
    std::map<int, int> class_counts;
    
    // Model parameters for Gaussian NB
    std::map<int, std::vector<double>> class_means;    // Mean for each feature per class
    std::map<int, std::vector<double>> class_vars;     // Variance for each feature per class
    std::map<int, double> class_priors;                // Prior probabilities
    
    // Model parameters for Multinomial/Bernoulli NB
    std::map<int, std::vector<double>> feature_probs;  // Feature probabilities per class
    std::vector<double> feature_counts_total;          // Total feature counts
    
    // Feature scaling and preprocessing
    bool use_feature_scaling;
    std::vector<double> feature_means;
    std::vector<double> feature_stds;
    
    // Training history and metrics
    std::vector<double> training_accuracy_history;
    std::vector<double> log_likelihood_history;
    
    // Logging
    bool enable_logging;
    std::string data_name;
    
    // Internal methods
    void compute_class_statistics();
    void compute_gaussian_parameters();
    void compute_multinomial_parameters();
    void compute_bernoulli_parameters();
    void scale_features(std::vector<std::vector<double>>& X);
    std::vector<double> scale_query_point(const std::vector<double>& point) const;
    double gaussian_probability(double x, double mean, double variance) const;
    double log_gaussian_probability(double x, double mean, double variance) const;
    std::vector<int> get_unique_classes(const std::vector<int>& y) const;
    void validate_parameters() const;

public:
    // Constructor
    NaiveBayes(NaiveBayesType type = NaiveBayesType::GAUSSIAN,
               SmoothingType smooth = SmoothingType::LAPLACE,
               double alpha = 1.0,
               bool feature_scaling = true);
    
    // Core training and prediction
    void fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
    int predict(const std::vector<double>& features) const;
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;
    
    // Probability predictions
    std::vector<double> predict_proba(const std::vector<double>& features) const;
    std::vector<std::vector<double>> predict_proba(const std::vector<std::vector<double>>& X) const;
    double predict_log_proba(const std::vector<double>& features, int class_label) const;
    
    // Evaluation metrics
    double evaluate_accuracy(const std::vector<std::vector<double>>& X, const std::vector<int>& y) const;
    double evaluate_log_likelihood(const std::vector<std::vector<double>>& X, const std::vector<int>& y) const;
    std::map<std::string, double> classification_report(const std::vector<std::vector<double>>& X, 
                                                        const std::vector<int>& y) const;
    
    // Cross-validation
    double cross_validate(const std::vector<std::vector<double>>& X, 
                         const std::vector<int>& y, 
                         int k_folds = 5) const;
    
    // Getters for model parameters
    const std::vector<int>& get_classes() const;
    const std::map<int, double>& get_class_priors() const;
    const std::map<int, std::vector<double>>& get_class_means() const;
    const std::map<int, std::vector<double>>& get_class_variances() const;
    const std::vector<double>& get_training_accuracy_history() const;
    const std::vector<double>& get_log_likelihood_history() const;
    
    // Setters for hyperparameters
    void set_naive_bayes_type(NaiveBayesType type);
    void set_smoothing(SmoothingType smooth, double alpha = 1.0);
    void set_feature_scaling(bool enable);
    void set_logging(bool enable, const std::string& dataset_name = "default");
    
    // Model persistence
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);
    
    // Advanced analysis
    void print_model_summary() const;
    void print_feature_importance() const;
    std::vector<double> get_feature_importance() const;
    
    // Utility methods
    int get_num_features() const;
    int get_num_classes() const;
    bool is_fitted() const;
};
