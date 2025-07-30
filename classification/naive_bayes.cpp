#include "naive_bayes.h"
#include "../utils/log_manager.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <set>

NaiveBayes::NaiveBayes(NaiveBayesType type, SmoothingType smooth, double alpha, bool feature_scaling)
    : nb_type(type), smoothing(smooth), smoothing_alpha(alpha), 
      use_feature_scaling(feature_scaling), enable_logging(false), data_name("default") {
    
    validate_parameters();
}

void NaiveBayes::validate_parameters() const {
    if (smoothing != SmoothingType::NONE && smoothing_alpha <= 0.0) {
        throw std::invalid_argument("Smoothing alpha must be positive when smoothing is enabled");
    }
}

std::vector<int> NaiveBayes::get_unique_classes(const std::vector<int>& y) const {
    std::set<int> unique_set(y.begin(), y.end());
    return std::vector<int>(unique_set.begin(), unique_set.end());
}

void NaiveBayes::scale_features(std::vector<std::vector<double>>& X) {
    if (!use_feature_scaling || X.empty()) return;
    
    int n_features = X[0].size();
    feature_means.resize(n_features, 0.0);
    feature_stds.resize(n_features, 1.0);
    
    // Compute means
    for (int j = 0; j < n_features; ++j) {
        double sum = 0.0;
        for (const auto& sample : X) {
            sum += sample[j];
        }
        feature_means[j] = sum / X.size();
    }
    
    // Compute standard deviations
    for (int j = 0; j < n_features; ++j) {
        double sum_sq_diff = 0.0;
        for (const auto& sample : X) {
            double diff = sample[j] - feature_means[j];
            sum_sq_diff += diff * diff;
        }
        feature_stds[j] = std::sqrt(sum_sq_diff / X.size());
        if (feature_stds[j] < 1e-10) feature_stds[j] = 1.0;
    }
    
    // Apply scaling
    for (auto& sample : X) {
        for (int j = 0; j < n_features; ++j) {
            sample[j] = (sample[j] - feature_means[j]) / feature_stds[j];
        }
    }
}

std::vector<double> NaiveBayes::scale_query_point(const std::vector<double>& point) const {
    if (!use_feature_scaling) return point;
    
    std::vector<double> scaled_point = point;
    for (size_t j = 0; j < point.size(); ++j) {
        scaled_point[j] = (point[j] - feature_means[j]) / feature_stds[j];
    }
    return scaled_point;
}

void NaiveBayes::compute_class_statistics() {
    unique_classes = get_unique_classes(y_train);
    class_counts.clear();
    class_priors.clear();
    
    // Count samples per class
    for (int class_label : y_train) {
        class_counts[class_label]++;
    }
    
    // Compute prior probabilities
    int total_samples = y_train.size();
    for (int class_label : unique_classes) {
        class_priors[class_label] = static_cast<double>(class_counts[class_label]) / total_samples;
    }
}

void NaiveBayes::compute_gaussian_parameters() {
    int n_features = X_train[0].size();
    
    class_means.clear();
    class_vars.clear();
    
    for (int class_label : unique_classes) {
        std::vector<double> means(n_features, 0.0);
        std::vector<double> variances(n_features, 0.0);
        
        // Collect samples for this class
        std::vector<std::vector<double>> class_samples;
        for (size_t i = 0; i < y_train.size(); ++i) {
            if (y_train[i] == class_label) {
                class_samples.push_back(X_train[i]);
            }
        }
        
        // Compute means
        for (int j = 0; j < n_features; ++j) {
            double sum = 0.0;
            for (const auto& sample : class_samples) {
                sum += sample[j];
            }
            means[j] = sum / class_samples.size();
        }
        
        // Compute variances
        for (int j = 0; j < n_features; ++j) {
            double sum_sq_diff = 0.0;
            for (const auto& sample : class_samples) {
                double diff = sample[j] - means[j];
                sum_sq_diff += diff * diff;
            }
            variances[j] = sum_sq_diff / class_samples.size();
            
            // Add small epsilon to avoid division by zero
            if (variances[j] < 1e-9) {
                variances[j] = 1e-9;
            }
        }
        
        class_means[class_label] = means;
        class_vars[class_label] = variances;
    }
}

void NaiveBayes::compute_multinomial_parameters() {
    int n_features = X_train[0].size();
    
    feature_probs.clear();
    feature_counts_total.assign(n_features, 0.0);
    
    // Compute total feature counts
    for (const auto& sample : X_train) {
        for (int j = 0; j < n_features; ++j) {
            feature_counts_total[j] += sample[j];
        }
    }
    
    for (int class_label : unique_classes) {
        std::vector<double> class_feature_counts(n_features, 0.0);
        double class_total_count = 0.0;
        
        // Count features for this class
        for (size_t i = 0; i < y_train.size(); ++i) {
            if (y_train[i] == class_label) {
                for (int j = 0; j < n_features; ++j) {
                    class_feature_counts[j] += X_train[i][j];
                    class_total_count += X_train[i][j];
                }
            }
        }
        
        // Compute probabilities with smoothing
        std::vector<double> probs(n_features);
        for (int j = 0; j < n_features; ++j) {
            if (smoothing == SmoothingType::LAPLACE) {
                probs[j] = (class_feature_counts[j] + smoothing_alpha) / 
                          (class_total_count + smoothing_alpha * n_features);
            } else if (smoothing == SmoothingType::LIDSTONE) {
                probs[j] = (class_feature_counts[j] + smoothing_alpha) / 
                          (class_total_count + smoothing_alpha * n_features);
            } else {
                probs[j] = class_feature_counts[j] / class_total_count;
            }
        }
        
        feature_probs[class_label] = probs;
    }
}

void NaiveBayes::compute_bernoulli_parameters() {
    int n_features = X_train[0].size();
    
    feature_probs.clear();
    
    for (int class_label : unique_classes) {
        std::vector<double> class_feature_counts(n_features, 0.0);
        int class_sample_count = 0;
        
        // Count binary features for this class
        for (size_t i = 0; i < y_train.size(); ++i) {
            if (y_train[i] == class_label) {
                class_sample_count++;
                for (int j = 0; j < n_features; ++j) {
                    if (X_train[i][j] > 0.5) {  // Threshold for binary
                        class_feature_counts[j] += 1.0;
                    }
                }
            }
        }
        
        // Compute probabilities with smoothing
        std::vector<double> probs(n_features);
        for (int j = 0; j < n_features; ++j) {
            if (smoothing == SmoothingType::LAPLACE) {
                probs[j] = (class_feature_counts[j] + smoothing_alpha) / 
                          (class_sample_count + 2 * smoothing_alpha);
            } else if (smoothing == SmoothingType::LIDSTONE) {
                probs[j] = (class_feature_counts[j] + smoothing_alpha) / 
                          (class_sample_count + 2 * smoothing_alpha);
            } else {
                probs[j] = class_feature_counts[j] / class_sample_count;
            }
        }
        
        feature_probs[class_label] = probs;
    }
}

double NaiveBayes::gaussian_probability(double x, double mean, double variance) const {
    double exponent = -0.5 * ((x - mean) * (x - mean)) / variance;
    double denominator = std::sqrt(2 * M_PI * variance);
    return std::exp(exponent) / denominator;
}

double NaiveBayes::log_gaussian_probability(double x, double mean, double variance) const {
    double exponent = -0.5 * ((x - mean) * (x - mean)) / variance;
    double log_denominator = 0.5 * std::log(2 * M_PI * variance);
    return exponent - log_denominator;
}

void NaiveBayes::fit(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    if (X.empty() || y.empty() || X.size() != y.size()) {
        throw std::invalid_argument("Invalid training data");
    }
    
    // Store training data
    X_train = X;
    y_train = y;
    
    // Scale features if enabled
    scale_features(X_train);
    
    // Compute class statistics
    compute_class_statistics();
    
    // Compute model parameters based on type
    switch (nb_type) {
        case NaiveBayesType::GAUSSIAN:
            compute_gaussian_parameters();
            break;
        case NaiveBayesType::MULTINOMIAL:
            compute_multinomial_parameters();
            break;
        case NaiveBayesType::BERNOULLI:
            compute_bernoulli_parameters();
            break;
    }
    
    // Log training session
    if (enable_logging) {
        std::string nb_type_str;
        switch (nb_type) {
            case NaiveBayesType::GAUSSIAN: nb_type_str = "Gaussian"; break;
            case NaiveBayesType::MULTINOMIAL: nb_type_str = "Multinomial"; break;
            case NaiveBayesType::BERNOULLI: nb_type_str = "Bernoulli"; break;
        }
        
        std::string log_path = LogManager::generate_log_path("naive_bayes", data_name, "training");
        std::ofstream log_file(log_path, std::ios::app);
        if (log_file.is_open()) {
            log_file << LogManager::create_session_header("Naive Bayes (" + nb_type_str + ")", 
                                                         "Training on " + std::to_string(X.size()) + " samples, " + 
                                                         std::to_string(X[0].size()) + " features, " + 
                                                         std::to_string(unique_classes.size()) + " classes");
            
            log_file << "Naive Bayes Type: " << nb_type_str << "\n";
            log_file << "Smoothing: ";
            switch (smoothing) {
                case SmoothingType::NONE: log_file << "None"; break;
                case SmoothingType::LAPLACE: log_file << "Laplace"; break;
                case SmoothingType::LIDSTONE: log_file << "Lidstone"; break;
            }
            log_file << " (alpha=" << smoothing_alpha << ")\n";
            log_file << "Feature Scaling: " << (use_feature_scaling ? "Enabled" : "Disabled") << "\n";
            
            log_file << "\nClass Priors:\n";
            for (int class_label : unique_classes) {
                log_file << "  Class " << class_label << ": " << class_priors[class_label] 
                        << " (" << class_counts[class_label] << " samples)\n";
            }
            
            log_file << std::string(80, '=') << "\n";
            log_file.close();
        }
    }
    
    std::string nb_type_str = (nb_type == NaiveBayesType::GAUSSIAN) ? "Gaussian" :
                              (nb_type == NaiveBayesType::MULTINOMIAL) ? "Multinomial" : "Bernoulli";
    
    std::cout << "✅ Naive Bayes (" << nb_type_str << ") fitted with " << X.size() << " samples, " 
              << X[0].size() << " features, " << unique_classes.size() << " classes" << std::endl;
}

int NaiveBayes::predict(const std::vector<double>& features) const {
    if (!is_fitted()) {
        throw std::runtime_error("Model must be fitted before prediction");
    }
    
    std::vector<double> scaled_features = scale_query_point(features);
    
    double best_log_prob = -std::numeric_limits<double>::infinity();
    int best_class = unique_classes[0];
    
    for (int class_label : unique_classes) {
        double log_prob = std::log(class_priors.at(class_label));
        
        switch (nb_type) {
            case NaiveBayesType::GAUSSIAN:
                for (size_t j = 0; j < scaled_features.size(); ++j) {
                    log_prob += log_gaussian_probability(scaled_features[j], 
                                                       class_means.at(class_label)[j],
                                                       class_vars.at(class_label)[j]);
                }
                break;
                
            case NaiveBayesType::MULTINOMIAL:
                for (size_t j = 0; j < scaled_features.size(); ++j) {
                    if (scaled_features[j] > 0) {
                        log_prob += scaled_features[j] * std::log(feature_probs.at(class_label)[j]);
                    }
                }
                break;
                
            case NaiveBayesType::BERNOULLI:
                for (size_t j = 0; j < scaled_features.size(); ++j) {
                    double feature_prob = feature_probs.at(class_label)[j];
                    if (scaled_features[j] > 0.5) {
                        log_prob += std::log(feature_prob);
                    } else {
                        log_prob += std::log(1.0 - feature_prob);
                    }
                }
                break;
        }
        
        if (log_prob > best_log_prob) {
            best_log_prob = log_prob;
            best_class = class_label;
        }
    }
    
    return best_class;
}

std::vector<int> NaiveBayes::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<int> predictions;
    predictions.reserve(X.size());
    
    for (const auto& sample : X) {
        predictions.push_back(predict(sample));
    }
    
    return predictions;
}

std::vector<double> NaiveBayes::predict_proba(const std::vector<double>& features) const {
    if (!is_fitted()) {
        throw std::runtime_error("Model must be fitted before prediction");
    }
    
    std::vector<double> scaled_features = scale_query_point(features);
    std::vector<double> log_probs(unique_classes.size());
    
    // Compute log probabilities for each class
    for (size_t i = 0; i < unique_classes.size(); ++i) {
        int class_label = unique_classes[i];
        double log_prob = std::log(class_priors.at(class_label));
        
        switch (nb_type) {
            case NaiveBayesType::GAUSSIAN:
                for (size_t j = 0; j < scaled_features.size(); ++j) {
                    log_prob += log_gaussian_probability(scaled_features[j], 
                                                       class_means.at(class_label)[j],
                                                       class_vars.at(class_label)[j]);
                }
                break;
                
            case NaiveBayesType::MULTINOMIAL:
                for (size_t j = 0; j < scaled_features.size(); ++j) {
                    if (scaled_features[j] > 0) {
                        log_prob += scaled_features[j] * std::log(feature_probs.at(class_label)[j]);
                    }
                }
                break;
                
            case NaiveBayesType::BERNOULLI:
                for (size_t j = 0; j < scaled_features.size(); ++j) {
                    double feature_prob = feature_probs.at(class_label)[j];
                    if (scaled_features[j] > 0.5) {
                        log_prob += std::log(feature_prob);
                    } else {
                        log_prob += std::log(1.0 - feature_prob);
                    }
                }
                break;
        }
        
        log_probs[i] = log_prob;
    }
    
    // Convert to probabilities using log-sum-exp trick
    double max_log_prob = *std::max_element(log_probs.begin(), log_probs.end());
    std::vector<double> probs(unique_classes.size());
    double sum_exp = 0.0;
    
    for (size_t i = 0; i < log_probs.size(); ++i) {
        probs[i] = std::exp(log_probs[i] - max_log_prob);
        sum_exp += probs[i];
    }
    
    // Normalize
    for (double& prob : probs) {
        prob /= sum_exp;
    }
    
    return probs;
}

std::vector<std::vector<double>> NaiveBayes::predict_proba(const std::vector<std::vector<double>>& X) const {
    std::vector<std::vector<double>> all_probs;
    all_probs.reserve(X.size());
    
    for (const auto& sample : X) {
        all_probs.push_back(predict_proba(sample));
    }
    
    return all_probs;
}

double NaiveBayes::evaluate_accuracy(const std::vector<std::vector<double>>& X, const std::vector<int>& y) const {
    if (X.size() != y.size()) {
        throw std::invalid_argument("X and y size mismatch");
    }
    
    std::vector<int> predictions = predict(X);
    int correct = 0;
    
    for (size_t i = 0; i < y.size(); ++i) {
        if (predictions[i] == y[i]) {
            correct++;
        }
    }
    
    return static_cast<double>(correct) / y.size();
}

double NaiveBayes::evaluate_log_likelihood(const std::vector<std::vector<double>>& X, const std::vector<int>& y) const {
    if (X.size() != y.size()) {
        throw std::invalid_argument("X and y size mismatch");
    }
    
    double total_log_likelihood = 0.0;
    
    for (size_t i = 0; i < X.size(); ++i) {
        std::vector<double> probs = predict_proba(X[i]);
        
        // Find the probability for the true class
        auto it = std::find(unique_classes.begin(), unique_classes.end(), y[i]);
        if (it != unique_classes.end()) {
            size_t class_idx = std::distance(unique_classes.begin(), it);
            total_log_likelihood += std::log(probs[class_idx] + 1e-15);  // Add small epsilon
        }
    }
    
    return total_log_likelihood / X.size();
}

std::map<std::string, double> NaiveBayes::classification_report(const std::vector<std::vector<double>>& X, 
                                                               const std::vector<int>& y) const {
    std::vector<int> predictions = predict(X);
    std::map<std::string, double> metrics;
    
    // Count true positives, false positives, false negatives for each class
    std::map<int, int> tp, fp, fn;
    
    for (int class_label : unique_classes) {
        tp[class_label] = fp[class_label] = fn[class_label] = 0;
    }
    
    for (size_t i = 0; i < y.size(); ++i) {
        int true_class = y[i];
        int pred_class = predictions[i];
        
        if (true_class == pred_class) {
            tp[true_class]++;
        } else {
            fn[true_class]++;
            fp[pred_class]++;
        }
    }
    
    // Compute overall metrics
    double total_tp = 0, total_fp = 0, total_fn = 0;
    for (int class_label : unique_classes) {
        total_tp += tp[class_label];
        total_fp += fp[class_label];
        total_fn += fn[class_label];
    }
    
    metrics["accuracy"] = total_tp / (total_tp + total_fp + total_fn);
    
    // Macro-averaged precision, recall, f1
    double macro_precision = 0, macro_recall = 0, macro_f1 = 0;
    for (int class_label : unique_classes) {
        double precision = tp[class_label] / static_cast<double>(tp[class_label] + fp[class_label] + 1e-15);
        double recall = tp[class_label] / static_cast<double>(tp[class_label] + fn[class_label] + 1e-15);
        double f1 = 2 * precision * recall / (precision + recall + 1e-15);
        
        macro_precision += precision;
        macro_recall += recall;
        macro_f1 += f1;
    }
    
    metrics["precision"] = macro_precision / unique_classes.size();
    metrics["recall"] = macro_recall / unique_classes.size();
    metrics["f1_score"] = macro_f1 / unique_classes.size();
    
    return metrics;
}

double NaiveBayes::cross_validate(const std::vector<std::vector<double>>& X, 
                                 const std::vector<int>& y, 
                                 int k_folds) const {
    if (k_folds <= 1) {
        throw std::invalid_argument("k_folds must be greater than 1");
    }
    
    std::vector<double> fold_accuracies;
    int n_samples = X.size();
    int fold_size = n_samples / k_folds;
    
    for (int fold = 0; fold < k_folds; ++fold) {
        // Create train/test split
        std::vector<std::vector<double>> X_train_fold, X_test_fold;
        std::vector<int> y_train_fold, y_test_fold;
        
        for (int i = 0; i < n_samples; ++i) {
            if (i >= fold * fold_size && i < (fold + 1) * fold_size) {
                X_test_fold.push_back(X[i]);
                y_test_fold.push_back(y[i]);
            } else {
                X_train_fold.push_back(X[i]);
                y_train_fold.push_back(y[i]);
            }
        }
        
        // Train and evaluate fold model
        NaiveBayes fold_model(nb_type, smoothing, smoothing_alpha, use_feature_scaling);
        fold_model.fit(X_train_fold, y_train_fold);
        double accuracy = fold_model.evaluate_accuracy(X_test_fold, y_test_fold);
        fold_accuracies.push_back(accuracy);
    }
    
    // Return average accuracy
    return std::accumulate(fold_accuracies.begin(), fold_accuracies.end(), 0.0) / fold_accuracies.size();
}

// Getters
const std::vector<int>& NaiveBayes::get_classes() const { return unique_classes; }
const std::map<int, double>& NaiveBayes::get_class_priors() const { return class_priors; }
const std::map<int, std::vector<double>>& NaiveBayes::get_class_means() const { return class_means; }
const std::map<int, std::vector<double>>& NaiveBayes::get_class_variances() const { return class_vars; }
const std::vector<double>& NaiveBayes::get_training_accuracy_history() const { return training_accuracy_history; }
const std::vector<double>& NaiveBayes::get_log_likelihood_history() const { return log_likelihood_history; }

// Setters
void NaiveBayes::set_naive_bayes_type(NaiveBayesType type) {
    nb_type = type;
}

void NaiveBayes::set_smoothing(SmoothingType smooth, double alpha) {
    smoothing = smooth;
    smoothing_alpha = alpha;
    validate_parameters();
}

void NaiveBayes::set_feature_scaling(bool enable) {
    use_feature_scaling = enable;
}

void NaiveBayes::set_logging(bool enable, const std::string& dataset_name) {
    enable_logging = enable;
    data_name = dataset_name;
}

// Model persistence
void NaiveBayes::save_model(const std::string& filename) const {
    if (!is_fitted()) {
        throw std::runtime_error("Model must be fitted before saving");
    }
    
    std::ofstream out(filename);
    if (!out) {
        throw std::runtime_error("Failed to open file to save model.");
    }
    
    // Save model type and parameters
    out << static_cast<int>(nb_type) << " " << static_cast<int>(smoothing) << " " << smoothing_alpha << "\n";
    out << use_feature_scaling << "\n";
    
    // Save classes
    out << unique_classes.size() << "\n";
    for (int class_label : unique_classes) {
        out << class_label << " ";
    }
    out << "\n";
    
    // Save class priors
    for (int class_label : unique_classes) {
        out << class_priors.at(class_label) << " ";
    }
    out << "\n";
    
    // Save model parameters based on type
    if (nb_type == NaiveBayesType::GAUSSIAN) {
        int n_features = class_means.begin()->second.size();
        out << n_features << "\n";
        
        for (int class_label : unique_classes) {
            for (double mean : class_means.at(class_label)) {
                out << mean << " ";
            }
            out << "\n";
            for (double var : class_vars.at(class_label)) {
                out << var << " ";
            }
            out << "\n";
        }
    } else {
        int n_features = feature_probs.begin()->second.size();
        out << n_features << "\n";
        
        for (int class_label : unique_classes) {
            for (double prob : feature_probs.at(class_label)) {
                out << prob << " ";
            }
            out << "\n";
        }
    }
    
    // Save feature scaling parameters
    if (use_feature_scaling) {
        for (double mean : feature_means) {
            out << mean << " ";
        }
        out << "\n";
        for (double std : feature_stds) {
            out << std << " ";
        }
        out << "\n";
    }
    
    out.close();
    std::cout << "Naive Bayes model saved to " << filename << std::endl;
}

void NaiveBayes::load_model(const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
        throw std::runtime_error("Failed to open file to load model.");
    }
    
    // Load model type and parameters
    int nb_type_int, smoothing_int;
    in >> nb_type_int >> smoothing_int >> smoothing_alpha;
    nb_type = static_cast<NaiveBayesType>(nb_type_int);
    smoothing = static_cast<SmoothingType>(smoothing_int);
    in >> use_feature_scaling;
    
    // Load classes
    size_t n_classes;
    in >> n_classes;
    unique_classes.resize(n_classes);
    for (size_t i = 0; i < n_classes; ++i) {
        in >> unique_classes[i];
    }
    
    // Load class priors
    class_priors.clear();
    for (int class_label : unique_classes) {
        double prior;
        in >> prior;
        class_priors[class_label] = prior;
    }
    
    // Load model parameters
    int n_features;
    in >> n_features;
    
    if (nb_type == NaiveBayesType::GAUSSIAN) {
        class_means.clear();
        class_vars.clear();
        
        for (int class_label : unique_classes) {
            std::vector<double> means(n_features), vars(n_features);
            for (int j = 0; j < n_features; ++j) {
                in >> means[j];
            }
            for (int j = 0; j < n_features; ++j) {
                in >> vars[j];
            }
            class_means[class_label] = means;
            class_vars[class_label] = vars;
        }
    } else {
        feature_probs.clear();
        
        for (int class_label : unique_classes) {
            std::vector<double> probs(n_features);
            for (int j = 0; j < n_features; ++j) {
                in >> probs[j];
            }
            feature_probs[class_label] = probs;
        }
    }
    
    // Load feature scaling parameters
    if (use_feature_scaling) {
        feature_means.resize(n_features);
        feature_stds.resize(n_features);
        for (int j = 0; j < n_features; ++j) {
            in >> feature_means[j];
        }
        for (int j = 0; j < n_features; ++j) {
            in >> feature_stds[j];
        }
    }
    
    in.close();
    std::cout << "Naive Bayes model loaded from " << filename << std::endl;
}

// Analysis methods
void NaiveBayes::print_model_summary() const {
    if (!is_fitted()) {
        std::cout << "Model not fitted yet." << std::endl;
        return;
    }
    
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "NAIVE BAYES MODEL SUMMARY\n";
    std::cout << std::string(60, '=') << "\n";
    
    std::cout << "Type: ";
    switch (nb_type) {
        case NaiveBayesType::GAUSSIAN: std::cout << "Gaussian"; break;
        case NaiveBayesType::MULTINOMIAL: std::cout << "Multinomial"; break;
        case NaiveBayesType::BERNOULLI: std::cout << "Bernoulli"; break;
    }
    std::cout << "\n";
    
    std::cout << "Smoothing: ";
    switch (smoothing) {
        case SmoothingType::NONE: std::cout << "None"; break;
        case SmoothingType::LAPLACE: std::cout << "Laplace"; break;
        case SmoothingType::LIDSTONE: std::cout << "Lidstone"; break;
    }
    std::cout << " (alpha=" << smoothing_alpha << ")\n";
    
    std::cout << "Feature Scaling: " << (use_feature_scaling ? "Enabled" : "Disabled") << "\n";
    std::cout << "Number of Classes: " << unique_classes.size() << "\n";
    std::cout << "Number of Features: " << get_num_features() << "\n";
    
    std::cout << "\nClass Priors:\n";
    for (int class_label : unique_classes) {
        std::cout << "  Class " << class_label << ": " << std::fixed << std::setprecision(4) 
                  << class_priors.at(class_label) << "\n";
    }
    
    std::cout << std::string(60, '=') << "\n";
}

std::vector<double> NaiveBayes::get_feature_importance() const {
    if (!is_fitted()) {
        throw std::runtime_error("Model must be fitted to get feature importance");
    }
    
    int n_features = get_num_features();
    std::vector<double> importance(n_features, 0.0);
    
    if (nb_type == NaiveBayesType::GAUSSIAN) {
        // For Gaussian NB, use inverse of average variance as importance
        for (int j = 0; j < n_features; ++j) {
            double avg_variance = 0.0;
            for (int class_label : unique_classes) {
                avg_variance += class_vars.at(class_label)[j];
            }
            avg_variance /= unique_classes.size();
            importance[j] = 1.0 / (avg_variance + 1e-10);
        }
    } else {
        // For Multinomial/Bernoulli, use variance of probabilities across classes
        for (int j = 0; j < n_features; ++j) {
            std::vector<double> probs_for_feature;
            for (int class_label : unique_classes) {
                probs_for_feature.push_back(feature_probs.at(class_label)[j]);
            }
            
            double mean_prob = std::accumulate(probs_for_feature.begin(), probs_for_feature.end(), 0.0) / probs_for_feature.size();
            double variance = 0.0;
            for (double prob : probs_for_feature) {
                variance += (prob - mean_prob) * (prob - mean_prob);
            }
            variance /= probs_for_feature.size();
            importance[j] = variance;
        }
    }
    
    return importance;
}

void NaiveBayes::print_feature_importance() const {
    std::vector<double> importance = get_feature_importance();
    
    std::cout << "\nFeature Importance:\n";
    for (size_t j = 0; j < importance.size(); ++j) {
        std::cout << "  Feature " << j << ": " << std::fixed << std::setprecision(6) 
                  << importance[j] << "\n";
    }
}

// Utility methods
int NaiveBayes::get_num_features() const {
    if (!is_fitted()) return 0;
    
    if (nb_type == NaiveBayesType::GAUSSIAN && !class_means.empty()) {
        return class_means.begin()->second.size();
    } else if (!feature_probs.empty()) {
        return feature_probs.begin()->second.size();
    }
    return 0;
}

int NaiveBayes::get_num_classes() const {
    return unique_classes.size();
}

bool NaiveBayes::is_fitted() const {
    return !unique_classes.empty() && !class_priors.empty();
}
