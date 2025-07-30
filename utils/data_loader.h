#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <vector>
#include <string>
#include <utility>

class DataLoader {
public:
    // Load regression data (features + continuous target)
    static std::pair<std::vector<std::vector<double>>, std::vector<double>> 
    load_regression_data(const std::string& filename);
    
    // Load classification data (features + discrete class labels)
    static std::pair<std::vector<std::vector<double>>, std::vector<int>> 
    load_classification_data(const std::string& filename);
    
    // Load polynomial data (single feature + target)
    static std::pair<std::vector<double>, std::vector<double>> 
    load_polynomial_data(const std::string& filename);
    
    // Load neural network regression data
    static std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
    load_neural_regression_data(const std::string& filename);
    
    // Helper methods
    static std::vector<std::string> split_line(const std::string& line, char delimiter = ',');
    static bool is_comment_or_empty(const std::string& line);
    static void print_data_info(const std::string& dataset_name, 
                               int samples, int features, int classes = -1);
};

#endif // DATA_LOADER_H
