#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <set>

std::pair<std::vector<std::vector<double>>, std::vector<double>> 
DataLoader::load_regression_data(const std::string& filename) {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (is_comment_or_empty(line)) continue;
        
        auto values = split_line(line);
        if (values.size() < 2) continue;
        
        std::vector<double> features;
        for (size_t i = 0; i < values.size() - 1; ++i) {
            features.push_back(std::stod(values[i]));
        }
        double target = std::stod(values.back());
        
        X.push_back(features);
        y.push_back(target);
    }
    
    file.close();
    
    if (!X.empty()) {
        print_data_info("Regression", X.size(), X[0].size());
    }
    
    return {X, y};
}

std::pair<std::vector<std::vector<double>>, std::vector<int>> 
DataLoader::load_classification_data(const std::string& filename) {
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::string line;
    std::set<int> unique_classes;
    
    while (std::getline(file, line)) {
        if (is_comment_or_empty(line)) continue;
        
        auto values = split_line(line);
        if (values.size() < 2) continue;
        
        std::vector<double> features;
        for (size_t i = 0; i < values.size() - 1; ++i) {
            features.push_back(std::stod(values[i]));
        }
        int class_label = std::stoi(values.back());
        
        X.push_back(features);
        y.push_back(class_label);
        unique_classes.insert(class_label);
    }
    
    file.close();
    
    if (!X.empty()) {
        print_data_info("Classification", X.size(), X[0].size(), unique_classes.size());
    }
    
    return {X, y};
}

std::pair<std::vector<double>, std::vector<double>> 
DataLoader::load_polynomial_data(const std::string& filename) {
    std::vector<double> X, y;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (is_comment_or_empty(line)) continue;
        
        auto values = split_line(line);
        if (values.size() != 2) continue;
        
        X.push_back(std::stod(values[0]));
        y.push_back(std::stod(values[1]));
    }
    
    file.close();
    
    print_data_info("Polynomial", X.size(), 1);
    return {X, y};
}

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> 
DataLoader::load_neural_regression_data(const std::string& filename) {
    std::vector<std::vector<double>> X;
    std::vector<std::vector<double>> y;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::string line;
    while (std::getline(file, line)) {
        if (is_comment_or_empty(line)) continue;
        
        auto values = split_line(line);
        if (values.size() < 2) continue;
        
        std::vector<double> features;
        for (size_t i = 0; i < values.size() - 1; ++i) {
            features.push_back(std::stod(values[i]));
        }
        std::vector<double> target = {std::stod(values.back())};
        
        X.push_back(features);
        y.push_back(target);
    }
    
    file.close();
    
    if (!X.empty()) {
        print_data_info("Neural Regression", X.size(), X[0].size());
    }
    
    return {X, y};
}

std::vector<std::string> DataLoader::split_line(const std::string& line, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        // Trim whitespace
        token.erase(0, token.find_first_not_of(" \t\r\n"));
        token.erase(token.find_last_not_of(" \t\r\n") + 1);
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    
    return tokens;
}

bool DataLoader::is_comment_or_empty(const std::string& line) {
    if (line.empty()) return true;
    
    // Find first non-whitespace character
    size_t first = line.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return true;
    
    return line[first] == '#';
}

void DataLoader::print_data_info(const std::string& dataset_name, 
                                int samples, int features, int classes) {
    std::cout << dataset_name << " data loaded: " << samples << " samples";
    if (features > 0) {
        std::cout << ", " << features << " features";
    }
    if (classes > 0) {
        std::cout << ", " << classes << " classes";
    }
    std::cout << std::endl;
}
