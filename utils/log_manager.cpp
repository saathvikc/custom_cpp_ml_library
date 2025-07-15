#include "log_manager.h"
#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <filesystem>

std::string LogManager::generate_log_path(const std::string& model_type, 
                                         const std::string& data_type, 
                                         const std::string& test_name) {
    std::string date = get_current_date();
    std::string timestamp = get_current_timestamp();
    
    // Create hierarchical log structure: logs/model_type/data_type/date/
    std::string log_dir = "logs/" + model_type + "/" + data_type + "/" + date + "/";
    
    // Ensure directory exists
    ensure_log_directory_exists(log_dir);
    
    // Create filename with timestamp
    std::string filename = test_name + "_" + timestamp + ".log";
    
    return log_dir + filename;
}

void LogManager::ensure_log_directory_exists(const std::string& path) {
    try {
        std::filesystem::create_directories(path);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not create log directory " << path << ": " << e.what() << std::endl;
    }
}

std::string LogManager::get_current_date() {
    auto now = std::time(nullptr);
    auto* tm = std::localtime(&now);
    
    std::ostringstream oss;
    oss << std::put_time(tm, "%Y-%m-%d");
    return oss.str();
}

std::string LogManager::get_current_timestamp() {
    auto now = std::time(nullptr);
    auto* tm = std::localtime(&now);
    
    std::ostringstream oss;
    oss << std::put_time(tm, "%H%M%S");
    return oss.str();
}

void LogManager::clean_old_logs(const std::string& base_path, int days_to_keep) {
    // Placeholder implementation for cleaning old logs
    // In a full implementation, this would check file dates and remove old ones
    std::cout << "Log cleanup not implemented yet for path: " << base_path 
              << " (keeping " << days_to_keep << " days)" << std::endl;
}

std::string LogManager::create_session_header(const std::string& model_type, 
                                             const std::string& session_description) {
    std::ostringstream header;
    
    header << std::string(80, '=') << "\n";
    header << "ML TRAINING SESSION LOG\n";
    header << std::string(80, '=') << "\n";
    header << "Model Type: " << model_type << "\n";
    header << "Session: " << session_description << "\n";
    header << "Date: " << get_current_date() << "\n";
    header << "Time: ";
    
    auto now = std::time(nullptr);
    auto* tm = std::localtime(&now);
    header << std::put_time(tm, "%H:%M:%S") << "\n";
    
    header << std::string(80, '-') << "\n";
    
    return header.str();
}
