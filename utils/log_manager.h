#pragma once
#include <string>
#include <ctime>
#include <iomanip>
#include <sstream>

class LogManager {
public:
    // Generate a standardized log path based on model type, data type, and date
    static std::string generate_log_path(const std::string& model_type, 
                                       const std::string& data_type = "default", 
                                       const std::string& test_name = "training");
    
    // Create directory structure if it doesn't exist
    static void ensure_log_directory_exists(const std::string& path);
    
    // Get current date string in YYYY-MM-DD format
    static std::string get_current_date();
    
    // Get current timestamp string
    static std::string get_current_timestamp();
    
    // Clean old log files (optional)
    static void clean_old_logs(const std::string& base_path, int days_to_keep = 7);
    
    // Create a session log header
    static std::string create_session_header(const std::string& model_type, 
                                           const std::string& session_description);
};
