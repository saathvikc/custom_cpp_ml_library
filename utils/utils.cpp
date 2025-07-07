#include "utils.h"
#include <fstream>
#include <sstream>
#include <iostream>

void load_csv(const std::string& filename, std::vector<double>& x, std::vector<double>& y) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open CSV file.\n";
        return;
    }

    std::string line;
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string x_val, y_val;
        if (getline(ss, x_val, ',') && getline(ss, y_val)) {
            x.push_back(std::stod(x_val));
            y.push_back(std::stod(y_val));
        }
    }

    file.close();
}
