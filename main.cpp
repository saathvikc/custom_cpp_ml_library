#include "regression/linear_regression.h"
#include "utils/utils.h"
#include <iostream>

int main() {
    std::vector<double> x, y;

    load_csv("data/sample.csv", x, y);
    if (x.empty()) {
        std::cerr << "No data loaded.\n";
        return 1;
    }

    LinearRegression model(0.01, 1000);
    model.fit(x, y);

    std::cout << "\nPrediction for x = 6.0: " << model.predict(6.0) << "\n";
    return 0;
}
