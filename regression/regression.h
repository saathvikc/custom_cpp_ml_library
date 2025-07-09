#pragma once

// Common enums and types used by both linear and logistic regression

enum class OptimizerType {
    SGD,
    MOMENTUM,
    ADAGRAD,
    ADAM
};

enum class RegularizationType {
    NONE,
    L1,
    L2,
    ELASTIC_NET
};
