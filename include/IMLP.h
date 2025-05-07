#pragma once

#include <iostream>
#include <vector>

// Interface for the MLP
class IMLP {
public:
    virtual std::vector<double> forward(const std::vector<double>& input) = 0;
    virtual void backward(const std::vector<double>& target) = 0;
    virtual ~IMLP() = default;
};
