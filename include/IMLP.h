#pragma once

#include <iostream>
#include <vector>

// Interface for the MLP
class IMLP {
public:
    virtual std::vector<double> forward(const std::vector<double>& input) = 0;
    
    virtual double loss(const std::vector<double>& input, const std::vector<double>& target) = 0;
    virtual void train(const std::vector<std::vector<double>>& inputData, const std::vector<std::vector<double>>& outputData) = 0;
    
    virtual ~IMLP() = default;
};
