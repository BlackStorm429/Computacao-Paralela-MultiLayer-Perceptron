#ifndef MLP_H
#define MLP_H

#include <vector>
#include "IMLP.h"

class MLP : public IMLP {
public:
    std::vector<int> layers;
    std::vector<std::vector<double>> neurons;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> deltas;

    MLP(const int* layerSizes, double learningRate = 0.1);
    MLP(const MLP& other);

    std::vector<double> forward(const std::vector<double>& input);
    void backward(const std::vector<double>& target);
    void updateWeights(const std::vector<double>& input);
    double loss(const std::vector<double>& input, const std::vector<double>& target);
    std::vector<std::vector<std::vector<double>>> getWeights();
    void setWeights(const std::vector<std::vector<std::vector<double>>>& newWeights);
};

#endif // MLP_H
