#pragma once

#include "../src/models/MLP.cpp"
#include <vector>

class MLP_CUDA : public MLP {
public:
    MLP_CUDA(const MLP& mlp_base);

    MLP_CUDA(const int* layerSizes,
             int batch_size,
             double lr = 0.1,
             double acc_limit = 0.12);
    void train(const std::vector<std::vector<double>>& inputData,
               const std::vector<std::vector<double>>& outputData) override;

private:
    double acc_limit;
};
