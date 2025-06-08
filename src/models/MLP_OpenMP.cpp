#pragma once

#include "MLP.cpp"
#include <omp.h>
#include <vector>
#include <iostream>
#include <algorithm>

class MLP_OpenMP : public MLP {
private:
    int numThreads;
    std::vector<std::vector<double>> thread_gradients;

public:
    MLP_OpenMP(const int* layerSizes, int batch_size, int num_threads, double lr = 0.1)
        : MLP(layerSizes, batch_size, lr), numThreads(num_threads) {
        omp_set_num_threads(numThreads);
        thread_gradients.resize(numThreads);
        for (int i = 0; i < numThreads; ++i) {
            thread_gradients[i].resize(weights.size(), 0.0);
        }
    }

    MLP_OpenMP(const MLP& other, int num_threads)
        : MLP(other), numThreads(num_threads) {
        thread_gradients.resize(numThreads);
        for (int i = 0; i < numThreads; ++i) {
            thread_gradients[i].resize(weights.size(), 0.0);
        }
    }

    void train(const std::vector<std::vector<double>>& inputData,
               const std::vector<std::vector<double>>& outputData) override {
        if (inputData.empty() || outputData.empty() || inputData.size() != outputData.size()) {
            throw std::invalid_argument("Input or output data is empty or sizes mismatch.");
        }

        
        zero_gradients();
        for (int k = 0; k < (int)inputData.size(); k += batchSize) {
            int current_batch_size = std::min(batchSize, (int)inputData.size() - k);
            
            #pragma omp parallel num_threads(numThreads)
            {
                int tid = omp_get_thread_num();
                MLP local_mlp = MLP(*this);
                local_mlp.zero_gradients();
                
                #pragma omp for schedule(static)
                for (int i = 0; i < current_batch_size; ++i) { 
                    local_mlp.forward(inputData[k + i]);
                    local_mlp.compute_and_accumulate_gradients(outputData[k + i]);
                    
                }
                 
                std::copy(local_mlp.accumulated_gradients.begin(), local_mlp.accumulated_gradients.end(), 
                            thread_gradients[tid].begin());
                
            }

            for (size_t j = 0; j < accumulated_gradients.size(); ++j) {
                accumulated_gradients[j] = 0.0;
                for (int t = 0; t < numThreads; ++t) {
                    accumulated_gradients[j] += thread_gradients[t][j];
                }
            }
            
            apply_averaged_gradients(batchSize);
        }
    }
};
