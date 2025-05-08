#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <mpi/mpi.h>
#include "IMLP.h"
#include "MLP_OpenMP.cpp"

class MLP_MPI : public IMLP {
private:
    double learningRate = 0.1;
    
    // MPI variables
    int mpiSize = 1;
    int numThreads = 1;

    MLP_OpenMP openMP;

public:
    MLP_MPI(const int* layerSizes, double learningRate = 0.1, int mpiSize = 1, int numThreads = 1)
    : openMP(layerSizes, learningRate, numThreads)
    {
        this->learningRate = learningRate;
        this->mpiSize = mpiSize;
        this->numThreads = numThreads;
    }


    
    std::vector<double> forward(const std::vector<double>& input) {
        return openMP.forward(input);
    }
    
    void backward(const std::vector<double>& target) {
        return openMP.backward(target);
    }
    
    void train(const std::vector<std::vector<double>>& inputData,
        const std::vector<std::vector<double>>& outputData) override {
    if (inputData.size() != outputData.size()) {
        throw std::invalid_argument("Input and output data sizes do not match.");
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //add noise to the weights
    for (size_t l = 0; l < openMP.centralReplica.weights.size(); ++l) {
        for (size_t j = 0; j < openMP.centralReplica.weights[l].size(); ++j) {
            for (size_t k = 0; k < openMP.centralReplica.weights[l][j].size(); ++k) {
                openMP.centralReplica.weights[l][j][k] += ((double(std::rand()) / RAND_MAX - 0.5) * 2) * learningRate;
            }
        }
    }

    // 2. Train locally using OpenMP
    openMP.train(inputData, outputData);

    // 3. Compute local loss
    double localLoss = openMP.loss(inputData[0], outputData[0]);

    // 4. Gather all losses to rank 0
    std::vector<double> allLosses;
    if (rank == 0) {
        allLosses.resize(mpiSize);
    }
    MPI_Gather(&localLoss, 1, MPI_DOUBLE, allLosses.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 5. Send all weights to rank 0
    std::vector<double> localWeights;

    //flatten the weights
    std::vector<std::vector<std::vector<double>>> weights = openMP.getWeights();
    int weightSize = 0;
    for (const auto& layer : weights) {
        for (const auto& neuron : layer) {
            weightSize += neuron.size();
        }
    }

    for (const auto& layer : weights) {
        for (const auto& neuron : layer) {
            localWeights.insert(localWeights.end(), neuron.begin(), neuron.end());
        }
    }
    

    std::vector<double> allWeights;
    if (rank == 0) {
        allWeights.resize(weightSize * mpiSize);
    }

    MPI_Gather(localWeights.data(), weightSize, MPI_DOUBLE,
                allWeights.data(), weightSize, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // 6. Rank 0 picks best model and sets it as central
    if (rank == 0) {
        int bestIndex = 0;
        for (int i = 1; i < mpiSize; ++i) {
            if (allLosses[i] < allLosses[bestIndex]) {
                bestIndex = i;
            }
        }

        std::vector<double> bestWeights(allWeights.begin() + bestIndex * weightSize,
                                        allWeights.begin() + (bestIndex + 1) * weightSize);

        std::vector<int> layerSizes;
        for (const auto& layer : weights) {
            layerSizes.push_back(layer.size());
        }
        //unflatten the weights
        std::vector<std::vector<std::vector<double>>> bestWeightsUnflattened;
        for (size_t i = 0; i < layerSizes.size(); ++i) {
            std::vector<std::vector<double>> layer(layerSizes[i]);
            for (size_t j = 0; j < layerSizes[i]; ++j) {
                layer[j].resize(layerSizes[i]);
                for (size_t k = 0; k < layerSizes[i]; ++k) {
                    layer[j][k] = bestWeights[i * layerSizes[i] + j * layerSizes[i] + k];
                }
            }
            bestWeightsUnflattened.push_back(layer);
        }
        openMP.setWeights(bestWeightsUnflattened);
        
    }

    // 7. Broadcast the best weights to all processes
    std::vector<std::vector<std::vector<double>>> weights;
    if (rank == 0) {
        weights = openMP.getWeights();
    }

    std::vector<double> broadcastWeights;
    std::vector<int> layerSizes;
    for (const auto& layer : weights) {
        layerSizes.push_back(layer.size());
    }
    //flatten the weights
    for (const auto& layer : weights) {
        for (const auto& neuron : layer) {
            broadcastWeights.insert(broadcastWeights.end(), neuron.begin(), neuron.end());
        }
    }


    MPI_Bcast(broadcastWeights.data(), weightSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(layerSizes.data(), layerSizes.size(), MPI_INT, 0, MPI_COMM_WORLD);
    
    // 8. Unflatten the weights
    if (rank != 0) {
        std::vector<std::vector<std::vector<double>>> newWeights;
        for (size_t i = 0; i < layerSizes.size(); ++i) {
            std::vector<std::vector<double>> layer(layerSizes[i]);
            for (size_t j = 0; j < layerSizes[i]; ++j) {
                layer[j].resize(layerSizes[i]);
                for (size_t k = 0; k < layerSizes[i]; ++k) {
                    layer[j][k] = broadcastWeights[i * layerSizes[i] + j * layerSizes[i] + k];
                }
            }
            newWeights.push_back(layer);
        }
        openMP.setWeights(newWeights);
    }
}
    
    
};