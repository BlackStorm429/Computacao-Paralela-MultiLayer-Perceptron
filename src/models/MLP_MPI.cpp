#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
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

    std::vector<double> centralWeights;
    std::vector<std::vector<std::vector<double>>> centralReplicaWeights = openMP.getWeights();
    for (const auto& layer : centralReplicaWeights) {
        for (const auto& neuron : layer) {
            centralWeights.insert(centralWeights.end(), neuron.begin(), neuron.end());
        }
    }

    int centralWeightSize = centralWeights.size();
    MPI_Bcast(&centralWeightSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    centralWeights.resize(centralWeightSize);
    MPI_Bcast(centralWeights.data(), centralWeightSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int rank, mpiSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);

    std::vector<std::vector<std::vector<double>>> unflattenedWeights;
    size_t index = 0;
    for (const auto& layer : centralReplicaWeights) {
        std::vector<std::vector<double>> unflattenedLayer;
        for (const auto& neuron : layer) {
            std::vector<double> unflattenedNeuron(neuron.size());
            for (size_t k = 0; k < neuron.size(); ++k) {
                unflattenedNeuron[k] = centralWeights[index++];
            }
            unflattenedLayer.push_back(unflattenedNeuron);
        }
        unflattenedWeights.push_back(unflattenedLayer);
    }
    centralReplicaWeights = unflattenedWeights;

    MLP_OpenMP openMPInstance(openMP);
    openMPInstance.setWeights(centralReplicaWeights);

    for (size_t l = 0; l < openMPInstance.centralReplica.weights.size(); ++l) {
        for (size_t j = 0; j < openMPInstance.centralReplica.weights[l].size(); ++j) {
            for (size_t k = 0; k < openMPInstance.centralReplica.weights[l][j].size(); ++k) {
                openMPInstance.centralReplica.weights[l][j][k] += ((double(std::rand()) / RAND_MAX - 0.5) * 2) * learningRate;
            }
        }
    }

    openMPInstance.train(inputData, outputData);
    double localLoss = openMPInstance.loss(inputData[0], outputData[0]);

    std::vector<double> allLosses;
    if (rank == 0) {
        allLosses.resize(mpiSize);
    }
    MPI_Gather(&localLoss, 1, MPI_DOUBLE,
               rank == 0 ? allLosses.data() : nullptr, 1, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    std::vector<double> localWeights;
    std::vector<std::vector<std::vector<double>>> weights = openMPInstance.getWeights();
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
               rank == 0 ? allWeights.data() : nullptr, weightSize, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

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
        std::vector<int> neuronSizes;
        for (const auto& layer : weights) {
            layerSizes.push_back(layer.size());
            neuronSizes.push_back(layer[0].size());
        }

        std::vector<std::vector<std::vector<double>>> bestWeightsUnflattened;
        size_t index = 0;
        for (size_t i = 0; i < layerSizes.size(); ++i) {
            std::vector<std::vector<double>> layer(layerSizes[i], std::vector<double>(neuronSizes[i]));
            for (int j = 0; j < layerSizes[i]; ++j) {
                for (int k = 0; k < neuronSizes[i]; ++k) {
                    layer[j][k] = bestWeights[index++];
                }
            }
            bestWeightsUnflattened.push_back(layer);
        }
        openMP.setWeights(bestWeightsUnflattened);
    }

    std::vector<std::vector<std::vector<double>>> mainWeights;
    if (rank == 0) {
        mainWeights = openMP.getWeights();
    }

    std::vector<double> broadcastWeights;
    std::vector<int> layerSizes;
    std::vector<int> neuronSizes;

    if (rank == 0) {
        for (const auto& layer : mainWeights) {
            layerSizes.push_back(layer.size());
            neuronSizes.push_back(layer[0].size());
            for (const auto& neuron : layer) {
                broadcastWeights.insert(broadcastWeights.end(), neuron.begin(), neuron.end());
            }
        }
    }

    if (rank != 0) {
        int numLayers;
        MPI_Bcast(&numLayers, 1, MPI_INT, 0, MPI_COMM_WORLD);
        layerSizes.resize(numLayers);
        neuronSizes.resize(numLayers);
    } else {
        int numLayers = layerSizes.size();
        MPI_Bcast(&numLayers, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Bcast(layerSizes.data(), layerSizes.size(), MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(neuronSizes.data(), neuronSizes.size(), MPI_INT, 0, MPI_COMM_WORLD);

    weightSize = 0;
    for (size_t i = 0; i < layerSizes.size(); ++i) {
        weightSize += layerSizes[i] * neuronSizes[i];
    }
    if (rank != 0) {
        broadcastWeights.resize(weightSize);
    }

    MPI_Bcast(broadcastWeights.data(), weightSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        std::vector<std::vector<std::vector<double>>> newWeights;
        size_t index = 0;
        for (size_t i = 0; i < layerSizes.size(); ++i) {
            std::vector<std::vector<double>> layer(layerSizes[i], std::vector<double>(neuronSizes[i]));
            for (int j = 0; j < layerSizes[i]; ++j) {
                for (int k = 0; k < neuronSizes[i]; ++k) {
                    layer[j][k] = broadcastWeights[index++];
                }
            }
            newWeights.push_back(layer);
        }
        openMP.setWeights(newWeights);
    }
}
    
};