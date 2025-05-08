#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include "MLP.cpp"
#include "IMLP.h"
#include <limits>

class MLP_OpenMP : public IMLP {
private:
    std::vector<int> layers;
    double learningRate = 0.1;
    int numReplicas = 4;
    std::vector<MLP> replicas;
    
    
    public:
    MLP centralReplica;

    MLP_OpenMP(const int* layerSizes, double learningRate = 0.1, int numReplicas = 4) : centralReplica(layerSizes, learningRate) {
        this->numReplicas = numReplicas;
        this->learningRate = learningRate;
    }
    
    std::vector<double> forward(const std::vector<double>& input) {
       return centralReplica.forward(input); 
    }
    
    void backward(const std::vector<double>& target) {
        centralReplica.backward(target);
    }
    
    double loss(const std::vector<double>& input, const std::vector<double>& target) {
        return centralReplica.loss(input, target);
    }

    void setWeights(const std::vector<std::vector<std::vector<double>>>& weights) {
        centralReplica.weights = weights;
    }

    std::vector<std::vector<std::vector<double>>> getWeights() const {
        return centralReplica.weights;
    }

    void train(const std::vector<std::vector<double>>& inputData,
               const std::vector<std::vector<double>>& outputData) {

        std::vector<MLP> replicas;

        for (int i = 0; i < numReplicas; ++i) {
            replicas.push_back(MLP(centralReplica));
            //add noise
            for (size_t l = 0; l < replicas[i].weights.size(); ++l) {
                for (size_t j = 0; j < replicas[i].weights[l].size(); ++j) {
                    for (size_t k = 0; k < replicas[i].weights[l][j].size(); ++k) {
                        replicas[i].weights[l][j][k] += ((double(std::rand()) / RAND_MAX - 0.5) * 2) * learningRate;
                    }
                }
            }
        }
        
        #pragma omp parallel for num_threads(numReplicas)
        for (int i = 0; i < numReplicas; ++i) {
            replicas[i].train(inputData, outputData);
        }

        //Get the best replica
        double bestLoss = std::numeric_limits<double>::max();
        int bestIndex = 0;
        for (int i = 0; i < numReplicas; ++i) {
            double loss = replicas[i].loss(inputData[0], outputData[0]);
            if (loss < bestLoss) {
                bestLoss = loss;
                bestIndex = i;
            }
        }
        //Update the central replica with the best replica
        centralReplica = replicas[bestIndex];
    }


};