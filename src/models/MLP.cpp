#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include "IMLP.h"

class MLP : public IMLP {
private:

    double learningRate = 0.1;

    double activation(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double activationDerivative(double x) {
        return x * (1 - x);
    }

    double computeNeuronInput(const std::vector<double>& prevLayer, const std::vector<double>& weightRow) {
        double sum = weightRow[prevLayer.size()];
        for (size_t i = 0; i < prevLayer.size(); ++i) {
            sum += weightRow[i] * prevLayer[i];
        }
        return sum;
    }

public:
    std::vector<int> layers;
    std::vector<std::vector<double>> neurons;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> deltas;

    MLP(const int* layerSizes, double learningRate = 0.1) : learningRate(learningRate) {
        std::srand(static_cast<unsigned int>(std::time(0)));
        for (int i = 0; layerSizes[i] != 0; ++i) {
            layers.push_back(layerSizes[i]);
            neurons.push_back(std::vector<double>(layerSizes[i], 0.0));
            deltas.push_back(std::vector<double>(layerSizes[i], 0.0));
        }

        for (size_t i = 1; i < layers.size(); ++i) {
            weights.push_back(std::vector<std::vector<double>>(layers[i], std::vector<double>(layers[i - 1] + 1)));
            for (int j = 0; j < layers[i]; ++j) {
                for (int k = 0; k <= layers[i - 1]; ++k) {
                    weights[i - 1][j][k] = (double(std::rand()) / RAND_MAX - 0.5) * 2;
                }
            }
        }
    }

    MLP(const MLP& other)
        : layers(other.layers),
          neurons(other.neurons),
          weights(other.weights),
          deltas(other.deltas),
          learningRate(other.learningRate) {
    }

    std::vector<double> forward(const std::vector<double>& input) {
        if ((int)input.size() != layers[0]) {
            throw std::invalid_argument("Input size does not match the first layer size.");
        }
        neurons[0] = input;
        for (int i = 1; i < (int) layers.size(); ++i) {
            for (int j = 0; j < layers[i]; ++j) {
                double sum = computeNeuronInput(neurons[i - 1], weights[i - 1][j]);
                neurons[i][j] = activation(sum);
            }
        }
        return neurons.back();
    }

    double loss(const std::vector<double>& input, const std::vector<double>& target) {
        if ((int)input.size() != layers[0]) {
            throw std::invalid_argument("Input size does not match the first layer size.");
        }
        if ((int)target.size() != layers.back()) {
            throw std::invalid_argument("Target size does not match the output layer size.");
        }
        forward(input);
        double totalLoss = 0.0;
        for (size_t i = 0; i < target.size(); ++i) {
            totalLoss += 0.5 * std::pow(target[i] - neurons.back()[i], 2);
        }
        return totalLoss;
    }

    

    void backward(const std::vector<double>& target) {
        if ((int)target.size() != layers.back()) {
            throw std::invalid_argument("Target size does not match the output layer size.");
        }

        int L = layers.size() - 1;
        for (int i = 0; i < layers[L]; ++i) {
            deltas[L][i] = (target[i] - neurons[L][i]) * activationDerivative(neurons[L][i]);
        }

        for (int l = L - 1; l > 0; --l) {
            for (int i = 0; i < layers[l]; ++i) {
                double error = 0.0;
                for (int j = 0; j < layers[l + 1]; ++j) {
                    error += weights[l][j][i] * deltas[l + 1][j];
                }
                deltas[l][i] = error * activationDerivative(neurons[l][i]);
            }
        }

        for (size_t l = 1; l < layers.size(); ++l) {
            for (int i = 0; i < layers[l]; ++i) {
                for (int j = 0; j < layers[l - 1]; ++j) {
                    weights[l - 1][i][j] += learningRate * deltas[l][i] * neurons[l - 1][j];
                }      
            }
        }
    }

    void train(const std::vector<std::vector<double>>& inputData,
               const std::vector<std::vector<double>>& outputData) {
       
        for (size_t i = 0; i < inputData.size(); ++i) {
            forward(inputData[i]);
            backward(outputData[i]);
        }
        
    }
};
