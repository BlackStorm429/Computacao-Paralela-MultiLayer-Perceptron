#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <numeric> 
#include <stdexcept> 
#include "IMLP.h" 

class MLP : public IMLP {
protected:
    double learningRate = 0.1;
    int batchSize = 100;

    double activation(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }

    double activationDerivative(double activated_x) {
        return activated_x * (1.0 - activated_x);
    }

    double computeNeuronInput(const std::vector<double>& prevLayerOutputs, const std::vector<double>& neuronWeights) {
        double sum = neuronWeights.back(); 
        for (size_t i = 0; i < prevLayerOutputs.size(); ++i) {
            sum += neuronWeights[i] * prevLayerOutputs[i];
        }
        return sum;
    }

public:
    std::vector<int> layers; 
    std::vector<std::vector<double>> neurons; 
    std::vector<std::vector<std::vector<double>>> weights; 
    std::vector<std::vector<double>> deltas; 
    
    std::vector<std::vector<std::vector<double>>> accumulated_gradients;

    MLP(const int* layerSizes, int batch_size, double lr = 0.1) : learningRate(lr), batchSize(batch_size) {
        std::srand(static_cast<unsigned int>(std::time(nullptr)));
        for (int i = 0; layerSizes[i] != 0; ++i) {
            layers.push_back(layerSizes[i]);
            neurons.push_back(std::vector<double>(layerSizes[i], 0.0));
            deltas.push_back(std::vector<double>(layerSizes[i], 0.0));
        }

        for (size_t i = 0; i < layers.size() - 1; ++i) { 
            int neurons_in_current_layer = layers[i];
            int neurons_in_next_layer = layers[i+1];
            
            weights.push_back(std::vector<std::vector<double>>(neurons_in_next_layer, 
                               std::vector<double>(neurons_in_current_layer + 1))); 
            accumulated_gradients.push_back(std::vector<std::vector<double>>(neurons_in_next_layer,
                               std::vector<double>(neurons_in_current_layer + 1, 0.0))); 

            for (int j = 0; j < neurons_in_next_layer; ++j) {
                for (int k = 0; k <= neurons_in_current_layer; ++k) { 
                    weights[i][j][k] = (static_cast<double>(std::rand()) / RAND_MAX - 0.5) * 2.0; 
                }
            }
        }
    }

    MLP(const MLP& other)
        : layers(other.layers),
          neurons(other.neurons),
          weights(other.weights),
          deltas(other.deltas),
          accumulated_gradients(other.accumulated_gradients), 
          learningRate(other.learningRate) {
    }

    std::vector<double> forward(const std::vector<double>& input) {
        if (static_cast<int>(input.size()) != layers[0]) {
            throw std::invalid_argument("Input size does not match the first layer size.");
        }
        neurons[0] = input;
        for (size_t i = 1; i < layers.size(); ++i) { 
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
        const std::vector<double>& outputNeurons = neurons.back();
        for (size_t i = 0; i < target.size(); ++i) {
            totalLoss += 0.5 * std::pow(target[i] - outputNeurons[i], 2);
        }
        return totalLoss;
    }

    void compute_and_accumulate_gradients(const std::vector<double>& target) {
        if (static_cast<int>(target.size()) != layers.back()) {
            throw std::invalid_argument("Target size does not match the output layer size.");
        }

        int num_actual_layers = layers.size();
        int output_layer_idx = num_actual_layers - 1;

        for (int i = 0; i < layers[output_layer_idx]; ++i) {
            double output_neuron_val = neurons[output_layer_idx][i];
            deltas[output_layer_idx][i] = (target[i] - output_neuron_val) * activationDerivative(output_neuron_val);
        }

        for (int l = output_layer_idx - 1; l > 0; --l) { 
            for (int i = 0; i < layers[l]; ++i) { 
                double error_sum = 0.0;
                for (int j = 0; j < layers[l + 1]; ++j) { 
                    error_sum += weights[l][j][i] * deltas[l + 1][j];
                }
                deltas[l][i] = error_sum * activationDerivative(neurons[l][i]);
            }
        }

        for (size_t k = 0; k < weights.size(); ++k) { 
            int layer_idx_from = k;
            int layer_idx_to = k + 1;

            for (int i = 0; i < layers[layer_idx_to]; ++i) { 
                for (int j = 0; j < layers[layer_idx_from]; ++j) { 
                    accumulated_gradients[k][i][j] += deltas[layer_idx_to][i] * neurons[layer_idx_from][j];
                }
                accumulated_gradients[k][i][layers[layer_idx_from]] += deltas[layer_idx_to][i]; 
            }
        }
    }

    void zero_gradients() {
        for (size_t k = 0; k < accumulated_gradients.size(); ++k) {
            for (size_t i = 0; i < accumulated_gradients[k].size(); ++i) {
                std::fill(accumulated_gradients[k][i].begin(), accumulated_gradients[k][i].end(), 0.0);
            }
        }
    }

    void apply_averaged_gradients(size_t batch_size) {
        if (batch_size == 0) return; 

        for (size_t k = 0; k < weights.size(); ++k) { 
            int neurons_in_next_layer = layers[k+1]; 
            int neurons_in_current_layer = layers[k]; 

            for (int i = 0; i < neurons_in_next_layer; ++i) { 
                for (int j = 0; j < neurons_in_current_layer; ++j) { 
                    weights[k][i][j] += learningRate * (accumulated_gradients[k][i][j] / batch_size);
                }
                weights[k][i][neurons_in_current_layer] += learningRate * (accumulated_gradients[k][i][layers[k]] / batch_size);
            }
        }
    }

    void train(const std::vector<std::vector<double>>& inputData,
               const std::vector<std::vector<double>>& outputData) {
        
        if (inputData.empty() || outputData.empty() || inputData.size() != outputData.size()) {
             throw std::invalid_argument("Input or output data is empty or sizes mismatch.");
        }

        zero_gradients(); 

        for (size_t i = 0; i < inputData.size(); ++i) {
            forward(inputData[i]); 
            compute_and_accumulate_gradients(outputData[i]);
            if ((i + 1) % batchSize == 0 || i == inputData.size() - 1) {
                apply_averaged_gradients(batchSize);
                zero_gradients(); 
            }
        }
        
        apply_averaged_gradients(inputData.size());
    }
};