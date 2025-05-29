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

   int getWeightIndex(int layer, int neuron, int input) const {
      return weightOffsets[layer] + neuron * (layers[layer] + 1) + input;
   }

public:
   std::vector<int> layers;
   std::vector<int> neuronOffsets;
   std::vector<int> weightOffsets;

   std::vector<double> neurons;
   std::vector<double> deltas;
   std::vector<double> weights;
   std::vector<double> accumulated_gradients;

   MLP(const int* layerSizes, int batch_size, double lr = 0.1) : learningRate(lr), batchSize(batch_size) {
      std::srand(static_cast<unsigned int>(std::time(nullptr)));
      int neuronOffset = 0, weightOffset = 0;

      for (int i = 0; layerSizes[i] != 0; ++i) {
         layers.push_back(layerSizes[i]);
         neuronOffsets.push_back(neuronOffset);
         neuronOffset += layerSizes[i];
      }

      deltas.resize(neuronOffset);
      neurons.resize(neuronOffset);

      for (size_t i = 0; i < layers.size() - 1; ++i) {
         weightOffsets.push_back(weightOffset);
         int weightsPerLayer = layers[i+1] * (layers[i] + 1);
         weightOffset += weightsPerLayer;
      }

      weights.resize(weightOffset);
      accumulated_gradients.resize(weightOffset);

      for (double &w : weights) {
         w = (static_cast<double>(std::rand()) / RAND_MAX - 0.5) * 2.0;
      }
   }

   MLP(const MLP& other) 
      : layers(other.layers), neuronOffsets(other.neuronOffsets), weightOffsets(other.weightOffsets),
        neurons(other.neurons), deltas(other.deltas), weights(other.weights), accumulated_gradients(other.accumulated_gradients),
        learningRate(other.learningRate), batchSize(other.batchSize) {}

   std::vector<double> forward(const std::vector<double>& input) {
      if ((int)input.size() != layers[0]) {
         throw std::invalid_argument("Input size does not match the first layer size.");
      }
      std::copy(input.begin(), input.end(), neurons.begin());

      for (size_t l = 1; l < layers.size(); ++l) {
         int prevOffset = neuronOffsets[l-1];
         int currOffset = neuronOffsets[l];
         for (int j = 0; j < layers[l]; ++j) {
            double sum = weights[getWeightIndex(l-1, j, layers[l-1])];
            for (int k = 0; k < layers[l-1]; ++k) {
               sum += weights[getWeightIndex(l-1, j, k)] * neurons[prevOffset + k];
            }
            neurons[currOffset + j] = activation(sum);
         }
      }
      return std::vector<double>(neurons.begin() + neuronOffsets.back(), neurons.begin() + neuronOffsets.back() + layers.back());
   }

   void compute_and_accumulate_gradients(const std::vector<double>& target) {
      int outputLayerIdx = layers.size() - 1;
      int outputOffset = neuronOffsets[outputLayerIdx];

      for (int i = 0; i < layers[outputLayerIdx]; ++i) {
         double out = neurons[outputOffset + i];
         deltas[outputOffset + i] = (target[i] - out) * activationDerivative(out);
      }

      for (int l = outputLayerIdx - 1; l > 0; --l) {
         int currOffset = neuronOffsets[l];
         int nextOffset = neuronOffsets[l+1];
         for (int i = 0; i < layers[l]; ++i) {
            double error = 0.0;
            for (int j = 0; j < layers[l+1]; ++j) {
               error += weights[getWeightIndex(l, j, i)] * deltas[nextOffset + j];
            }
            deltas[currOffset + i] = error * activationDerivative(neurons[currOffset + i]);
         }
      }

      for (size_t l = 0; l < layers.size() - 1; ++l) {
         int fromOffset = neuronOffsets[l];
         int toOffset = neuronOffsets[l+1];
         for (int i = 0; i < layers[l+1]; ++i) {
            for (int j = 0; j < layers[l]; ++j) {
               accumulated_gradients[getWeightIndex(l, i, j)] += deltas[toOffset + i] * neurons[fromOffset + j];
            }
            accumulated_gradients[getWeightIndex(l, i, layers[l])] += deltas[toOffset + i]; // bias
         }
      }
   }

   void zero_gradients() {
      std::fill(accumulated_gradients.begin(), accumulated_gradients.end(), 0.0);
   }

   void apply_averaged_gradients(size_t batch_size) {
      if (batch_size == 0) return;
      for (size_t i = 0; i < weights.size(); ++i) {
         weights[i] += learningRate * (accumulated_gradients[i] / batch_size);
      }
   }

   void train(const std::vector<std::vector<double>>& inputData, const std::vector<std::vector<double>>& outputData) {
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
   }

    double loss(const std::vector<double>& input, const std::vector<double>& target) override {
        if (input.size() != layers[0] || target.size() != layers.back()) {
            throw std::invalid_argument("Input or target size does not match the network.");
        }
        forward(input);
        double total_loss = 0.0;
        int outputOffset = neuronOffsets.back();
        for (int i = 0; i < layers.back(); ++i) {
            double diff = neurons[outputOffset + i] - target[i];
            total_loss += diff * diff; // Mean Squared Error
        }
        return total_loss / layers.back();
    }

   const std::vector<double>& getNeurons() const { return neurons; }
   const std::vector<double>& getDeltas() const { return deltas; }
   const std::vector<double>& getWeights() const { return weights; }
   const std::vector<double>& getAccumulatedGradients() const { return accumulated_gradients; }

   void setNeurons(const std::vector<double>& n) { neurons = n; }
   void setDeltas(const std::vector<double>& d) { deltas = d; }
   void setWeights(const std::vector<double>& w) { weights = w; }
   void setAccumulatedGradients(const std::vector<double>& g) { accumulated_gradients = g; }
};
