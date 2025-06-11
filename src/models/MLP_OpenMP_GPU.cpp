#pragma once

#include <omp.h>
#include <vector>
#include <iostream>
#include <algorithm>

#include "MLP.cpp"

class MLP_OpenMP_GPU : public MLP {
    private:
    int acc_limit = 0.12;

    public:
        MLP_OpenMP_GPU(const int* layerSizes, int batch_size, double lr = 0.1, int acc_limit = 0.12)
            : MLP(layerSizes, batch_size, lr)
        {        }

        MLP_OpenMP_GPU(const MLP& other, int acc_limit = 0.12)
            : MLP(other)
        {

        }

        void train(const std::vector<std::vector<double>>& inputData, const std::vector<std::vector<double>>& outputData) override
    {
        if (inputData.empty() || outputData.empty() || inputData.size() != outputData.size()) {
            throw std::invalid_argument("Input or output data is empty or sizes mismatch.");
        }

        const size_t num_samples = inputData.size();
        const size_t input_size = layers[0];
        const size_t output_size = layers.back();
        const size_t num_layers = layers.size();
        const size_t weights_count = weights.size();
        const size_t neurons_count = neurons.size();
        const size_t deltas_count = deltas.size();
        const size_t gradients_count = accumulated_gradients.size();

        // Flatten the input and output data for easier GPU transfer
        std::vector<double> flat_input(num_samples * input_size);
        std::vector<double> flat_output(num_samples * output_size);

        for(size_t i = 0; i < num_samples; ++i) {
            std::copy(inputData[i].begin(), inputData[i].end(), flat_input.begin() + i * input_size);
            std::copy(outputData[i].begin(), outputData[i].end(), flat_output.begin() + i * output_size);
        }

        // Use pointers for capturing in the lambda for OpenMP target
        double* p_weights = weights.data();
        double* p_neurons = neurons.data();
        double* p_deltas = deltas.data();
        double* p_accumulated_gradients = accumulated_gradients.data();
        const int* p_layers = layers.data();
        const int* p_neuron_offsets = neuronOffsets.data();
        const int* p_weight_offsets = weightOffsets.data();
        const double* p_flat_input = flat_input.data();
        const double* p_flat_output = flat_output.data();
        const double lr = this->learningRate;
        const int b_size = this->batchSize;

        #pragma omp target data map(tofrom: p_weights[0:weights_count]) \
                                map(alloc: p_neurons[0:neurons_count], p_deltas[0:deltas_count], p_accumulated_gradients[0:gradients_count]) \
                                map(to: p_layers[0:num_layers], p_neuron_offsets[0:num_layers], p_weight_offsets[0:num_layers-1]) \
                                map(to: p_flat_input[0:num_samples * input_size], p_flat_output[0:num_samples * output_size])
        {
            for (int epoch = 0; epoch < 5; epoch++) {
                for (size_t i = 0; i < num_samples; i += b_size)
                {
                    const size_t current_batch_size = std::min((size_t)b_size, num_samples - i);

                    // Zero out gradients
                    #pragma omp target teams distribute parallel for
                    for (size_t j = 0; j < gradients_count; ++j) {
                        p_accumulated_gradients[j] = 0.0;
                    }

                    #pragma omp target teams distribute
                    for (size_t sample_idx = i; sample_idx < i + current_batch_size; ++sample_idx)
                    {
                        // FORWARD 
                        // Copy input for the current sample
                        #pragma omp parallel for
                        for (size_t k = 0; k < input_size; ++k) {
                            p_neurons[k] = p_flat_input[sample_idx * input_size + k];
                        }

                        for (size_t l = 1; l < num_layers; ++l) {
                            const int prev_offset = p_neuron_offsets[l - 1];
                            const int curr_offset = p_neuron_offsets[l];
                            const int prev_layer_size = p_layers[l - 1];
                            const int curr_layer_size = p_layers[l];
                            const int weight_off = p_weight_offsets[l - 1];

                            #pragma omp parallel for
                            for (int j = 0; j < curr_layer_size; ++j) {
                                double sum = p_weights[weight_off + j * (prev_layer_size + 1) + prev_layer_size];
                                for (int k = 0; k < prev_layer_size; ++k) {
                                    sum += p_weights[weight_off + j * (prev_layer_size + 1) + k] * p_neurons[prev_offset + k];
                                }
                                p_neurons[curr_offset + j] = 1.0 / (1.0 + exp(-sum));
                            }
                        }

                        // BACKWARD
                        // Calculate deltas for the output layer
                        const int output_layer_idx = num_layers - 1;
                        const int output_offset = p_neuron_offsets[output_layer_idx];
                        
                        #pragma omp parallel for
                        for (int j = 0; j < p_layers[output_layer_idx]; ++j) {
                            double out = p_neurons[output_offset + j];
                            double target = p_flat_output[sample_idx * output_size + j];
                            p_deltas[output_offset + j] = (target - out) * out * (1.0 - out);
                        }
                        
                        // Propagate deltas to hidden layers
                        for (int l = output_layer_idx - 1; l > 0; --l) {
                            const int curr_offset = p_neuron_offsets[l];
                            const int next_offset = p_neuron_offsets[l + 1];
                            const int curr_layer_size = p_layers[l];
                            const int next_layer_size = p_layers[l + 1];
                            const int weight_off = p_weight_offsets[l];

                            #pragma omp parallel for
                            for (int j = 0; j < curr_layer_size; ++j) {
                                double error = 0.0;
                                for (int k = 0; k < next_layer_size; ++k) {
                                    error += p_weights[weight_off + k * (curr_layer_size + 1) + j] * p_deltas[next_offset + k];
                                }
                                double activated_neuron = p_neurons[curr_offset + j];
                                p_deltas[curr_offset + j] = error * activated_neuron * (1.0 - activated_neuron);
                            }
                        }

                        for (size_t l = 0; l < num_layers - 1; ++l) {
                            const int from_offset = p_neuron_offsets[l];
                            const int to_offset = p_neuron_offsets[l + 1];
                            const int from_layer_size = p_layers[l];
                            const int to_layer_size = p_layers[l + 1];
                            const int weight_off = p_weight_offsets[l];

                            #pragma omp parallel for
                            for (int j = 0; j < to_layer_size; ++j) {
                                double delta_val = p_deltas[to_offset + j];
                                for (int k = 0; k < from_layer_size; ++k) {
                                    #pragma omp atomic
                                    p_accumulated_gradients[weight_off + j * (from_layer_size + 1) + k] += delta_val * p_neurons[from_offset + k];
                                }
                                #pragma omp atomic
                                p_accumulated_gradients[weight_off + j * (from_layer_size + 1) + from_layer_size] += delta_val; // Bias gradient
                            }
                        }
                    } // End of batch processing

                    #pragma omp target teams distribute parallel for
                    for (size_t j = 0; j < weights_count; ++j) {
                        p_weights[j] += lr * (p_accumulated_gradients[j] / current_batch_size);
                    }

                    } // End of epoch loop
            } // OMP target data region ends here. Data is automatically mapped back/deallocated.
        }
    }
};