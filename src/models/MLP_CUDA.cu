#include "MLP_CUDA.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>

// Device-side activation functions
__device__ double cuda_sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

__device__ double cuda_sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

// Kernel to copy input data to network's input layer
__global__ void input_kernel(double* neurons, const double* inputs, 
                            const int* neuron_offsets, int input_size, 
                            int total_neurons, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * input_size) return;
    
    int sample = idx / input_size;
    int feature = idx % input_size;
    neurons[sample * total_neurons + neuron_offsets[0] + feature] = inputs[sample * input_size + feature];
}

// Kernel for forward propagation (one layer)
__global__ void forward_kernel(int layer, double* neurons, const double* weights,
                              const int* layers, const int* neuron_offsets,
                              const int* weight_offsets, int total_neurons,
                              int batch_size) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample = blockIdx.y;
    
    if (sample >= batch_size) return;
    if (neuron_idx >= layers[layer]) return;
    
    int prev_layer_size = layers[layer-1];
    int weight_offset = weight_offsets[layer-1];
    int prev_neuron_offset = neuron_offsets[layer-1];
    int curr_neuron_offset = neuron_offsets[layer];
    
    double sum = weights[weight_offset + neuron_idx * (prev_layer_size + 1) + prev_layer_size]; // bias
    
    for (int i = 0; i < prev_layer_size; ++i) {
        double weight = weights[weight_offset + neuron_idx * (prev_layer_size + 1) + i];
        double activation = neurons[sample * total_neurons + prev_neuron_offset + i];
        sum += weight * activation;
    }
    
    neurons[sample * total_neurons + curr_neuron_offset + neuron_idx] = cuda_sigmoid(sum);
}

// Kernel for output layer deltas
__global__ void output_delta_kernel(double* deltas, const double* neurons, 
                                   const double* targets, const int* layers,
                                   const int* neuron_offsets, int output_size,
                                   int total_neurons, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size * output_size) return;
    
    int sample = idx / output_size;
    int neuron = idx % output_size;
    int output_offset = neuron_offsets[layers[0] - 1]; // Last layer offset
    
    double out_val = neurons[sample * total_neurons + output_offset + neuron];
    double target_val = targets[sample * output_size + neuron];
    deltas[sample * total_neurons + output_offset + neuron] = (target_val - out_val) * cuda_sigmoid_derivative(out_val);
}

// Kernel for hidden layer deltas
__global__ void hidden_delta_kernel(int layer, double* deltas, const double* neurons,
                                   const double* weights, const int* layers,
                                   const int* neuron_offsets, const int* weight_offsets,
                                   int total_neurons, int batch_size) {
    int neuron_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int sample = blockIdx.y;
    
    if (sample >= batch_size) return;
    if (neuron_idx >= layers[layer]) return;
    
    int next_layer_size = layers[layer+1];
    int weight_offset = weight_offsets[layer];
    int curr_offset = neuron_offsets[layer];
    int next_offset = neuron_offsets[layer+1];
    
    double error = 0.0;
    for (int j = 0; j < next_layer_size; ++j) {
        double weight = weights[weight_offset + j * (layers[layer] + 1) + neuron_idx];
        error += weight * deltas[sample * total_neurons + next_offset + j];
    }
    
    double neuron_val = neurons[sample * total_neurons + curr_offset + neuron_idx];
    deltas[sample * total_neurons + curr_offset + neuron_idx] = error * cuda_sigmoid_derivative(neuron_val);
}

// Kernel to accumulate gradients
__global__ void accumulate_gradients_kernel(int layer, double* gradients, 
                                           const double* deltas, const double* neurons,
                                           const int* layers, const int* neuron_offsets,
                                           const int* weight_offsets, int total_neurons,
                                           int batch_size) {
    int to_neuron = blockIdx.x * blockDim.x + threadIdx.x;
    int from_neuron = blockIdx.y * blockDim.y + threadIdx.y;
    int sample = blockIdx.z;
    
    if (sample >= batch_size) return;
    if (to_neuron >= layers[layer+1]) return;
    if (from_neuron >= layers[layer] + 1) return; // +1 for bias

    int weight_offset = weight_offsets[layer];
    int from_offset = neuron_offsets[layer];
    int to_offset = neuron_offsets[layer+1];
    
    double delta_val = deltas[sample * total_neurons + to_offset + to_neuron];
    double activation = (from_neuron < layers[layer]) ? 
                        neurons[sample * total_neurons + from_offset + from_neuron] : 
                        1.0; // bias term
    
    int weight_index = weight_offset + to_neuron * (layers[layer] + 1) + from_neuron;
    atomicAdd(&gradients[weight_index], delta_val * activation);
}

// Kernel to update weights
__global__ void update_weights_kernel(double* weights, const double* gradients, 
                                     int total_weights, double lr, int batch_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_weights) return;
    weights[idx] += lr * (gradients[idx] / batch_size);
}

// Constructor - Allocates device memory and copies initial data
MLP_CUDA::MLP_CUDA(const MLP& base_mlp) : MLP(base_mlp) {
    // Copy layer configuration
    total_neurons = neurons.size();
    total_weights = weights.size();
    num_layers = layers.size();
    
    // Allocate device memory
    cudaMalloc(&d_weights, total_weights * sizeof(double));
    cudaMalloc(&d_neurons, batchSize * total_neurons * sizeof(double));
    cudaMalloc(&d_deltas, batchSize * total_neurons * sizeof(double));
    cudaMalloc(&d_gradients, total_weights * sizeof(double));
    
    // Copy layer metadata
    cudaMalloc(&d_layers, num_layers * sizeof(int));
    cudaMalloc(&d_neuron_offsets, num_layers * sizeof(int));
    cudaMalloc(&d_weight_offsets, (num_layers-1) * sizeof(int));
    
    // Copy initial weights
    cudaMemcpy(d_weights, weights.data(), total_weights * sizeof(double), cudaMemcpyHostToDevice);
    
    // Copy metadata
    cudaMemcpy(d_layers, layers.data(), num_layers * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_neuron_offsets, neuronOffsets.data(), num_layers * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_offsets, weightOffsets.data(), (num_layers-1) * sizeof(int), cudaMemcpyHostToDevice);
}

// Destructor - Clean up device memory
MLP_CUDA::~MLP_CUDA() {
    cudaFree(d_weights);
    cudaFree(d_neurons);
    cudaFree(d_deltas);
    cudaFree(d_gradients);
    cudaFree(d_layers);
    cudaFree(d_neuron_offsets);
    cudaFree(d_weight_offsets);
}

// Training method with CUDA acceleration
void MLP_CUDA::train(const std::vector<std::vector<double>>& inputData, 
                    const std::vector<std::vector<double>>& outputData) {
    if (inputData.empty() || outputData.empty() || inputData.size() != outputData.size()) {
        throw std::invalid_argument("Invalid training data");
    }

    const int num_samples = inputData.size();
    const int input_size = layers[0];
    const int output_size = layers.back();
    
    // Flatten input and output data
    std::vector<double> flat_inputs;
    std::vector<double> flat_targets;
    for (int i = 0; i < num_samples; ++i) {
        flat_inputs.insert(flat_inputs.end(), inputData[i].begin(), inputData[i].end());
        flat_targets.insert(flat_targets.end(), outputData[i].begin(), outputData[i].end());
    }
    
    // Device memory for batch data
    double *d_batch_input, *d_batch_output;
    cudaMalloc(&d_batch_input, batchSize * input_size * sizeof(double));
    cudaMalloc(&d_batch_output, batchSize * output_size * sizeof(double));
    
    // Training loop
    for (int i = 0; i < num_samples; i += batchSize) {
        int current_batch_size = std::min(batchSize, num_samples - i);
        
        // Copy batch data to device
        cudaMemcpy(d_batch_input, flat_inputs.data() + i * input_size, 
                  current_batch_size * input_size * sizeof(double), 
                  cudaMemcpyHostToDevice);
        cudaMemcpy(d_batch_output, flat_targets.data() + i * output_size, 
                  current_batch_size * output_size * sizeof(double), 
                  cudaMemcpyHostToDevice);
        
        // Reset gradients
        cudaMemset(d_gradients, 0, total_weights * sizeof(double));
        
        // Forward pass: Copy input to network
        dim3 block_in(256);
        dim3 grid_in((current_batch_size * input_size + block_in.x - 1) / block_in.x);
        input_kernel<<<grid_in, block_in>>>(d_neurons, d_batch_input, 
                                          d_neuron_offsets, input_size,
                                          total_neurons, current_batch_size);
        cudaDeviceSynchronize();
        
        // Forward pass: Hidden and output layers
        for (int l = 1; l < num_layers; ++l) {
            dim3 block_f(256);
            dim3 grid_f((layers[l] + block_f.x - 1) / block_f.x, current_batch_size);
            forward_kernel<<<grid_f, block_f>>>(l, d_neurons, d_weights, 
                                              d_layers, d_neuron_offsets,
                                              d_weight_offsets, total_neurons,
                                              current_batch_size);
            cudaDeviceSynchronize();
        }
        
        // Backward pass: Output layer
        dim3 block_out(256);
        dim3 grid_out((current_batch_size * output_size + block_out.x - 1) / block_out.x);
        output_delta_kernel<<<grid_out, block_out>>>(d_deltas, d_neurons, d_batch_output,
                                                   d_layers, d_neuron_offsets,
                                                   output_size, total_neurons,
                                                   current_batch_size);
        cudaDeviceSynchronize();
        
        // Backward pass: Hidden layers
        for (int l = num_layers - 2; l > 0; --l) {
            dim3 block_h(256);
            dim3 grid_h((layers[l] + block_h.x - 1) / block_h.x, current_batch_size);
            hidden_delta_kernel<<<grid_h, block_h>>>(l, d_deltas, d_neurons, d_weights,
                                                   d_layers, d_neuron_offsets,
                                                   d_weight_offsets, total_neurons,
                                                   current_batch_size);
            cudaDeviceSynchronize();
        }
        
        // Accumulate gradients
        for (int l = 0; l < num_layers - 1; ++l) {
            dim3 block_acc(16, 16);
            dim3 grid_acc((layers[l+1] + block_acc.x - 1) / block_acc.x,
                        (layers[l] + 1 + block_acc.y - 1) / block_acc.y,
                        current_batch_size);
            accumulate_gradients_kernel<<<grid_acc, block_acc>>>(l, d_gradients, d_deltas, d_neurons,
                                                               d_layers, d_neuron_offsets,
                                                               d_weight_offsets, total_neurons,
                                                               current_batch_size);
            cudaDeviceSynchronize();
        }
        
        // Update weights
        dim3 block_up(256);
        dim3 grid_up((total_weights + block_up.x - 1) / block_up.x);
        update_weights_kernel<<<grid_up, block_up>>>(d_weights, d_gradients, 
                                                    total_weights, learningRate,
                                                    current_batch_size);
        cudaDeviceSynchronize();
    }
    
    // Copy final weights back to host
    cudaMemcpy(weights.data(), d_weights, total_weights * sizeof(double), 
              cudaMemcpyDeviceToHost);
    
    // Clean up
    cudaFree(d_batch_input);
    cudaFree(d_batch_output);
}
