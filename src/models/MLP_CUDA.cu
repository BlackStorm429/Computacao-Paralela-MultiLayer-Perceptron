#include "MLP_CUDA.h"

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
// Custom atomicAdd for double for older architectures
__device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

// Macro for checking CUDA errors
#define CUDA_CHECK(err) \
    if ((err) != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " code=" << err << " \"" << cudaGetErrorString(err) << "\"" \
                  << std::endl; \
        exit(EXIT_FAILURE); \
    }

// --------------------------------------
// Kernels (all __global__)
// --------------------------------------

__global__ void train_batch_process_kernel(
    const double* weights,
    const int* layers,
    const int* neuronOffsets,       // Offsets for a single sample's neuron array
    const int* weightOffsets,
    double* all_neurons,            // All neuron activations for the current batch
    double* all_deltas,             // All deltas for the current batch
    double* acc_grad,               // Accumulated gradients (shared across all samples in batch)
    const double* flat_input,       // Full flattened input dataset
    const double* flat_output,      // Full flattened output dataset
    int num_layers,
    int total_neurons_per_sample,
    int total_deltas_per_sample,
    int input_size,
    int output_size,
    int current_batch_size,
    int batch_start_idx)
{
   
    int sample_idx_in_batch = blockIdx.x;
    if (sample_idx_in_batch >= current_batch_size) return;

    
    int global_sample_idx = batch_start_idx + sample_idx_in_batch;

    
    double* current_neurons = all_neurons + sample_idx_in_batch * total_neurons_per_sample;
    double* current_deltas = all_deltas + sample_idx_in_batch * total_deltas_per_sample;

    
    for (int i = threadIdx.x; i < input_size; i += blockDim.x) {
        current_neurons[neuronOffsets[0] + i] = flat_input[global_sample_idx * input_size + i];
        printf("Input neuron %d for sample %d: %f\n", i, global_sample_idx, current_neurons[neuronOffsets[0] + i]);
    }
    __syncthreads(); 

  
    for (int l = 1; l < num_layers; ++l) {
        int prev_sz = layers[l - 1]; 
        int curr_sz = layers[l];     
        int off_prev = neuronOffsets[l - 1]; 
        int off_curr = neuronOffsets[l];     
        int woff = weightOffsets[l - 1];     

      
        for (int j = threadIdx.x; j < curr_sz; j += blockDim.x) {
            double sum = weights[woff + j * (prev_sz + 1) + prev_sz];
            for (int k = 0; k < prev_sz; ++k) {
                sum += weights[woff + j * (prev_sz + 1) + k] * current_neurons[off_prev + k];
            }
            // Sigmoid activation function
            current_neurons[off_curr + j] = 1.0 / (1.0 + exp(-sum));
        }
        __syncthreads(); 
    }

   
    int output_layer_idx = num_layers - 1;
    int output_layer_offset = neuronOffsets[output_layer_idx];

    for (int j = threadIdx.x; j < output_size; j += blockDim.x) {
        double out_neuron = current_neurons[output_layer_offset + j];
        double target_value = flat_output[global_sample_idx * output_size + j];
        // Delta = (target - output) * output * (1 - output) (for sigmoid)
        current_deltas[output_layer_offset + j] = (target_value - out_neuron) * out_neuron * (1.0 - out_neuron);
    }
    __syncthreads(); // Ensure all output deltas are computed before backpropagation starts

   
    for (int l = num_layers - 2; l > 0; --l) {
        int curr_sz = layers[l];     
        int next_sz = layers[l + 1]; 
        int off_curr = neuronOffsets[l];     
        int off_next = neuronOffsets[l + 1]; 
        int woff = weightOffsets[l];         

        
        for (int j = threadIdx.x; j < curr_sz; j += blockDim.x) {
            double error_sum = 0.0;
            for (int k = 0; k < next_sz; ++k) {
                
                error_sum += weights[woff + k * (curr_sz + 1) + j] * current_deltas[off_next + k];
            }
            double activation = current_neurons[off_curr + j];
            
            current_deltas[off_curr + j] = error_sum * activation * (1.0 - activation);
        }
        __syncthreads(); 
    }

    
    for (int l = 0; l < num_layers - 1; ++l) {
        int from_sz = layers[l];    
        int to_sz = layers[l + 1];  
        int off_from = neuronOffsets[l];     
        int off_to = neuronOffsets[l + 1];   
        int woff = weightOffsets[l];         

        
        for (int j = threadIdx.x; j < to_sz; j += blockDim.x) {
            double delta_val = current_deltas[off_to + j];
            
            for (int k = 0; k < from_sz; ++k) {
                atomicAdd(&acc_grad[woff + j * (from_sz + 1) + k],
                          delta_val * current_neurons[off_from + k]);
            }
           
            atomicAdd(&acc_grad[woff + j * (from_sz + 1) + from_sz], delta_val);
        }
    }
}


__global__ void update_weights_kernel(double* weights,
                                      const double* acc_grad,
                                      double lr,
                                      int batch_sz,
                                      size_t W)
{
    // Each thread updates a portion of the weights
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < W; i += blockDim.x * gridDim.x) {
        weights[i] += lr * (acc_grad[i] / batch_sz);
    }
}

// --------------------------------------
// MLP_CUDA methods
// --------------------------------------

MLP_CUDA::MLP_CUDA(const int* layerSizes,
                   int batch_size,
                   double lr)
    : MLP(layerSizes, batch_size, lr)
{}

MLP_CUDA::MLP_CUDA(const MLP& mlp_base) : MLP(mlp_base) {}

void MLP_CUDA::train(const std::vector<std::vector<double>>& inputData,
                     const std::vector<std::vector<double>>& outputData)
{
    if (inputData.empty() || outputData.empty() ||
        inputData.size() != outputData.size())
    {
        throw std::invalid_argument("Input/output size mismatch");
    }

    size_t N = inputData.size(); // Total number of samples
    size_t inZ = layers[0];     // Input layer size
    size_t outZ = layers.back(); // Output layer size
    size_t L = layers.size();    // Number of layers
    size_t W = weights.size();   // Total number of weights
    size_t T_per_sample = neurons.size(); // Total neurons for one sample
    size_t D_per_sample = deltas.size();  // Total deltas for one sample
    size_t G = accumulated_gradients.size(); // Total accumulated gradients size

    // Flatten input and output data for GPU transfer
    std::vector<double> flat_in(N * inZ);
    std::vector<double> flat_out(N * outZ);
    for (size_t i = 0; i < N; ++i) {
        std::copy(inputData[i].begin(), inputData[i].end(), flat_in.begin() + i * inZ);
        std::copy(outputData[i].begin(), outputData[i].end(), flat_out.begin() + i * outZ);
    }

   
    double *d_W, *d_all_neurons, *d_all_deltas, *d_G, *d_flat_in, *d_flat_out;
    int *d_Layers, *d_nOff, *d_wOff;

   
    CUDA_CHECK(cudaMalloc(&d_W,          W * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_all_neurons, batchSize * T_per_sample * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_all_deltas,  batchSize * D_per_sample * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_G,          G * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Layers,     L * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nOff,       L * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_wOff,       (L - 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_flat_in,    N * inZ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_flat_out,   N * outZ * sizeof(double)));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_W,          weights.data(),           W * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Layers,     layers.data(),            L * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nOff,       neuronOffsets.data(),     L * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wOff,       weightOffsets.data(),     (L - 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flat_in,    flat_in.data(),           N * inZ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_flat_out,   flat_out.data(),          N * outZ * sizeof(double), cudaMemcpyHostToDevice));

    const int BLOCK_SIZE = 256; // Standard block size

    for (int epoch = 0; epoch < 5; ++epoch) { // Loop over training epochs
        for (size_t start = 0; start < N; start += batchSize) { // Loop over mini-batches
            int current_batch_sz = std::min(batchSize, int(N - start));

            // Zero out accumulated gradients before processing a new batch
            int zero_grid = (G + BLOCK_SIZE - 1) / BLOCK_SIZE;
            cudaMemset(d_G, 0, G * sizeof(double));
            CUDA_CHECK(cudaDeviceSynchronize());

            int batch_grid = current_batch_sz; // Each block handles one sample
            train_batch_process_kernel<<<batch_grid, BLOCK_SIZE>>>(
                d_W, d_Layers, d_nOff, d_wOff,
                d_all_neurons, d_all_deltas, d_G,
                d_flat_in, d_flat_out,
                (int)L, (int)T_per_sample, (int)D_per_sample,
                (int)inZ, (int)outZ,
                current_batch_sz, (int)start
            );
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
            }
            CUDA_CHECK(cudaDeviceSynchronize()); 

       
            int update_grid = (W + BLOCK_SIZE - 1) / BLOCK_SIZE;
            update_weights_kernel<<<update_grid, BLOCK_SIZE>>>(
                d_W, d_G, learningRate, current_batch_sz, W
            );
            CUDA_CHECK(cudaDeviceSynchronize()); 
        }
    }

   
    CUDA_CHECK(cudaMemcpy(weights.data(), d_W,
                          W * sizeof(double),
                          cudaMemcpyDeviceToHost));

   
    cudaFree(d_W);
    cudaFree(d_all_neurons);
    cudaFree(d_all_deltas);
    cudaFree(d_G);
    cudaFree(d_Layers);
    cudaFree(d_nOff);
    cudaFree(d_wOff);
    cudaFree(d_flat_in);
    cudaFree(d_flat_out);
}
