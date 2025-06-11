#include "MLP_CUDA.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>

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

__global__ void zero_gradients_kernel(double* acc_grad, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) acc_grad[idx] = 0.0;
}

__global__ void copy_input_kernel(const double* flat_input,
                                  double*       neurons,
                                  int           sample_idx,
                                  int           input_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < input_size) {
        neurons[tid] = flat_input[sample_idx * input_size + tid];
    }
}

__global__ void forward_layer_kernel(const double* weights,
                                     const int*    layers,
                                     const int*    neuronOffsets,
                                     const int*    weightOffsets,
                                     double*       neurons,
                                     int           layer)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int prev_sz = layers[layer - 1];
    int curr_sz = layers[layer];
    if (j < curr_sz) {
        int off_prev = neuronOffsets[layer - 1];
        int off_curr = neuronOffsets[layer];
        int woff     = weightOffsets[layer - 1];
        double sum = weights[woff + j*(prev_sz+1) + prev_sz];  // bias
        for (int k = 0; k < prev_sz; ++k)
            sum += weights[woff + j*(prev_sz+1) + k] * neurons[off_prev + k];
        neurons[off_curr + j] = 1.0 / (1.0 + exp(-sum));
    }
}

__global__ void compute_output_deltas_kernel(const double* neurons,
                                             double*       deltas,
                                             const double* flat_output,
                                             int           sample_idx,
                                             int           output_size,
                                             const int*    neuronOffsets,
                                             int           num_layers)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int L = num_layers - 1;
    int off = neuronOffsets[L];
    if (j < output_size) {
        double out    = neurons[off + j];
        double target = flat_output[sample_idx * output_size + j];
        deltas[off + j] = (target - out) * out * (1.0 - out);
    }
}

__global__ void propagate_deltas_kernel(const double* weights,
                                        const int*    layers,
                                        const int*    neuronOffsets,
                                        const int*    weightOffsets,
                                        const double* deltas_in,
                                        const double* neurons,
                                        double*       deltas_out,
                                        int           layer)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int curr_sz = layers[layer];
    int next_sz = layers[layer + 1];
    if (j < curr_sz) {
        int off_curr = neuronOffsets[layer];
        int off_next = neuronOffsets[layer + 1];
        int woff     = weightOffsets[layer];
        double err = 0.0;
        for (int k = 0; k < next_sz; ++k)
            err += weights[woff + k*(curr_sz+1) + j] * deltas_in[off_next + k];
        double act = neurons[off_curr + j];
        deltas_out[off_curr + j] = err * act * (1.0 - act);
    }
}

__global__ void accumulate_gradients_kernel(const double* deltas,
                                            const double* neurons,
                                            double*       acc_grad,
                                            const int*    layers,
                                            const int*    neuronOffsets,
                                            const int*    weightOffsets,
                                            int           layer)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int from_sz = layers[layer];
    int to_sz   = layers[layer + 1];
    if (j < to_sz) {
        int off_from = neuronOffsets[layer];
        int off_to   = neuronOffsets[layer + 1];
        int woff     = weightOffsets[layer];
        double dv = deltas[off_to + j];
        for (int k = 0; k < from_sz; ++k) {
            atomicAdd(&acc_grad[woff + j*(from_sz+1) + k],
                      dv * neurons[off_from + k]);
        }
        atomicAdd(&acc_grad[woff + j*(from_sz+1) + from_sz], dv);
    }
}

__global__ void update_weights_kernel(double*       weights,
                                      const double* acc_grad,
                                      double        lr,
                                      int           batch_sz,
                                      size_t        W)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < W) {
        weights[idx] += lr * (acc_grad[idx] / batch_sz);
    }
}

// --------------------------------------
// MLP_CUDA methods
// --------------------------------------

MLP_CUDA::MLP_CUDA(const int* layerSizes,
                   int         batch_size,
                   double      lr,
                   double      acc_limit)
  : MLP(layerSizes, batch_size, lr), acc_limit(acc_limit)
{}

MLP_CUDA::MLP_CUDA(const MLP& mlp_base, const int acc_limit) : MLP(mlp_base), acc_limit(acc_limit) {}

void MLP_CUDA::train(const std::vector<std::vector<double>>& inputData,
                     const std::vector<std::vector<double>>& outputData)
{
    if (inputData.empty() || outputData.empty() ||
        inputData.size() != outputData.size())
    {
        throw std::invalid_argument("Input/output size mismatch");
    }

    size_t N    = inputData.size();
    size_t inZ  = layers[0];
    size_t outZ = layers.back();
    size_t L    = layers.size();
    size_t W    = weights.size();
    size_t T    = neurons.size();
    size_t D    = deltas.size();
    size_t G    = accumulated_gradients.size();

    std::vector<double> flat_in (N * inZ),
                        flat_out(N * outZ);
    for (size_t i = 0; i < N; ++i) {
        std::copy(inputData[i].begin(),  inputData[i].end(),  flat_in.begin()  + i*inZ);
        std::copy(outputData[i].begin(), outputData[i].end(), flat_out.begin() + i*outZ);
    }

    double *d_W, *d_X, *d_D, *d_G, *d_in, *d_out;
    int *d_Layers, *d_nOff, *d_wOff;

    CUDA_CHECK(cudaMalloc(&d_W,      W   * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_X,      T   * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_D,      D   * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_G,      G   * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Layers, L   * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nOff,   L   * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_wOff,   (L-1)* sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_in,   N*inZ * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out,  N*outZ * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_W,      weights.data(),       W   * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Layers, layers.data(),       L   * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nOff,   neuronOffsets.data(), L   * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wOff,   weightOffsets.data(), (L-1)* sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in,     flat_in.data(),      N*inZ * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out,    flat_out.data(),     N*outZ * sizeof(double), cudaMemcpyHostToDevice));

    const int BLOCK = 256;
    int grid;

    for (int epoch = 0; epoch < 5; ++epoch) {
        for (size_t start = 0; start < N; start += batchSize) {
            int bs = std::min(batchSize, int(N - start));

            // zero
            grid = (G + BLOCK - 1) / BLOCK;
            zero_gradients_kernel<<<grid,BLOCK>>>(d_G, G);
            CUDA_CHECK(cudaDeviceSynchronize());

            for (int s = start; s < start + bs; ++s) {
                grid = (inZ + BLOCK - 1) / BLOCK;
                copy_input_kernel<<<grid,BLOCK>>>(d_in, d_X, s, inZ);
                CUDA_CHECK(cudaDeviceSynchronize());

                for (int l = 1; l < L; ++l) {
                    grid = (layers[l] + BLOCK - 1) / BLOCK;
                    forward_layer_kernel<<<grid,BLOCK>>>
                      (d_W,d_Layers,d_nOff,d_wOff,d_X,l);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }

                grid = (outZ + BLOCK - 1) / BLOCK;
                compute_output_deltas_kernel<<<grid,BLOCK>>>
                  (d_X,d_D,d_out,s,outZ,d_nOff,(int)L);
                CUDA_CHECK(cudaDeviceSynchronize());

                for (int l = L-2; l > 0; --l) {
                    grid = (layers[l] + BLOCK - 1) / BLOCK;
                    propagate_deltas_kernel<<<grid,BLOCK>>>
                      (d_W,d_Layers,d_nOff,d_wOff,
                       d_D, d_X, d_D, l);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }

                for (int l = 0; l < L-1; ++l) {
                    grid = (layers[l+1] + BLOCK - 1) / BLOCK;
                    accumulate_gradients_kernel<<<grid,BLOCK>>>
                      (d_D,d_X,d_G,d_Layers,d_nOff,d_wOff,l);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
            }

            grid = (W + BLOCK - 1) / BLOCK;
            update_weights_kernel<<<grid,BLOCK>>>
              (d_W, d_G, learningRate, bs, W);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    CUDA_CHECK(cudaMemcpy(weights.data(), d_W,
                          W * sizeof(double),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_W);     cudaFree(d_X);
    cudaFree(d_D);     cudaFree(d_G);
    cudaFree(d_Layers);cudaFree(d_nOff);
    cudaFree(d_wOff);  cudaFree(d_in);
    cudaFree(d_out);
}
