#pragma once

#include "MLP.cpp"
#include <cstring>
#include <vector>
#include <cmath>
#include <omp.h>
#include <iostream>

// GPU-accelerated variant using OpenMP offloading
class MLP_OpenMP_GPU : public MLP {
private:
    int num_threads_;
    double learningRate_;
    std::vector<std::vector<double>> weights_;  // per-layer storage copied from base

public:
    MLP_OpenMP_GPU(const MLP &base_model, int num_threads)
        : MLP(base_model),
          num_threads_(num_threads),
          learningRate_(MLP::learningRate)
    {
        int layers_count = layers.size();
        weights_.reserve(layers_count - 1);
        for (int l = 0; l < layers_count - 1; ++l) {
            int in = layers[l];
            int out = layers[l + 1];
            int offset = weightOffsets[l];
            int stride = in + 1;
            weights_.emplace_back(stride * out);
            std::memcpy(weights_[l].data(), weights.data() + offset,
                        sizeof(double) * stride * out);
        }
        omp_set_num_threads(num_threads_);
    }

    // Override forward to use GPU-offloaded weights_
    std::vector<double> forward(const std::vector<double>& input) override {
        int L = static_cast<int>(layers.size()) - 1;
        std::vector<std::vector<double>> activ(layers.size());
        activ[0] = input;
        for (int l = 0; l < L; ++l) {
            int in_size = layers[l];
            int out_size = layers[l + 1];
            const double* Wl = weights_[l].data();
            activ[l+1].assign(out_size, 0.0);
            const double* in_ptr = activ[l].data();
            double* out_ptr = activ[l+1].data();
            // Offload dot-products
            #pragma omp target teams distribute parallel for \
                map(to: Wl[0:(in_size+1)*out_size], in_ptr[0:in_size]) \
                map(from: out_ptr[0:out_size])
            for (int j = 0; j < out_size; ++j) {
                double sum = Wl[j*(in_size+1) + in_size];
                for (int k = 0; k < in_size; ++k)
                    sum += Wl[j*(in_size+1) + k] * in_ptr[k];
                out_ptr[j] = 1.0 / (1.0 + std::exp(-sum));
            }
        }
        return activ.back();
    }

    // Train one epoch: update weights_ then sync back
    void train(const std::vector<std::vector<double>> &X,
               const std::vector<std::vector<double>> &Y) override
    {
        int N = static_cast<int>(X.size());
        int L = static_cast<int>(layers.size()) - 1;
        // Simple SGD per sample
        for (int i = 0; i < N; ++i) {
            // Forward
            std::vector<std::vector<double>> activ(layers.size());
            activ[0] = X[i];
            for (int l = 0; l < L; ++l) {
                int in_size = layers[l];
                int out_size = layers[l + 1];
                const double* Wl = weights_[l].data();
                activ[l+1].assign(out_size, 0.0);
                const double* in_ptr = activ[l].data();
                double* out_ptr = activ[l+1].data();
                #pragma omp target teams distribute parallel for \
                    map(to: Wl[0:(in_size+1)*out_size], in_ptr[0:in_size]) \
                    map(from: out_ptr[0:out_size])
                for (int j = 0; j < out_size; ++j) {
                    double sum = Wl[j*(in_size+1) + in_size];
                    for (int k = 0; k < in_size; ++k)
                        sum += Wl[j*(in_size+1) + k] * in_ptr[k];
                    out_ptr[j] = 1.0 / (1.0 + std::exp(-sum));
                }
            }
            // Backward
            std::vector<std::vector<double>> deltas(layers.size());
            deltas[L].resize(layers[L]);
            for (int j = 0; j < layers[L]; ++j) {
                double outv = activ[L][j];
                deltas[L][j] = (outv - Y[i][j]) * outv * (1 - outv);
            }
            for (int l = L; l > 0; --l) {
                int in_size = layers[l - 1];
                int out_size = layers[l];
                double* Wl = weights_[l - 1].data();
                double* prev_ptr = activ[l - 1].data();
                double* delta_ptr = deltas[l].data();
                #pragma omp target teams distribute parallel for \
                    map(tofrom: Wl[0:(in_size+1)*out_size], prev_ptr[0:in_size], delta_ptr[0:out_size])
                for (int j = 0; j < out_size; ++j) {
                    // update bias
                    Wl[j*(in_size+1) + in_size] -= learningRate_ * delta_ptr[j];
                    for (int k = 0; k < in_size; ++k) {
                        Wl[j*(in_size+1) + k] -= learningRate_ * delta_ptr[j] * prev_ptr[k];
                    }
                }
                if (l - 1 > 0) {
                    deltas[l-1].assign(layers[l-1], 0.0);
                    for (int k = 0; k < layers[l-1]; ++k) {
                        double sum = 0;
                        for (int j = 0; j < out_size; ++j) {
                            sum += Wl[j*(in_size+1) + k] * delta_ptr[j];
                        }
                        double act = activ[l-1][k];
                        deltas[l-1][k] = sum * act * (1 - act);
                    }
                }
            }
        }
        // Sync back to base-class weights
        int layers_count = layers.size();
        for (int l = 0; l < layers_count - 1; ++l) {
            int in_size = layers[l];
            int out_size = layers[l+1];
            int offset = weightOffsets[l];
            int stride = in_size + 1;
            std::memcpy(weights.data() + offset,
                        weights_[l].data(),
                        sizeof(double) * stride * out_size);
        }
    }
};