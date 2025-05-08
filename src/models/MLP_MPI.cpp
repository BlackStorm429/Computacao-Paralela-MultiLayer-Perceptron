#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <mpi.h>
#include "IMLP.h"

class MLP_MPI : public IMLP {
private:
    std::vector<int> layers;
    std::vector<std::vector<double>> neurons;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> deltas;
    double learningRate = 0.1;
    
    // MPI variables
    int mpiRank;
    int mpiSize;
    bool mpiInitializedByUs = false;
    
    double activation(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    
    double activationDerivative(double x) {
        return x * (1 - x);
    }
    
public:
    MLP_MPI(const int* layerSizes, double learningRate = 0.1) 
        : learningRate(learningRate) {
        
        // Initialize MPI if not already initialized
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized) {
            int argc = 0;
            char** argv = nullptr;
            MPI_Init(&argc, &argv);
            mpiInitializedByUs = true;
        }
        
        MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
        
        if (mpiRank == 0) {
            std::cout << "MPI: Usando " << mpiSize << " processo(s)\n";
        }
        
        // Use the same seed across all processes for reproducibility
        unsigned int seed = 12345;
        if (mpiRank == 0) {
            seed = static_cast<unsigned int>(std::time(0));
        }
        // Broadcast the seed from rank 0 to all processes
        MPI_Bcast(&seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        std::srand(seed);
        
        for (int i = 0; layerSizes[i] != 0; ++i) {
            layers.push_back(layerSizes[i]);
            neurons.push_back(std::vector<double>(layerSizes[i], 0.0));
            deltas.push_back(std::vector<double>(layerSizes[i], 0.0));
        }
        
        // Initialize weights
        for (size_t i = 1; i < layers.size(); ++i) {
            weights.push_back(std::vector<std::vector<double>>(layers[i], std::vector<double>(layers[i - 1] + 1)));
            
            for (int j = 0; j < layers[i]; ++j) {
                for (int k = 0; k <= layers[i - 1]; ++k) {
                    weights[i - 1][j][k] = (double(std::rand()) / RAND_MAX - 0.5) * 2;
                }
            }
        }
        
        // Synchronize initial weights across all processes
        syncWeights();
    }
    
    ~MLP_MPI() {
        // Only finalize MPI if we initialized it
        if (mpiInitializedByUs) {
            int finalized;
            MPI_Finalized(&finalized);
            if (!finalized) {
                MPI_Finalize();
            }
        }
    }
    
    // Synchronize weights across all MPI processes
    void syncWeights() {
        if (mpiSize <= 1) {
            return; // No need to sync if only one process
        }
        
        for (size_t l = 0; l < weights.size(); ++l) {
            for (size_t i = 0; i < weights[l].size(); ++i) {
                size_t weightsCount = weights[l][i].size();
                
                if (mpiRank == 0) {
                    // Process 0 sends its weights to all other processes
                    for (int dest = 1; dest < mpiSize; ++dest) {
                        MPI_Send(weights[l][i].data(), weightsCount, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
                    }
                } else {
                    // Other processes receive weights from process 0
                    MPI_Recv(weights[l][i].data(), weightsCount, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }
        }
        
        // Ensure all processes have completed the sync
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    std::vector<double> forward(const std::vector<double>& input) {
        if ((int)input.size() != layers[0]) {
            throw std::invalid_argument("Input size does not match the first layer size.");
        }
        
        neurons[0] = input;
        
        for (int i = 1; i < (int)layers.size(); ++i) {
            for (int j = 0; j < layers[i]; ++j) {
                double sum = weights[i - 1][j][layers[i - 1]]; // Bias
                
                for (int k = 0; k < layers[i - 1]; ++k) {
                    sum += weights[i - 1][j][k] * neurons[i - 1][k];
                }
                
                neurons[i][j] = activation(sum);
            }
        }
        
        return neurons.back();
    }
    
    void backward(const std::vector<double>& target) {
        if ((int)target.size() != layers.back()) {
            throw std::invalid_argument("Target size does not match the output layer size.");
        }
        
        int L = layers.size() - 1;
        
        // Calculate output layer deltas
        for (int i = 0; i < layers[L]; ++i) {
            deltas[L][i] = (target[i] - neurons[L][i]) * activationDerivative(neurons[L][i]);
        }
        
        // Backpropagate error
        for (int l = L - 1; l > 0; --l) {
            for (int i = 0; i < layers[l]; ++i) {
                double error = 0.0;
                
                for (int j = 0; j < layers[l + 1]; ++j) {
                    error += weights[l][j][i] * deltas[l + 1][j];
                }
                
                deltas[l][i] = error * activationDerivative(neurons[l][i]);
            }
        }
        
        // Update weights
        for (size_t l = 1; l < layers.size(); ++l) {
            for (int i = 0; i < layers[l]; ++i) {
                for (int j = 0; j <= layers[l - 1]; ++j) {
                    double update = learningRate * deltas[l][i];
                    if (j < layers[l - 1]) {
                        update *= neurons[l - 1][j];
                    }
                    weights[l - 1][i][j] += update;
                }
            }
        }
    }
    
    void train(const std::vector<std::vector<double>>& inputData,
               const std::vector<std::vector<double>>& outputData) {
        if (inputData.size() != outputData.size()) {
            throw std::invalid_argument("Input and output data sizes do not match.");
        }
        
        // Process each sample sequentially
        for (size_t i = 0; i < inputData.size(); ++i) {
            forward(inputData[i]);
            backward(outputData[i]);
        }
        
        // Synchronize weights across all processes
        syncWeightsAllReduce();
    }
    
    // More efficient weight synchronization using MPI_Allreduce
    void syncWeightsAllReduce() {
        if (mpiSize <= 1) {
            return; // No need to sync if only one process
        }
        
        for (size_t l = 0; l < weights.size(); ++l) {
            for (size_t i = 0; i < weights[l].size(); ++i) {
                size_t weightsCount = weights[l][i].size();
                std::vector<double> tempWeights(weightsCount);
                
                // Sum all weights across processes and distribute the result
                MPI_Allreduce(weights[l][i].data(), tempWeights.data(), weightsCount, 
                             MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                
                // Average the weights
                for (size_t j = 0; j < weightsCount; ++j) {
                    weights[l][i][j] = tempWeights[j] / mpiSize;
                }
            }
        }
    }
    
    // Distribute data across processes and train in parallel with OpenMP within each process
    void trainParallel(const std::vector<std::vector<double>>& inputData,
                       const std::vector<std::vector<double>>& outputData) {
        if (inputData.size() != outputData.size()) {
            throw std::invalid_argument("Input and output data sizes do not match.");
        }
        
        int numSamples = inputData.size();
        
        // Calculate local sample range for each process
        int samplesPerProcess = numSamples / mpiSize;
        int startIdx = mpiRank * samplesPerProcess;
        int endIdx = (mpiRank == mpiSize - 1) ? numSamples : startIdx + samplesPerProcess;
        
        // Create local weight updates
        std::vector<std::vector<std::vector<double>>> localWeightUpdates;
        for (size_t l = 0; l < weights.size(); ++l) {
            std::vector<std::vector<double>> layerUpdates;
            for (size_t i = 0; i < weights[l].size(); ++i) {
                layerUpdates.push_back(std::vector<double>(weights[l][i].size(), 0.0));
            }
            localWeightUpdates.push_back(layerUpdates);
        }
        
        // Process local samples using OpenMP for within-process parallelism
        #pragma omp parallel
        {
            // Each thread needs its own copies of neurons and deltas
            std::vector<std::vector<double>> threadNeurons = neurons;
            std::vector<std::vector<double>> threadDeltas = deltas;
            
            // Thread-local weight updates
            std::vector<std::vector<std::vector<double>>> threadWeightUpdates;
            for (size_t l = 0; l < weights.size(); ++l) {
                std::vector<std::vector<double>> layerUpdates;
                for (size_t i = 0; i < weights[l].size(); ++i) {
                    layerUpdates.push_back(std::vector<double>(weights[l][i].size(), 0.0));
                }
                threadWeightUpdates.push_back(layerUpdates);
            }
            
            // Parallel processing of samples
            #pragma omp for
            for (int sample = startIdx; sample < endIdx; ++sample) {
                // Forward pass with thread-local data
                threadNeurons[0] = inputData[sample];
                
                for (int i = 1; i < (int)layers.size(); ++i) {
                    for (int j = 0; j < layers[i]; ++j) {
                        double sum = weights[i - 1][j][layers[i - 1]]; // Bias
                        
                        for (int k = 0; k < layers[i - 1]; ++k) {
                            sum += weights[i - 1][j][k] * threadNeurons[i - 1][k];
                        }
                        
                        threadNeurons[i][j] = activation(sum);
                    }
                }
                
                // Backward pass with thread-local data
                int L = layers.size() - 1;
                
                // Output layer deltas
                for (int i = 0; i < layers[L]; ++i) {
                    threadDeltas[L][i] = (outputData[sample][i] - threadNeurons[L][i]) * 
                                      activationDerivative(threadNeurons[L][i]);
                }
                
                // Hidden layers
                for (int l = L - 1; l > 0; --l) {
                    for (int i = 0; i < layers[l]; ++i) {
                        double error = 0.0;
                        
                        for (int j = 0; j < layers[l+1]; ++j) {
                            error += weights[l][j][i] * threadDeltas[l+1][j];
                        }
                        
                        threadDeltas[l][i] = error * activationDerivative(threadNeurons[l][i]);
                    }
                }
                
                // Accumulate weight updates in thread-local structures
                for (size_t l = 1; l < layers.size(); ++l) {
                    for (int i = 0; i < layers[l]; ++i) {
                        for (int j = 0; j < layers[l-1]; ++j) {
                            threadWeightUpdates[l-1][i][j] += learningRate * threadDeltas[l][i] * threadNeurons[l-1][j];
                        }
                        // Bias
                        threadWeightUpdates[l-1][i][layers[l-1]] += learningRate * threadDeltas[l][i];
                    }
                }
            }
            
            // Combine thread-local updates into process-local updates
            #pragma omp critical
            {
                for (size_t l = 0; l < weights.size(); ++l) {
                    for (size_t i = 0; i < weights[l].size(); ++i) {
                        for (size_t j = 0; j < weights[l][i].size(); ++j) {
                            localWeightUpdates[l][i][j] += threadWeightUpdates[l][i][j];
                        }
                    }
                }
            }
        }
        
        // Synchronize and apply weight updates across all MPI processes
        for (size_t l = 0; l < weights.size(); ++l) {
            for (size_t i = 0; i < weights[l].size(); ++i) {
                size_t weightsCount = weights[l][i].size();
                std::vector<double> globalUpdates(weightsCount);
                
                // Sum all weight updates across processes
                MPI_Allreduce(localWeightUpdates[l][i].data(), globalUpdates.data(), 
                              weightsCount, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                
                // Apply the updates
                for (size_t j = 0; j < weightsCount; ++j) {
                    weights[l][i][j] += globalUpdates[j];
                }
            }
        }
    }
    
    // Train for multiple epochs with data distribution
    void train(const std::vector<std::vector<double>>& inputData,
               const std::vector<std::vector<double>>& outputData,
               int epochs) {
        for (int e = 0; e < epochs; ++e) {
            // Use parallel training for better performance
            trainParallel(inputData, outputData);
            
            // Only rank 0 prints progress
            if (mpiRank == 0 && e % 10 == 0) {
                std::cout << "Época " << e << " concluída." << std::endl;
            }
        }
    }
};