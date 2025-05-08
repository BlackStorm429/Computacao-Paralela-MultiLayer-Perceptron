#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <omp.h>
#include "IMLP.h"

class MLP_OpenMP : public IMLP {
private:
    std::vector<int> layers;
    std::vector<std::vector<double>> neurons;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> deltas;
    double learningRate = 0.1;
    
    double activation(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    
    double activationDerivative(double x) {
        return x * (1 - x);
    }
    
public:
    MLP_OpenMP(const int* layerSizes, double learningRate = 0.1) 
        : learningRate(learningRate) {
        
        std::srand(static_cast<unsigned int>(std::time(0)));
        
        for (int i = 0; layerSizes[i] != 0; ++i) {
            layers.push_back(layerSizes[i]);
            neurons.push_back(std::vector<double>(layerSizes[i], 0.0));
            deltas.push_back(std::vector<double>(layerSizes[i], 0.0));
        }
        
        // Inicializar pesos - usando um único seed para garantir reprodutibilidade
        unsigned int seed = static_cast<unsigned int>(std::time(0));
        for (size_t i = 1; i < layers.size(); ++i) {
            weights.push_back(std::vector<std::vector<double>>(layers[i], std::vector<double>(layers[i - 1] + 1)));
            
            for (int j = 0; j < layers[i]; ++j) {
                for (int k = 0; k <= layers[i - 1]; ++k) {
                    // Usar o mesmo gerador para todos os threads
                    std::srand(seed + j * layers[i - 1] + k);
                    weights[i - 1][j][k] = (double(std::rand()) / RAND_MAX - 0.5) * 2;
                }
            }
        }
    }
    
    std::vector<double> forward(const std::vector<double>& input) {
        if ((int)input.size() != layers[0]) {
            throw std::invalid_argument("Input size does not match the first layer size.");
        }
        
        neurons[0] = input;
        
        for (int i = 1; i < (int)layers.size(); ++i) {
            #pragma omp parallel for
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
        
        // Cálculo dos deltas na camada de saída
        #pragma omp parallel for
        for (int i = 0; i < layers[L]; ++i) {
            deltas[L][i] = (target[i] - neurons[L][i]) * activationDerivative(neurons[L][i]);
        }
        
        // Propagação do erro para camadas anteriores
        for (int l = L - 1; l > 0; --l) {
            #pragma omp parallel for
            for (int i = 0; i < layers[l]; ++i) {
                double error = 0.0;
                
                for (int j = 0; j < layers[l + 1]; ++j) {
                    error += weights[l][j][i] * deltas[l + 1][j];
                }
                
                deltas[l][i] = error * activationDerivative(neurons[l][i]);
            }
        }
        
        // Atualização dos pesos
        for (size_t l = 1; l < layers.size(); ++l) {
            #pragma omp parallel for collapse(2)
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
        
        // Processar cada amostra sequencialmente, mas com paralelismo interno
        for (size_t i = 0; i < inputData.size(); ++i) {
            forward(inputData[i]);
            backward(outputData[i]);
        }
    }
    
    // Versão paralela da função train, otimizada para mini-batches
    void trainParallel(const std::vector<std::vector<double>>& inputData,
                   const std::vector<std::vector<double>>& outputData) {
        if (inputData.size() != outputData.size()) {
            throw std::invalid_argument("Input and output data sizes do not match.");
        }
        
        int numSamples = inputData.size();
        int numThreads = omp_get_max_threads();
        
        // Paralelizar por mini-lotes
        #pragma omp parallel
        {
            // Obter ID do thread e total de threads
            int threadID = omp_get_thread_num();
            int numThreads = omp_get_num_threads();
            
            // Cópias locais para cada thread
            std::vector<std::vector<double>> localNeurons = neurons;
            std::vector<std::vector<double>> localDeltas = deltas;
            std::vector<std::vector<std::vector<double>>> localWeightUpdates;
            
            // Inicializar matriz de atualizações de pesos locais
            for (size_t l = 0; l < weights.size(); ++l) {
                std::vector<std::vector<double>> layerUpdates;
                for (size_t i = 0; i < weights[l].size(); ++i) {
                    layerUpdates.push_back(std::vector<double>(weights[l][i].size(), 0.0));
                }
                localWeightUpdates.push_back(layerUpdates);
            }
            
            // Cada thread processa seu próprio conjunto de amostras
            #pragma omp for
            for (int sample = 0; sample < numSamples; ++sample) {
                // Forward pass com cálculos locais
                localNeurons[0] = inputData[sample];
                
                for (size_t l = 1; l < layers.size(); ++l) {
                    for (int i = 0; i < layers[l]; ++i) {
                        double sum = weights[l-1][i][layers[l-1]]; // Bias
                        
                        for (int j = 0; j < layers[l-1]; ++j) {
                            sum += weights[l-1][i][j] * localNeurons[l-1][j];
                        }
                        
                        localNeurons[l][i] = activation(sum);
                    }
                }
                
                // Backward pass para calcular gradientes
                int L = layers.size() - 1;
                
                // Camada de saída
                for (int i = 0; i < layers[L]; ++i) {
                    localDeltas[L][i] = (outputData[sample][i] - localNeurons[L][i]) * 
                                      activationDerivative(localNeurons[L][i]);
                }
                
                // Camadas ocultas
                for (int l = L - 1; l > 0; --l) {
                    for (int i = 0; i < layers[l]; ++i) {
                        double error = 0.0;
                        
                        for (int j = 0; j < layers[l+1]; ++j) {
                            error += weights[l][j][i] * localDeltas[l+1][j];
                        }
                        
                        localDeltas[l][i] = error * activationDerivative(localNeurons[l][i]);
                    }
                }
                
                // Acumular atualizações de pesos localmente
                for (size_t l = 1; l < layers.size(); ++l) {
                    for (int i = 0; i < layers[l]; ++i) {
                        for (int j = 0; j < layers[l-1]; ++j) {
                            localWeightUpdates[l-1][i][j] += learningRate * localDeltas[l][i] * localNeurons[l-1][j];
                        }
                        // Bias
                        localWeightUpdates[l-1][i][layers[l-1]] += learningRate * localDeltas[l][i];
                    }
                }
            }
            
            // Aplicar atualizações de peso de forma sincronizada
            #pragma omp critical
            {
                for (size_t l = 0; l < weights.size(); ++l) {
                    for (size_t i = 0; i < weights[l].size(); ++i) {
                        for (size_t j = 0; j < weights[l][i].size(); ++j) {
                            weights[l][i][j] += localWeightUpdates[l][i][j];
                        }
                    }
                }
            }
        }
    }
    
    // Sobrecarga do método train para permitir múltiplas épocas
    void train(const std::vector<std::vector<double>>& inputData,
               const std::vector<std::vector<double>>& outputData,
               int epochs) {
        for (int e = 0; e < epochs; ++e) {
            // Usar o treinamento paralelo para melhor desempenho
            trainParallel(inputData, outputData);
            
            // Imprimir progresso a cada 10 épocas
            if (e % 10 == 0) {
                std::cout << "Época " << e << " concluída." << std::endl;
            }
        }
    }
};