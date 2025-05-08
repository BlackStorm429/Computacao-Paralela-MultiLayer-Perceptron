#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "IMLP.h"
#include <algorithm>
#include <random>
#include <chrono>

// Tester class
class MLPTester {
    private:
        IMLP& mlp;
    
    public:

        MLPTester(IMLP& mlpInstance)
            : mlp(mlpInstance) {}
    
        void train(int epochs, std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& expectedOutputs) {
            auto start = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < epochs; ++i) {
                mlp.train(inputs, expectedOutputs);
                std::cout << "Epoch: " << i + 1 << " Accuracy: " << accuracy(inputs, expectedOutputs) * 100 << "%" << std::endl;
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "Training completed in " << duration << " ms" << std::endl;
        }

    
        float accuracy(std::vector<std::vector<double>>& testInputs,
                  const std::vector<std::vector<double>>& expectedOutputs) {

            int correct = 0;
            for (size_t i = 0; i < testInputs.size(); ++i) {
                std::vector<double> output = mlp.forward(testInputs[i]);
                if (output.size() == 1) { // Binary classification
                    if (output[0] > 0.5 && expectedOutputs[i][0] == 1.0) {
                        correct++;
                    } else if (output[0] <= 0.5 && expectedOutputs[i][0] == 0.0) {
                        correct++;
                    }
                } else { // Multi-class classification
                    int predicted = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
                    int expected = std::distance(expectedOutputs[i].begin(), std::max_element(expectedOutputs[i].begin(), expectedOutputs[i].end()));
                    if (predicted == expected) {
                        correct++;
                    }
                }
            }

            return static_cast<float>(correct) / testInputs.size();
        }

        
    };
    