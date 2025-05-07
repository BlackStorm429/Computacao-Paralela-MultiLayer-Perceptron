#pragma once
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "IMLP.h"
#include <algorithm>

// Tester class
class MLPTester {
    private:
        IMLP& mlp;
    
    public:

        static void splitTestTrain(const std::vector<std::vector<double>>& inputData, const std::vector<std::vector<double>>& outputData,
                                   std::vector<std::vector<double>>& inputTrain, std::vector<std::vector<double>>& outputTrain,
                                      std::vector<std::vector<double>>& inputTest, std::vector<std::vector<double>>& outputTest,
                                   double trainRatio) {

                                    {
                                        size_t total = inputData.size();
                                        if (total != outputData.size()) {
                                            std::cerr << "Input and output data size mismatch." << std::endl;
                                            return;
                                        }

                                        std::vector<size_t> indices(total);
                                        for (size_t i = 0; i < total; ++i) {
                                            indices[i] = i;
                                        }

                                        std::srand(static_cast<unsigned int>(std::time(nullptr)));
                                        std::random_shuffle(indices.begin(), indices.end());

                                        size_t trainSize = static_cast<size_t>(total * trainRatio);
                                        for (size_t i = 0; i < total; ++i) {
                                            if (i < trainSize) {
                                                inputTrain.push_back(inputData[indices[i]]);
                                                outputTrain.push_back(outputData[indices[i]]);
                                            } else {
                                                inputTest.push_back(inputData[indices[i]]);
                                                outputTest.push_back(outputData[indices[i]]);
                                            }
                                        }
                                    }
                                   }


        static void normalizeData(std::vector<std::vector<double>>& data) {
            for (size_t i = 0; i < data.size(); ++i) {
                double maxVal = *std::max_element(data[i].begin(), data[i].end());
                double minVal = *std::min_element(data[i].begin(), data[i].end());
                for (size_t j = 0; j < data[i].size(); ++j) {
                    data[i][j] = (data[i][j] - minVal) / (maxVal - minVal);
                }
            }
        }
    

        MLPTester(IMLP& mlpInstance)
            : mlp(mlpInstance) {}
    
        void train(int epochs, std::vector<std::vector<double>>& inputs, std::vector<std::vector<double>>& expectedOutputs) {
            for (int e = 0; e < epochs; ++e) {
                for (size_t i = 0; i < inputs.size(); ++i) {
                    mlp.forward(inputs[i]);
                    mlp.backward(expectedOutputs[i]);
                }
            }
        }

    
        float accuracy(std::vector<std::vector<double>>& testInputs,
                  const std::vector<std::vector<double>>& expectedOutputs) {

            int correct = 0;
            for (size_t i = 0; i < testInputs.size(); ++i) {
                std::vector<double> output = mlp.forward(testInputs[i]);
                if (output[0] > 0.5 && expectedOutputs[i][0] == 1.0) {
                    correct++;
                } else if (output[0] <= 0.5 && expectedOutputs[i][0] == 0.0) {
                    correct++;
                }
            }

            return static_cast<float>(correct) / testInputs.size();
        }

        
    };
    