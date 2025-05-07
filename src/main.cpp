#include "util/MLPTester.cpp"
#include "models/MLP.cpp"
#include "util/parser.cpp"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <stdexcept>
#include <cctype>
#include <algorithm>
#include <chrono>



using namespace std;



int main() {
    // Load the dataset
    std::vector<std::vector<double>> inputs, expectedOutputs;
    loadDB("dataset/diabetes_balanced.csv", inputs, expectedOutputs);
    
    int layers[] = {8, 6, 6, 1, 0}; // 0-terminated array

    // Create an instance of MLP
    MLP mlp(layers, 0.1);

    
    std::vector<std::vector<double>> Xtrain, Xtest, Ytrain, Ytest;

    // Split the data into training and testing sets
    splitTestTrain(inputs, expectedOutputs, Xtrain, Ytrain, Xtest, Ytest, 0.8);




    // Create a tester instance
    MLPTester tester(mlp);

    std::cout << "Initial Accuracy: " << tester.accuracy(Xtest, Ytest) << std::endl;

    tester.train(500, Xtrain, Ytrain);
    


    return 0;
}