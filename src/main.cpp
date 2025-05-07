#include "util/MLPTester.cpp"
#include "models/MLP.cpp"
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

vector<vector<string>> CSVparser(const string& path) {
    vector<vector<string>> data;
    ifstream file(path);
    string line;

    while (getline(file, line)) {
        vector<string> row;
        stringstream ss(line);
        string value;
        
        while (getline(ss, value, ',')) {
            row.push_back(value);
        }
        
        data.push_back(row);
    }
    
    return data;
}
void normalizeData(vector<vector<double>>& data) {
    int cols = data[0].size();
    for (int i = 0; i < cols; ++i) {
        double maxVal = data[0][i];
        double minVal = data[0][i];

        for (size_t j = 1; j < data.size(); ++j) {
            if (data[j][i] > maxVal) {
                maxVal = data[j][i];
            }
            if (data[j][i] < minVal) {
                minVal = data[j][i];
            }
        }

        if (maxVal == minVal) {
            cout << "Warning: All values in column " << i << " are the same. Skipping normalization." << endl;
            continue; // Avoid division by zero
        }

        for (size_t j = 0; j < data.size(); ++j) {
            data[j][i] = (data[j][i] - minVal) / (maxVal - minVal);
        }
    }
}

void loadDB(const string& path, vector<vector<double>>& X, vector<vector<double>>& Y) {
    vector<vector<string>> rawData = CSVparser(path);
    unordered_map<string, double> stringToDouble;
    double nextMapping = 0.0;

    // Remove the header
    if (!rawData.empty()) {
        rawData.erase(rawData.begin());
    }

    for (const auto& row : rawData) {
        vector<double> xRow, yRow;

        for (size_t i = 0; i < row.size(); ++i) {
            string val = row[i];
            double num;

            try {
                num = stod(val);
            } catch (...) {
                if (stringToDouble.find(val) == stringToDouble.end()) {
                    stringToDouble[val] = nextMapping++;
                }
                num = stringToDouble[val];
            }

            if (i < row.size() - 1)
                xRow.push_back(num);
            else {
                yRow.push_back(num);
            }
        }

        X.push_back(xRow);
        Y.push_back(yRow);
    }

    // Normalize the data
    normalizeData(X);
    normalizeData(Y);
}





int main() {
    // Load the dataset
    std::vector<std::vector<double>> inputs, expectedOutputs;
    loadDB("dataset/diabetes_balanced.csv", inputs, expectedOutputs);
    
    // Define the network structure
    int numInputs = inputs[0].size();
    int numOutputs = expectedOutputs[0].size();
    int layers[] = {numInputs, 6, 6, numOutputs, 0}; // 0-terminated array

    // Create an instance of MLP
    MLP mlp(layers);

    
    std::vector<std::vector<double>> Xtrain, Xtest, Ytrain, Ytest;

    // Split the data into training and testing sets
    MLPTester::splitTestTrain(inputs, expectedOutputs, Xtrain, Ytrain, Xtest, Ytest, 0.8);


    // Create a tester instance
    MLPTester tester(mlp);

    std::cout << "Initial Accuracy: " << tester.accuracy(Xtest, Ytest) << std::endl;

    // Train the MLP
    int epochs = 1000;
    for (int i = 0; i < epochs; i++) {
        //Start timer
        auto start = std::chrono::high_resolution_clock::now();
        tester.train(1, Xtrain, Ytrain);
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "Epoch " << i << " Accuracy: " << tester.accuracy(Xtest, Ytest);
        std::cout << " Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }


    return 0;
}