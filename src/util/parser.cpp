#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <stdexcept>
#include <cctype>
#include <algorithm>
#include <iostream>
#include <string>
#include <random>
#include <ctime>

using namespace std;

static vector<vector<string>> CSVparser(const string& path) {
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

static void normalizeData(vector<vector<double>>& data) {
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

static void loadDB(const string& path, vector<vector<double>>& X, vector<vector<double>>& Y) {
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


static void splitTestTrain(const std::vector<std::vector<double>>& inputData, const std::vector<std::vector<double>>& outputData,
    std::vector<std::vector<double>>& inputTrain, std::vector<std::vector<double>>& outputTrain,
       std::vector<std::vector<double>>& inputTest, std::vector<std::vector<double>>& outputTest,
    double trainRatio) {

     
    size_t total = inputData.size();
    if (total != outputData.size()) {
        throw std::invalid_argument("Input and output data size mismatch.");
    }

    std::vector<size_t> indices(total);
    for (size_t i = 0; i < total; ++i) {
        indices[i] = i;
    }

    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    std::shuffle(indices.begin(), indices.end(), std::default_random_engine(std::rand()));

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
