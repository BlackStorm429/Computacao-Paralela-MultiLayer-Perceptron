#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <regex>
#include <numeric>
#include "mlp.h"
using namespace std;

const int MAX_EPOCHS = 100;
const double LEARNING_RATE = 0.01;

int mapStringToInt(const string& str, unordered_map<string,int>& dict, int& nextCode) {
    auto it = dict.find(str);
    if(it == dict.end()) dict[str] = nextCode++;
    return dict[str];
}

// Improved split function to handle special cases
vector<string> split(const string& line) {
    // First, normalize "No Info" to "No_Info" if present
    string normalized_line = regex_replace(line, regex("No Info"), "No_Info");
    
    vector<string> toks;
    istringstream ss(normalized_line);
    string tok;
    while(ss >> tok) toks.push_back(tok);
    return toks;
}

// Fixed serialization function
void serialize(const MultiLayerPerceptron& mlp, vector<double>& buf) {
    buf.clear();
    int L = mlp.GetLayerCount();
    for (int l = 1; l < L; ++l) {
        int N = mlp.GetLayerSize(l);
        int M = mlp.GetLayerSize(l - 1);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                buf.push_back(mlp.GetWeight(l, i, j));
            }
            // Add bias (which is implicitly 0 in this implementation)
            buf.push_back(0.0);
        }
    }
}

// Fixed deserialization function
void deserialize(MultiLayerPerceptron& mlp, const vector<double>& buf) {
    if (buf.empty()) return;
    
    int idx = 0;
    int L = mlp.GetLayerCount();
    for(int l = 1; l < L; ++l) {
        int N = mlp.GetLayerSize(l);
        int M = mlp.GetLayerSize(l - 1);
        for(int i = 0; i < N; ++i) {
            for(int j = 0; j < M; ++j) {
                if (idx < buf.size()) {
                    mlp.SetWeight(l, i, j, buf[idx++]);
                }
            }
            // Skip bias value (not explicitly used in this implementation)
            if (idx < buf.size()) idx++;
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Set fixed random seed for reproducibility
    if (rank == 0) {
        InitializeRandoms();
    }

    vector<vector<double>> X;
    vector<int> y;
    int n_samples = 0, n_feats = 0;
    
    // Read data from stdin on rank 0
    if (rank == 0) {
        string line;
        unordered_map<string, int> dict;
        int nextCode = 0;
        
        if (argc > 1) {
            // Read from file if specified
            ifstream infile(argv[1]);
            if (!infile.is_open()) {
                cerr << "Error: Could not open file " << argv[1] << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            while (getline(infile, line)) {
                auto toks = split(line);
                if (toks.size() < 2) continue;
                
                vector<double> sample;
                for (size_t i = 0; i + 1 < toks.size(); ++i) {
                    try { sample.push_back(stod(toks[i])); }
                    catch(...) { sample.push_back(mapStringToInt(toks[i], dict, nextCode)); }
                }
                
                X.push_back(sample);
                
                try { y.push_back(stoi(toks.back())); }
                catch(...) { y.push_back(mapStringToInt(toks.back(), dict, nextCode)); }
            }
        } else {
            // Read from stdin
            while (getline(cin, line)) {
                auto toks = split(line);
                if (toks.size() < 2) continue;
                
                vector<double> sample;
                for (size_t i = 0; i + 1 < toks.size(); ++i) {
                    try { sample.push_back(stod(toks[i])); }
                    catch(...) { sample.push_back(mapStringToInt(toks[i], dict, nextCode)); }
                }
                
                X.push_back(sample);
                
                try { y.push_back(stoi(toks.back())); }
                catch(...) { y.push_back(mapStringToInt(toks.back(), dict, nextCode)); }
            }
        }
        
        if (X.empty()) {
            cerr << "Error: No data was read" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        n_samples = X.size();
        n_feats = X[0].size();
        
        cout << "Data loaded: " << n_samples << " samples with " << n_feats << " features" << endl;
        
        // Print some stats about class distribution
        unordered_map<int, int> class_counts;
        for (int label : y) {
            class_counts[label]++;
        }
        cout << "Class distribution:" << endl;
        for (const auto& pair : class_counts) {
            cout << "  Class " << pair.first << ": " << pair.second << " samples" << endl;
        }
    }

    // Broadcast data dimensions to all processes
    MPI_Bcast(&n_samples, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n_feats, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Prepare for data broadcast
    vector<double> flat_X;
    if (rank == 0) {
        flat_X.resize(n_samples * n_feats);
        for(int i = 0; i < n_samples; ++i) {
            for(int j = 0; j < n_feats; ++j) {
                flat_X[i * n_feats + j] = X[i][j];
            }
        }
    } else {
        flat_X.resize(n_samples * n_feats);
        y.resize(n_samples);
    }

    // Broadcast data to all processes
    MPI_Bcast(flat_X.data(), n_samples * n_feats, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(y.data(), n_samples, MPI_INT, 0, MPI_COMM_WORLD);

    // Reconstruct X from flat_X on non-root processes
    if (rank != 0) {
        X.resize(n_samples);
        for (int i = 0; i < n_samples; ++i) {
            X[i].resize(n_feats);
            for (int j = 0; j < n_feats; ++j) {
                X[i][j] = flat_X[i * n_feats + j];
            }
        }
    }
    
    // Determine number of classes
    int n_classes = 0;
    for(int v : y) n_classes = max(n_classes, v + 1);
    
    // Create MLP model
    int topology[] = { n_feats, 24, n_classes };
    MultiLayerPerceptron mlp(3, topology);
    mlp.dEta = LEARNING_RATE;
    mlp.dAlpha = 0.9;
    mlp.dGain = 1.0;
    
    // Initialize weights on rank 0 and broadcast to all processes
    if (rank == 0) {
        mlp.RandomWeights();
    }
    
    vector<double> weights_buf;
    int weights_count = 0;
    
    if (rank == 0) {
        serialize(mlp, weights_buf);
        weights_count = weights_buf.size();
    }
    
    MPI_Bcast(&weights_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        weights_buf.resize(weights_count);
    }
    
    MPI_Bcast(weights_buf.data(), weights_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    deserialize(mlp, weights_buf);
    
    // Prepare for training
    double sum_acc = 0.0;
    int converged_at = MAX_EPOCHS;
    auto t0 = chrono::high_resolution_clock::now();
    
    // Create arrays for training
    vector<double> input(n_feats);
    vector<double> target(n_classes, 0.0);
    vector<double> output(n_classes);
    
    // Training loop
    for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
        // Create a thread-local copy of the MLP for training
        MultiLayerPerceptron local_mlp = mlp;
        
        // Each process handles a subset of samples
        for (int i = rank; i < n_samples; i += size) {
            for (int j = 0; j < n_feats; ++j) {
                input[j] = X[i][j];
            }
            
            fill(target.begin(), target.end(), 0.0);
            if (y[i] >= 0 && y[i] < n_classes) {
                target[y[i]] = 1.0;
            }
            
            local_mlp.Simulate(input.data(), output.data(), target.data(), true);
        }
        
        // Gather and average weights from all processes
        vector<double> local_weights;
        serialize(local_mlp, local_weights);
        
        vector<double> aggregated_weights(weights_count, 0.0);
        MPI_Allreduce(local_weights.data(), aggregated_weights.data(), weights_count, 
                     MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        // Average the weights
        for (auto &w : aggregated_weights) {
            w /= size;
        }
        
        // Update the model with averaged weights
        deserialize(mlp, aggregated_weights);
        
        // Calculate accuracy on rank 0
        if (rank == 0) {
            int correct = 0;
            for (int i = 0; i < n_samples; ++i) {
                for (int j = 0; j < n_feats; ++j) {
                    input[j] = X[i][j];
                }
                
                int predicted = mlp.Predict(input.data());
                if (predicted == y[i]) correct++;
            }
            
            double acc = 100.0 * correct / n_samples;
            sum_acc += acc;
            
            cout << "Epoch " << epoch + 1 << "/" << MAX_EPOCHS << ": Accuracy = " << acc << "%" << endl;
            
            if (acc >= 99.9 && converged_at == MAX_EPOCHS) {
                converged_at = epoch + 1;
                // No need to break here - we'll continue for all MAX_EPOCHS to maintain compatibility
            }
        }
        
        // Broadcast convergence status
        MPI_Bcast(&converged_at, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    auto t1 = chrono::high_resolution_clock::now();
    
    if (rank == 0) {
        double time_ms = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();
        double time_s = time_ms / 1000.0;
        double avg_acc = sum_acc / MAX_EPOCHS;
        
        cout << "Execution time: " << time_s << " seconds" << endl;
        cout << "Average accuracy across epochs: " << avg_acc << "%" << endl;
        cout << "Training converged after " << converged_at << " epochs" << endl;
        cout << "Configuration: " << size << " MPI processes" << endl;
        cout << "Total time (ms): " << time_ms << endl;
    }
    
    MPI_Finalize();
    return 0;
}