/*

    Versão OpenMP - Resultados de tempo (em milissegundos) no servidor parcode:

    Threads | Tempo (ms)
    --------|-----------
    1       | 1696
    2       | 1635
    4       | 1517
    8       | 1840

*/

#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <stdlib.h>
#include <random>
#include <omp.h>
#include "../mlp/mlp.h"
#include <string.h>


using namespace std;

const int MAX_EPOCHS = 100;

int mapStringToInt(const string& str, unordered_map<string,int>& dict, int& nextCode) {
    auto it = dict.find(str);
    if (it == dict.end()) dict[str] = nextCode++;
    return dict[str];
}

vector<string> split(const string& line) {
    vector<string> tokens;
    istringstream iss(line);
    string token;
    while (iss >> token) tokens.push_back(token);
    return tokens;
}

void shuffle_data(vector<vector<double>>& X, vector<int>& y) {
    vector<int> idx(X.size());
    iota(idx.begin(), idx.end(), 0);
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(idx.begin(), idx.end(), default_random_engine(seed));
    auto Xcopy = X;
    auto ycopy = y;
    for (size_t i = 0; i < idx.size(); ++i) {
        X[i] = Xcopy[idx[i]];
        y[i] = ycopy[idx[i]];
    }
}

void split_data(const vector<vector<double>>& X, const vector<int>& y,
                vector<vector<double>>& X_train, vector<int>& y_train,
                vector<vector<double>>& X_test,  vector<int>& y_test,
                double train_ratio = 0.8) {
    size_t n = X.size();
    size_t n_train = static_cast<size_t>(train_ratio * n);
    X_train.assign(X.begin(), X.begin() + n_train);
    y_train.assign(y.begin(), y.begin() + n_train);
    X_test.assign(X.begin() + n_train, X.end());
    y_test.assign(y.begin() + n_train, y.end());
}

void normalize_data(vector<vector<double>>& X_train, vector<vector<double>>& X_test) {
    size_t m = X_train[0].size();
    vector<double> mn(m, numeric_limits<double>::max());
    vector<double> mx(m, numeric_limits<double>::lowest());
    for (auto& sample : X_train) {
        for (size_t j = 0; j < m; ++j) {
            mn[j] = min(mn[j], sample[j]);
            mx[j] = max(mx[j], sample[j]);
        }
    }
    for (auto* set : {&X_train, &X_test}) {
        for (auto& sample : *set) {
            for (size_t j = 0; j < m; ++j) {
                if (mx[j] > mn[j])
                    sample[j] = (sample[j] - mn[j]) / (mx[j] - mn[j]);
                else sample[j] = 0.0;
            }
        }
    }
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
    if (argc < 2) {
        cerr << "Uso: " << argv[0] << " [num_threads] < data.txt" << endl;
        return 1;
    }
    int num_threads = stoi(argv[1]);
    omp_set_num_threads(num_threads);

    string line;
    vector<vector<double>> X;
    vector<int> y;
    unordered_map<string,int> dict;
    int nextCode = 0;
    while (getline(cin, line)) {
        auto toks = split(line);
        if (toks.size() < 2) continue;
        vector<double> sample;
        for (size_t i = 0; i + 1 < toks.size(); ++i) {
            try { sample.push_back(stod(toks[i])); }
            catch(...) { sample.push_back(mapStringToInt(toks[i], dict, nextCode)); }
        }
        try { y.push_back(stoi(toks.back())); }
        catch(...) { y.push_back(mapStringToInt(toks.back(), dict, nextCode)); }
        X.push_back(sample);
    }
    if (X.empty()) { cerr << "Sem dados!" << endl; return 1; }

    // size_t n = X.size();
    size_t dims = X[0].size();

    shuffle_data(X, y);
    vector<vector<double>> X_train, X_test;
    vector<int> y_train, y_test;
    split_data(X, y, X_train, y_train, X_test, y_test);

    int n_classes = *max_element(y.begin(), y.end()) + 1;
    int topology[] = { (int)dims, 24, n_classes };

    MultiLayerPerceptron central(3, topology);


    int converged_epoch = MAX_EPOCHS;
    double sum_acc = 0.0;
    
    auto t0 = chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < MAX_EPOCHS; ++epoch) {
        int hits = 0;
        vector<vector<double>> weights_buf(num_threads);

        #pragma omp parallel reduction(+:hits) shared(X_train,y_train)
        {
            vector<double> input(dims), output(n_classes), target(n_classes);
            int tid = omp_get_thread_num();
            MultiLayerPerceptron mlp(3, topology);
            mlp.RandomWeights();
            mlp.ScaleWeights(0.1 / num_threads);
            mlp.AddWeightsFrom(central);
            mlp.ScaleWeights(1.0 / (1.1 * num_threads));

            //int local_hits = 0;
            #pragma omp for schedule(static)
            for (int i = 0; i < (int)X_train.size(); ++i) {
                std::copy(X_train[i].begin(), X_train[i].end(), input.begin());
                std::fill(target.begin(), target.end(), 0.0);
                target[y_train[i]] = 1.0;
                mlp.Simulate(input.data(), output.data(), target.data(), true);
                
                int pred = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
                if (pred == y_train[i]) hits++;
            }
            serialize(mlp, weights_buf[tid]);
        }

        central.ScaleWeights(0);
        // Agregar réplicas e normalizar
        for (int t = 1; t < num_threads; ++t) {
            MultiLayerPerceptron replica(3, topology);
            deserialize(replica, weights_buf[t]);
            central.AddWeightsFrom(replica);
        }
        central.ScaleWeights(1.0 / num_threads);
        

        double acc = 100.0 * hits / X_train.size();
        cout << "Epoca " << epoch + 1 << "/" << MAX_EPOCHS << ": Acuracia = " << acc << "%" << endl;

        sum_acc += acc;
        if (acc >= 99.9) { converged_epoch = epoch + 1; break; }
    }

    auto t1 = chrono::high_resolution_clock::now();
    double elapsed = chrono::duration_cast<chrono::milliseconds>(t1 - t0).count();

    cout << "Media acuracia: " << (sum_acc / converged_epoch) << "%\n";
    cout << "Convergiu apos: " << converged_epoch << " epocas\n";
    cout << "Tempo de execucao (" << num_threads << " threads): " << elapsed << " ms\n";

 
    return 0;
}