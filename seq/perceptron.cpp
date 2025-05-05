#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>
#include "../mlp/mlp.h"  // Inclusão da biblioteca MLP

using namespace std;

const int MAX_EPOCHS = 100;
const double LEARNING_RATE = 0.01;

// Função para mapear strings para inteiros
int mapStringToInt(const string& str, unordered_map<string, int>& dict, int& nextCode) {
    if (dict.find(str) == dict.end()) {
        dict[str] = nextCode++;
    }
    return dict[str];
}

// Embaralha X e y em conjunto
void shuffle_data(vector<vector<double>>& X, vector<int>& y) {
    if (X.size() != y.size()) {
        cerr << "Erro: tamanhos diferentes entre X e y." << endl;
        return;
    }

    vector<pair<vector<double>, int>> combined(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        combined[i] = make_pair(X[i], y[i]);
    }

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(combined.begin(), combined.end(), default_random_engine(seed));

    for (size_t i = 0; i < X.size(); ++i) {
        X[i] = combined[i].first;
        y[i] = combined[i].second;
    }
}

// Função para separar os dados de treino e teste
void split_data(const vector<vector<double>>& X, const vector<int>& y,
                vector<vector<double>>& X_train, vector<int>& y_train,
                vector<vector<double>>& X_test, vector<int>& y_test,
                double treino_ratio = 0.8) {
    size_t n_amostras = X.size();
    size_t n_treino = static_cast<size_t>(treino_ratio * n_amostras);

    X_train.assign(X.begin(), X.begin() + n_treino);
    y_train.assign(y.begin(), y.begin() + n_treino);
    X_test.assign(X.begin() + n_treino, X.end());
    y_test.assign(y.begin() + n_treino, y.end());
}

// Função para dividir os dados
vector<string> split(const string& line) {
    vector<string> tokens;
    istringstream iss(line);
    string token;
    while (iss >> token) tokens.push_back(token);
    return tokens;
}

// Normalizar os dados Min-Max
void normalize_data(vector<vector<double>>& X_train, vector<vector<double>>& X_test) {
    size_t n_atributos = X_train[0].size();

    vector<double> min_vals(n_atributos, numeric_limits<double>::max());
    vector<double> max_vals(n_atributos, numeric_limits<double>::lowest());

    // Encontrar o valor mínimo e máximo para cada atributo
    for (const auto& amostra : X_train) {
        for (size_t j = 0; j < n_atributos; ++j) {
            min_vals[j] = min(min_vals[j], amostra[j]);
            max_vals[j] = max(max_vals[j], amostra[j]);
        }
    }

    // Normalizar os dados de treinamento e teste
    for (auto& amostra : X_train) {
        for (size_t j = 0; j < n_atributos; ++j) {
            amostra[j] = (amostra[j] - min_vals[j]) / (max_vals[j] - min_vals[j]);
        }
    }

    for (auto& amostra : X_test) {
        for (size_t j = 0; j < n_atributos; ++j) {
            amostra[j] = (amostra[j] - min_vals[j]) / (max_vals[j] - min_vals[j]);
        }
    }
}

int main() {
    string linha;
    vector<vector<double>> X;
    vector<int> y;

    unordered_map<string, int> stringToInt;
    int nextCode = 0;

    // Leitura dos dados da entrada padrão
    while (getline(cin, linha)) {
        vector<string> tokens = split(linha);
        if (tokens.size() < 2) continue;

        vector<double> amostra;
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            try {
                amostra.push_back(stod(tokens[i]));
            } catch (...) {
                amostra.push_back(mapStringToInt(tokens[i], stringToInt, nextCode));
            }
        }

        try {
            y.push_back(stoi(tokens.back()));
        } catch (...) {
            y.push_back(mapStringToInt(tokens.back(), stringToInt, nextCode));
        }

        X.push_back(amostra);
    }

    if (X.empty()) {
        cerr << "Arquivo vazio ou inválido." << endl;
        return 1;
    }

    size_t n_amostras = X.size();
    size_t n_atributos = X[0].size();

    // Embaralhar os dados
    shuffle_data(X, y);

    // Dividir os dados em 80% treino e 20% teste
    vector<vector<double>> X_train, X_test;
    vector<int> y_train, y_test;
    split_data(X, y, X_train, y_train, X_test, y_test);

    // Normalizar os dados de entrada
    normalize_data(X_train, X_test);

    // Definindo topologia da rede: [entrada, oculta, saída]
    int n_classes = *max_element(y.begin(), y.end()) + 1;
    int topologia[] = { static_cast<int>(n_atributos), 24, n_classes };
    MultiLayerPerceptron mlp(3, topologia);

    mlp.dEta = LEARNING_RATE;
    mlp.dAlpha = 0.9;
    mlp.dGain = 1.0;

    vector<double> input(n_atributos);
    vector<double> output(n_classes);
    vector<double> target(n_classes, 0.0);

    // Declarações antes do laço de treinamento
    double soma_acuracia = 0.0;
    int convergiu_em = MAX_EPOCHS;

    auto inicio = chrono::high_resolution_clock::now();

    // Laço de treinamento
    for (int epoca = 0; epoca < MAX_EPOCHS; ++epoca) {
        int acertos = 0;
        for (size_t i = 0; i < X_train.size(); ++i) {
            fill(target.begin(), target.end(), 0.0);
            target[y_train[i]] = 1.0;

            for (size_t j = 0; j < n_atributos; ++j) {
                input[j] = X_train[i][j];
            }

            mlp.Simulate(input.data(), output.data(), target.data(), true);

            int pred = distance(output.begin(), max_element(output.begin(), output.end()));
            if (pred == y_train[i]) acertos++;
        }

        double acc = static_cast<double>(acertos) / X_train.size() * 100.0;
        soma_acuracia += acc;

        if (acc >= 99.9) {
            convergiu_em = epoca + 1;
            break;
        }
    }

    auto fim = chrono::high_resolution_clock::now();
    double tempo = chrono::duration_cast<chrono::milliseconds>(fim - inicio).count();

    float media_acuracia = soma_acuracia / convergiu_em;

    cout << "Media de Acuracia ao longo das epocas: " << media_acuracia << "%" << endl;
    cout << "Treinamento convergiu apos " << convergiu_em << " epocas." << endl;
    cout << "Tempo de execucao: " << tempo << " ms" << endl;

    // Avaliação no conjunto de teste
    int acertos_teste = 0;
    for (size_t i = 0; i < X_test.size(); ++i) {
        fill(target.begin(), target.end(), 0.0);
        target[y_test[i]] = 1.0;

        for (size_t j = 0; j < n_atributos; ++j) {
            input[j] = X_test[i][j];
        }

        mlp.Simulate(input.data(), output.data(), target.data(), false);

        int pred = distance(output.begin(), max_element(output.begin(), output.end()));
        if (pred == y_test[i]) acertos_teste++;
    }

    double acc_teste = static_cast<double>(acertos_teste) / X_test.size() * 100.0;
    cout << "Acuracia no conjunto de teste: " << acc_teste << "%" << endl;

    return 0;
}
