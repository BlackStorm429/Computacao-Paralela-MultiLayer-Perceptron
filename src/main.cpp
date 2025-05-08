#include "util/MLPTester.cpp"
#include "util/parser.cpp"
#include "models/MLP.cpp"
#include "models/MLP_OpenMP.cpp"
#include "models/MLP_MPI.cpp"

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
#include <omp.h>  // necessário para omp_set_num_threads
#include <mpi.h>  // necessário para MPI

using namespace std;

int main(int argc, char* argv[]) {
    // Inicializa MPI primeiro
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Configurar threads OpenMP se fornecido o argumento --threads
    for (int i = 1; i < argc - 1; i++) {
        if (string(argv[i]) == "--threads") {
            try {
                int num_threads = stoi(argv[i+1]);
                if (num_threads > 0) {
                    omp_set_num_threads(num_threads);
                    if (rank == 0) {
                        cout << "OpenMP: Usando " << num_threads << " thread(s) por processo\n";
                    }
                } else if (rank == 0) {
                    cerr << "Número inválido de threads.\n";
                }
            } catch (...) {
                if (rank == 0) {
                    cerr << "Uso correto: mpirun -np <número_processos> ./mlp_mpi --threads <número_threads>\n";
                }
            }
            break;
        }
    }

    // Load the dataset - apenas o processo 0 carrega os dados
    std::vector<std::vector<double>> inputs, expectedOutputs;
    if (rank == 0) {
        loadDB("dataset/diabetes_balanced.csv", inputs, expectedOutputs);
        cout << "Dataset carregado com " << inputs.size() << " amostras\n";
    }
    
    // Broadcast do tamanho dos dados para todos os processos
    int dataSize = 0;
    if (rank == 0) {
        dataSize = inputs.size();
    }
    MPI_Bcast(&dataSize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Processos não-zero inicializam seus vetores com o tamanho correto
    if (rank != 0) {
        inputs.resize(dataSize);
        expectedOutputs.resize(dataSize);
        
        // Inicializa o tamanho dos vetores internos
        // (apenas como exemplo, em uma implementação real precisaríamos transferir os dados)
        if (dataSize > 0) {
            int inputDim = 8;  // Diabetes dataset tem 8 features
            int outputDim = 1;  // Uma saída (classificação binária)
            
            for (int i = 0; i < dataSize; i++) {
                inputs[i].resize(inputDim);
                expectedOutputs[i].resize(outputDim);
            }
        }
    }
    
    // Broadcast dos dados para todos os processos
    // Numa implementação completa, faria broadcast real dos dados
    // Este é um ponto onde MPI seria necessário para compartilhar os dados
    // entre os processos. Para simplicidade, estamos assumindo que todos os processos têm os dados.
    
    int layers[] = {8, 6, 6, 1, 0}; // 0-terminated array

    // Create an instance of sequencial MLP
    MLP mlp(layers, 0.0001);

    // Create an instance of openMP MLP
    MLP_OpenMP openMP(layers, 0.0001);

    // Create an instance of MPI MLP
    MLP_MPI mpi(layers, 0.0001);
    
    std::vector<std::vector<double>> Xtrain, Xtest, Ytrain, Ytest;


    splitTestTrain(inputs, expectedOutputs, Xtrain, Ytrain, Xtest, Ytest, 0.8);
    
    
   
    
    // Broadcast dos dados de treino/teste
    // Em uma implementação real, usando MPI_Bcast para todas as matrizes
    
    // Create tester instances
    MLPTester sequentialTester(mlp);
    MLPTester openMPTester(openMP);
    MLPTester mpiTester(mpi);

    sequentialTester.train(1000, 0.95, Xtrain, Ytrain);
    openMPTester.train(1000, 0.95, Xtrain, Ytrain);
    mpiTester.train(1000, 0.95, Xtrain, Ytrain);


    return 0;
}