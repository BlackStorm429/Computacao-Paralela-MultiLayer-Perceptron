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
        loadMNist("dataset/mnist/t10k-images.idx3-ubyte", "dataset/mnist/t10k-labels.idx1-ubyte", inputs, expectedOutputs);
        cout << "Dataset carregado com " << inputs.size() << " amostras\n";
    }

    int inputSize = inputs[0].size();
    int outputSize = expectedOutputs[0].size();
    std::cout << "Tamanho da entrada: " << inputSize << ", Tamanho da saída: " << outputSize << std::endl;
    
    std::vector<std::vector<double>> Xtrain, Xtest, Ytrain, Ytest;
    splitTestTrain(inputs, expectedOutputs, Xtrain, Ytrain, Xtest, Ytest, 0.8);
    
    
    int layers[] = {inputSize, inputSize/4, inputSize/8, outputSize, 0}; // 0-terminated array
    MLP mlp(layers, 400, 0.01);
    int64_t sequential_duration, openmp_duration, mpi_duration;

    {
        std::cout << "MLP OpenMP:\n";
        MLP_OpenMP openMP(mlp, 4);
        MLPTester openMPTester(openMP);
        openmp_duration = openMPTester.train(1000, 0.15, Xtrain, Ytrain);
    }
    
    {
        std::cout << "MLP Sequêncial:\n";
        MLPTester sequentialTester(mlp);
        sequential_duration = sequentialTester.train(1000, 0.15, Xtrain, Ytrain);
    }

    std::cout << "Duração do treinamento sequencial: " << sequential_duration << " ms\n";
    std::cout << "Duração do treinamento OpenMP: " << openmp_duration << " ms\n";

    std::cout << "Speedup:\n";
    std::cout << "OpenMP: " << static_cast<double>(sequential_duration) / openmp_duration << "x\n";

    {
        std::cout << "MLP MPI:\n";
        MLP_MPI mpi(layers, 0.01, 400, 2);
        MLPTester mpiTester(mpi);
        mpiTester.train(10000, 0.2, Xtrain, Ytrain);
    }
     
    return 0;
}