#include "util/MLPTester.cpp"
#include "util/parser.cpp"
#include "models/MLP.cpp"
#include "models/MLP_OpenMP.cpp"
#include "models/MLP_MPI.cpp" 

#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>   
#include <mpi.h>   

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Configura número de threads via argumento
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
                    cerr << "Uso correto: mpirun -np <n_proc> ./meu_programa --threads <n_threads>\n";
                }
            }
            break;
        }
    }

    vector<vector<double>> inputs, expectedOutputs;
    if (rank == 0) {
        loadMNist("dataset/mnist/t10k-images.idx3-ubyte",
                  "dataset/mnist/t10k-labels.idx1-ubyte",
                  inputs, expectedOutputs);
        cout << "Dataset MNIST carregado com " << inputs.size() << " amostras\n";
    }

    vector<vector<double>> Xtrain, Xtest, Ytrain, Ytest;
    if (rank == 0) {
        splitTestTrain(inputs, expectedOutputs, Xtrain, Ytrain, Xtest, Ytest, 0.8);
    }

    int64_t sequential_duration = 0, openmp_duration = 0, mpi_duration = 0;

    if (rank == 0) {
        int inputSize = inputs[0].size();
        int outputSize = expectedOutputs[0].size();
        cout << "Tamanho da entrada: " << inputSize 
             << ", Tamanho da saída: \n" << outputSize << endl;

        int layers[] = { inputSize, inputSize/4, inputSize/8, outputSize, 0 };

        // Treino OpenMP
        cout << "Treinamento MLP OpenMP:\n";
        MLP mlp_base(layers, 400, 0.01);
        MLP_OpenMP openMP_net(mlp_base, omp_get_max_threads());
        MLPTester openMPTester(openMP_net);
        int64_t openmp_duration = openMPTester.train(1000, 0.15, Xtrain, Ytrain);

        // Treino Sequencial
        cout << "Treinamento MLP Sequencial:\n";
        MLPTester seqTester(mlp_base);
        int64_t sequential_duration = seqTester.train(1000, 0.15, Xtrain, Ytrain);

        cout << "Duração sequencial: " << sequential_duration << " ms\n";
        cout << "Duração OpenMP: " << openmp_duration << " ms\n";
        cout << "Speedup OpenMP: " << static_cast<double>(sequential_duration) / openmp_duration << "x\n\n";
    
        int inputSize = inputs[0].size();
        int outputSize = expectedOutputs[0].size();
        int layers[] = { inputSize, inputSize/4, inputSize/8, outputSize, 0 };
        int batchSize = 100;
        double lr = 0.01;
        int numThreads = 4;

        // Treinamento MPI
        cout << "Treinamento MLP MPI com " << size << " processos e " 
             << numThreads << " threads por processo:\n";
        
        MLP_MPI mpiNet(layers, batchSize, lr, size, numThreads);
        MLPTester mpiTester(mpiNet);
        mpiTester.train(10000, 0.2, Xtrain, Ytrain);
        int64_t mpi_duration = mpiTester.train(1000, 0.15, Xtrain, Ytrain);

        // Print comparativo OpenMP vs MPI
        cout << "Duração OpenMP: " << openmp_duration << " ms\n";
        cout << "Duração MPI: "      << mpi_duration  << " ms\n";
        cout << "Speedup (OpenMP/MPI): " << static_cast<double>(openmp_duration) / mpi_duration << "x\n";
    }

    MPI_Finalize();
    return 0;
}
