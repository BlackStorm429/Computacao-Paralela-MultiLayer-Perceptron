#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>   
#include <mpi.h> 

#include "util/MLPTester.cpp"
#include "util/parser.cpp"
#include "models/MLP.cpp"
#include "models/MLP_OpenMP.cpp"
#include "models/MLP_MPI.cpp"
#include "models/MLP_OpenMP_GPU.cpp"

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Configura número de threads via argumento
    for (int i = 1; i < argc - 1; i++) 
    {
        if (string(argv[i]) == "--threads") 
        {
            try 
            {
                int num_threads = stoi(argv[i+1]);

                if (num_threads > 0) 
                {
                    omp_set_num_threads(num_threads);

                    if (rank == 0) 
                    {
                        cout << "OpenMP: Usando " << num_threads << " thread(s) por processo\n";
                    }
                } 
                else if (rank == 0) 
                {
                    cerr << "Número inválido de threads.\n";
                }
            } 
            catch (...) 
            {
                if (rank == 0) 
                {
                    cerr << "Uso correto: mpirun -np <n_proc> ./meu_programa --threads <n_threads>\n";
                }
            }

            break;
        }
    }

    vector<vector<double>> inputs, expectedOutputs;

    if (rank == 0) 
    {
        loadMNist("dataset/mnist/t10k-images.idx3-ubyte", "dataset/mnist/t10k-labels.idx1-ubyte", inputs, expectedOutputs);
                  
        cout << "\nDataset MNIST carregado com " << inputs.size() << " amostras\n";
    }

    vector<vector<double>> Xtrain, Xtest, Ytrain, Ytest;

    if (rank == 0) 
    {
        splitTestTrain(inputs, expectedOutputs, Xtrain, Ytrain, Xtest, Ytest, 0.8);
    }

    int64_t sequential_duration = 0, openmp_duration = 0, mpi_duration = 0, gpu_openmp_duration = 0;

    if (rank == 0) 
    {
        int inputSize = inputs[0].size();
        int outputSize = expectedOutputs[0].size();

        cout << "Tamanho da entrada: " << inputSize  << ", Tamanho da saída: " << outputSize << endl;

        const int layers[] = { inputSize, inputSize/4, inputSize/8, outputSize, 0 };
        const double acc_limit = 0.12;
        const int max_epochs = 25;
        MLP mlp_base(layers, 1200, 0.01);


        
        // Treinamento OpenMP GPU
        {
            cout << "\nTreinamento MLP OpenMP GPU\n\n";
            
            MLP_OpenMP_GPU gpuNet(mlp_base);
            MLPTester gpuTester(gpuNet);
            gpu_openmp_duration = gpuTester.train(max_epochs, acc_limit, Xtrain, Ytrain);
        }

        // Treino OpenMP
        {
            cout << "\nTreinamento MLP OpenMP com " << omp_get_max_threads() << " threads\n\n";

            MLP_OpenMP openMP_net(mlp_base, omp_get_max_threads());
            MLPTester openMPTester(openMP_net);
            int64_t openmp_duration = openMPTester.train(max_epochs, acc_limit, Xtrain, Ytrain);
        }
        
        // Treino Sequencial
        {
            cout << "\nTreinamento MLP Sequencial:\n\n";
    
            MLPTester seqTester(mlp_base);
            int64_t sequential_duration = seqTester.train(max_epochs, acc_limit, Xtrain, Ytrain);
        }

        cout << "\nDuração sequencial: " << sequential_duration << " ms\n";
        cout << "Duração OpenMP: " << openmp_duration << " ms\n";
        cout << "Speedup OpenMP: " << static_cast<double>(sequential_duration) / openmp_duration << "x\n";

        cout << "\nDuração OpenMP: " << openmp_duration << " ms\n"; 
        cout << "Duração OpenMP GPU: " << gpu_openmp_duration << " ms\n";
        cout << "Speedup GPU/OpenMP: " << static_cast<double>(openmp_duration) / gpu_openmp_duration << "x\n";
    
        int batchSize = 100;
        double lr = 0.01;
        int numThreads = 4;

        // Treinamento MPI
        {
            cout << "\nTreinamento MLP MPI com " << size << " processos e " << numThreads << " threads por processo:\n\n";
            
            MLP_MPI mpiNet(layers, batchSize, lr, size, numThreads);
            MLPTester mpiTester(mpiNet);
            int64_t mpi_duration = mpiTester.train(max_epochs, acc_limit, Xtrain, Ytrain);
        }

        // Print comparativo OpenMP vs MPI
        cout << "\nDuração OpenMP: " << openmp_duration << " ms\n";
        cout << "Duração MPI: "      << mpi_duration  << " ms\n";
        cout << "Speedup (OpenMP/MPI): " << static_cast<double>(openmp_duration) / mpi_duration << "x\n";
    }

    MPI_Finalize();

    return 0;
}
