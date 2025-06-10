#include <iostream>
#include <vector>
#include <cstdlib>
#include <mpi.h>

#include "IMLP.h"
#include "MLP_OpenMP.cpp"

class MLP_MPI : public IMLP {
    private:
        int mpiSize;
        int numThreads;
        double learningRate;

        MLP_OpenMP openMP;

    public:
        MLP_MPI(const int* layerSizes, int batchSize, double learningRate = 0.1, int mpiSize = 1, int numThreads = 1)
        : learningRate(learningRate), mpiSize(mpiSize), numThreads(numThreads), openMP(layerSizes, batchSize, numThreads, learningRate) {}

        std::vector<double> forward(const std::vector<double>& input) override 
        {
            return openMP.forward(input);
        }

        double loss(const std::vector<double>& input, const std::vector<double>& target) override
        {
            return openMP.loss(input, target);
        }

        void train(const std::vector<std::vector<double>>& inputData, const std::vector<std::vector<double>>& outputData) override
        {
            if (inputData.size() != outputData.size()) 
            {
                throw std::invalid_argument("Input e output não têm o mesmo tamanho.");
            }

            int rank;

            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            int worldSize;

            MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

            std::vector<double> masterFlatWeights;

            if (rank == 0) 
            {
                masterFlatWeights = openMP.getWeights();
            }

            int weightSize = 0;

            if (rank == 0) 
            {
                weightSize = static_cast<int>(masterFlatWeights.size());
            }

            MPI_Bcast(&weightSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

            std::vector<double> localFlatWeights(weightSize);

            if (rank == 0) 
            {
                localFlatWeights = masterFlatWeights;
            }

            MPI_Bcast(localFlatWeights.data(), weightSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            for (int i = 0; i < weightSize; ++i) 
            {
                double noise = ((double(std::rand()) / RAND_MAX - 0.5) * 2.0) * learningRate;

                localFlatWeights[i] += noise;
            }

            openMP.setWeights(localFlatWeights);
            openMP.train(inputData, outputData);

            double localLoss = openMP.loss(inputData[0], outputData[0]);

            std::vector<double> allLosses;

            if (rank == 0) 
            {
                allLosses.resize(worldSize);
            }

            MPI_Gather(&localLoss, 1, MPI_DOUBLE, rank == 0 ? allLosses.data() : nullptr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            std::vector<double> localTrainedFlat = openMP.getWeights();
            std::vector<double> allTrainedWeights;

            if (rank == 0) 
            {
                allTrainedWeights.resize(weightSize * worldSize);
            }

            MPI_Gather(localTrainedFlat.data(), weightSize, MPI_DOUBLE, rank == 0 ? allTrainedWeights.data() : nullptr, weightSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            std::vector<double> bestFlatWeights(weightSize);

            if (rank == 0) 
            {
                int bestRank = 0;

                for (int r = 1; r < worldSize; ++r) 
                {
                    if (allLosses[r] < allLosses[bestRank]) 
                    {
                        bestRank = r;
                    }
                }

                int offset = bestRank * weightSize;

                for (int i = 0; i < weightSize; ++i) 
                {
                    bestFlatWeights[i] = allTrainedWeights[offset + i];
                }
            }

            MPI_Bcast(bestFlatWeights.data(), weightSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            openMP.setWeights(bestFlatWeights);
        }
};