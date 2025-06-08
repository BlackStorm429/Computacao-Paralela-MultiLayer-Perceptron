#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

// =========
// 1) Tipos
// =========
using float_t = float;

// Sigmoid e derivada
__device__ inline float_t sigmoid_cuda(float_t x) {
    return 1.0f / (1.0f + expf(-x));
}
__device__ inline float_t sigmoid_deriv_cuda(float_t y) {
    // y = sigmoid(x)
    return y * (1.0f - y);
}

// Estrutura da rede (buffers na CPU e GPU)
struct MLPcu {
    int D, H, C;    // dims

    // Pesos e vieses (host)
    float_t *h_W1, *h_b1, *h_W2, *h_b2;
    // Pesos e vieses (device)
    float_t *d_W1, *d_b1, *d_W2, *d_b2;

    // Gradientes (host)
    float_t *h_dW1, *h_db1, *h_dW2, *h_db2;
    // Gradientes (device)
    float_t *d_dW1, *d_db1, *d_dW2, *d_db2;

    MLPcu(int dim_in, int dim_hid, int dim_out)
        : D(dim_in), H(dim_hid), C(dim_out)
    {
        size_t size_W1 = H * D * sizeof(float_t);
        size_t size_b1 = H * sizeof(float_t);
        size_t size_W2 = C * H * sizeof(float_t);
        size_t size_b2 = C * sizeof(float_t);

        // 1) Aloca e inicializa host
        h_W1 = (float_t*)malloc(size_W1);
        h_b1 = (float_t*)malloc(size_b1);
        h_W2 = (float_t*)malloc(size_W2);
        h_b2 = (float_t*)malloc(size_b2);

        h_dW1 = (float_t*)malloc(size_W1);
        h_db1 = (float_t*)malloc(size_b1);
        h_dW2 = (float_t*)malloc(size_W2);
        h_db2 = (float_t*)malloc(size_b2);

        std::mt19937 gen(12345);
        std::normal_distribution<float_t> dist1(0.0f, 1.0f / sqrtf(D));
        std::normal_distribution<float_t> dist2(0.0f, 1.0f / sqrtf(H));

        for (int i = 0; i < H*D; i++) h_W1[i] = dist1(gen);
        for (int i = 0; i < H; i++)    h_b1[i] = 0.0f;
        for (int i = 0; i < C*H; i++) h_W2[i] = dist2(gen);
        for (int i = 0; i < C; i++)    h_b2[i] = 0.0f;

        // zera gradientes host
        memset(h_dW1, 0, size_W1);
        memset(h_db1, 0, size_b1);
        memset(h_dW2, 0, size_W2);
        memset(h_db2, 0, size_b2);

        // 2) Aloca device
        cudaMalloc(&d_W1, size_W1);
        cudaMalloc(&d_b1, size_b1);
        cudaMalloc(&d_W2, size_W2);
        cudaMalloc(&d_b2, size_b2);

        cudaMalloc(&d_dW1, size_W1);
        cudaMalloc(&d_db1, size_b1);
        cudaMalloc(&d_dW2, size_W2);
        cudaMalloc(&d_db2, size_b2);

        // 3) Copia de H→D
        cudaMemcpy(d_W1, h_W1, size_W1, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b1, h_b1, size_b1, cudaMemcpyHostToDevice);
        cudaMemcpy(d_W2, h_W2, size_W2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b2, h_b2, size_b2, cudaMemcpyHostToDevice);

        // Gradientes device começam em zero
        cudaMemset(d_dW1, 0, size_W1);
        cudaMemset(d_db1, 0, size_b1);
        cudaMemset(d_dW2, 0, size_W2);
        cudaMemset(d_db2, 0, size_b2);
    }

    ~MLPcu() {
        free(h_W1); free(h_b1); free(h_W2); free(h_b2);
        free(h_dW1); free(h_db1); free(h_dW2); free(h_db2);
        cudaFree(d_W1); cudaFree(d_b1); cudaFree(d_W2); cudaFree(d_b2);
        cudaFree(d_dW1); cudaFree(d_db1); cudaFree(d_dW2); cudaFree(d_db2);
    }

    // Função auxiliar para sincronizar gradientes (D→H) e zerar no device:
    void sync_gradients_to_host() {
        size_t size_W1 = H * D * sizeof(float_t);
        size_t size_b1 = H * sizeof(float_t);
        size_t size_W2 = C * H * sizeof(float_t);
        size_t size_b2 = C * sizeof(float_t);

        cudaMemcpy(h_dW1, d_dW1, size_W1, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_db1, d_db1, size_b1, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_dW2, d_dW2, size_W2, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_db2, d_db2, size_b2, cudaMemcpyDeviceToHost);

        // Zera gradientes no device para próxima iteração
        cudaMemset(d_dW1, 0, size_W1);
        cudaMemset(d_db1, 0, size_b1);
        cudaMemset(d_dW2, 0, size_W2);
        cudaMemset(d_db2, 0, size_b2);
    }

    // Função de atualização de parâmetros (host)
    void update_params_host(float_t lr, int batch_size) {
        float_t inv_bs = lr / batch_size;
        for (int i = 0; i < H*D; i++) h_W1[i] -= inv_bs * h_dW1[i];
        for (int i = 0; i < H;   i++) h_b1[i] -= inv_bs * h_db1[i];
        for (int i = 0; i < C*H; i++) h_W2[i] -= inv_bs * h_dW2[i];
        for (int i = 0; i < C;   i++) h_b2[i] -= inv_bs * h_db2[i];

        // Copia os novos pesos para device
        cudaMemcpy(d_W1, h_W1, H*D*sizeof(float_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b1, h_b1, H*sizeof(float_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_W2, h_W2, C*H*sizeof(float_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b2, h_b2, C*sizeof(float_t), cudaMemcpyHostToDevice);
    }
};

// ====================
// 2) Dataset Simples
// ====================
struct DataSet {
    int N, D, C;
    float_t *d_X; // [N][D] (device)
    int     *d_Y; // [N]     (device)
    DataSet(int n, int d, int c) : N(n), D(d), C(c) {
        size_t sx = N * D * sizeof(float_t);
        size_t sy = N * sizeof(int);
        // aloca host temporário para geração sintética
        std::vector<float_t> h_X(N*D);
        std::vector<int>     h_Y(N);
        std::mt19937 gen(0);
        std::uniform_real_distribution<float_t> dist(0.0f, 1.0f);
        for (int i = 0; i < N*D; i++) h_X[i] = dist(gen);
        for (int i = 0; i < N; i++)   h_Y[i] = gen() % C;

        // aloca device
        cudaMalloc(&d_X, sx);
        cudaMalloc(&d_Y, sy);
        // copia host → device
        cudaMemcpy(d_X, h_X.data(), sx, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Y, h_Y.data(), sy, cudaMemcpyHostToDevice);
    }
    ~DataSet() {
        cudaFree(d_X);
        cudaFree(d_Y);
    }
};

// =======================================================
// 3) Kernels CUDA: forward, backward e acúmulo de gradientes
// =======================================================

// 3.1) Forward kernel: calcula hidden_out e output_out
//      - X_batch: [B][D]
//      - W1: [H][D], b1: [H]
//      - hidden_out: [B][H]
//      - W2: [C][H], b2: [C]
//      - output_out: [B][C]
__global__ void kernel_forward(const float_t *X_batch, int B,
                               const float_t *W1, const float_t *b1,
                               float_t *hidden_out,
                               const float_t *W2, const float_t *b2,
                               float_t *output_out,
                               int D, int H, int C)
{
    int i = blockIdx.x;    // índice de amostra no batch
    int t = threadIdx.x;   // usaremos threads para neurônios

    // 1) calcular hidden_out (cada thread calcula um neurônio j ∈ [0,H))
    if (i < B) {
        // bloco de threads de tamanho >= H (ou dividido em sub-laços)
        if (t < H) {
            float_t sum = 0.0f;
            for (int k = 0; k < D; k++) {
                sum += W1[t * D + k] * X_batch[i * D + k];
            }
            sum += b1[t];
            hidden_out[i * H + t] = sigmoid_cuda(sum);
        }
    }
    __syncthreads();

    // 2) calcular output_out (cada thread calcula um neurônio j ∈ [0,C))
    //    Reorganizamos: usamos o mesmo bloco, mas threads extras calculam saídas.
    //    Para simplicidade, assumimos numThreads >= max(H,C).
    if (i < B) {
        if (t < C) {
            float_t sum = 0.0f;
            for (int k = 0; k < H; k++) {
                sum += W2[t * H + k] * hidden_out[i * H + k];
            }
            sum += b2[t];
            output_out[i * C + t] = sigmoid_cuda(sum);
        }
    }
}

// 3.2) Backward kernel: 
//      - Produz: δ_out [B][C], δ_hidden [B][H]
//      - E acumula no device: dW2, db2, dW1, db1
__global__ void kernel_backward(const float_t *X_batch, const int *Y_batch,
                                const float_t *hidden_out, const float_t *output_out,
                                int B, 
                                const float_t *W2, 
                                float_t *dW1, float_t *db1, 
                                float_t *dW2, float_t *db2,
                                int D, int H, int C)
{
    // Vamos partir em etapas:
    //  a) calcular δ_out[i][j], para i< B, j< C
    //  b) acumular gradientes de W2, b2
    //  c) calcular δ_hidden[i][k], acumular gradientes W1, b1

    int i = blockIdx.x;   // índice da amostra
    int t = threadIdx.x;  // thread = neurônio

    extern __shared__ float_t sdata[]; 
    // Usaremos sdata para armazenar δ_out[i][*] ou δ_hidden[i][*] temporariamente.

    // (a) δ_out
    if (i < B && t < C) {
        float_t y_true = (t == Y_batch[i]) ? 1.0f : 0.0f;
        float_t o = output_out[i * C + t];
        sdata[t] = (o - y_true) * sigmoid_deriv_cuda(o);
    }
    __syncthreads();

    // (b) acumular dW2 e db2: 
    // Cada thread que quiser pode acumular para cada W2[t][k]:
    // dW2[t][k] += δ_out[i][t] * hidden_out[i][k], para todos k < H
    if (i < B && t < C) {
        float_t delta_o = sdata[t];
        // db2
        atomicAdd(&db2[t], delta_o);
        // dW2[t][*]
        for (int k = 0; k < H; k++) {
            float_t val = delta_o * hidden_out[i * H + k];
            atomicAdd(&dW2[t * H + k], val);
        }
    }
    __syncthreads();

    // (c) δ_hidden[i][k] = ( Σ_j W2[j][k] * δ_out[i][j] ) * sigmoid’(hidden_out[i][k])
    if (i < B && t < H) {
        float_t sum = 0.0f;
        for (int j = 0; j < C; j++) {
            sum += W2[j * H + t] * sdata[j];
        }
        float_t h = hidden_out[i * H + t];
        float_t delta_h = sum * sigmoid_deriv_cuda(h);
        // armazena em sdata (reaproveitando)
        sdata[t] = delta_h;
    }
    __syncthreads();

    // acumular dW1 e db1
    if (i < B && t < H) {
        float_t delta_h = sdata[t];
        // db1
        atomicAdd(&db1[t], delta_h);
        // dW1[t][*]
        for (int k = 0; k < D; k++) {
            float_t val = delta_h * X_batch[i * D + k];
            atomicAdd(&dW1[t * D + k], val);
        }
    }
}

// ========================
// 4) Loop de Treino CUDA
// ========================

void train_cuda(MLPcu &net, DataSet &ds,
                int epochs, int batch_size, float_t lr)
{
    int N = ds.N;
    int D = ds.D;
    int C = ds.C;
    int H = net.H;
    int num_batches = (N + batch_size - 1) / batch_size;

    // Buffers intermediários (device):
    float_t *d_hidden, *d_output;
    // Máximo espaço: batch_size × maior(H, C)
    size_t size_hidden = batch_size * H * sizeof(float_t);
    size_t size_output = batch_size * C * sizeof(float_t);
    cudaMalloc(&d_hidden, size_hidden);
    cudaMalloc(&d_output, size_output);

    for (int epoch = 0; epoch < epochs; epoch++) {
        auto t0 = std::chrono::high_resolution_clock::now();

        // Zera gradientes device
        cudaMemset(net.d_dW1, 0, H*D*sizeof(float_t));
        cudaMemset(net.d_db1, 0, H*sizeof(float_t));
        cudaMemset(net.d_dW2, 0, C*H*sizeof(float_t));
        cudaMemset(net.d_db2, 0, C*sizeof(float_t));

        for (int b = 0; b < num_batches; b++) {
            int start = b * batch_size;
            int end   = std::min(start + batch_size, N);
            int B     = end - start;

            // PONHA INDICES para o batch no device:
            float_t *X_batch = ds.d_X + start * D;
            int     *Y_batch = ds.d_Y + start;

            // 1) Forward
            // Cada bloco = 1 amostra; cada bloco com numThreads ≥ max(H,C)
            int threads = std::max(H, C);
            int blocks  = B;
            // tamanho de shared mem = max(H,C) floats
            int shmem   = threads * sizeof(float_t);
            kernel_forward<<<blocks, threads, 0>>>(
                X_batch, B,
                net.d_W1, net.d_b1, d_hidden,
                net.d_W2, net.d_b2, d_output,
                D, H, C);
            cudaDeviceSynchronize();

            // 2) Backward
            kernel_backward<<<blocks, threads, sizeof(float_t) * max(H,C)>>>(
                X_batch, Y_batch,
                d_hidden, d_output,
                B,
                net.d_W2,
                net.d_dW1, net.d_db1,
                net.d_dW2, net.d_db2,
                D, H, C);
            cudaDeviceSynchronize();
        }

        // 3) Copia gradientes p/ host e atualiza pesos
        net.sync_gradients_to_host();
        net.update_params_host(lr, N);

        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "[CUDA] Epoch " << epoch
                  << " concluído em " << elapsed << " s\n";
    }

    cudaFree(d_hidden);
    cudaFree(d_output);
}

// ========
// 5) Main
// ========
int main(int argc, char *argv[]) {
    int N       = 10000;    // número total de amostras
    int D       = 128;
    int H       = 256;
    int C       = 10;
    int epochs  = 20;
    int batch   = 128;
    float_t lr  = 0.01f;

    // 1) Gera dataset sintético no device
    DataSet ds(N, D, C);

    // 2) Cria rede
    MLPcu net(D, H, C);

    // 3) Treina
    train_cuda(net, ds, epochs, batch, lr);

    return 0;
}
