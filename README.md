# Computação Paralela - Projetos 01 e 02

## MLP Multicamadas

Repositório para os Projetos 01 e 02 da disciplina de Computação Paralela da Pontifícia Universidade Católica de Minas Gerais (PUC Minas).

Implementação paralela de uma Rede Neural Multicamadas (MLP) com versões:
- Sequencial
- OpenMP para CPU
- MPI (com OpenMP)
- OpenMP para GPU
- CUDA

O objetivo é classificar o dataset de imagens de dígitos manuscritos MNIST (60k de amostras de treino e 10k de amostras de teste).

---

## Pré-requisitos

- **Compilador C++17** (g++ 9+ ou Clang 12+)
- **OpenMP** 4.5+
- **MPI** (OpenMPI 4.1+ ou MPICH 3.3+)
- **CUDA Toolkit** (11.0+), para OpenMP (GPU) e CUDA
- **Make** para automatização da compilação

---

## Instalação e Execução

### Compilação

No diretório raiz do projeto:

```bash
make clean
make
```

Isso irá gerar, depois de remover qualquer executável anterior, o executável `run.out` com todas as versões (sequencial, OpenMP para CPU, MPI, OpenMP para GPU e CUDA).

### Execução

```bash
./run.out
```

O programa executa sequencialmente os testes de todas as versões implementadas, exibindo o tempo de execução e speedup de cada uma.

---

## Arquitetura da MLP

### Topologia

Definida no código, exemplo:

```cpp
int layers[] = {784, 196, 98, 10}; // 784 entradas, 196 hidden layers (98 neurônios cada), 10 saídas
```

### Hiperparâmetros

Batch size, taxa de aprendizado, número de épocas e limite de acurácia são definidos no código principal.

### Características Principais

- **Funções de Ativação**:
    - Sigmoid aproximada para cálculo rápido.
    - Derivada otimizada para propagação reversa paralela.

- **Inicialização de Pesos**:
    - Inicialização com normalização.
    - Paralelização na inicialização (OpenMP).

### Fluxo de treinamento

    1. Carregamento e normalização do MNIST.
    2. Divisão treino/teste (80/20).
    3. Treinamento sequencial, OpenMP, MPI, OpenMP GPU e CUDA.
    4. Exibição dos tempos e speedups.

---

## Implementação

### Sequencial

- Treinamento tradicional, sem paralelismo.

### OpenMP

- Paralelização do cálculo de gradientes e propagação.
- Uso de thread-local storage para gradientes.
- Redução dos gradientes ao final de cada batch.

### MPI (com OpenMP)

- Cada processo MPI executa um treinamento independente com OpenMP.
- Sincronização dos melhores pesos entre processos ao final de cada época.

### OpenMP para GPU

- Utiliza diretivas OpenMP target para offload em GPU (se disponível).
- Processamento em batches.

### CUDA

- Implementação dedicada para GPU usando CUDA.
- Processamento em batches e paralelização total do forward/backward.

---

## Resultados Obtidos

|   Versão   |      Configuração      | Tempo (ms) | Speedup | Acurácia |
|------------|------------------------|------------|---------|----------|
| Sequencial | 1 core                 | 63120      | 1x      | 18.21%   |
| OpenMP     | 12 threads             | 66529      | 0.95x   | 18.21%   |
| MPI+OpenMP | 1 processo, 12 threads | 65869      | 0.96x   | 18.21%   |
| OpenMP GPU | 1 processo, 12 threads | 62049      | 1.02x   | 23.5%    |
| CUDA       | 1 processo, 12 threads | 17320      | 3.64x   | 23.5%    |

*Baseado em até 28 épocas com dataset de 10.000 amostras*

---

## Estrutura do Código

### Estrutura de Diretórios

- `src/` — Código-fonte principal
- `src/models/` — Implementações das diferentes versões do MLP
- `src/util/` — Utilitários (parser, tester)
- `include/` — Headers das interfaces e modelos
- `dataset/mnist/` — Arquivos da base de dados MNIST (`t10k-images.idx3-ubyte`, `t10k-labels.idx1-ubyte`)

### Diagrama de Classes

```
MLP_CUDA (Interface)
IMLP (Interface)
├── MLP (Sequencial)
│   ├── MLP_OpenMP
│   ├── MLP_MPI
│   ├── MLP_OpenMP_GPU
│   └── MLP_CUDA
```

### Componentes Críticos

- **MLPTester**: Responsável pela avaliação de desempenho.
- **CSVparser**: Pré-processamento eficiente de dados.
- **Normalização**: Adaptativa por coluna.

---

## Considerações Finais

### Lições Aprendidas

- Paralelização de loops internos traz ganhos significativos.
- Redução de precisão na comunicação MPI economiza 50% de banda.
- Balanceamento dinâmico é essencial para datasets desbalanceados.

### Limitações

- Overhead de comunicação em redes lentas.
- Dependência de tamanho de batch para eficiência.

### Melhorias Futuras

- Implementação de momentum para gradientes.
- Suporte a GPUs via OpenAC.

### Principais mudanças

#### Decisões de Implementação OpenMP

- Flattening transforma input e output em 1D arrays.
- Treinamento com Mini-Batchs facilitam paralelismo e aceleram a execução do programa.
- Parallel Zeroing: Utiliza o OpenMP para setar todos os gradientes acumulados antes de cada batch.
- Forward Pass copia o input para o array de neurônios e computa a ativação.
- Backward Pass propaga os deltas calculados para trás pelas camadas ocultas.
- Acumulação de Gradiente: Para cada peso, acumula o gradiente usando operações atômicas.

#### Decisões de Implementação CUDA

- Forward pass calcula cada camada usando função sigmoide.
- Backward pass calcula os deltas (gradientes locais) de trás pra frente.
- Acumulação de gradiente usando AtomicAdd.
- Sincronização para garantir que todas as threads terminam na mesma época.
- Mini-Batchs facilitam paralelismo e aceleram a execução do programa.

## Observações

- O dataset MNIST deve estar na pasta `dataset/mnist/` (`dataset/mnist/t10k-images.idx3-ubyte`, `dataset/mnist/t10k-labels.idx1-ubyte`).
- O código já carrega e divide o dataset automaticamente.
- Não é necessário passar argumentos de modelo: todas as versões são executadas em sequência.
- Caso deseje definir o número de threads para as versões OpenMP ou MPI+OpenMP, utilize o argumento `--threads N` (exemplo: `./run.out --threads 8`).
- O programa executa, em sequência: CUDA, OpenMP GPU, MPI+OpenMP, OpenMP e Sequencial, exibindo os tempos e speedups ao final.

---
