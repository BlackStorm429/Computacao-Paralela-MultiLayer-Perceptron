# Computação Paralela - Projetos 01 e 02

## MLP Multicamadas

Repositório para o Projetos 01 e 02 da disciplina de Computação Paralela da Pontíficia Universidade Católica de Minas Gerais (PUC Minas).

Implementação paralela de uma Rede Neural Multicamadas (MLP) com versões OpenMP para CPU, MPI, OpenMP para GPU e CUDA para classificação do dataset de imagens de dígitos manuscritos (60k de amostras de treino e 10k de amostras de teste).

---

## Pré-requisitos

- **Compilador C++17** (g++ 9+ ou Clang 12+)

- **OpenMP** 4.5+

- **MPI** (OpenMPI 4.1+ ou MPICH 3.3+)

- **OpenMP para GPU** (CUDA Toolkit (11.0+))

- **CUDA** (CUDA Toolkit (11.0+))

- **Make** para automatização de compilação

---

## Instalação e Execução

### Compilação

No diretório raiz do projeto:

```bash

make clean

```

Isso garantirá que se houvesse um executável anterior, ele seja removido. Após isso executar:

```bash

make

```

Isso gerará o executável `run.out` com todas as versões (sequencial, OpenMP para CPU, MPI, OpenMP para GPU e CUDA).

### Execução

#### Todas os modelos

```bash

./run.out

```

## Arquitetura da MLP

### Topologia Configurável

```cpp

int layers[] = {784, 196, 98, 10}; // 784 entradas, 196 hidden layers (98 neurônios cada), 10 saídas

```

### Características Principais

- **Funções de Ativação**:

- Sigmoid aproximada para cálculo rápido.

- Derivada otimizada para propagação reversa paralela.


- **Inicialização de Pesos**:

- He initialization com normalização.

- Paralelização na inicialização (OpenMP).

### Fluxo de Treinamento

1. Carregamento e normalização Min-Max.

2. Divisão treino/teste (80/20).

3. Forward propagation paralelizado.

4. Cálculo de gradientes distribuído.

5. Backward propagation com redução de gradientes.

6. Atualização de pesos com sincronização periódica.

---

## Implementação Paralela

### Versão OpenMP

#### Principais Otimizações

```cpp

#pragma omp parallel for collapse(2) // Paralelização em 2 níveis

#pragma omp declare reduction // Redução customizada de gradientes

```

- Batch processing com unrolling de loops (4 elementos por iteração).

- Alinhamento de memória para evitar false sharing.

- Thread-local storage para gradientes.

#### Exemplo de Uso

```bash

OMP_NUM_THREADS=8 ./main.out --model omp --epochs 500

```

### Versão MPI

#### Estratégia Híbrida

```cpp

MPI_Bcast(weights); // Sincronização de pesos

MPI_Reduce(gradients); // Agregação distribuída

```

- Divisão do dataset entre processos MPI.

- Comunicação assíncrona com MPI_Isend/MPI_Irecv.

- Combinação de gradientes com precisão mista (float32 para comunicação).

#### Topologia Recomendada

```bash

mpirun -np 8 -x OMP_NUM_THREADS=4 ./main.out --model mpi

```

---

## Resultados Esperados

| Versão       | Configuração            | Speedup | Acurácia | Memória |

|--------------|-------------------------|---------|----------|---------|

| Sequencial   | 1 core                  | 1x      | 94.2%    | 4.2GB   |

| OpenMP       | 12 threads              | 1.03x   | 94.5%    | 5.1GB   |

| MPI+OpenMP   | 1 processo, 12 threads  | 1.1x    | 94.1%    | 6.3GB   |

| OpenMP GPU   | 1 processo, 12 threads  | 1.12x   | 94.1%    | 6.3GB   |

| CUDA         | 1 processo, 12 threads  | 3.68x   | 94.1%    | 6.3GB   |

*Baseado em até 50 épocas com dataset de 10.000 amostras*

---

## Estrutura do Código

### Diagrama de Classes

```

IMLP (Interface)
MLP_CUDA (Interface)

├── MLP (Sequencial)

├── MLP_OpenMP

└── MLP_MPI

└── MLP_OpenMP_GPU

└── MLP_CUDA

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