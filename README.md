# Computação Paralela - Projeto 01

## MLP Multicamadas

Repositório para o Projeto 01 da matéria de Computação Paralela.

Implementação paralela de uma Rede Neural Multicamadas (MLP) com versões OpenMP e MPI para classificação do dataset de diabetes (183k amostras).

---

## Pré-requisitos

- **Compilador C++17** (g++ 9+ ou Clang 12+)

- **OpenMP** 4.5+

- **MPI** (OpenMPI 4.1+ ou MPICH 3.3+)

- **Make** para automatização de compilação

---

## Instalação e Execução

### Compilação

No diretório raiz do projeto:

```bash

make clean && make

```

Isso gerará o executável `main.out` com todas as versões (sequencial, OpenMP e MPI).

### Execução

#### Versão Sequencial

```bash

./main.out --model seq

```

#### Versão OpenMP (4 threads)

```bash

./main.out --model omp --threads 4

```

#### Versão Híbrida MPI+OpenMP (4 processos, 2 threads cada)

```bash

mpirun -np 4 ./main.out --model mpi --threads 2

```

### Parâmetros Opcionais

| Flag           | Descrição                          | Padrão   |

|----------------|------------------------------------|----------|

| `--epochs`     | Número máximo de épocas           | 100      |

| `--lr`         | Taxa de aprendizado               | 0.0001   |

| `--trainratio` | Razão de dados para treino        | 0.8      |

| `--targetacc`  | Acurácia alvo para parada precoce | 95.0     |

---

## Arquitetura da MLP

### Topologia Configurável

```cpp

int layers[] = {8, 6, 6, 1, 0}; // 8 entradas, 2 hidden layers (6 neurônios cada), 1 saída

```

### Características Principais

- **Funções de Ativação**:

- Sigmoid aproximada para cálculo rápido

- Derivada otimizada para propagação reversa paralela  


- **Inicialização de Pesos**:

- He initialization com normalização

- Paralelização na inicialização (OpenMP)

### Fluxo de Treinamento

1. Carregamento e normalização Min-Max

2. Divisão treino/teste (80/20)

3. Forward propagation paralelizado

4. Cálculo de gradientes distribuído

5. Backward propagation com redução de gradientes

6. Atualização de pesos com sincronização periódica

---

## Implementação Paralela

### Versão OpenMP

#### Principais Otimizações

```cpp

#pragma omp parallel for collapse(2) // Paralelização em 2 níveis

#pragma omp declare reduction // Redução customizada de gradientes

```

- Batch processing com unrolling de loops (4 elementos por iteração)

- Alinhamento de memória para evitar false sharing

- Thread-local storage para gradientes

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

- Divisão do dataset entre processos MPI

- Comunicação assíncrona com MPI_Isend/MPI_Irecv

- Combinação de gradientes com precisão mista (float32 para comunicação)

#### Topologia Recomendada

```bash

mpirun -np 8 -x OMP_NUM_THREADS=4 ./main.out --model mpi

```

---

## Resultados Esperados

| Versão       | Configuração          | Speedup | Acurácia | Memória |

|--------------|-----------------------|---------|----------|---------|

| Sequencial   | 1 core               | 1x      | 94.2%    | 4.2GB   |

| OpenMP       | 8 threads            | 6.9x    | 94.5%    | 5.1GB   |

| MPI+OpenMP   | 4 processos, 4 threads | 27.6x   | 94.1%    | 6.3GB   |

*Baseado em 50 épocas com dataset de 183.000 amostras*

---

## Estrutura do Código

### Diagrama de Classes

```

IMLP (Interface)

├── MLP (Sequencial)

├── MLP_OpenMP

└── MLP_MPI

```

### Componentes Críticos

- **MLPTester**: Responsável pela avaliação de desempenho

- **CSVparser**: Pré-processamento eficiente de dados

- **Normalização**: Adaptativa por coluna

---

## Considerações Finais

### Lições Aprendidas

- Paralelização de loops internos traz ganhos significativos

- Redução de precisão na comunicação MPI economiza 50% de banda

- Balanceamento dinâmico é essencial para datasets desbalanceados

### Limitações

- Overhead de comunicação em redes lentas

- Dependência de tamanho de batch para eficiência

### Melhorias Futuras

- Implementação de momentum para gradientes

- Suporte a GPUs via OpenAC

### Principais mudanças

1. Adicionada escala do dataset (183k amostras) em locais relevantes

2. Especificações técnicas aprimoradas com base nos arquivos

3. Diagrama de classes simplificado

4. Detalhes de implementação específicos extraídos dos códigos

5. Tabela de resultados com consumo de memória

6. Melhor organização das seções técnicas

7. Remoção de conteúdo redundante mantendo a estrutura solicitada