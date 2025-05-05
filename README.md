# Computação Paralela - Projeto 01

Repositório para o Projeto 01 da matéria de Computação Paralela.

Código base: implementação do algoritmo Perceptron simples em C++ com versões sequencial, OpenMP e MPI.  

# Instruções

## perceptron.cpp  
- Para compilar  
    ```bash
    g++ -o perceptron perceptron.cpp
    ```
- Para executar  
    ```bash
    ./perceptron.exe <dataset.txt>
    ```

## perceptron_openMP.cpp  
- Para compilar  
    ```bash
    g++ -fopenmp -o perceptron_OMP perceptron_OMP.cpp
    ```
- Para executar  
    ```bash
    ./perceptron_OMP.exe [número_de_threads] <dataset.txt>
    ```

## perceptron_MPI.cpp  
- Para compilar  
    ```bash
    mpic++ -o perceptron_MPI perceptron_MPI.cpp
    ```
- Para executar  
    ```bash
    mpirun -np [número_de_processos] ./perceptron_MPI.exe <dataset.txt>
    ```

# Visão Geral do Algoritmo Perceptron  

O Perceptron é um algoritmo de aprendizado supervisionado utilizado para classificadores binários. Ele ajusta pesos baseados no erro entre a saída prevista e a esperada.  

Funcionamento básico:

1. Inicializa os pesos com valores aleatórios pequenos.

2. Para cada amostra do conjunto de dados:
   - Calcula a saída do modelo (soma ponderada + função de ativação).
   - Atualiza os pesos se a saída estiver errada.

3. Repete o processo por várias épocas até convergência ou número máximo de iterações.

# Estrutura do Código

## Classe Perceptron (ou estrutura de funções)
- **Principais Atributos:**
    - `weights`: vetor de pesos sinápticos.
    - `learning_rate`: taxa de aprendizado.
    - `epochs`: número máximo de iterações.

- **Principais Métodos:**
    - `predict(inputs)`: calcula a saída binária para uma entrada.
    - `train(dataset)`: treina o modelo ajustando os pesos iterativamente.
    - `evaluate()`: avalia a acurácia após o treinamento.

## Função Main

- Carrega os dados do arquivo.
- Instancia e treina o Perceptron.
- Exibe métricas como número de acertos ou erro médio quadrático.

# OpenMP  

## Comparação entre o Código Original e o Código Paralelizado

### 1. Controle de Threads
- Leitura da quantidade de threads via argumento de linha de comando.
- `omp_set_num_threads(num_threads);`

### 2. Paralelização da Fase de Treinamento
- Utilização de `#pragma omp parallel for` para distribuir o treinamento em múltiplas amostras.
- Cuidado com variáveis compartilhadas como `weights`, uso de `critical` ou `reduction`.

### 3. Medição de Tempo
- Cronometragem da execução com `chrono` para comparar desempenho com a versão sequencial.
- Impressão do número de threads usados.

### 4. Reprodutibilidade
- Uso de `srand(0)` para fixar a aleatoriedade e permitir testes justos de desempenho.

# MPI  

## Distribuição dos Dados
- O dataset é dividido entre os processos usando `MPI_Scatter`.

## Treinamento Paralelo
- Cada processo realiza treinamento com sua parte do dataset.
- Os pesos são sincronizados com `MPI_Reduce` ou `MPI_Allreduce`.

## Coleta de Resultados
- O processo mestre coleta os pesos atualizados e calcula a saída final.
- Pode-se usar `MPI_Gather` para consolidar métricas de desempenho.

# Considerações para a Apresentação

### Explicação das Modificações:
- OpenMP: paralelização de loops de treinamento e predição.
- MPI: distribuição de dados entre processos e sincronização dos pesos.

### Benefícios do Paralelismo:
- Redução no tempo de treinamento.
- Aumento de escalabilidade com datasets maiores.

### Boas Práticas:
- Evitar condições de corrida em regiões paralelas.
- Uso de reduções e regiões críticas.

### Limitações:
- O Perceptron só resolve problemas linearmente separáveis.
- O ganho com paralelismo depende do tamanho do dataset.

### Lições Aprendidas:
- A compreensão do fluxo de dados é essencial para uma paralelização eficiente.
- A divisão cuidadosa de tarefas e sincronização entre threads/processos é crucial para resultados corretos.
