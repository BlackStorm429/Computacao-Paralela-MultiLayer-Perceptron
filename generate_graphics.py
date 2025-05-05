import matplotlib.pyplot as plt
import numpy as np

# Dados
tempos_sequencial = [11.337]
tempos_openmp = [1.696, 1.635, 1.517, 1.840]
tempos_mpi = [0.812, 0.957, 1.269]
config_mpi = ["1P, 4T", "2P, 2T", "4P, 0T"]

# Cria subplots lado a lado
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Gráfico 1 - Sequencial
axs[0].plot([120000] * len(tempos_sequencial), tempos_sequencial,
            color='deepskyblue', label='Sequencial')

# Adiciona ponto visível
axs[0].scatter([120000], tempos_sequencial, color='deepskyblue', edgecolor='white', s=100, zorder=5)

axs[0].set_title("Tempo de Execução Sequencial")
axs[0].set_xlabel("Tamanho do Dataset")
axs[0].set_ylabel("Tempo de Execução (s)")
axs[0].grid(True, linestyle='--', alpha=0.5)
axs[0].legend(facecolor='lightgray')

# Gráfico 2 - OpenMP
axs[1].plot([1, 2, 4, 8], tempos_openmp,
            color='limegreen', label='OpenMP')

# Linha pontilhada de referência (Sequencial)
axs[1].axhline(y=tempos_sequencial[0], color='deepskyblue', linestyle='--', label='Sequencial (referência)')

axs[1].set_title("Tempo de Execução OpenMP")
axs[1].set_xlabel("Número de Threads")
axs[1].set_ylabel("Tempo de Execução (s)")
axs[1].grid(True, linestyle='--', alpha=0.5)
axs[1].legend(facecolor='lightgray')

# Gráfico 3 - MPI
axs[2].plot(config_mpi, tempos_mpi,
            color='orange', label='MPI')

# Linha pontilhada de referência (Sequencial)
axs[2].axhline(y=tempos_sequencial[0], color='deepskyblue', linestyle='--', label='Sequencial (referência)')

axs[2].set_title("Tempo de Execução MPI")
axs[2].set_xlabel("Configuração de Processos e Threads")
axs[2].set_ylabel("Tempo de Execução (s)")
axs[2].grid(True, linestyle='--', alpha=0.5)
axs[2].legend(facecolor='lightgray')

plt.tight_layout()
plt.show()
