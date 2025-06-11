# Compiladores
CXX = mpicxx
NVCC = nvcc

# Flags de compilação
CXXFLAGS = -O3 -Wall -fopenmp -std=c++17 -Iinclude
NVCCFLAGS = -arch=sm_70 -O3 -Xcompiler -fopenmp -std=c++17 -Iinclude

# Bibliotecas
LDFLAGS = -fopenmp -lcudart -lmpi

# Diretórios
SRC_DIR = src
OBJ_DIR = obj
INCLUDE_DIR = include
MODELS_DIR = src/models
UTIL_DIR = src/util

# Fontes
SOURCES_CPP = $(wildcard $(SRC_DIR)/*.cpp) \
              $(wildcard $(MODELS_DIR)/*.cpp) \
              $(wildcard $(UTIL_DIR)/*.cpp)
              
SOURCES_CU = $(wildcard $(MODELS_DIR)/*.cu)

# Objetos
OBJECTS_CPP = $(SOURCES_CPP:%.cpp=$(OBJ_DIR)/%.o)
OBJECTS_CU = $(SOURCES_CU:%.cu=$(OBJ_DIR)/%.o)

# Executável
TARGET = mlp_runner

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJECTS_CPP) $(OBJECTS_CU)
	@echo "Linking $@..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "Build successful!"

# Regra para arquivos .cpp
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	@echo "Compiling $<..."
	@$(CXX) $(CXXFLAGS) -c $< -o $@

# Regra para arquivos .cu (CUDA)
$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	@echo "Compiling CUDA $<..."
	@$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	@rm -rf $(OBJ_DIR) $(TARGET)
	@echo "Cleaned build artifacts"
