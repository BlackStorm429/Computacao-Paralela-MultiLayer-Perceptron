# Compiladores
CXX = mpic++
NVCC = nvcc

# Flags de compilação
CXXFLAGS = -O3 -Wall -fopenmp -std=c++17 -Iinclude
NVCCFLAGS = -arch=sm_70 -O3 -Xcompiler -fopenmp -std=c++17 -Iinclude

# Bibliotecas
LDFLAGS = -fopenmp -lmpi

# Diretórios
SRC_DIR = src
OBJ_DIR = build
INCLUDE_DIR = include
MODELS_DIR = src/models
UTIL_DIR = src/util

# Fontes C++
SOURCES_CPP := $(wildcard $(SRC_DIR)/*.cpp) \
			   $(wildcard $(MODELS_DIR)/*.cpp) \
			   $(wildcard $(UTIL_DIR)/*.cpp)

# CUDA habilitado por padrão
ifndef CUDA
CUDA := 1
endif

# Objetos C++
OBJECTS_CPP := $(SOURCES_CPP:%.cpp=$(OBJ_DIR)/%.o)

ifeq ($(CUDA),1)
SOURCES_CU := $(wildcard $(MODELS_DIR)/*.cu)
OBJECTS_CU := $(SOURCES_CU:%.cu=$(OBJ_DIR)/%.o)
TARGET_OBJS := $(OBJECTS_CPP) $(OBJECTS_CU)
LDFLAGS += -lcudart
else
TARGET_OBJS := $(OBJECTS_CPP)
endif

# Executável
TARGET = run.out

.PHONY: all nocuda clean run run-nocuda

all: $(TARGET)

nocuda:
	$(MAKE) CUDA=0 all

run: all
	./$(TARGET)

run-nocuda:
	$(MAKE) CUDA=0
	./$(TARGET)

$(TARGET): $(TARGET_OBJS)
	@echo "Linking $@..."
	@$(CXX) $^ -o $@ $(LDFLAGS)
	@echo "Build successful!"

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	@echo "Compiling $<..."
	@$(CXX) $(CXXFLAGS) -c $< -o $@

ifeq ($(CUDA),1)
$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	@echo "Compiling CUDA $<..."
	@$(NVCC) $(NVCCFLAGS) -c $< -o $@
endif
