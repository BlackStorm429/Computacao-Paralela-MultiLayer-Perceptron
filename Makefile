# Makefile

# Enable CUDA by default (override with `make CUDA=0`)
CUDA ?= 1

# Compilers
CXX       := mpic++
NVCC      := nvcc

# Compile flags
CXXFLAGS  := -O3 -fopenmp -std=c++17 -Iinclude
NVCCFLAGS := -arch=sm_70 -O3 -Xcompiler -fopenmp -std=c++17 -Iinclude

# Linker flags
LDFLAGS := -fopenmp -lmpi
ifeq ($(CUDA),1)
	LDFLAGS += -lcudart
endif

# Directories
SRC_DIR := src
OBJ_DIR := build

# Find source files
CPP_SRCS := $(shell find $(SRC_DIR) -type f -name '*.cpp')
CU_SRCS  := $(shell find $(SRC_DIR)/models -type f -name '*.cu')

# Map sources -> objects
CPP_OBJS := $(CPP_SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
CU_OBJS  := $(CU_SRCS:$(SRC_DIR)/%.cu=$(OBJ_DIR)/%.o)
OBJS     := $(CPP_OBJS) $(CU_OBJS)

# Default target
all: run.out

# Link and run
run.out: $(OBJS)
	@echo "Linking → run.out"
	$(CXX) $^ $(LDFLAGS) -o $@

# C++ compilation
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	@echo " CXX    $< → $@"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# CUDA compilation
ifeq ($(CUDA),1)
$(OBJ_DIR)/models/%.o: $(SRC_DIR)/models/%.cu
	@mkdir -p $(@D)
	@echo " NVCC   $< → $@"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
endif

# Clean up
clean:
	rm -rf $(OBJ_DIR) run.out

.PHONY: all run.out clean
