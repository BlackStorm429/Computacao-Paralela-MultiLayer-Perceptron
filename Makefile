# Compiler: use the MPI C++ wrapper which links MPI libraries.
CC = g++
CFLAGS = -Wall -O2 -fopenmp -Iinclude

SRC_DIR = src
# Compile every .cpp file in the src directory.
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
# Generate corresponding .o files.
OBJS := $(SRCS:.cpp=.o)

TARGET = main.cpp

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

# Pattern rule to compile .cpp files into .o object files.
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)