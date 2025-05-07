CXX      := mpic++
CXXFLAGS := -std=c++11

SOURCES := perceptron_MPI.cpp ../mlp/mlp.cc
OBJECTS := $(SOURCES:.cpp=.o)
TARGET  := perceptron_MPI.out

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: run
run: $(TARGET)
	mpirun -np 4 $(TARGET) ../dataset/diabetes_balanced.txt

clean:
	rm -f $(OBJECTS) $(TARGET)
