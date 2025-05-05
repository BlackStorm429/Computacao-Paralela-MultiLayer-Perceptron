CXX      := mpic++

SOURCES := perceptron_MPI.cpp ../mlp/mlp.cc
OBJECTS := $(SOURCES:.cpp=.o)
TARGET  := perceptron_MPI

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $^ -o $@

%.o: %.cpp
	$(CXX) -c $< -o $@

.PHONY: run
run: $(TARGET)
	mpirun -np 4 $(TARGET) < ../dataset/diabetes.txt

clean:
	rm -f $(OBJECTS) $(TARGET)
