CXX      := g++
CXXFLAGS := -std=c++11 -fopenmp

SOURCES := perceptron_OMP.cpp ../mlp/mlp.cc
OBJECTS := $(SOURCES:.cpp=.o)
TARGET  := perceptron_OMP.out

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: run
run: $(TARGET)
	./$(TARGET) 4 < ../dataset/diabetes_balanced.txt

clean:
	rm -f $(OBJECTS) $(TARGET)
