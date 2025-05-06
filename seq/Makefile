CXX      := g++
CXXFLAGS := -std=c++11 -fopenmp

SOURCES := perceptron.cpp ../mlp/mlp.cc
OBJECTS := $(SOURCES:.cpp=.o)
TARGET  := perceptron

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: run
run: $(TARGET)
	./$(TARGET) < ../dataset/diabetes.txt

clean:
	rm -f $(OBJECTS) $(TARGET)
