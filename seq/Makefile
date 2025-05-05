CXX      := g++

SOURCES := perceptron.cpp ../mlp/mlp.cc
OBJECTS := $(SOURCES:.cpp=.o)
TARGET  := perceptron

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $^ -o $@

%.o: %.cpp
	$(CXX) -c $< -o $@

.PHONY: run
run: $(TARGET)
	./$(TARGET) < ../dataset/diabetes.txt

clean:
	rm -f $(OBJECTS) $(TARGET)
