# Compiler: use the MPI C++ wrapper which links MPI libraries.
CC = mpic++
CFLAGS = -O2 -fopenmp -Iinclude 

SRC_DIR = src
HDR_DIR = include

# Compile every .cpp file in the src directory.
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
# Generate corresponding .o files.
OBJS := $(SRCS:.cpp=.o)
# Dependency files generated alongside object files.
DEPS := $(OBJS:.o=.d)

TARGET = main.out

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ -o $@

# Pattern rule to compile .cpp files into .o object files.
%.o: %.cpp
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET) $(DEPS)

# Include dependency files if they exist.
-include $(DEPS)