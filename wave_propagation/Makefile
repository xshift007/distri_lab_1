CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -fopenmp -std=c++17
LDFLAGS = -fopenmp
TARGET = wave_propagation

SOURCES = main.cpp Node.cpp Network.cpp WavePropagator.cpp           NetworkStatePrinter.cpp PerformanceMetrics.cpp Benchmark.cpp
OBJECTS = $(SOURCES:.cpp=.o)
HEADERS = Node.h Network.h WavePropagator.h NetworkStatePrinter.h           PerformanceMetrics.h Benchmark.h

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJECTS)

.PHONY: all clean
