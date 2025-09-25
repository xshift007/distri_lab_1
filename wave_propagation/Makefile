CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -fopenmp -std=c++17
LDFLAGS = -fopenmp
TARGET = wave_propagation

SOURCES = main.cpp Node.cpp Network.cpp WavePropagator.cpp Integrator.cpp MetricsCalculator.cpp Benchmark.cpp Visualizer.cpp
HEADERS = Node.h Network.h WavePropagator.h Integrator.h MetricsCalculator.h Benchmark.h Visualizer.h

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o *.dat *.png

benchmark: $(TARGET)
	./$(TARGET) -benchmark

analysis: $(TARGET)
	./$(TARGET) -analysis

.PHONY: all clean benchmark analysis