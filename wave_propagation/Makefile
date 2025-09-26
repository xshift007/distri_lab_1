CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -fopenmp -std=c++17
LDFLAGS = -fopenmp

TARGET = wave_propagation
SOURCES = main.cpp Node.cpp Network.cpp WavePropagator.cpp \
          MetricsCalculator.cpp Benchmark.cpp Visualizer.cpp
HEADERS = Node.h Network.h WavePropagator.h MetricsCalculator.h \
          Benchmark.h Visualizer.h

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o results/*.dat results/*.png

benchmark: $(TARGET)
	./$(TARGET) --benchmark

analysis: $(TARGET)
	./$(TARGET) --analysis

.PHONY: clean benchmark analysis
