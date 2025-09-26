CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -fopenmp -std=c++17
LDFLAGS = -fopenmp

TARGET = wave_propagation
SOURCES = main.cpp Node.cpp Network.cpp WavePropagator.cpp           MetricsCalculator.cpp Benchmark.cpp Visualizer.cpp
HEADERS = Node.h Network.h WavePropagator.h MetricsCalculator.h           Benchmark.h Visualizer.h

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o results/*.dat results/*.png

benchmark: $(TARGET)
	./$(TARGET) --benchmark --threads 1,2,4,8

analysis: $(TARGET)
	python3 scripts/plot_speedup.py && python3 scripts/plot_benchmarks.py

.PHONY: clean benchmark analysis
