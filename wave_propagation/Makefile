CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -fopenmp -std=c++17
LDFLAGS = -fopenmp

TARGET = wave_propagation
SOURCES = main.cpp Node.cpp Network.cpp WavePropagator.cpp Benchmark.cpp
HEADERS = Types.h Node.h Network.h WavePropagator.h Benchmark.h

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o
	rm -rf results

.PHONY: clean benchmark analysis

benchmark: $(TARGET)
	@mkdir -p results
	./$(TARGET) --benchmark

analysis: $(TARGET)
	@mkdir -p results results/frames
	@echo "[analysis] corriendo escalamiento..."
	@./$(TARGET) --network 2d --Lx 256 --Ly 256 --steps 800 --schedule dynamic --chunk auto \
	        --threads 8 --energy-accum reduction --noise pernode --S0 0.05 \
	        --omega-mu 8.0 --omega-sigma 1.5 --collapse2 --dump-frames --frame-every 20
	@python3 scripts/plot_speedup.py
	@python3 scripts/plot_chunk.py
	@python3 scripts/make_video.py results/frames out.gif
