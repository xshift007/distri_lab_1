CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -fopenmp -std=c++17
LDFLAGS = -fopenmp

TARGET = wave_propagation
SOURCES = main.cpp Node.cpp Network.cpp WavePropagator.cpp Benchmark.cpp
HEADERS = Types.h Node.h Network.h WavePropagator.h Benchmark.h

PY ?= python3
FPS ?= 20
VIDEO_FORMAT ?= mp4
FRAMES_DIR := results/frames
VIDEOS_DIR := videos

$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o
	rm -rf results $(VIDEOS_DIR)

.PHONY: clean benchmark analysis frames_clean videos_dir video1d video2d video_all

benchmark: $(TARGET)
	@mkdir -p results
	./$(TARGET) --benchmark

analysis: $(TARGET)
	@mkdir -p results
	@$(PY) scripts/plot_speedup.py
	@$(PY) scripts/plot_time_vs_chunk.py

frames_clean:
	@rm -rf $(FRAMES_DIR) && mkdir -p $(FRAMES_DIR)

videos_dir:
	@mkdir -p $(VIDEOS_DIR)

# 1D
video1d: $(TARGET) videos_dir frames_clean
	@./$(TARGET) --network 1d --N 256 --steps 200 --threads 8 \
	  --schedule dynamic --chunk auto --noise global --S0 0.2 --omega 6.0 \
	  --dump-frames --frame-every 1
	@$(PY) scripts/make_video.py "$(FRAMES_DIR)" --outdir "$(VIDEOS_DIR)" \
	  --fps $(FPS) --format $(VIDEO_FORMAT)

# 2D
video2d: $(TARGET) videos_dir frames_clean
	@./$(TARGET) --network 2d --Lx 64 --Ly 64 --steps 200 --threads 8 \
	  --schedule dynamic --chunk auto --noise single --S0 0.3 \
	  --omega-mu 8.0 --omega-sigma 1.0 \
	  --dump-frames --frame-every 2
	@$(PY) scripts/make_video.py "$(FRAMES_DIR)" --outdir "$(VIDEOS_DIR)" \
	  --fps $(FPS) --format $(VIDEO_FORMAT)

video_all: video1d video2d
