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
	rm -rf results videos

.PHONY: clean benchmark analysis amdahl

benchmark: $(TARGET)
	@mkdir -p results
	./$(TARGET) --benchmark

analysis: $(TARGET)
        @mkdir -p results
        @$(PY) scripts/plot_speedup.py
        @$(PY) scripts/plot_time_vs_chunk.py
        @$(PY) scripts/plot_amdahl.py

amdahl: $(TARGET)
        @$(PY) scripts/plot_amdahl.py --input results/scaling.dat --output results/amdahl.png

# =====================[ Video helpers ]=====================
# Puedes sobreescribir estas variables al invocar make:
PY            ?= python3
FPS           ?= 20
VIDEO_FORMAT  ?= mp4
FRAMES_DIR    := results/frames
VIDEOS_DIR    := videos

# Presets de visualización (puedes overridearlos desde CLI)
VIDEO1D_FLAGS ?= --mode 1d --downsample 1500 --zoom1d 250 --mark-peak --zero-center-1d
VIDEO2D_FLAGS ?= --mode 2d --norm robust --robust-pct 1 --interp bilinear --colorbar --zero-center-2d --ema 0.6

.PHONY: video1d video2d video_all frames_clean videos_dir

frames_clean:
	@rm -rf $(FRAMES_DIR) && mkdir -p $(FRAMES_DIR)

videos_dir:
	@mkdir -p $(VIDEOS_DIR)

# --------------------- 1D: línea amplitud vs nodo -----------------------
video1d: $(TARGET) videos_dir frames_clean
	@./$(TARGET) --network 1d --N 256 --steps 200 --threads 8 \
	--schedule dynamic --chunk auto --noise global --S0 0.2 --omega 6.0 \
	--dump-frames --frame-every 1
	@$(PY) scripts/make_video.py "$(FRAMES_DIR)" --outdir "$(VIDEOS_DIR)" \
	--fps $(FPS) --format $(VIDEO_FORMAT) $(VIDEO1D_FLAGS)

# ----------------------- 2D: mapa de calor ------------------------------
video2d: $(TARGET) videos_dir frames_clean
	@./$(TARGET) --network 2d --Lx 64 --Ly 64 --steps 200 --threads 8 \
	--schedule dynamic --chunk auto --noise single --S0 0.3 \
	--omega-mu 8.0 --omega-sigma 1.0 \
	--dump-frames --frame-every 2
	@$(PY) scripts/make_video.py "$(FRAMES_DIR)" --outdir "$(VIDEOS_DIR)" \
	--fps $(FPS) --format $(VIDEO_FORMAT) $(VIDEO2D_FLAGS)

video_all: video1d video2d
# ===========================================================
