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
	@python3 scripts/plot_speedup.py || true
	@python3 scripts/plot_chunk.py || true
	@mkdir -p videos
	@python3 scripts/make_video.py results/frames --mode 2d --outdir videos \
		--fps 12 --norm robust --robust-pct 1 --interp bilinear --colorbar

# =====================[ Video helpers ]=====================
# Puedes sobreescribir estas variables al invocar make:
PY            ?= python3
FPS           ?= 12
VIDEO_FORMAT  ?= gif            # gif | mp4
FRAMES_DIR    ?= results/frames
VIDEOS_DIR    ?= videos

# Presets de visualización (puedes overridearlos desde CLI)
VIDEO1D_FLAGS ?= --mode 1d --downsample 1500 --zoom1d 250 --mark-peak --zero-center-1d
VIDEO2D_FLAGS ?= --mode 2d --norm robust --robust-pct 1 --interp bilinear --colorbar --zero-center-2d --ema 0.6

.PHONY: video1d video2d video_all frames_clean videos_dir

frames_clean:
	@rm -rf "$(FRAMES_DIR)"
	@mkdir -p "$(FRAMES_DIR)"

videos_dir:
	@mkdir -p "$(VIDEOS_DIR)"

# --------------------- 1D: línea amplitud vs nodo -----------------------
video1d: $(TARGET) videos_dir frames_clean
	@echo "[video1d] ejecutando simulacion 1D y volcando frames..."
	@./$(TARGET) --network 1d --Lx 100 --Ly 1 --steps 50 \
	  --threads 8 --schedule dynamic --chunk auto \
	  --noise off --S0 0.0 --omega 0.0 \
	  --dump-frames --frame-every 1
	@echo "[video1d] creando video..."
	@$(PY) scripts/make_video.py "$(FRAMES_DIR)" --outdir "$(VIDEOS_DIR)" \
	  --fps $(FPS) --format $(VIDEO_FORMAT) $(VIDEO1D_FLAGS)

# ----------------------- 2D: mapa de calor ------------------------------
video2d: $(TARGET) videos_dir frames_clean
	@echo "[video2d] ejecutando simulacion 2D y volcando frames..."
	@./$(TARGET) --network 2d --Lx 64 --Ly 64 --steps 50 \
	  --threads 8 --schedule dynamic --chunk auto \
	  --noise single --S0 0.3 --omega-mu 8.0 --omega-sigma 1.0 \
	  --dump-frames --frame-every 1
	@echo "[video2d] creando video..."
	@$(PY) scripts/make_video.py "$(FRAMES_DIR)" --outdir "$(VIDEOS_DIR)" \
	  --fps $(FPS) --format $(VIDEO_FORMAT) $(VIDEO2D_FLAGS)

video_all: video1d video2d
# ===========================================================
