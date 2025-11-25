# =========================[ Compilación ]=========================
CXX       = g++
CXXFLAGS  = -Wall -Wextra -O3 -march=native -fopenmp -std=c++17
LDFLAGS   = -fopenmp

TARGET  = wave_propagation
SOURCES = main.cpp Node.cpp Network.cpp WavePropagator.cpp Benchmark.cpp
HEADERS = Types.h Node.h Network.h WavePropagator.h Benchmark.h

# =========================[ Python & Paths ]======================
PY           ?= python3     # cambia a 'python' si tu entorno lo requiere
RESULTS_DIR  := results
FRAMES_DIR   := $(RESULTS_DIR)/frames
VIDEOS_DIR   := videos

# =========================[ Video presets ]=======================
# Puedes overridear desde CLI, p.ej.: VIDEO_FORMAT=gif make video2d
VIDEO_FORMAT ?= mp4          # gif | mp4
FPS          ?= 20
VIDEO_DPI    ?= 120
VIDEO_INTERP ?= bilinear     # nearest | bilinear | none

# Presets de visualización (SOLO FLAGS SOPORTADOS POR scripts/make_video.py)
VIDEO1D_FLAGS ?= --mode 1d --downsample 1500 --zoom1d 250 --mark-peak --zero-center-1d
VIDEO2D_FLAGS ?= --mode 2d --interp $(VIDEO_INTERP) --colorbar

# =========================[ Reglas base ]=========================
$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	$(PY) -c "import shutil, os, glob; [os.remove(f) for f in glob.glob('*.o')] + [os.remove(f) for f in glob.glob('$(TARGET)') if os.path.exists(f)] + [os.remove(f) for f in glob.glob('$(TARGET).exe') if os.path.exists(f)]; shutil.rmtree('$(RESULTS_DIR)', ignore_errors=True); shutil.rmtree('$(VIDEOS_DIR)', ignore_errors=True)"

.PHONY: clean benchmark analysis amdahl help \
        video1d video2d video_all frames_clean videos_dir matrix analyze_matrix

help:
	@echo "Targets principales:"
	@echo "  make                  -> compila $(TARGET)"
	@echo "  make benchmark        -> corre benchmarks y deja .dat en $(RESULTS_DIR)"
	@echo "  make analysis         -> genera speedup.png, efficiency.png, time_vs_chunk_dynamic.png y amdahl.png"
	@echo "  make video1d          -> genera frames 1D y arma video (gif/mp4)"
	@echo "  make video2d          -> genera frames 2D y arma video (gif/mp4)"
	@echo "  make matrix           -> corre scripts/run_matrix.py y deja CSV en $(RESULTS_DIR)"
	@echo "  make analyze_matrix   -> genera speedup_* y time_vs_chunk_dynamic_*.png desde el CSV"
	@echo "Variables útiles (override por CLI): VIDEO_FORMAT=gif FPS=22 VIDEO_INTERP=nearest"

# =========================[ Bench & Analysis ]====================
benchmark: $(TARGET)
	@mkdir -p $(RESULTS_DIR)
	./$(TARGET) --benchmark

analysis: $(TARGET)
	@mkdir -p $(RESULTS_DIR)
	@$(PY) scripts/plot_speedup.py
	@$(PY) scripts/plot_time_vs_chunk.py
	@$(PY) scripts/plot_amdahl.py
	@echo "Listo: $(RESULTS_DIR)/speedup.png, $(RESULTS_DIR)/efficiency.png, $(RESULTS_DIR)/time_vs_chunk_dynamic.png, $(RESULTS_DIR)/amdahl.png"

amdahl: $(TARGET)
	@mkdir -p $(RESULTS_DIR)
	@$(PY) scripts/plot_amdahl.py

# =========================[ Helpers video ]=======================
frames_clean:
	@rm -rf $(FRAMES_DIR) && mkdir -p $(FRAMES_DIR)

videos_dir:
	@mkdir -p $(VIDEOS_DIR)

# =========================[ Videos ]==============================
# 1D: línea amplitud vs nodo
video1d: $(TARGET) videos_dir frames_clean
	@echo "[video1d] generando frames 1D..."
	./$(TARGET) --network 1d --N 256 --steps 200 --threads 8 \
		--schedule dynamic --chunk auto \
		--noise off \
		--dump-frames --frame-every 1
	@echo "[video1d] construyendo video..."
	$(PY) scripts/make_video.py "$(FRAMES_DIR)" \
		--outdir "$(VIDEOS_DIR)" --format $(VIDEO_FORMAT) \
		--fps $(FPS) --dpi $(VIDEO_DPI) $(VIDEO1D_FLAGS)
	@echo "[video1d] listo: $(VIDEOS_DIR)/video_1d.$(VIDEO_FORMAT)"

# 2D: mapa de calor
video2d: $(TARGET) videos_dir frames_clean
	@echo "[video2d] generando frames 2D..."
	./$(TARGET) --network 2d --Lx 64 --Ly 64 --steps 200 --threads 8 \
		--schedule dynamic --chunk auto \
		--noise off \
		--dump-frames --frame-every 2
	@echo "[video2d] construyendo video..."
	$(PY) scripts/make_video.py "$(FRAMES_DIR)" \
		--outdir "$(VIDEOS_DIR)" --format $(VIDEO_FORMAT) \
		--fps $(FPS) --dpi $(VIDEO_DPI) $(VIDEO2D_FLAGS)
	@echo "[video2d] listo: $(VIDEOS_DIR)/video_2d.$(VIDEO_FORMAT)"

video_all: video1d video2d

# =========================[ Matriz de pruebas ]===================
matrix:
	@mkdir -p $(RESULTS_DIR)
	@$(PY) scripts/run_matrix.py

analyze_matrix:
	@$(PY) scripts/analyze_matrix.py

demo_bonus: $(TARGET)
	@echo "Corriendo simulacion con Ruido por Nodo (Bonus)..."
	./$(TARGET) --network 2d --Lx 200 --Ly 200 --steps 1000 \
		--dt 0.05 --D 0.1 --gamma 0.005 \
		--noise pernode --S0 1.0 --omega-mu 10.0 --omega-sigma 2.0 \
		--threads 4
	@echo "Graficando energia..."
	@$(PY) scripts/plot_energy.py
	@echo "Resultado en results/energy_plot.png"
