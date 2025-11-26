# =========================[ Compilación ]=========================
CXX       = g++
CXXFLAGS  = -Wall -Wextra -O3 -march=native -fopenmp -std=c++17
LDFLAGS   = -fopenmp

TARGET  = wave_propagation
SOURCES = main.cpp Node.cpp Network.cpp WavePropagator.cpp Benchmark.cpp
HEADERS = Types.h Node.h Network.h WavePropagator.h Benchmark.h

# =========================[ Python & Paths ]======================
PY           ?= python3
RESULTS_DIR  := results
FRAMES_DIR   := $(RESULTS_DIR)/frames
VIDEOS_DIR   := videos

# =========================[ Reglas Principales ]===================
$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	$(PY) -c "import shutil, os, glob; [os.remove(f) for f in glob.glob('*.o')] + [os.remove(f) for f in glob.glob('$(TARGET)') if os.path.exists(f)] + [os.remove(f) for f in glob.glob('$(TARGET).exe') if os.path.exists(f)]; shutil.rmtree('$(RESULTS_DIR)', ignore_errors=True); shutil.rmtree('$(VIDEOS_DIR)', ignore_errors=True)"

.PHONY: clean benchmark analysis amdahl help \
        video1d video2d video_all frames_clean videos_dir matrix analyze_matrix \
        graphs videos

help:
	@echo "Targets:"
	@echo "  make            -> Compila el proyecto"
	@echo "  make graphs     -> Ejecuta benchmarks y genera TODOS los gráficos"
	@echo "  make videos     -> Genera videos HQ 1D y 2D automáticamente"
	@echo "  make clean      -> Limpia todo"

# =========================[ Utilidades ]===========================
# Crea carpetas de forma compatible (Windows/Linux) usando Python
dirs:
	@$(PY) -c "import os; os.makedirs('$(RESULTS_DIR)', exist_ok=True); os.makedirs('$(FRAMES_DIR)', exist_ok=True); os.makedirs('$(VIDEOS_DIR)', exist_ok=True)"

frames_clean:
	@$(PY) -c "import shutil, os; shutil.rmtree('$(FRAMES_DIR)', ignore_errors=True); os.makedirs('$(FRAMES_DIR)', exist_ok=True)"

# =========================[ Automatización ]=======================

# 1. Generar Gráficos (Rendimiento + Energía)
graphs: $(TARGET) dirs
	@echo "--- [1/2] Benchmarks de Rendimiento ---"
	./$(TARGET) --benchmark
	@$(PY) scripts/plot_speedup.py
	@$(PY) scripts/plot_time_vs_chunk.py
	@$(PY) scripts/plot_amdahl.py
	@echo "--- [2/2] Simulación de Energía ---"
	./$(TARGET) --network 2d --Lx 200 --Ly 200 --steps 1000 \
		--dt 0.05 --D 0.1 --gamma 0.005 \
		--noise pernode --S0 1.0 --omega-mu 10.0 --omega-sigma 2.0 \
		--threads 4
	@$(PY) scripts/plot_energy.py
	@echo ">>> Gráficos listos en $(RESULTS_DIR)/"

# 2. Generar Videos HQ
videos: $(TARGET) dirs
	@echo "=== Video 1D (HQ) ==="
	$(MAKE) frames_clean
	./$(TARGET) --network 1d --N 200 --steps 500 --dump-frames --frame-every 5 --noise single --S0 2.0
	$(PY) scripts/make_video.py "$(FRAMES_DIR)" --mode 1d --format gif --fps 20 --outdir "$(VIDEOS_DIR)" --zero-center \
		--mark-peak --interp bilinear --cmap viridis --colorbar --zoom1d 0.5

	@echo "=== Video 2D (3D Surface) ==="
	$(MAKE) frames_clean
	@# AUMENTADO A 500 STEPS PARA VIDEO MÁS LARGO
	./$(TARGET) --network 2d --Lx 100 --Ly 100 --steps 500 --dump-frames --frame-every 5 --noise single --S0 5.0
	$(PY) scripts/make_video.py "$(FRAMES_DIR)" --mode 2d --format gif --fps 15 --outdir "$(VIDEOS_DIR)" --zero-center \
		--mark-peak --interp bilinear --cmap viridis --colorbar

	@echo ">>> Videos listos en $(VIDEOS_DIR)/"

# Targets legacy (por si los necesitas)
benchmark: graphs
analysis: graphs
amdahl: graphs

video1d: videos
video2d: videos

matrix:
	@$(PY) -c "import os; os.makedirs('$(RESULTS_DIR)', exist_ok=True)"
	@$(PY) scripts/run_matrix.py

analyze_matrix:
	@$(PY) scripts/analyze_matrix.py
