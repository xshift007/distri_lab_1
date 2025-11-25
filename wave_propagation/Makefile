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
VIDEO_FORMAT ?= mp4       # gif | mp4
FPS          ?= 20
VIDEO_DPI    ?= 120
VIDEO_INTERP ?= bilinear  # nearest | bilinear | none

# Flags para el script clásico (make_video.py)
VIDEO1D_FLAGS ?= --mode 1d --downsample 1500 --zoom1d 250 --mark-peak --zero-center-1d
VIDEO2D_FLAGS ?= --mode 2d --interp $(VIDEO_INTERP) --colorbar

# =========================[ Reglas base ]=========================
$(TARGET): $(SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES) $(LDFLAGS)

clean:
	$(PY) -c "import shutil, os, glob; [os.remove(f) for f in glob.glob('*.o')] + [os.remove(f) for f in glob.glob('$(TARGET)') if os.path.exists(f)] + [os.remove(f) for f in glob.glob('$(TARGET).exe') if os.path.exists(f)]; shutil.rmtree('$(RESULTS_DIR)', ignore_errors=True); shutil.rmtree('$(VIDEOS_DIR)', ignore_errors=True)"

.PHONY: clean benchmark analysis amdahl help \
        video1d video2d video_all frames_clean videos_dir matrix analyze_matrix \
        graphs videos

help:
	@echo "Targets principales:"
	@echo "  make                  -> compila $(TARGET)"
	@echo "  make graphs           -> [NUEVO] Genera TODOS los gráficos (speedup, eficiencia, energía)"
	@echo "  make videos           -> [NUEVO] Genera los videos HQ (1D y 2D) automáticamente"
	@echo "  make benchmark        -> corre benchmarks y deja .dat en $(RESULTS_DIR)"
	@echo "  make analysis         -> procesa datos y genera graficos de rendimiento"
	@echo "  make clean            -> limpia binarios y carpetas de salida"

# =========================[ Bench & Analysis ]====================
benchmark: $(TARGET)
	@$(PY) -c "import os; os.makedirs('$(RESULTS_DIR)', exist_ok=True)"
	./$(TARGET) --benchmark

analysis: $(TARGET)
	@$(PY) -c "import os; os.makedirs('$(RESULTS_DIR)', exist_ok=True)"
	@$(PY) scripts/plot_speedup.py
	@$(PY) scripts/plot_time_vs_chunk.py
	@$(PY) scripts/plot_amdahl.py
	@echo "Listo: Gráficos de rendimiento generados en $(RESULTS_DIR)/"

amdahl: $(TARGET)
	@$(PY) -c "import os; os.makedirs('$(RESULTS_DIR)', exist_ok=True)"
	@$(PY) scripts/plot_amdahl.py

# =========================[ Helpers video ]=======================
frames_clean:
	@$(PY) -c "import shutil, os; shutil.rmtree('$(FRAMES_DIR)', ignore_errors=True); os.makedirs('$(FRAMES_DIR)', exist_ok=True)"

videos_dir:
	@$(PY) -c "import os; os.makedirs('$(VIDEOS_DIR)', exist_ok=True)"

# =========================[ Videos Clásicos ]======================
# Usan scripts/make_video.py original
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
	@$(PY) -c "import os; os.makedirs('$(RESULTS_DIR)', exist_ok=True)"
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

# =========================[ AUTOMATIZACIÓN TOTAL ]=================

# 1. Generar TODOS los gráficos (Rendimiento + Física)
graphs: $(TARGET)
	@$(PY) -c "import os; os.makedirs('$(RESULTS_DIR)', exist_ok=True)"
	@echo "--- [1/2] Generando datos de rendimiento (esto puede tardar) ---"
	$(MAKE) benchmark
	@echo "--- Graficando Speedup, Eficiencia y Amdahl ---"
	@$(PY) scripts/plot_speedup.py
	@$(PY) scripts/plot_time_vs_chunk.py
	@$(PY) scripts/plot_amdahl.py
	@echo "--- [2/2] Simulando sistema para análisis de energía ---"
	./$(TARGET) --network 2d --Lx 200 --Ly 200 --steps 1000 \
		--dt 0.05 --D 0.1 --gamma 0.005 \
		--noise pernode --S0 1.0 --omega-mu 10.0 --omega-sigma 2.0 \
		--threads 4
	@echo "--- Graficando Energía ---"
	@$(PY) scripts/plot_energy.py
	@echo ">>> LISTO: Revisa la carpeta $(RESULTS_DIR)/ para ver todos los gráficos."

# 2. Generar TODOS los videos HQ (Requiere scripts/make_video_3d_v2.py)
videos: $(TARGET) videos_dir
	@echo "=== Generando Video 1D (Estilo HQ con Relleno) ==="
	$(MAKE) frames_clean
	./$(TARGET) --network 1d --N 200 --steps 500 --dump-frames --frame-every 5 --noise single --S0 2.0
	$(PY) scripts/make_video_3d_v2.py "$(FRAMES_DIR)" --mode 1d --format gif --fps 20 --outdir "$(VIDEOS_DIR)"
	
	@echo "=== Generando Video 2D (Estilo 3D Surface) ==="
	$(MAKE) frames_clean
	./$(TARGET) --network 2d --Lx 100 --Ly 100 --steps 300 --dump-frames --frame-every 5 --noise single --S0 5.0
	$(PY) scripts/make_video_3d_v2.py "$(FRAMES_DIR)" --mode 2d --format gif --fps 15 --outdir "$(VIDEOS_DIR)"
	
	@echo ">>> LISTO: Revisa la carpeta $(VIDEOS_DIR)/ para ver video_1d_HQ.gif y video_2d_HQ.gif"