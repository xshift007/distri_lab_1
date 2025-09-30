# Laboratorio 1 — Versión LEAN (USACH)

**Objetivo:** simulador de propagación en redes **1D/2D** con C++ + OpenMP, diseño OO simple, **solo** cláusulas necesarias.

## Diseño (OO mínimo y claro)
- `Node`: amplitud actual y previa (doble buffer), vecinos.
- `Network`: genera topologías **1D** (línea) o **2D** (malla 4-neighbors); guarda `D` y `gamma`.
- `WavePropagator`: loop temporal **fused** (una región paralela), update + energía (`reduction`) + commit.
- `Benchmark`: tiempos, **speedup**, **eficiencia**, **tiempo vs chunk** (dynamic).

## Compilar
```bash
make
```

## Ejecutar simulación (ejemplo)
```bash
./wave_propagation --network 2d --Lx 256 --Ly 256 --steps 1000 --schedule dynamic --chunk auto --threads 8
```
Salida: `results/energy_trace.dat`

## Benchmarks
```bash
make benchmark
python3 scripts/plot_speedup.py
python3 scripts/plot_chunk.py
python3 scripts/plot_amdahl.py
```
Genera:
- `results/scaling.dat` → `speedup.png` y `efficiency.png`
  - Formato: `threads mean_time std_time speedup speedup_err efficiency efficiency_err`
- `results/time_vs_chunk_dynamic.dat` → `time_vs_chunk_dynamic.png`
- `results/amdahl.png` → comparación `S_p` medido vs. predicción de Amdahl (las desviaciones provienen de caché, NUMA y ruido del sistema).

## Cómo generar video
- **1D:** `make video1d`
- **2D:** `make video2d`

Tips: puedes aumentar `--frame-every` si quieres que avance más lento y ajustar la resolución para mantener buena calidad (requisito del curso).

## Parámetros recomendados
- `--schedule dynamic` con `--chunk auto` (heurística elige 256 para dynamic).
- Si la carga es homogénea, `static` también sirve; `guided` es intermedio.

## Justificación (para el informe)
- **POO**: 4 clases con responsabilidades claras (no todo en `main`, no “una clase gigante”).
- **Cláusulas OpenMP**: `parallel for` + `schedule(X,chunk)` y `reduction` para energía. Otras (barrier/single) no se usan **porque** el **doble buffer** elimina dependencias cruzadas.
- **Alcance**: **solo 1D/2D** por instrucción del docente.
- **Métricas**: ≥10 repeticiones → `T̄ ± σ`, `S_p`, `E_p`, y gráficos mínimos (speedup/eficiencia/tiempo vs chunk).

> Fuente global opcional: `S(t)=S0*sin(ωt)`, misma para todos los nodos (confirmado en clase).
