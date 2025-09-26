# Simulador de Propagación de Ondas en Redes (C++ + OpenMP)

Laboratorio 1 — Programación Paralela con OpenMP  
**Departamento de Ingeniería Informática — USACH**

Esta **plantilla completa** incluye el esqueleto del proyecto, descripciones, Makefile, CLI, scripts de gráficos
y archivos de salida para que puedas empezar **de inmediato**. Cumple con el enfoque **POO**,
demuestra cláusulas clave de **OpenMP** y sigue la **política de flexibilidad** del profesor.

---

## 1) Contenidos

```
wave_propagation_template/
├─ main.cpp
├─ Node.h / Node.cpp
├─ Network.h / Network.cpp
├─ WavePropagator.h / WavePropagator.cpp
├─ MetricsCalculator.h / MetricsCalculator.cpp
├─ Benchmark.h / Benchmark.cpp
├─ Visualizer.h / Visualizer.cpp
├─ Makefile
├─ scripts/
│  └─ plot_speedup.py
└─ results/        # salidas .dat y .png
```

---

## 2) Modelo y numerics (resumen)

Ecuación (difusión amortiguada con fuente común):
\[
\frac{dA_i}{dt} = D \sum_{j \in \mathcal{N}(i)} (A_j - A_i) - \gamma A_i + S(t)
\]
Avance de **Euler explícito**:
\[
A_i^{t+\Delta t} = A_i^{t} + \Delta t \big[ D \sum_{j}(A_j^t - A_i^t) - \gamma A_i^t + S(t) \big]
\]

**Estabilidad práctica**: usa \(\Delta t \lesssim \frac{c}{D \cdot d_{\max} + \gamma}\) con \(c\in[0.2,0.5]\).  
Si ves “explosión” numérica, reduce \(\Delta t\).

---

## 3) OpenMP (qué y por qué)

- `schedule(static|dynamic|guided, chunk)`: balanceo de carga vs overhead.
- `reduction(+:E)`: **preferido** para agregados (energía).
- `atomic` / `critical`: demostrar contención y alternativas.
- `collapse(2)`: útil en malla 2D (ejemplo incluido).
- `single`: coordinación dentro de una región paralela.
- `barrier` / `nowait`: sincronización por fases (usar **con criterio**).
- `task`: ejemplo segmentado por granos (micro‑benchmark).

> **Flexibilidad**: si omites alguna cláusula/método, **justifica** en tu reporte por qué tu diseño no la requiere.

---

## 4) Compilación y ejecución

### Requisitos
- **g++** con **OpenMP** (`-fopenmp`) o **MSVC** con `/openmp`.
- Estándar C++17.

### Linux / WSL / MSYS2
```bash
make              # compila
./wave_propagation --network 2d --Lx 100 --Ly 100 --steps 200 \
  --schedule dynamic --chunk 64 --sync reduction --threads 8
make benchmark    # corre matriz de experimentos y genera results/*.dat
python3 scripts/plot_speedup.py
```

### Windows (MSVC)
- Habilita **/openmp** y **/std:c++17** en Propiedades del proyecto.
- Adapta Makefile si usas `nmake` o compila desde el IDE.

---

## 5) Datos y gráficos

- `results/scaling.dat` → `p  mean_T  std_T  S  std_S  E  std_E`
- `results/schedule_vs_chunk.dat` → `schedule  chunk  mean_T  std_T`
- `results/sync_methods.dat` → `method  mean_T  std_T`
- `results/energy_trace.dat` → `t  E(t)`

Genera figuras con `scripts/plot_speedup.py` (**Matplotlib**, sin estilos específicos).

---

## 6) Reporte (≤10 págs — guía)

1. Resumen; 2. Introducción; 3. Diseño POO; 4. Paralelización y **justificación** de cláusulas usadas/omitidas;
5. Metodología experimental; 6. Resultados con barras de error; 7. Ajuste Amdahl y discusión de \(f\);
8. Conclusiones; 9. Trabajo futuro; Apéndice reproducibilidad.

---

## 7) Siguientes pasos sugeridos

- Completar topologías **random** y **small‑world** (placeholders incluidos).
- Validar estabilidad variando `dt`, `D`, `γ`.
- Extender *tasks* y comparar con `parallel for`.
- Añadir **.bat**/**.ps1** para Windows si lo deseas.
