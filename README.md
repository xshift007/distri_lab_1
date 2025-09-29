# Laboratorio 1 — Simulación de propagación de ondas

Este repositorio contiene la implementación del laboratorio 1 de Programación Paralela. El objetivo es simular la propagación amortiguada de una perturbación en redes regulares 1D y 2D utilizando C++17 y OpenMP, respetando el diseño orientado a objetos solicitado en la pauta.

La solución incluye:
- Generación de redes regulares y manejo del estado de cada nodo mediante doble buffer (`Node`, `Network`).
- Iterador temporal paralelizado con OpenMP para aplicar difusión, amortiguación y una fuente senoidal global (`WavePropagator`).
- Módulo de benchmarking para obtener medias, desviaciones estándar, speedup y eficiencia, además de barrer chunks dinámicos (`Benchmark`).
- Scripts de apoyo para producir los gráficos requeridos a partir de los datos crudos en `results/`.

## Estructura del proyecto

```
.
├── README.md              ← Este documento
└── wave_propagation/
    ├── *.cpp, *.h         ← Código fuente C++17 con OpenMP
    ├── Makefile           ← Objetivos `make`, `make benchmark`, `make clean`
    ├── results/           ← Se crea en tiempo de ejecución (datos y gráficos)
    └── scripts/           ← Scripts Python para graficar
```

## Requisitos previos

### Herramientas de compilación
- **g++** (o cualquier compilador compatible con C++17) con soporte para OpenMP.
- **make**.

### Herramientas de análisis opcional
- **Python 3.8+** con los paquetes `numpy` y `matplotlib` para ejecutar los scripts de graficado.

En distribuciones basadas en Debian/Ubuntu puedes instalarlos con:
```bash
sudo apt-get install g++ make python3 python3-numpy python3-matplotlib
```

## Compilación

1. Mueve al directorio principal del código fuente:
   ```bash
   cd wave_propagation
   ```
2. Compila la aplicación (por defecto genera el binario `wave_propagation`):
   ```bash
   make
   ```
3. Si deseas limpiar los artefactos de compilación y resultados previos:
   ```bash
   make clean
   ```

La compilación utiliza `-O3 -fopenmp -std=c++17` y enlaza automáticamente con OpenMP.

## Ejecución de una simulación

Una vez compilado, ejecuta el binario desde `wave_propagation/`:
```bash
./wave_propagation [opciones]
```
Las opciones disponibles son:

| Opción | Descripción |
|--------|-------------|
| `--network {1d,2d}` | Selecciona la topología regular (por defecto `2d`). |
| `--N <n>` | Tamaño de la red 1D (solo si `--network 1d`). |
| `--Lx <nx> --Ly <ny>` | Dimensiones de la malla 2D (solo si `--network 2d`). |
| `--D <valor>` | Coeficiente de difusión. |
| `--gamma <valor>` | Factor de amortiguación. |
| `--dt <valor>` | Paso de tiempo del integrador de Euler explícito. |
| `--steps <iter>` | Número de pasos de simulación. |
| `--S0 <valor>` | Amplitud de la fuente senoidal global (0 = desactivada). |
| `--omega <valor>` | Frecuencia angular de la fuente senoidal. |
| `--schedule {static,dynamic,guided}` | Estrategia de scheduling para el `parallel for` principal. |
| `--chunk <n\|auto>` | Tamaño de chunk para OpenMP. Con `auto` se calcula una heurística en tiempo de ejecución. |
| `--threads <n>` | Fija el número de hilos (alternativa a `OMP_NUM_THREADS`). |
| `--benchmark` | Ejecuta las campañas de benchmarking en lugar de una simulación simple. |
| `--help` | Muestra el resumen de uso y termina. |

**Ejemplo 1D:**
```bash
./wave_propagation --network 1d --N 5000 --steps 2000 --schedule static --threads 4
```

**Ejemplo 2D con fuente senoidal y selección automática de chunk:**
```bash
./wave_propagation --network 2d --Lx 256 --Ly 256 \
                   --steps 1500 --D 0.08 --gamma 0.015 --dt 0.01 \
                   --S0 0.1 --omega 0.5 --schedule dynamic --chunk auto --threads 8
```

Cada ejecución normal genera el archivo `results/energy_trace.dat` con la energía promedio por paso de tiempo y deja un mensaje `OK. Resultados en results/` en la terminal.

## Benchmarks automatizados

Para reproducir las mediciones del informe:
```bash
make benchmark
```
Esto ejecuta el binario en modo `--benchmark`, lo que produce en `results/` los siguientes archivos:
- `scaling.dat`: columnas `threads mean_time std_time speedup speedup_err efficiency efficiency_err`.
- `time_vs_chunk_dynamic.dat`: tiempos medios y desviaciones para chunks dinámicos representativos.

Puedes ajustar el número de hilos modificando la lista `plist` en `main.cpp` si tu máquina soporta más/menos núcleos.

## Post-procesamiento y gráficos

Con los datos generados, ejecuta los scripts de la carpeta `scripts/`:
```bash
python3 scripts/plot_speedup.py
python3 scripts/plot_chunk.py
```
Los gráficos se guardarán en `results/speedup.png`, `results/efficiency.png` y `results/time_vs_chunk_dynamic.png`.

## Notas adicionales
- El programa crea la carpeta `results/` de manera automática si no existe.
- Gracias al doble buffer, el bucle principal evita condiciones de carrera y no requiere barreras adicionales.
- Si omites `--threads`, OpenMP utilizará el valor de `OMP_NUM_THREADS` o, en su defecto, el máximo de hilos disponibles.
- La fuente externa `S(t) = S0 · sin(ωt)` se aplica de forma homogénea a todos los nodos; establece `S0 = 0` para desactivarla.

Con este README deberías poder compilar, ejecutar, medir y documentar el laboratorio completo sin necesidad de consultar archivos adicionales.
