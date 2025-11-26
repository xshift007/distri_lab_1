# Laboratorio 1 — Propagación de ondas con C++17 y OpenMP

Este documento resume el proyecto de simulación de propagación amortiguada de ondas en redes 1D y 2D. Incluye instrucciones para compilar, ejecutar, medir rendimiento y generar videos, además de detallar el diseño general de la solución y cómo utilizar la herramienta de visualización mejorada.

## 1 Descripción del problema y objetivos

Se debe implementar un simulador que modele la evolución temporal de una perturbación propagándose a través de una red regular de nodos con interacciones de sus vecinos. Cada nodo representa un punto donde se almacena una amplitud que evoluciona según un esquema de difusión y amortiguamiento. El simulador debe soportar redes 1D (línea de `N` nodos) y 2D (malla rectangular de `L_x × L_y` nodos).

El algoritmo se debe diseñar de manera orientada a objetos y paralelizar con OpenMP usando solo las cláusulas necesarias para evitar condiciones de carrera. Se pide además generar un informe con análisis de performance (speedup, eficiencia y Ley de Amdahl) y visualizar la energía del sistema y la evolución de la onda mediante gráficos y videos.

## 2 Diseño y arquitectura del código

El diseño se organiza en cuatro clases principales situadas en `wave_propagation/`:

| Clase            | Función                                                            |
|------------------|---------------------------------------------------------------------|
| **`Node`**       | Almacena la amplitud actual y la amplitud previa (doble buffer) de un nodo y la lista de índices de sus vecinos. Expone métodos para leer o escribir la amplitud actual o previa y para commit del doble buffer. |
| **`Network`**    | Crea la topología de la red 1D o 2D y almacena parámetros físicos globales: coeficiente de difusión `D` y amortiguamiento `γ`. Incluye constructores sobrecargados: uno para redes 1D (`N`) y otro para redes 2D (`Lx`, `Ly`) que inicializan la conectividad apropiada. Permite inicializar nodos con valores iniciales o un impulso en el centro. |
| **`WavePropagator`** | Encapsula el bucle temporal de la simulación. Dentro de una única región `#pragma omp parallel` itera en el tiempo actualizando las amplitudes de todos los nodos con un `#pragma omp for` sobre el vector de nodos. Utiliza la operación `reduction` para sumar la energía global y, al final de cada paso, realiza el commit del doble buffer. Soporta fuente sinusoidal global opcional de forma S(t) = S0 · sin(ω t). |
| **`Benchmark`**  | Ejecuta la simulación repetidas veces para medir tiempos, calcula medias y desviaciones estándar, y a partir de ellas deriva speedup y eficiencia. Contiene lógica para explorar el efecto del tamaño de chunk dinámico y genera archivos `.dat` con los resultados de cada campaña. |

El `main.cpp` contiene la lógica para parsear argumentos (usando un pequeño analizador propio), construir la red y lanzar la simulación o los benchmarks. La opción `--benchmark` activa las campañas de rendimiento; de lo contrario, se ejecuta una simulación simple.

El doble buffer evita condiciones de carrera: en cada paso de tiempo se leen las amplitudes previas de los vecinos (`a_prev`) y se escriben nuevas amplitudes (`a`). Al final de cada iteración, un commit copia `a` en `a_prev`. De esta manera, distintos hilos pueden leer y escribir nodos diferentes sin interferencia. Solo las reducciones de energía requieren sincronización (vía `reduction(+:E_global)`).

## 3 Requisitos y dependencias

### Compilación

- **g++** compatible con C++17 y con soporte para OpenMP (por ejemplo, `g++ -fopenmp`).
- **make** para usar el Makefile.

### Scripts de análisis y gráficos (opcional)

Se proporcionan scripts en `wave_propagation/scripts` para crear gráficos y videos. Requieren:

- Python 3.8 o superior.
- Paquetes: `numpy`, `matplotlib`, `tqdm`, `imageio` (y opcionalmente `scipy` para suavizado). En Debian/Ubuntu pueden instalarse mediante:

```bash
sudo apt-get install python3 python3-numpy python3-matplotlib python3-tqdm
pip install imageio
```

## 4 Compilación del simulador

Desde el directorio raíz `wave_propagation`, ejecuta:

```bash
make
```

Esto genera el binario `wave_propagation` en el mismo directorio. Para limpiar artefactos y resultados previos:

```bash
make clean
```

## 5 Ejecución de simulaciones

Una vez compilado, el programa se ejecuta así:

```bash
./wave_propagation [opciones]
```

Las opciones más importantes son:

| Opción                            | Descripción |
|----------------------------------|-------------|
| `--network {1d,2d}`              | Selecciona red 1D o 2D (por defecto 2D). |
| `--N N`                          | Número de nodos en 1D. |
| `--Lx Lx --Ly Ly`                | Dimensiones de la malla 2D. |
| `--D valor`                      | Coeficiente de difusión `D` (default 0.1). |
| `--gamma valor`                  | Coeficiente de amortiguamiento `γ` (default 0.0). |
| `--dt valor`                     | Paso temporal Δt para el integrador (default 0.1). |
| `--steps iter`                   | Número de iteraciones de la simulación. |
| `--S0 valor`                     | Amplitud de la fuente sinusoidal global (0 la desactiva). |
| `--omega valor`                  | Frecuencia angular de la fuente sinusoidal (ω). |
| `--schedule {static,dynamic,guided,auto}` | Tipo de schedule para el bucle paralelo principal. |
| `--chunk n \| auto`             | Tamaño de chunk para `schedule(dynamic)` o `guided`. Si se usa `auto`, se estima una heurística (256 en nuestras pruebas). |
| `--threads n`                    | Número de hilos a usar (puede reemplazar a `OMP_NUM_THREADS`). |
| `--noise {none,single,pernode}` | Tipo de ruido inicial (para excitación aleatoria). |
| `--dump-frames`                  | Guarda un archivo `results/frames/amp_tXXXX.txt` cada `--frame-every` pasos para generar videos. |
| `--frame-every n`                | Intervalo de pasos entre frames (por defecto 1). |
| `--benchmark`                    | Ejecuta las campañas de benchmarking en lugar de una simulación simple. |
| `--help`                         | Muestra la ayuda detallada y sale. |

Ejemplo 1D:

```bash
./wave_propagation --network 1d --N 5000 --steps 2000 --schedule static --threads 4
```

Ejemplo 2D con ruido local y excitación sinusoidal:

```bash
./wave_propagation --network 2d --Lx 200 --Ly 200 \
                  --steps 1000 --D 0.1 --gamma 0.005 --dt 0.05 \
                  --noise pernode --S0 1.0 --omega 10.0 \
                  --schedule dynamic --chunk auto --threads 8 \
                  --dump-frames --frame-every 10
```

Durante la ejecución normal se imprimirá `OK. Resultados en results/` y se guardará un archivo `results/energy_trace.dat` con la energía media en cada paso. Si se activó `--dump-frames`, se crearán además los archivos `results/frames/amp_tXXXX.txt` o `.csv` para cada frame.

## 6 Medición de rendimiento y benchmarking

Para reproducir los experimentos de rendimiento reportados en el informe, se provee el objetivo de make:

```bash
make benchmark
```

Esto hace lo siguiente:

1. Ejecuta el binario con la opción `--benchmark`. El código de `Benchmark` se encarga de medir el tiempo de simulación para diferentes números de hilos y registrar las medias y desviaciones estándar en `results/scaling.dat`. También explora distintos tamaños de chunk dinámico y guarda los resultados en `results/time_vs_chunk_dynamic.dat`.
2. Corre los scripts de graficado en `scripts/` para producir imágenes listas para incluir en el informe: `speedup.png`, `efficiency.png`, `amdahl.png` y `time_vs_chunk_dynamic.png`. Estas se guardarán en `results/`.
3. Finalmente, realiza una simulación 2D larga para obtener la evolución de la energía y genera `results/energy_plot.png`.

El archivo `results/scaling.dat` contiene columnas en el orden:
```
threads  mean_time  std_time  speedup  speedup_err  efficiency  efficiency_err
```
Los errores (`*_err`) se calculan mediante propagación de la incertidumbre usando las desviaciones estándar de los tiempos.

El script `scripts/plot_amdahl.py` ajusta los puntos de speedup a la predicción de la Ley de Amdahl y genera la figura `amdahl.png` indicando la fracción serial estimada `f`.

## 7 Generación de videos con visualización mejorada

Una vez que la simulación ha producido los archivos de frames (`--dump-frames`), se puede convertir la secuencia en un video animado usando el script mejorado `scripts/make_video.py`. Este script soporta visualizaciones 1D y 2D/3D con múltiples opciones:

- `--mark-peak` resalta el nodo de mayor amplitud en cada frame.
- `--zoom1d FRAC` en modo 1D centra el gráfico en torno al pico y muestra solo una fracción FRAC del dominio (entre 0 y 1).
- `--interp {nearest,bilinear,bicubic}` selecciona el método de interpolación para suavizar las superficies en 2D/3D.
- `--cmap NOMBRE` elige un colormap perceptualmente uniforme (recomendado `viridis` o `cividis`).
- `--colorbar` añade una barra de colores indicando la escala de amplitudes.
- `--zero-center` centra la escala de color en 0 para resaltar valores positivos y negativos.
- `--outdir` y `--format` determinan el directorio y formato del video (por ejemplo, `gif` o `mp4`).

Ejemplo para generar un GIF de una simulación 2D previamente realizada:

```bash
python3 scripts/make_video.py results/frames --mode 2d --format gif --fps 15 --outdir videos \
    --zero-center --mark-peak --interp bilinear --cmap viridis --colorbar
```

Para 1D, se puede usar `--zoom1d` para enfocar la región activa:

```bash
python3 scripts/make_video.py results/frames --mode 1d --format gif --fps 20 --outdir videos \
    --zero-center --mark-peak --zoom1d 0.5
```

El Makefile ya incluye un target `make videos` que automatiza la generación de un video 1D y uno 2D de ejemplo con parámetros preconfigurados. Estos comandos ejecutan simulaciones breves, guardan los frames y luego llaman al script de video con opciones que se ha comprobado que proporcionan buena calidad visual (`bilinear`, `viridis`, `colorbar`, etc.).

### Detalles de implementación de `make_video.py`

El script contiene tres clases internas:

1. **`Renderer1D`**: genera gráficos 1D (curva y área bajo la curva) con opciones de marcado de pico y zoom. Se utiliza la configuración global de Matplotlib sin especificar colores absolutos salvo para destacar el pico.
2. **`Renderer3D`**: crea superficies 3D para visualizar las mallas 2D. Para evitar que las amplitudes pequeñas queden planas, recorta alrededor de la región activa y escala la altura del gráfico de manera que el pico ocupe una proporción significativa del eje z; los colores se asignan según la amplitud física, y se puede añadir barra de colores.
3. **`Renderer2D`** (no activado por defecto): opcionalmente puede dibujar los datos 2D como heatmap 2D si se decide prescindir de la vista 3D.

Estas clases cargan los frames desde archivos `.txt` o `.csv`, realizan el downsampling e interpolación solicitados y convierten cada figura en un array RGB que se pasa a `imageio` para crear el video final.

## 8 Interpretación de los resultados

Una vez obtenidos los gráficos de speedup y eficiencia, verás que el speedup crece al aumentar los hilos pero se estanca antes de llegar al valor ideal. La Ley de Amdahl permite estimar la fracción serial de la aplicación (f ≈ 0.07 en nuestras pruebas), lo cual limita el speedup máximo. La eficiencia disminuye con más hilos porque la parte serial domina y la sobrecarga de coordinación entre hilos crece.

En cuanto a la visualización de la energía (gráfico de la sección anterior), un valor de γ mayor que 0 produce un decaimiento exponencial de la energía en el tiempo, confirmando que la implementación conserva la física del modelo. El uso de una fuente global (`S0` y `ω`) permite excitar la onda de manera persistente y observar cómo se estabiliza el sistema.

## 9 Conclusiones y recomendaciones

El proyecto demuestra que con un diseño limpio orientado a objetos y el uso adecuado de OpenMP es posible paralelizar una simulación física con eficiencia razonable. El doble buffer y las reducciones permiten evitar condiciones de carrera. Se observa que la fusión del bucle temporal en una sola región paralela reduce el overhead de creación de hilos.

Para maximizar el rendimiento:

- Emplea `schedule(static)` o `dynamic` con un tamaño de chunk moderado (alrededor de 256) cuando la carga por iteración es homogénea.
- Ajusta el número de hilos acorde al hardware disponible; demasiados hilos pueden reducir la eficiencia.
- Si la red es grande o las variaciones de amplitud son pequeñas, usa las opciones de recorte y escalado de `make_video.py` para mejorar la visibilidad en los videos.

Este README proporciona todas las instrucciones necesarias para compilar, ejecutar, medir y visualizar la simulación. Para detalles más técnicos del algoritmo y análisis de rendimiento, consulta el informe en la carpeta `results/` generado a partir del laboratorio.