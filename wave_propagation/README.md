# Simulador de Propagación de Ondas en Redes

Este proyecto implementa un simulador de propagación de ondas en redes de nodos
utilizando programación paralela con **OpenMP** en C++. El objetivo es estudiar
el comportamiento de una señal que se transmite entre nodos conectados y
analizar el rendimiento de distintas configuraciones paralelas.

## Estructura del proyecto

```
wave_propagation/
├── main.cpp                # Programa principal
├── Node.h / Node.cpp       # Clase Node para representar nodos
├── Network.h / Network.cpp # Clase Network para manejar la topología
├── WavePropagator.h/.cpp   # Clase WavePropagator para la integración temporal
├── Integrator.h/.cpp       # Clase Integrator (envoltorio de WavePropagator)
├── MetricsCalculator.h/.cpp# Cálculo de métricas de performance
├── Benchmark.h/.cpp        # Ejecución de benchmarks y generación de datos
├── Visualizer.h/.cpp       # Generación de gráficas a partir de datos
├── Makefile                # Script de compilación
└── README.md               # Este archivo
```

## Compilación

Para compilar el programa es necesario tener instalado `g++` con soporte
para OpenMP y C++17. Desde la raíz del proyecto `wave_propagation`, ejecute:

```bash
make
```

Esto generará un ejecutable llamado `wave_propagation`.

## Uso

El programa se puede ejecutar en tres modos:

1. **Simulación básica (por defecto)**

   Ejecuta una pequeña simulación de 100 nodos durante 10 pasos de tiempo y
   muestra la energía total al final.

   ```bash
   ./wave_propagation
   ```

2. **Benchmark de escalabilidad**

   Ejecuta un benchmark variando el número de hilos desde 1 hasta el número
   máximo disponible en el sistema. Repite cada medición 5 veces y calcula
   promedios y desviaciones estándar. Los resultados se guardan en
   `benchmark_results.dat` y se genera una gráfica de speedup en `speedup.png`.

   ```bash
   ./wave_propagation -benchmark
   ```

3. **Análisis de resultados**

   A partir de un archivo de resultados (`benchmark_results.dat`) existente,
   genera las gráficas correspondientes. Actualmente se produce el gráfico
   de speedup. Se asume que el archivo `benchmark_results.dat` ya existe.

   ```bash
   ./wave_propagation -analysis
   ```

4. **Benchmark de schedules**

   Ejecuta pruebas que comparan diferentes tipos de `schedule` de OpenMP
   (`static`, `dynamic`, `guided`) y varios tamaños de `chunk`. Los resultados se
   guardan en `schedule_results.dat`. Actualmente la aplicación no genera
   automáticamente las gráficas para este benchmark; se pueden utilizar los
   scripts de Python proporcionados (o el método `Visualizer`) para procesar
   dicho archivo.

   ```bash
   ./wave_propagation -schedule
   ```

## Limpieza

Para limpiar el directorio de build y eliminar archivos generados (ejecutable,
datos y gráficas), utilice:

```bash
make clean
```

## Notas

- Este proyecto es un ejemplo didáctico inspirado en un laboratorio universitario.
- Las implementaciones de métodos avanzados (por ejemplo, variaciones de
  scheduling o sincronización) se incluyen principalmente como demostración.
- Para una experimentación más completa, considere ajustar el tamaño de la red,
  el número de pasos de tiempo y el número de repeticiones en los benchmarks.