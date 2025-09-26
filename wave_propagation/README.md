# Simulador de propagación de ondas

Este laboratorio modela la propagación de una señal sobre redes regulares mediante un
enfoque orientado a objetos. El código final se organiza en clases con responsabilidades
claras y acota la experimentación a topologías 1D y 2D, tal como permiten las
instrucciones de la ayudantía.

## Arquitectura orientada a objetos

- **Node** encapsula el estado de cada vértice (amplitud actual, vecinos y valor
  provisional). Expone operaciones para actualizar la amplitud a partir de la ecuación
de difusión.
- **Network** administra la topología (línea 1D o grilla 2D) y los parámetros físicos.
  Ofrece utilidades para inicializar conexiones, reiniciar la condición inicial y aplicar
  los valores calculados.
- **WavePropagator** realiza la integración temporal (Euler explícito) con una única
  región paralela que reutiliza hilos y soporta los _schedules_ estático, dinámico y
  guiado.
- **NetworkStatePrinter**, **PerformanceMetrics** y **Benchmark** separan la
  visualización, el cálculo de métricas (promedio, desviación estándar, _speedup_,
  eficiencia, fracción serial) y la ejecución de experimentos.

Esta división evita concentrar la lógica en `main.cpp` o en una sola clase, y mantiene el
proyecto estrictamente dentro del paradigma orientado a objetos.

## Flexibilidad de OpenMP

Se emplean tres variantes de `schedule` (`static`, `dynamic`, `guided`) y la cláusula
`runtime` para permitir que el tipo de planificación se configure desde las funciones de
medición. No se incluyen cláusulas adicionales (por ejemplo `collapse`, regiones `task`
o sincronizaciones `atomic/critical`) porque el foco del informe está en el impacto del
_scheduling_. Esta omisión está permitida por la flexibilidad indicada en la pauta y se
encuentra documentada en el código.

## Topologías consideradas

Las rutinas de inicialización de la clase `Network` sólo construyen redes en línea (1D) y
grilas regulares (2D). El `main` demuestra ambos casos:

1. Visualización de la propagación paso a paso en una línea 1D de 10 nodos.
2. Visualización equivalente en una grilla 2D de 4×4 nodos.
3. Benchmark de rendimiento sobre una línea 1D de 10 000 nodos para analizar _speedup_ y
eficiencia con 1–4 hilos.

## Compilación y ejecución

```bash
make
./wave_propagation
```

Compilar con `-fopenmp` (incluido en el `Makefile`) es obligatorio para habilitar las
secciones paralelas.
