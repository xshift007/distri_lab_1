// Benchmark.h
// Clase encargada de ejecutar diferentes experimentos de rendimiento sobre
// la simulación de propagación de ondas. Mide tiempos de ejecución con
// diferentes configuraciones de hilos y cláusulas de OpenMP, calcula métricas
// (speedup, eficiencia, error de propagación) y guarda los resultados en
// archivos .dat para su posterior visualización.

#pragma once

#include "Network.h"
#include "WavePropagator.h"
#include "Integrator.h"
#include "MetricsCalculator.h"

#include <string>

class Benchmark {
private:
    int iterations;        // Número de pasos de tiempo a simular en cada ejecución
    int repetitions;       // Número de repeticiones para promediar tiempos

    // Ejecuta una simulación completa (propagando la onda) y devuelve el tiempo de ejecución
    double runSimulation(Network &net, int steps);

    // Ejecuta integrateEuler con sync_type dado y devuelve el tiempo de ejecución
    double runIntegration(WavePropagator &prop, int sync_type, int steps);

public:
    Benchmark(int iters = 100, int reps = 5);
    // Ejecución de un benchmark de escalabilidad variando el número de hilos
    void runScalabilityBenchmark(int maxThreads, const std::string &filename);

    // Benchmark para comparar tipos de scheduling y tamaños de chunk
    void runScheduleBenchmark(const std::string &filename);
};