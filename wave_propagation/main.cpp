// main.cpp
// Programa principal para el simulador de propagación de ondas en redes. Permite
// ejecutar la simulación básica, benchmarks de escalabilidad y análisis de
// resultados desde línea de comandos.

#include "Network.h"
#include "WavePropagator.h"
#include "Integrator.h"
#include "MetricsCalculator.h"
#include "Benchmark.h"
#include "Visualizer.h"

#include <iostream>
#include <string>
#include <omp.h>

int main(int argc, char **argv) {
    // Establecemos un valor razonable de hilos por defecto (máximo disponible)
    int maxThreads = omp_get_max_threads();

    if (argc >= 2) {
        std::string arg1 = argv[1];
        if (arg1 == "-benchmark") {
            std::cout << "Ejecutando benchmark de escalabilidad..." << std::endl;
            Benchmark bench(100, 5); // 100 iteraciones, 5 repeticiones
            bench.runScalabilityBenchmark(maxThreads, "benchmark_results.dat");
            std::cout << "Benchmark completado. Datos guardados en benchmark_results.dat" << std::endl;
            // Generar gráfica de speedup
            Visualizer::plotSpeedup("benchmark_results.dat", "speedup.png");
            std::cout << "Gráfico generado: speedup.png" << std::endl;
            return 0;
        }
        else if (arg1 == "-schedule") {
            std::cout << "Ejecutando benchmark de schedules..." << std::endl;
            Benchmark bench(50, 3); // menos iteraciones por rapidez
            bench.runScheduleBenchmark("schedule_results.dat");
            std::cout << "Benchmark de schedules completado. Datos guardados en schedule_results.dat" << std::endl;
            // El usuario puede generar manualmente las gráficas a partir de schedule_results.dat usando Python
            return 0;
        }
        else if (arg1 == "-analysis") {
            std::cout << "Generando análisis y gráficas a partir de benchmark_results.dat..." << std::endl;
            // Para este ejemplo se genera únicamente el gráfico de speedup
            Visualizer::plotSpeedup("benchmark_results.dat", "speedup.png");
            std::cout << "Análisis completado." << std::endl;
            return 0;
        }
    }
    // Ejecución por defecto: simulación simple de un pequeño número de pasos
    std::cout << "Ejecutando simulación básica de propagación de ondas..." << std::endl;
    int size = 100;          // número de nodos para demostración
    double D = 0.1;
    double gamma = 0.01;
    double dt = 0.01;
    int steps = 10;
    // Crear red 1D regular e inicializar amplitudes
    Network net(size, D, gamma, dt);
    net.initializeRegularNetwork(1);
    WavePropagator propagator(&net, dt);
    propagator.parallelInitializationSingle();
    // Ejecutar algunos pasos de integración
    for (int i = 0; i < steps; ++i) {
        propagator.integrateEuler();
    }
    // Calcular energía final
    double energy = propagator.calculateEnergy();
    std::cout << "Simulación finalizada. Energía total: " << energy << std::endl;
    return 0;
}