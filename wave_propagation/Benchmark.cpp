// Benchmark.cpp
// Implementación de la clase Benchmark. Esta clase proporciona métodos para
// ejecutar y medir experimentos de rendimiento sobre la simulación de
// propagación de ondas con OpenMP.

#include "Benchmark.h"
#include <fstream>
#include <iostream>
#include <omp.h>

Benchmark::Benchmark(int iters, int reps) : iterations(iters), repetitions(reps) {}

double Benchmark::runSimulation(Network &net, int steps) {
    double start = omp_get_wtime();
    for (int i = 0; i < steps; ++i) {
        net.propagateWaves();
    }
    double end = omp_get_wtime();
    return end - start;
}

double Benchmark::runIntegration(WavePropagator &prop, int sync_type, int steps) {
    double start = omp_get_wtime();
    for (int i = 0; i < steps; ++i) {
        prop.integrateEuler(sync_type);
    }
    double end = omp_get_wtime();
    return end - start;
}

void Benchmark::runScalabilityBenchmark(int maxThreads, const std::string &filename) {
    // Abrir archivo para escribir resultados
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "No se pudo abrir el archivo de salida: " << filename << std::endl;
        return;
    }
    // Escribir cabecera
    out << "#threads\tT_mean\tT_std\tSpeedup\tSigma_S\tEfficiency\tSigma_E\tSerialFraction\tAmdahlPred" << std::endl;

    // Determinar parámetros físicos por defecto
    int size = 1000;              // Número de nodos (ajustar si se desea)
    double D = 0.1;               // Coeficiente de difusión
    double gamma = 0.01;          // Coeficiente de amortiguación
    double dt = 0.01;             // Paso de tiempo

    // Vector para almacenar tiempos seriales en p=1 para cálculo de speedup
    double serial_mean = 0.0;
    double serial_std = 0.0;
    std::vector<double> serial_times;

    // Primera pasada: p=1 para obtener T1
    {
        std::vector<double> times;
        omp_set_num_threads(1);
        for (int rep = 0; rep < repetitions; ++rep) {
            // Crear y configurar red
            Network net(size, D, gamma, dt);
            net.initializeRegularNetwork(1);
            // Inicializar amplitudes
            // Para reproducibilidad, se establece la amplitud inicial a 1.0 en todos los nodos
            WavePropagator wp(&net, dt);
            wp.parallelInitializationSingle();
            double t = runSimulation(net, iterations);
            times.push_back(t);
        }
        serial_mean = MetricsCalculator::computeMean(times);
        serial_std = MetricsCalculator::computeStdDev(times, serial_mean);
        serial_times = times;
        // Registrar resultado para p=1
        double speedup = 1.0;
        double sigmaSpeed = 0.0;
        double efficiency = 1.0;
        double sigmaEff = 0.0;
        // Fracción serial estimada usando Sp=1 -> f=1 (todo serial)
        double f = 1.0;
        double amdahlPred = 1.0;
        out << 1 << "\t" << serial_mean << "\t" << serial_std << "\t" << speedup << "\t" << sigmaSpeed
            << "\t" << efficiency << "\t" << sigmaEff << "\t" << f << "\t" << amdahlPred << std::endl;
    }

    // Para p > 1
    for (int p = 2; p <= maxThreads; ++p) {
        std::vector<double> times;
        omp_set_num_threads(p);
        for (int rep = 0; rep < repetitions; ++rep) {
            Network net(size, D, gamma, dt);
            net.initializeRegularNetwork(1);
            WavePropagator wp(&net, dt);
            wp.parallelInitializationSingle();
            double t = runSimulation(net, iterations);
            times.push_back(t);
        }
        double meanT = MetricsCalculator::computeMean(times);
        double stdT = MetricsCalculator::computeStdDev(times, meanT);
        double speedup = MetricsCalculator::computeSpeedup(serial_mean, meanT);
        // Calcular error en speedup usando propagación de errores
        double sigmaSpeed = MetricsCalculator::computeSpeedupError(serial_mean, serial_std, meanT, stdT);
        double efficiency = MetricsCalculator::computeEfficiency(speedup, p);
        double sigmaEff = MetricsCalculator::computeEfficiencyError(speedup, sigmaSpeed, p);
        // Estimar fracción serial a partir del speedup observado usando la ley de Amdahl
        // f = (1/Sp - 1/p)/(1 - 1/p)
        double f = 0.0;
        if (p > 1) {
            double invSp = (speedup > 0.0) ? 1.0 / speedup : 0.0;
            f = (invSp - 1.0 / static_cast<double>(p)) / (1.0 - 1.0 / static_cast<double>(p));
            if (f < 0.0) f = 0.0;
            if (f > 1.0) f = 1.0;
        }
        double amdahlPred = MetricsCalculator::computeAmdahl(f, p);
        out << p << "\t" << meanT << "\t" << stdT << "\t" << speedup << "\t" << sigmaSpeed
            << "\t" << efficiency << "\t" << sigmaEff << "\t" << f << "\t" << amdahlPred << std::endl;
    }
    out.close();
}

// Ejecuta benchmarks para evaluar el impacto de distintos tipos de scheduling y tamaños de chunk
void Benchmark::runScheduleBenchmark(const std::string &filename) {
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "No se pudo abrir el archivo de salida: " << filename << std::endl;
        return;
    }
    out << "#schedule_type\tchunk_size\tT_mean\tT_std" << std::endl;
    // Parámetros de la simulación
    int size = 1000;
    double D = 0.1;
    double gamma = 0.01;
    double dt = 0.01;
    // Lista de tipos de scheduling y nombre legible
    std::vector<ScheduleType> scheds = { ScheduleType::Static, ScheduleType::Dynamic, ScheduleType::Guided };
    std::vector<std::string> schedNames = { "static", "dynamic", "guided" };
    std::vector<int> chunks = { 0, 1, 10, 50, 100 };
    for (size_t s = 0; s < scheds.size(); ++s) {
        ScheduleType schedType = scheds[s];
        for (int chunk : chunks) {
            std::vector<double> times;
            for (int rep = 0; rep < repetitions; ++rep) {
                Network net(size, D, gamma, dt);
                net.initializeRegularNetwork(1);
                WavePropagator wp(&net, dt);
                wp.parallelInitializationSingle();
                double start = omp_get_wtime();
                for (int iter = 0; iter < iterations; ++iter) {
                    net.propagateWaves(schedType, chunk);
                }
                double end = omp_get_wtime();
                times.push_back(end - start);
            }
            double meanT = MetricsCalculator::computeMean(times);
            double stdT = MetricsCalculator::computeStdDev(times, meanT);
            out << schedNames[s] << "\t" << chunk << "\t" << meanT << "\t" << stdT << std::endl;
        }
    }
    out.close();
}