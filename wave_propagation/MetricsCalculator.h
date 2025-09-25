// MetricsCalculator.h
// Clase que implementa métodos estáticos para el cálculo de métricas de
// performance y análisis de aplicaciones paralelas: speedup, eficiencia,
// fracción serial y predicción de la Ley de Amdahl. Incluye además
// utilidades para calcular promedios y desviaciones estándar.

#pragma once

#include <vector>

class MetricsCalculator {
public:
    // Calcula el promedio de una serie de datos
    static double computeMean(const std::vector<double> &data);

    // Calcula la desviación estándar de una serie de datos dada su media
    static double computeStdDev(const std::vector<double> &data, double mean);

    // Calcula el speedup a partir del tiempo serial y el tiempo paralelo
    static double computeSpeedup(double T1, double Tp);

    // Calcula la eficiencia a partir del speedup y el número de hilos
    static double computeEfficiency(double speedup, int p);

    // Calcula la propagación de errores en el speedup
    static double computeSpeedupError(double T1, double sigma1, double Tp, double sigmaP);

    // Calcula la propagación de errores en la eficiencia
    static double computeEfficiencyError(double speedup, double sigmaSpeedup, int p);

    // Calcula la fracción serial a partir de tiempos de ejecución
    static double computeSerialFraction(double Tserial, double Ttotal);

    // Predicción teórica de speedup según la Ley de Amdahl (fracción serial f y número de hilos p)
    static double computeAmdahl(double f, int p);
};