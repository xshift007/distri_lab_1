// MetricsCalculator.cpp
// Implementación de los métodos estáticos de MetricsCalculator.

#include "MetricsCalculator.h"
#include <cmath>

double MetricsCalculator::computeMean(const std::vector<double> &data) {
    if (data.empty()) return 0.0;
    double sum = 0.0;
    for (double v : data) sum += v;
    return sum / static_cast<double>(data.size());
}

double MetricsCalculator::computeStdDev(const std::vector<double> &data, double mean) {
    if (data.size() < 2) return 0.0;
    double var = 0.0;
    for (double v : data) var += (v - mean) * (v - mean);
    var /= static_cast<double>(data.size());
    return std::sqrt(var);
}

double MetricsCalculator::computeSpeedup(double T1, double Tp) {
    if (Tp == 0.0) return 0.0;
    return T1 / Tp;
}

double MetricsCalculator::computeEfficiency(double speedup, int p) {
    if (p <= 0) return 0.0;
    return speedup / static_cast<double>(p);
}

double MetricsCalculator::computeSpeedupError(double T1, double sigma1, double Tp, double sigmaP) {
    if (T1 == 0.0 || Tp == 0.0) return 0.0;
    double Sp = T1 / Tp;
    double rel1 = sigma1 / T1;
    double relP = sigmaP / Tp;
    return Sp * std::sqrt(rel1 * rel1 + relP * relP);
}

double MetricsCalculator::computeEfficiencyError(double /*speedup*/, double sigmaSpeedup, int p) {
    if (p <= 0) return 0.0;
    return sigmaSpeedup / static_cast<double>(p);
}

double MetricsCalculator::computeSerialFraction(double Tserial, double Ttotal) {
    if (Ttotal == 0.0) return 0.0;
    return Tserial / Ttotal;
}

double MetricsCalculator::computeAmdahl(double f, int p) {
    if (p <= 0) return 0.0;
    // Fórmula: 1 / (f + (1 - f) / p)
    return 1.0 / (f + (1.0 - f) / static_cast<double>(p));
}