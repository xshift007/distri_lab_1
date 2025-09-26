#ifndef PERFORMANCEMETRICS_H
#define PERFORMANCEMETRICS_H

#include <vector>

struct Measurement {
    double mean{0.0};
    double stddev{0.0};
};

struct ParallelMetrics {
    int threads{0};
    Measurement time;
    double speedup{0.0};
    double speedupError{0.0};
    double efficiency{0.0};
    double efficiencyError{0.0};
    double serialFraction{1.0};
    double amdahlPrediction{1.0};
};

class PerformanceMetrics {
public:
    static Measurement computeStatistics(const std::vector<double>& samples);
    static ParallelMetrics computeParallelMetrics(int threads,
                                                  const Measurement& baseline,
                                                  const Measurement& sample);
};

#endif
