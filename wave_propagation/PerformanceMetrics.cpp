#include "PerformanceMetrics.h"

#include <cmath>
#include <numeric>

Measurement PerformanceMetrics::computeStatistics(const std::vector<double>& samples) {
    Measurement stats;
    if (samples.empty()) {
        return stats;
    }

    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    stats.mean = sum / static_cast<double>(samples.size());

    double variance = 0.0;
    for (double value : samples) {
        double diff = value - stats.mean;
        variance += diff * diff;
    }
    variance /= static_cast<double>(samples.size());
    stats.stddev = std::sqrt(variance);

    return stats;
}

ParallelMetrics PerformanceMetrics::computeParallelMetrics(int threads,
                                                           const Measurement& baseline,
                                                           const Measurement& sample) {
    ParallelMetrics metrics;
    metrics.threads = threads;
    metrics.time = sample;

    if (threads <= 0) {
        return metrics;
    }

    if (threads == 1) {
        metrics.speedup = 1.0;
        metrics.efficiency = 1.0;
        metrics.serialFraction = 1.0;
        metrics.amdahlPrediction = 1.0;
        return metrics;
    }

    if (baseline.mean <= 0.0 || sample.mean <= 0.0) {
        return metrics;
    }

    metrics.speedup = baseline.mean / sample.mean;
    metrics.efficiency = metrics.speedup / static_cast<double>(threads);

    double relErrBaseline = (baseline.mean != 0.0) ? baseline.stddev / baseline.mean : 0.0;
    double relErrSample = (sample.mean != 0.0) ? sample.stddev / sample.mean : 0.0;
    double relErrSpeedup = std::sqrt(relErrBaseline * relErrBaseline + relErrSample * relErrSample);
    metrics.speedupError = metrics.speedup * relErrSpeedup;
    metrics.efficiencyError = metrics.speedupError / static_cast<double>(threads);

    double serialCandidate = ((sample.mean / baseline.mean) - 1.0 / static_cast<double>(threads)) /
                             (1.0 - 1.0 / static_cast<double>(threads));
    if (serialCandidate < 0.0) {
        serialCandidate = 0.0;
    }
    if (serialCandidate > 1.0) {
        serialCandidate = 1.0;
    }
    metrics.serialFraction = serialCandidate;
    metrics.amdahlPrediction = 1.0 /
        (metrics.serialFraction + (1.0 - metrics.serialFraction) / static_cast<double>(threads));

    return metrics;
}

