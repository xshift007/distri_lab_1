#include "Benchmark.h"

#include <iomanip>
#include <iostream>
#include <omp.h>

#include "Network.h"
#include "NetworkStatePrinter.h"
#include "WavePropagator.h"

Benchmark::Benchmark(int maxThreads_, int repeatsSmall_, int repeatsLarge_)
    : maxThreads(maxThreads_),
      repeatsSmall(repeatsSmall_),
      repeatsLarge(repeatsLarge_),
      printer(nullptr) {}

void Benchmark::setPrinter(NetworkStatePrinter* printerInstance) {
    printer = printerInstance;
}

int Benchmark::getMaxThreads() const {
    return maxThreads;
}

int Benchmark::getRepeatsSmall() const {
    return repeatsSmall;
}

int Benchmark::getRepeatsLarge() const {
    return repeatsLarge;
}

void Benchmark::runDemonstration(Network& network, WavePropagator& propagator, const std::string& title) {
    std::cout << title << std::endl;
    applyCentralImpulse(network);
    omp_set_num_threads(1);
    if (printer != nullptr) {
        propagator.integrateEulerSchedule(0, -1, printer);
    } else {
        propagator.integrateEulerSchedule(0);
    }
    std::cout << std::endl;
}

ScheduleResults Benchmark::measureSchedule(Network& network, WavePropagator& propagator,
                                            int scheduleType, int repeats, const std::string& scheduleName) {
    ScheduleResults results;
    results.scheduleName = scheduleName;

    Measurement baseline;
    bool baselineComputed = false;

    for (int threads = 1; threads <= maxThreads; ++threads) {
        omp_set_num_threads(threads);
        std::vector<double> samples = runTimedSamples(network, propagator, scheduleType, repeats);
        Measurement stats = PerformanceMetrics::computeStatistics(samples);
        if (!baselineComputed) {
            baseline = stats;
            baselineComputed = true;
        }
        ParallelMetrics metrics = PerformanceMetrics::computeParallelMetrics(threads, baseline, stats);
        results.metrics.push_back(metrics);
    }

    return results;
}

void Benchmark::printScheduleResults(const ScheduleResults& results) const {
    std::cout << "\nSchedule " << results.scheduleName << ":" << std::endl;
    std::cout << "#threads\tT_mean\tT_std\tSpeedup\tSigma_S\tEfficiency\tSigma_E\tSerialFraction\tAmdahlPred" << std::endl;
    std::cout.setf(std::ios::fixed);

    for (const ParallelMetrics& metrics : results.metrics) {
        std::cout << metrics.threads << "\t"
                  << std::setprecision(9) << metrics.time.mean << "\t"
                  << std::setprecision(9) << metrics.time.stddev << "\t"
                  << std::setprecision(5) << metrics.speedup << "\t"
                  << std::setprecision(6) << metrics.speedupError << "\t"
                  << std::setprecision(6) << metrics.efficiency << "\t"
                  << std::setprecision(7) << metrics.efficiencyError << "\t"
                  << std::setprecision(6) << metrics.serialFraction << "\t"
                  << std::setprecision(6) << metrics.amdahlPrediction
                  << std::endl;
    }
}

void Benchmark::applyCentralImpulse(Network& network, double amplitude) const {
    network.resetValues(0.0);
    if (network.size() == 0) {
        return;
    }

    if (network.getTopology() == Network::Topology::Line1D) {
        int center = network.size() / 2;
        network.setValueAt(center, amplitude);
    } else {
        int centerRow = network.getRows() / 2;
        int centerCol = network.getCols() / 2;
        int index = network.indexFromCoordinates(centerRow, centerCol);
        network.setValueAt(index, amplitude);
    }
}

std::vector<double> Benchmark::runTimedSamples(Network& network, WavePropagator& propagator,
                                               int scheduleType, int repeats) const {
    std::vector<double> samples;
    samples.reserve(repeats);

    for (int r = 0; r < repeats; ++r) {
        applyCentralImpulse(network);
        double start = omp_get_wtime();
        propagator.integrateEulerSchedule(scheduleType);
        double end = omp_get_wtime();
        samples.push_back(end - start);
    }

    return samples;
}

