#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <string>
#include <vector>

#include "PerformanceMetrics.h"

class Network;
class WavePropagator;
class NetworkStatePrinter;

struct ScheduleResults {
    std::string scheduleName;
    std::vector<ParallelMetrics> metrics;
};

class Benchmark {
public:
    Benchmark(int maxThreads, int repeatsSmall, int repeatsLarge);

    void setPrinter(NetworkStatePrinter* printerInstance);

    int getMaxThreads() const;
    int getRepeatsSmall() const;
    int getRepeatsLarge() const;

    void runDemonstration(Network& network, WavePropagator& propagator, const std::string& title);

    ScheduleResults measureSchedule(Network& network, WavePropagator& propagator,
                                    int scheduleType, int repeats, const std::string& scheduleName);

    void printScheduleResults(const ScheduleResults& results) const;

    void applyCentralImpulse(Network& network, double amplitude = 1.0) const;

private:
    std::vector<double> runTimedSamples(Network& network, WavePropagator& propagator,
                                        int scheduleType, int repeats) const;

    int maxThreads;
    int repeatsSmall;
    int repeatsLarge;
    NetworkStatePrinter* printer;
};

#endif
