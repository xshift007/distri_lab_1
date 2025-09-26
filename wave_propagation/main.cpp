#include <iostream>
#include <omp.h>

#include "Benchmark.h"
#include "Network.h"
#include "NetworkStatePrinter.h"
#include "WavePropagator.h"

int main() {
    omp_set_dynamic(0);

    const double diffusion = 0.1;
    const double damping = 0.01;
    const double timeStep = 0.01;

    Benchmark benchmark(/*maxThreads*/4, /*repeatsSmall*/10, /*repeatsLarge*/5);
    NetworkStatePrinter printer(std::cout);
    printer.setPrecision(4);
    benchmark.setPrinter(&printer);

    std::cout << "=== Escenario 1D pequeno ===" << std::endl;
    Network lineSmall(10, diffusion, damping, timeStep);
    lineSmall.buildLine();
    WavePropagator propagatorLineSmall(lineSmall, 20);
    benchmark.runDemonstration(lineSmall, propagatorLineSmall,
                               "Simulacion paso a paso con schedule estatico");

    const char* scheduleNames[] = {"Estatico", "Dinamico", "Guiado"};
    for (int schedule = 0; schedule < 3; ++schedule) {
        ScheduleResults results = benchmark.measureSchedule(lineSmall, propagatorLineSmall,
                                                            schedule, benchmark.getRepeatsSmall(),
                                                            scheduleNames[schedule]);
        benchmark.printScheduleResults(results);
    }

    std::cout << "\n=== Escenario 2D pequeno ===" << std::endl;
    Network gridSmall(16, diffusion, damping, timeStep);
    gridSmall.buildGrid(4, 4);
    WavePropagator propagatorGridSmall(gridSmall, 15);
    benchmark.runDemonstration(gridSmall, propagatorGridSmall,
                               "Simulacion 2D paso a paso con schedule estatico");

    std::cout << "\n=== Escenario 1D mediano ===" << std::endl;
    Network lineMedium(10000, diffusion, damping, timeStep);
    lineMedium.buildLine();
    WavePropagator propagatorLineMedium(lineMedium, 1000);
    for (int schedule = 0; schedule < 3; ++schedule) {
        ScheduleResults results = benchmark.measureSchedule(lineMedium, propagatorLineMedium,
                                                            schedule, benchmark.getRepeatsLarge(),
                                                            scheduleNames[schedule]);
        benchmark.printScheduleResults(results);
    }

    return 0;
}

