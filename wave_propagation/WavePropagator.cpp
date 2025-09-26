#include "WavePropagator.h"

#include <omp.h>

#include "Network.h"
#include "NetworkStatePrinter.h"

WavePropagator::WavePropagator(Network& net, int steps)
    : network(net), timeSteps(steps) {}

void WavePropagator::setSteps(int steps) {
    timeSteps = steps;
}

int WavePropagator::getSteps() const {
    return timeSteps;
}

void WavePropagator::integrateEuler() {
    integrateEulerSchedule(0);
}

void WavePropagator::integrateEulerSchedule(int scheduleType, int chunkSize,
                                            NetworkStatePrinter* printer,
                                            int visualizeLimit) {
    omp_set_dynamic(0);

    omp_sched_t scheduleKind;
    int chunk = chunkSize;
    switch (scheduleType) {
        case 1:
            scheduleKind = omp_sched_dynamic;
            if (chunk < 1) {
                chunk = 1;
            }
            break;
        case 2:
            scheduleKind = omp_sched_guided;
            if (chunk < 1) {
                chunk = 1;
            }
            break;
        default:
            scheduleKind = omp_sched_static;
            if (chunk < 1) {
                chunk = 0;
            }
            break;
    }
    omp_set_schedule(scheduleKind, chunk);

    std::vector<Node>& nodes = network.nodesMutable();
    const int totalNodes = network.size();
    const double diffusion = network.getDiffusionCoeff();
    const double damping = network.getDampingCoeff();
    const double dt = network.getTimeStep();

    #pragma omp parallel
    {
        for (int step = 0; step < timeSteps; ++step) {
            #pragma omp for schedule(runtime)
            for (int i = 0; i < totalNodes; ++i) {
                nodes[i].computeNewValue(diffusion, damping, dt, nodes);
            }

            #pragma omp single
            {
                network.applyNewValues();
                if (printer != nullptr && totalNodes <= visualizeLimit) {
                    printer->printStep(network, step + 1);
                }
            }
        }
    }
}

