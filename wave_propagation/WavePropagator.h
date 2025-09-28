#pragma once
#include <string>
#include "Types.h"
#include "Network.h"

class WavePropagator {
    Network* net_;
    double dt_, S0_, omega_;
    double tcur_ = 0.0;
public:
    WavePropagator(Network* n, double dt, double S0, double omega)
        : net_(n), dt_(dt), S0_(S0), omega_(omega) {}

    void run_fused(int steps, ScheduleType st, int chunk, const std::string& energy_out);

    // Access
    double time() const { return tcur_; }
};
