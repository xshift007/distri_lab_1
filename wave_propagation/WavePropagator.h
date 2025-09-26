#pragma once
#include "Network.h"
#include <cmath>
#include <string>

enum class ScheduleType { Static=0, Dynamic=1, Guided=2 };
enum class SyncMethod   { Reduction=0, Atomic=1, Critical=2 };

class WavePropagator {
    Network* net_;
    double dt_;
    double S0_;
    double omega_;
    double tcur_ = 0.0;
    double lastprivate_val_ = 0.0;
public:
    WavePropagator(Network* n, double dt, double S0=0.0, double omega=0.0)
    : net_(n), dt_(dt), S0_(S0), omega_(omega) {}
    void step_schedule(ScheduleType st, int chunk);
    void energy_reduction(double& E);
    void energy_atomic(double& E);
    void energy_critical(double& E);
    void commit_with_barrier();
    void commit_with_nowait_example();
    void process_with_tasks(int steps, int grain, SyncMethod sm, const char* energy_trace_path);
    void demo_data_clauses(const char* out_path);
    void run(int steps, ScheduleType st, int chunk, SyncMethod sm, const char* energy_trace_path);
};
