#pragma once
#include "Network.h"

class MetricsCalculator {
public:
    static void energy_reduction(const Network& net, double& E);
    static void energy_atomic(const Network& net, double& E);
    static void energy_critical(const Network& net, double& E);
};
