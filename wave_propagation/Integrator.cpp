// Integrator.cpp
// Implementación de la clase Integrator que delega en WavePropagator la
// integración temporal.

#include "Integrator.h"

Integrator::Integrator(WavePropagator *wp) : propagator(wp) {}

void Integrator::runEuler() {
    if (propagator) {
        propagator->integrateEuler();
    }
}

void Integrator::runEuler(int sync_type) {
    if (propagator) {
        propagator->integrateEuler(sync_type);
    }
}

void Integrator::runEuler(int sync_type, bool use_barrier) {
    if (propagator) {
        propagator->integrateEuler(sync_type, use_barrier);
    }
}