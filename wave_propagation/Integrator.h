// Integrator.h
// Clase de utilidad para encapsular la integración temporal. Esta clase
// delega en WavePropagator la implementación de los distintos métodos de
// integración, ofreciendo una interfaz sencilla para ejecutar los
// experimentos desde otras partes del programa.

#pragma once

#include "WavePropagator.h"

class Integrator {
private:
    WavePropagator *propagator;

public:
    explicit Integrator(WavePropagator *wp);
    // Ejecuta la integración básica
    void runEuler();
    // Ejecuta la integración con tipo de sincronización
    void runEuler(int sync_type);
    // Ejecuta la integración con tipo de sincronización y barrera
    void runEuler(int sync_type, bool use_barrier);
};