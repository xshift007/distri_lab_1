// WavePropagator.h
// Clase responsable de la integración temporal y cálculo de métricas físicas
// utilizando la red suministrada. Implementa sobrecarga de métodos para
// diferentes tipos de sincronización (atomic, critical, nowait) y
// cálculos de energía usando varias cláusulas de OpenMP.

#pragma once

#include "Network.h"
#include <vector>

class WavePropagator {
private:
    Network *network;         // Referencia a la red sobre la cual operar
    double time_step;         // Paso de tiempo local (puede diferir del de la red)

public:
    // Constructor que recibe un puntero a la red y un paso de tiempo
    WavePropagator(Network *net, double dt);

    // Integración básica mediante método de Euler utilizando scheduling por defecto
    void integrateEuler();

    // Integración con sincronización específica (atomic=0, critical=1, nowait=2)
    void integrateEuler(int sync_type);

    // Integración con sincronización y uso de barrera opcional
    void integrateEuler(int sync_type, bool use_barrier);

    // Cálculo de energía total (sumatorio de amplitud^2) usando reducción
    double calculateEnergy();

    // Cálculo de energía con método específico (reduce=0, atomic=1)
    double calculateEnergy(int method);

    // Cálculo de energía con método y uso de variables private/shared
    double calculateEnergy(int method, bool use_private);

    // Procesamiento básico de nodos (puede ser usado para preprocesar la red)
    void processNodes();

    // Procesamiento de nodos con tipo de ejecución (task=0, parallel_for=1)
    void processNodes(int task_type);

    // Procesamiento de nodos con tipo de ejecución y uso de cláusula single
    void processNodes(int task_type, bool use_single);

    // Simula fases con barrera explícita entre etapas
    void simulatePhasesBarrier();

    // Inicialización paralela con cláusula single
    void parallelInitializationSingle();

    // Cálculo de métricas usando firstprivate
    double calculateMetricsFirstprivate();

    // Cálculo del estado final usando lastprivate
    double calculateFinalStateLastprivate();
};