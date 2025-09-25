// WavePropagator.cpp
// Implementación de los métodos de la clase WavePropagator. Esta clase
// encapsula la lógica de integración y cálculo de métricas físicas sobre una
// Network, con variaciones de cláusulas OpenMP demostrando distintos
// comportamientos de sincronización.

#include "WavePropagator.h"
#include <omp.h>
#include <cmath>
#include <iostream>

WavePropagator::WavePropagator(Network *net, double dt)
    : network(net), time_step(dt) {
    if (network) {
        // Aseguramos que la red utilice el mismo paso de tiempo
        network->setTimeStep(dt);
    }
}

void WavePropagator::integrateEuler() {
    if (!network) return;
    // Utiliza el método de propagación por defecto de la red
    network->propagateWaves();
}

void WavePropagator::integrateEuler(int sync_type) {
    if (!network) return;
    // Ejecuta la actualización de Euler dentro de una región paralela
    #pragma omp parallel
    {
        // Llama al método auxiliar de la red con tipo de sincronización
        network->eulerUpdate(sync_type, true);
    }
}

void WavePropagator::integrateEuler(int sync_type, bool use_barrier) {
    if (!network) return;
    #pragma omp parallel
    {
        network->eulerUpdate(sync_type, use_barrier);
    }
}

double WavePropagator::calculateEnergy() {
    if (!network) return 0.0;
    const auto &nodes = network->getNodes();
    int N = static_cast<int>(nodes.size());
    double total_energy = 0.0;
    // Reducción de suma paralela
    #pragma omp parallel for reduction(+:total_energy)
    for (int i = 0; i < N; ++i) {
        double amp = nodes[i].getAmplitude();
        total_energy += amp * amp;
    }
    return total_energy;
}

double WavePropagator::calculateEnergy(int method) {
    // method: 0 = reducción, 1 = atomic
    if (!network) return 0.0;
    const auto &nodes = network->getNodes();
    int N = static_cast<int>(nodes.size());
    double total = 0.0;
    if (method == 1) {
        // Suma con atomic
        #pragma omp parallel
        {
            double local = 0.0;
            #pragma omp for nowait
            for (int i = 0; i < N; ++i) {
                double amp = nodes[i].getAmplitude();
                local += amp * amp;
            }
            #pragma omp atomic
            total += local;
        }
    } else {
        // Reducción por defecto
        #pragma omp parallel for reduction(+:total)
        for (int i = 0; i < N; ++i) {
            double amp = nodes[i].getAmplitude();
            total += amp * amp;
        }
    }
    return total;
}

double WavePropagator::calculateEnergy(int method, bool use_private) {
    // use_private controla si se utilizan variables private o shared
    if (!network) return 0.0;
    const auto &nodes = network->getNodes();
    int N = static_cast<int>(nodes.size());
    double total = 0.0;
    if (method == 1) {
        // atomic
        #pragma omp parallel
        {
            double local_sum = use_private ? 0.0 : total;
            #pragma omp for nowait
            for (int i = 0; i < N; ++i) {
                double amp = nodes[i].getAmplitude();
                local_sum += amp * amp;
            }
            #pragma omp atomic
            total += local_sum;
        }
    } else {
        // reduce
        if (use_private) {
            #pragma omp parallel
            {
                double priv = 0.0;
                #pragma omp for nowait
                for (int i = 0; i < N; ++i) {
                    double amp = nodes[i].getAmplitude();
                    priv += amp * amp;
                }
                #pragma omp atomic
                total += priv;
            }
        } else {
            #pragma omp parallel for reduction(+:total)
            for (int i = 0; i < N; ++i) {
                double amp = nodes[i].getAmplitude();
                total += amp * amp;
            }
        }
    }
    return total;
}

void WavePropagator::processNodes() {
    if (!network) return;
    const auto &nodes = network->getNodes();
    int N = static_cast<int>(nodes.size());
    // Proceso básico: recorre y calcula la suma de grados (se usa como ejemplo)
    int total_degree = 0;
    #pragma omp parallel for reduction(+:total_degree)
    for (int i = 0; i < N; ++i) {
        total_degree += nodes[i].getDegree();
    }
    // Para demostrar el uso, imprimimos la suma total de grados (solo un hilo)
    #pragma omp single
    {
        std::cout << "Suma total de grados de nodos: " << total_degree << std::endl;
    }
}

void WavePropagator::processNodes(int task_type) {
    if (!network) return;
    const auto &nodes = network->getNodes();
    int N = static_cast<int>(nodes.size());
    if (task_type == 0) {
        // Crear tareas para procesar cada nodo
        #pragma omp parallel
        {
            #pragma omp single
            {
                for (int i = 0; i < N; ++i) {
                    #pragma omp task firstprivate(i)
                    {
                        // Trabajo simulado: calcula la raíz cuadrada de la amplitud
                        volatile double val = std::sqrt(nodes[i].getAmplitude());
                        (void)val;
                    }
                }
            }
        }
    } else {
        // Uso de parallel for para procesar nodos
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            volatile double val = std::sqrt(nodes[i].getAmplitude());
            (void)val;
        }
    }
}

void WavePropagator::processNodes(int task_type, bool use_single) {
    if (!network) return;
    const auto &nodes = network->getNodes();
    int N = static_cast<int>(nodes.size());
    // task_type: 0 -> usar tareas, 1 -> usar parallel for
    if (task_type == 0) {
        // Creación de tareas. Se usa single para crear las tareas en un solo hilo,
        // independientemente de use_single, ya que las tareas se ejecutarán en
        // paralelo por otros hilos disponibles.
        #pragma omp parallel
        {
            #pragma omp single
            {
                for (int i = 0; i < N; ++i) {
                    #pragma omp task firstprivate(i)
                    {
                        volatile double val = std::sqrt(nodes[i].getAmplitude());
                        (void)val;
                    }
                }
            }
        }
    } else {
        // parallel for: se puede controlar si toda la operación se realiza en un
        // único hilo o en paralelo.
        #pragma omp parallel
        {
            if (use_single) {
                // Ejecutar de forma secuencial en un único hilo
                #pragma omp single
                {
                    for (int i = 0; i < N; ++i) {
                        volatile double val = std::sqrt(nodes[i].getAmplitude());
                        (void)val;
                    }
                }
            } else {
                #pragma omp for
                for (int i = 0; i < N; ++i) {
                    volatile double val = std::sqrt(nodes[i].getAmplitude());
                    (void)val;
                }
            }
        }
    }
}

void WavePropagator::simulatePhasesBarrier() {
    if (!network) return;
    int N = network->getSize();
    // Simulación simple de dos fases con barrera explícita
    #pragma omp parallel
    {
        // Fase 1: copiar amplitudes previas en paralelo
        #pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            // Copiamos la amplitud actual a previous_amplitude
            // Para modificar el vector de nodos usamos const_cast, ya que getNodes() devuelve un const&
            auto &nodesRef = const_cast<std::vector<Node>&>(network->getNodes());
            nodesRef[i].setPreviousAmplitude(nodesRef[i].getAmplitude());
        }
        // Sincronización explícita entre fases
        #pragma omp barrier
        // Fase 2: actualización de amplitudes utilizando eulerUpdate con nowait
        // Aquí usamos sync_type=2 (nowait) para evitar la barrera implícita al final del for
        network->eulerUpdate(2, false);
        // Sincronizar antes de calcular energía
        #pragma omp barrier
        // Fase 3: cálculo de energía (solo un hilo)
        #pragma omp single
        {
            double energy = calculateEnergy();
            std::cout << "Energía total (con barrera): " << energy << std::endl;
        }
    }
}

void WavePropagator::parallelInitializationSingle() {
    if (!network) return;
    auto &nodes = const_cast<std::vector<Node>&>(network->getNodes());
    int N = static_cast<int>(nodes.size());
    // Inicializar amplitudes en paralelo pero asegurando que la constante inicial
    // sea establecida por un solo hilo
    #pragma omp parallel
    {
        #pragma omp single
        {
            // Establecemos un valor inicial distinto para todos los nodos
            for (int i = 0; i < N; ++i) {
                nodes[i].setAmplitude(1.0);
            }
        }
    }
}

double WavePropagator::calculateMetricsFirstprivate() {
    if (!network) return 0.0;
    const auto &nodes = network->getNodes();
    int N = static_cast<int>(nodes.size());
    double result = 0.0;
    // Ejemplo: suma de amplitudes usando firstprivate en el acumulador
    #pragma omp parallel for firstprivate(result)
    for (int i = 0; i < N; ++i) {
        result += nodes[i].getAmplitude();
    }
    // Debido a firstprivate, cada hilo tiene su propia copia y al finalizar no se combina.
    // Para ilustrar el resultado final, devolvemos la copia final del hilo maestro.
    return result;
}

double WavePropagator::calculateFinalStateLastprivate() {
    if (!network) return 0.0;
    const auto &nodes = network->getNodes();
    int N = static_cast<int>(nodes.size());
    double last_value = 0.0;
    // Ejemplo: encontrar la amplitud del último nodo usando lastprivate
    #pragma omp parallel for lastprivate(last_value)
    for (int i = 0; i < N; ++i) {
        // Simplemente asignamos la amplitud; el valor del último hilo quedará en last_value
        last_value = nodes[i].getAmplitude();
    }
    return last_value;
}