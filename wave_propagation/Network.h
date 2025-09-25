// Network.h
// Clase que modela la topología de la red de nodos. Gestiona la creación de
// conexiones (vecindad) entre nodos y proporciona funciones para simular la
// propagación de ondas sobre la red utilizando OpenMP. Incluye diferentes
// variaciones de scheduling y cláusulas de OpenMP mediante sobrecarga de
// funciones.

#pragma once

#include <vector>
#include <random>
#include "Node.h"

// Enumeración de tipos de scheduling para mayor legibilidad.
enum class ScheduleType {
    Static = 0,
    Dynamic = 1,
    Guided = 2
};

class Network {
private:
    std::vector<Node> nodes;       // Lista de nodos de la red
    int network_size;               // Número total de nodos
    double diffusion_coeff;         // Coeficiente de difusión D
    double damping_coeff;           // Coeficiente de amortiguación γ
    double time_step;               // Paso de tiempo Δt
    int dim_x;                      // Dimensión X (para redes 2D)
    int dim_y;                      // Dimensión Y (para redes 2D)

    // Inicializa conexiones en red 1D (cada nodo conectado a su vecino previo y siguiente)
    void initialize1D();

    // Inicializa conexiones en red 2D (malla rectangular)
    void initialize2D(int width, int height);

    // Genera conexiones aleatorias entre nodos
    void initializeRandom();

    // Actualiza todas las amplitudes de los nodos mediante el método de Euler explícito.
    // Esta función es utilizada por propagateWaves() y sus sobrecargas.
    // Los argumentos sync_type controlan la sincronización (atomic=0, critical=1, nowait=2), y
    // use_barrier indica si se aplica barrera entre fases. Declarada en la sección pública.

public:
    // Constructor que inicializa la red con el número de nodos, coeficientes físicos y paso de tiempo
    Network(int size, double diff_coeff, double damp_coeff, double dt);

    // Inicializa la red según el tipo: 1D o 2D; para 2D se pasa dimensión de un lado
    void initializeRegularNetwork(int dimensions = 1);

    // Inicializa una red completamente aleatoria
    void initializeRandomNetwork();

    // Propagación básica utilizando el scheduling por defecto (estático) y sincronización atómica
    void propagateWaves();

    // Expone el método de Euler explícito para ser utilizado por WavePropagator u otras clases.
    void eulerUpdate(int sync_type = 0, bool use_barrier = true);

    // Propagación con tipo de scheduling especificado (estático=0, dinámico=1, guiado=2)
    void propagateWaves(ScheduleType schedule_type);

    // Propagación con tipo de scheduling y tamaño de chunk
    void propagateWaves(ScheduleType schedule_type, int chunk_size);

    // Propagación utilizando la cláusula collapse para loops anidados (solo válido en 2D)
    void propagateWavesCollapse();

    // Acceso a los nodos (lectura)
    const std::vector<Node>& getNodes() const;

    // Devuelve el número de nodos
    int getSize() const;

    // Permite modificar el paso de tiempo
    void setTimeStep(double dt);
};