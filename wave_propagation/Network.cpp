// Network.cpp
// Implementación de la clase Network. Gestiona la topología de la red y
// proporciona métodos para propagar una onda en paralelo utilizando OpenMP con
// distintos tipos de scheduling.

#include "Network.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <omp.h>

Network::Network(int size, double diff_coeff, double damp_coeff, double dt)
    : network_size(size), diffusion_coeff(diff_coeff), damping_coeff(damp_coeff), time_step(dt), dim_x(0), dim_y(0) {
    // Crear nodos con amplitud inicial cero
    nodes.reserve(network_size);
    for (int i = 0; i < network_size; ++i) {
        nodes.emplace_back(i, 0.0);
    }
}

void Network::initialize1D() {
    // Conectar cada nodo con su vecino izquierdo y derecho
    for (int i = 0; i < network_size; ++i) {
        if (i > 0) {
            nodes[i].addNeighbor(i - 1);
        }
        if (i < network_size - 1) {
            nodes[i].addNeighbor(i + 1);
        }
    }
}

void Network::initialize2D(int width, int height) {
    dim_x = width;
    dim_y = height;
    // Conectar cada nodo con sus vecinos de la malla
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            // Vecino izquierdo
            if (x > 0) {
                nodes[idx].addNeighbor(idx - 1);
            }
            // Vecino derecho
            if (x < width - 1) {
                nodes[idx].addNeighbor(idx + 1);
            }
            // Vecino arriba
            if (y > 0) {
                nodes[idx].addNeighbor(idx - width);
            }
            // Vecino abajo
            if (y < height - 1) {
                nodes[idx].addNeighbor(idx + width);
            }
        }
    }
}

void Network::initializeRandom() {
    // Conexiones aleatorias: cada nodo se conecta con un número aleatorio de vecinos
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist_n(1, 4);          // número de vecinos
    std::uniform_int_distribution<int> dist_id(0, network_size - 1);

    // Limpiar cualquier vecindad existente. En esta implementación los nodos
    // no exponen un método para limpiar su lista de vecinos, por lo que no
    // eliminamos conexiones previas. En un escenario real se debería
    // proporcionar una función clearNeighbors() en Node para reiniciar la
    // conectividad.

    // Generar conexiones bidireccionales
    for (int i = 0; i < network_size; ++i) {
        int num_neighbors = dist_n(gen);
        for (int k = 0; k < num_neighbors; ++k) {
            int j = dist_id(gen);
            if (j == i) continue;
            // Añadir vecino i-j si no existe ya (evitar duplicados básicos)
            bool exists = false;
            for (int n_id : nodes[i].getNeighbors()) {
                if (n_id == j) { exists = true; break; }
            }
            if (!exists) {
                nodes[i].addNeighbor(j);
                nodes[j].addNeighbor(i);
            }
        }
    }
}

void Network::initializeRegularNetwork(int dimensions) {
    if (dimensions <= 1) {
        initialize1D();
    } else {
        // Suponemos una red cuadrada; si size no es un cuadrado perfecto se usa la raíz
        int width = static_cast<int>(std::sqrt(network_size));
        if (width * width != network_size) {
            width = dimensions;
        }
        int height = network_size / width;
        initialize2D(width, height);
    }
}

void Network::initializeRandomNetwork() {
    initializeRandom();
}

// Función auxiliar para la integración mediante Euler con diferentes tipos de
// sincronización (atomic=0, critical=1, nowait=2). Este método no gestiona
// scheduling, ya que se utiliza principalmente desde las funciones
// propagateWaves() que sí controlan la distribución de iteraciones.
void Network::eulerUpdate(int sync_type, bool use_barrier) {
    int N = network_size;
    // Copiar amplitudes actuales a amplitudes previas
    #pragma omp for schedule(static)
    for (int i = 0; i < N; ++i) {
        nodes[i].setPreviousAmplitude(nodes[i].getAmplitude());
    }
    if (use_barrier) {
        #pragma omp barrier
    }

    // Calcular nuevas amplitudes
    #pragma omp for schedule(static)
    for (int i = 0; i < N; ++i) {
        double Ai = nodes[i].getPreviousAmplitude();
        double sum = 0.0;
        const auto &neigh = nodes[i].getNeighbors();
        for (int nbr : neigh) {
            double Aj = nodes[nbr].getPreviousAmplitude();
            sum += (Aj - Ai);
        }
        // Término de difusión y amortiguación; ignoramos fuentes externas aquí
        double delta = diffusion_coeff * sum - damping_coeff * Ai;
        double new_val = Ai + time_step * delta;
        // Sincronización en la escritura de la amplitud
        switch (sync_type) {
        case 1: // critical
            #pragma omp critical
            {
                nodes[i].setAmplitude(new_val);
            }
            break;
        case 2: // nowait (sin secciones críticas, simplemente actualizar)
            nodes[i].setAmplitude(new_val);
            break;
        default: // atomic (no se puede aplicar a llamadas de función, pero la variable es única por iteración)
            nodes[i].setAmplitude(new_val);
            break;
        }
    }
}

void Network::propagateWaves() {
    // Schedule estático por defecto
    int N = network_size;
    #pragma omp parallel
    {
        // Copiar amplitudes actuales
        #pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            nodes[i].setPreviousAmplitude(nodes[i].getAmplitude());
        }
        // Actualizar amplitudes con schedule static
        #pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            double Ai = nodes[i].getPreviousAmplitude();
            double sum = 0.0;
            const auto &neigh = nodes[i].getNeighbors();
            for (int nbr : neigh) {
                double Aj = nodes[nbr].getPreviousAmplitude();
                sum += (Aj - Ai);
            }
            double delta = diffusion_coeff * sum - damping_coeff * Ai;
            double new_val = Ai + time_step * delta;
            nodes[i].setAmplitude(new_val);
        }
    }
}

void Network::propagateWaves(ScheduleType schedule_type) {
    // Delegate to overload with default chunk_size = 0 (OpenMP chooses)
    propagateWaves(schedule_type, 0);
}

void Network::propagateWaves(ScheduleType schedule_type, int chunk_size) {
    int N = network_size;
    // Configurar chunk_size por defecto si no se especifica
    int chunk = (chunk_size > 0) ? chunk_size : 0;
    switch (schedule_type) {
    case ScheduleType::Static:
        #pragma omp parallel
        {
            // Copiar amplitudes
            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) {
                nodes[i].setPreviousAmplitude(nodes[i].getAmplitude());
            }
            #pragma omp for schedule(static, chunk)
            for (int i = 0; i < N; ++i) {
                double Ai = nodes[i].getPreviousAmplitude();
                double sum = 0.0;
                const auto &neigh = nodes[i].getNeighbors();
                for (int nbr : neigh) {
                    double Aj = nodes[nbr].getPreviousAmplitude();
                    sum += (Aj - Ai);
                }
                double delta = diffusion_coeff * sum - damping_coeff * Ai;
                double new_val = Ai + time_step * delta;
                nodes[i].setAmplitude(new_val);
            }
        }
        break;
    case ScheduleType::Dynamic:
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) {
                nodes[i].setPreviousAmplitude(nodes[i].getAmplitude());
            }
            #pragma omp for schedule(dynamic, chunk)
            for (int i = 0; i < N; ++i) {
                double Ai = nodes[i].getPreviousAmplitude();
                double sum = 0.0;
                const auto &neigh = nodes[i].getNeighbors();
                for (int nbr : neigh) {
                    double Aj = nodes[nbr].getPreviousAmplitude();
                    sum += (Aj - Ai);
                }
                double delta = diffusion_coeff * sum - damping_coeff * Ai;
                double new_val = Ai + time_step * delta;
                nodes[i].setAmplitude(new_val);
            }
        }
        break;
    case ScheduleType::Guided:
        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) {
                nodes[i].setPreviousAmplitude(nodes[i].getAmplitude());
            }
            #pragma omp for schedule(guided, chunk)
            for (int i = 0; i < N; ++i) {
                double Ai = nodes[i].getPreviousAmplitude();
                double sum = 0.0;
                const auto &neigh = nodes[i].getNeighbors();
                for (int nbr : neigh) {
                    double Aj = nodes[nbr].getPreviousAmplitude();
                    sum += (Aj - Ai);
                }
                double delta = diffusion_coeff * sum - damping_coeff * Ai;
                double new_val = Ai + time_step * delta;
                nodes[i].setAmplitude(new_val);
            }
        }
        break;
    default:
        propagateWaves();
        break;
    }
}

void Network::propagateWavesCollapse() {
    // Este método aplica la cláusula collapse para colapsar dos loops anidados. Es
    // útil únicamente para redes 2D. Si no es 2D, delega en propagateWaves().
    if (dim_x <= 0 || dim_y <= 0) {
        propagateWaves();
        return;
    }
    // Dimensiones
    int width = dim_x;
    int height = dim_y;
    // Copiar amplitudes previas
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int idx = 0; idx < network_size; ++idx) {
            nodes[idx].setPreviousAmplitude(nodes[idx].getAmplitude());
        }
        // Actualizar amplitudes con loops colapsados (2D)
        #pragma omp for collapse(2) schedule(static)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int i = y * width + x;
                double Ai = nodes[i].getPreviousAmplitude();
                double sum = 0.0;
                const auto &neigh = nodes[i].getNeighbors();
                for (int nbr : neigh) {
                    double Aj = nodes[nbr].getPreviousAmplitude();
                    sum += (Aj - Ai);
                }
                double delta = diffusion_coeff * sum - damping_coeff * Ai;
                double new_val = Ai + time_step * delta;
                nodes[i].setAmplitude(new_val);
            }
        }
    }
}

const std::vector<Node>& Network::getNodes() const {
    return nodes;
}

int Network::getSize() const {
    return network_size;
}

void Network::setTimeStep(double dt) {
    time_step = dt;
}