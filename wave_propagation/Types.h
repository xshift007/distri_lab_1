#pragma once  // Esto es para que se compile una sola vez y no ocurran errores

#include <string>

enum class ScheduleType {      //Enumeracion con Scope para evitar colisiones de nombre
    Static,         // opciones del planificador
    Dynamic,
    Guided
};

enum class NoiseMode { Off = 0, Global, PerNode, Single };
enum class EnergyAccum { Reduction = 0, Atomic, Critical };

struct RunParams {
    // parámetros de topología / simulación
    std::string network = "2d"; // {1d,2d}
    int N = 10000;              // tamaño 1D
    int Lx = 100, Ly = 100;     // tamaño 2D
    double D = 0.1;             // difusión
    double gamma = 0.01;        // amortiguamiento
    double dt = 0.01;           // paso de tiempo
    int steps = 200;            // pasos a simular

    // fuente base
    double S0 = 0.0;
    double omega = 0.0;

    // ruido
    NoiseMode noise = NoiseMode::Off;
    double omega_mu = 10.0;
    double omega_sigma = 1.0;
    int noise_node = -1;

    // openmp / scheduling
    ScheduleType schedule = ScheduleType::Dynamic;
    int chunk = 32;
    bool chunk_auto = false;
    int threads = 0; // 0 => usar configuracion por defecto de OMP
    bool fused = true;
    bool taskloop = false;
    int grain = 4096;

    // acumulación de energía
    EnergyAccum energyAccum = EnergyAccum::Reduction;

    // opciones extra
    bool collapse2 = false;
    bool dump_frames = false;
    int frame_every = 10;
    bool do_bench = false;
    std::string energy_out = "results/energy_trace.dat";
};
