#include "WavePropagator.h"
#include <omp.h>
#include <fstream>
#include <filesystem>
#include <vector>
#include <cmath>

void WavePropagator::run_fused(int steps, ScheduleType st, int chunk, const std::string& energy_out){
    auto& nodes = net_->data();
    const int N = net_->size();
    const double D = net_->diffusion();
    const double g = net_->damping();

    std::vector<std::pair<int,double>> trace;
    trace.reserve(steps);

    const double dt = dt_;
    double local_t = tcur_;
    double E_global = 0.0;

    #pragma omp parallel default(none) \
        shared(nodes, N, D, g, st, chunk, steps, dt, local_t, S0_, omega_, E_global, trace)
    {
        for (int it=0; it<steps; ++it){
            double s_val = 0.0;
            #pragma omp single
            { s_val = (omega_!=0.0) ? (S0_*std::sin(omega_*local_t)) : S0_; E_global = 0.0; }

            // Update (leer prev, escribir actual)
            if (st == ScheduleType::Static){
                #pragma omp for schedule(static,chunk)
                for (int i=0;i<N;++i){
                    double ai = nodes[i].getPrev();
                    const auto& nb = nodes[i].neighbors();
                    double acc=0.0; for (int j: nb) acc += (nodes[j].getPrev() - ai);
                    nodes[i].set( ai + dt*(D*acc - g*ai + s_val) );
                }
            } else if (st == ScheduleType::Dynamic){
                #pragma omp for schedule(dynamic,chunk)
                for (int i=0;i<N;++i){
                    double ai = nodes[i].getPrev();
                    const auto& nb = nodes[i].neighbors();
                    double acc=0.0; for (int j: nb) acc += (nodes[j].getPrev() - ai);
                    nodes[i].set( ai + dt*(D*acc - g*ai + s_val) );
                }
            } else { // Guided
                #pragma omp for schedule(guided,chunk)
                for (int i=0;i<N;++i){
                    double ai = nodes[i].getPrev();
                    const auto& nb = nodes[i].neighbors();
                    double acc=0.0; for (int j: nb) acc += (nodes[j].getPrev() - ai);
                    nodes[i].set( ai + dt*(D*acc - g*ai + s_val) );
                }
            }

            // Energía y commit en paralelo
            #pragma omp for reduction(+:E_global) nowait
            for (int i=0;i<N;++i){
                double a = nodes[i].get();
                E_global += a*a;
            }
            #pragma omp for nowait
            for (int i=0;i<N;++i) nodes[i].commit();

            #pragma omp single
            { trace.emplace_back(it+1, E_global); local_t += dt; }

            #pragma omp barrier
        }
    }

    // Dump energía (fuera del paralelo)
    if (!energy_out.empty()){
        std::filesystem::create_directories("results");
        std::ofstream f(energy_out);
        if (f){
            f << "# step E\n";
            for (auto &kv : trace) f << kv.first << " " << kv.second << "\n";
        }
    }
    tcur_ = local_t;
}
