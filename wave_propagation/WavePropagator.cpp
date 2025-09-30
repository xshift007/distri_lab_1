#include "WavePropagator.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <omp.h>

WavePropagator::WavePropagator(Network& net, const RunParams& params)
    : net_(net), params_(params), rng_(std::random_device{}()), norm_(params_.omega_mu, params_.omega_sigma)
{
    if (params_.noise == NoiseMode::PerNode){
        omega_i_.resize(net_.size());
        for (double& w : omega_i_) w = norm_(rng_);
    } else if (params_.noise == NoiseMode::Single){
        single_idx_ = params_.noise_node;
        if (single_idx_ < 0 || single_idx_ >= net_.size()){
            single_idx_ = net_.is2D()
                ? ((net_.Ly()/2) * net_.Lx() + (net_.Lx()/2))
                : (net_.Lx()/2);
        }
        omega_i_.assign(net_.size(), 0.0);
        if (single_idx_ >= 0 && single_idx_ < net_.size()){
            omega_i_[single_idx_] = norm_(rng_);
        }
    }
}

double WavePropagator::source_val(int idx, double time) const{
    switch (params_.noise){
        case NoiseMode::Off:
            return (params_.omega != 0.0) ? params_.S0 * std::sin(params_.omega * time) : params_.S0;
        case NoiseMode::Global:
            return params_.S0 * std::sin(params_.omega * time);
        case NoiseMode::PerNode:
            if (idx >= 0 && idx < (int)omega_i_.size())
                return params_.S0 * std::sin(omega_i_[idx] * time);
            return 0.0;
        case NoiseMode::Single:
            if (idx == single_idx_ && idx >= 0 && idx < (int)omega_i_.size())
                return params_.S0 * std::sin(omega_i_[idx] * time);
            return 0.0;
    }
    return 0.0;
}

void WavePropagator::dump_energy(std::ofstream& fe, int step, double E){
    if (step == 1) fe << "# step\tE\n";
    fe << step << "\t" << std::setprecision(12) << E << "\n";
}

void WavePropagator::dump_frame_1d(int step){
    char name[256];
    std::snprintf(name, sizeof(name), "results/frames/amp_t%06d.dat", step);
    std::ofstream f(name);
    if (!f) return;
    auto& nodes = net_.data();
    for (int x=0; x<net_.Lx(); ++x){
        int idx = x;
        if (idx >= 0 && idx < (int)nodes.size())
            f << nodes[idx].get() << "\n";
    }
}

void WavePropagator::dump_frame_2d(int step){
    char name[256];
    std::snprintf(name, sizeof(name), "results/frames/amp_t%06d.csv", step);
    std::ofstream f(name);
    if (!f) return;
    auto& nodes = net_.data();
    const int Lx = net_.Lx();
    const int Ly = net_.Ly();
    for (int y=0; y<Ly; ++y){
        for (int x=0; x<Lx; ++x){
            int idx = y*Lx + x;
            double val = (idx >= 0 && idx < (int)nodes.size()) ? nodes[idx].get() : 0.0;
            f << val;
            if (x+1<Lx) f << ',';
        }
        f << '\n';
    }
}

void WavePropagator::run(const std::string& energy_out){
    auto& nodes = net_.data();
    const int N = net_.size();
    const double D = net_.diffusion();
    const double g = net_.damping();

    std::ofstream energy_file;
    if (!energy_out.empty()){
        std::filesystem::path p(energy_out);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
        energy_file.open(energy_out);
    }

    if (params_.dump_frames){
        std::filesystem::create_directories("results/frames");
    }

    const int chunk = params_.chunk > 0 ? params_.chunk : 1;
    const int grain = params_.grain > 0 ? params_.grain : 1;

    double dt = params_.dt;
    double local_t = tcur_;
    double time_for_step = 0.0;
    double E_global = 0.0;

    const int Lx = net_.Lx();
    const int Ly = net_.Ly();

    #pragma omp parallel shared(time_for_step, E_global)
    {
        for (int it=0; it<params_.steps; ++it){
            #pragma omp single
            {
                time_for_step = local_t;
                E_global = 0.0;
            }

            auto update_index = [&](int idx){
                double ai = nodes[idx].getPrev();
                const auto& nb = nodes[idx].neighbors();
                double acc = 0.0;
                for (int j : nb){
                    acc += (nodes[j].getPrev() - ai);
                }
                double s = source_val(idx, time_for_step);
                nodes[idx].set(ai + dt*(D*acc - g*ai + s));
            };

            if (params_.taskloop){
                if (net_.is2D()){
                    #pragma omp single
                    {
                        #pragma omp taskloop grainsize(grain)
                        for (int y=0; y<Ly; ++y){
                            for (int x=0; x<Lx; ++x){
                                int idx = y*Lx + x;
                                update_index(idx);
                            }
                        }
                        #pragma omp taskwait
                    }
                } else {
                    #pragma omp single
                    {
                        #pragma omp taskloop grainsize(grain)
                        for (int i=0; i<N; ++i){
                            update_index(i);
                        }
                        #pragma omp taskwait
                    }
                }
            } else if (net_.is2D()){
                if (params_.collapse2){
                    if (params_.schedule == ScheduleType::Static){
                        #pragma omp for schedule(static, chunk) collapse(2)
                        for (int y=0; y<Ly; ++y){
                            for (int x=0; x<Lx; ++x){
                                int idx = y*Lx + x;
                                update_index(idx);
                            }
                        }
                    } else if (params_.schedule == ScheduleType::Dynamic){
                        #pragma omp for schedule(dynamic, chunk) collapse(2)
                        for (int y=0; y<Ly; ++y){
                            for (int x=0; x<Lx; ++x){
                                int idx = y*Lx + x;
                                update_index(idx);
                            }
                        }
                    } else {
                        #pragma omp for schedule(guided, chunk) collapse(2)
                        for (int y=0; y<Ly; ++y){
                            for (int x=0; x<Lx; ++x){
                                int idx = y*Lx + x;
                                update_index(idx);
                            }
                        }
                    }
                } else {
                    if (params_.schedule == ScheduleType::Static){
                        #pragma omp for schedule(static, chunk)
                        for (int y=0; y<Ly; ++y){
                            for (int x=0; x<Lx; ++x){
                                int idx = y*Lx + x;
                                update_index(idx);
                            }
                        }
                    } else if (params_.schedule == ScheduleType::Dynamic){
                        #pragma omp for schedule(dynamic, chunk)
                        for (int y=0; y<Ly; ++y){
                            for (int x=0; x<Lx; ++x){
                                int idx = y*Lx + x;
                                update_index(idx);
                            }
                        }
                    } else {
                        #pragma omp for schedule(guided, chunk)
                        for (int y=0; y<Ly; ++y){
                            for (int x=0; x<Lx; ++x){
                                int idx = y*Lx + x;
                                update_index(idx);
                            }
                        }
                    }
                }
            } else {
                if (params_.schedule == ScheduleType::Static){
                    #pragma omp for schedule(static, chunk)
                    for (int i=0; i<N; ++i){
                        update_index(i);
                    }
                } else if (params_.schedule == ScheduleType::Dynamic){
                    #pragma omp for schedule(dynamic, chunk)
                    for (int i=0; i<N; ++i){
                        update_index(i);
                    }
                } else {
                    #pragma omp for schedule(guided, chunk)
                    for (int i=0; i<N; ++i){
                        update_index(i);
                    }
                }
            }

            if (params_.energyAccum == EnergyAccum::Reduction){
                #pragma omp for reduction(+:E_global)
                for (int i=0; i<N; ++i){
                    double a = nodes[i].get();
                    E_global += a*a;
                }
            } else if (params_.energyAccum == EnergyAccum::Atomic){
                #pragma omp for
                for (int i=0; i<N; ++i){
                    double e = nodes[i].get();
                    e *= e;
                    #pragma omp atomic
                    E_global += e;
                }
            } else {
                double local_sum = 0.0;
                #pragma omp for
                for (int i=0; i<N; ++i){
                    double a = nodes[i].get();
                    local_sum += a*a;
                }
                #pragma omp critical
                {
                    E_global += local_sum;
                }
            }

            #pragma omp for
            for (int i=0; i<N; ++i){
                nodes[i].commit();
            }

            #pragma omp single
            {
                if (energy_file){
                    dump_energy(energy_file, it+1, E_global);
                }
                if (params_.dump_frames && params_.frame_every>0 && (it % params_.frame_every == 0)){
                    if (net_.is2D()) dump_frame_2d(it);
                    else dump_frame_1d(it);
                }
                local_t += dt;
            }
        }
    }

    tcur_ = local_t;
    if (energy_file){
        energy_file.flush();
    }
}
