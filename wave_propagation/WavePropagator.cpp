#include "WavePropagator.h"
#include <omp.h>
#include <fstream>

void WavePropagator::step_schedule(ScheduleType st, int chunk) {
    auto& nodes = net_->data();
    const int N = net_->size();
    const double D = net_->diffusion();
    const double g = net_->damping();
    double s_val = 0.0;
    #pragma omp parallel
    {
        #pragma omp single
        { s_val = (omega_!=0.0) ? (S0_ * std::sin(omega_*tcur_)) : S0_; }
        if (net_->is2D()) {
            const int Lx = net_->Lx();
            const int Ly = net_->Ly();
            if (st == ScheduleType::Static) {
                #pragma omp for collapse(2) schedule(static,chunk) nowait
                for (int y=0; y<Ly; ++y)
                    for (int x=0; x<Lx; ++x) {
                        int i = y*Lx + x;
                        double ai = nodes[i].getPrev();
                        double acc = 0.0;
                        const auto& nbrs = nodes[i].neighbors();
                        for (int j : nbrs) acc += nodes[j].getPrev() - ai;
                        double next = ai + dt_ * ( D*acc - g*ai + s_val );
                        nodes[i].setAmplitude(next);
                    }
            } else if (st == ScheduleType::Dynamic) {
                #pragma omp for collapse(2) schedule(dynamic,chunk) nowait
                for (int y=0; y<Ly; ++y)
                    for (int x=0; x<Lx; ++x) {
                        int i = y*Lx + x;
                        double ai = nodes[i].getPrev();
                        double acc = 0.0;
                        const auto& nbrs = nodes[i].neighbors();
                        for (int j : nbrs) acc += nodes[j].getPrev() - ai;
                        double next = ai + dt_ * ( D*acc - g*ai + s_val );
                        nodes[i].setAmplitude(next);
                    }
            } else {
                #pragma omp for collapse(2) schedule(guided,chunk) nowait
                for (int y=0; y<Ly; ++y)
                    for (int x=0; x<Lx; ++x) {
                        int i = y*Lx + x;
                        double ai = nodes[i].getPrev();
                        double acc = 0.0;
                        const auto& nbrs = nodes[i].neighbors();
                        for (int j : nbrs) acc += nodes[j].getPrev() - ai;
                        double next = ai + dt_ * ( D*acc - g*ai + s_val );
                        nodes[i].setAmplitude(next);
                    }
            }
        } else {
            if (st == ScheduleType::Static) {
                #pragma omp for schedule(static,chunk) nowait
                for (int i=0; i<N; ++i) {
                    double ai = nodes[i].getPrev();
                    double acc = 0.0;
                    const auto& nbrs = nodes[i].neighbors();
                    for (int j : nbrs) acc += nodes[j].getPrev() - ai;
                    double next = ai + dt_ * ( D*acc - g*ai + s_val );
                    nodes[i].setAmplitude(next);
                }
            } else if (st == ScheduleType::Dynamic) {
                #pragma omp for schedule(dynamic,chunk) nowait
                for (int i=0; i<N; ++i) {
                    double ai = nodes[i].getPrev();
                    double acc = 0.0;
                    const auto& nbrs = nodes[i].neighbors();
                    for (int j : nbrs) acc += nodes[j].getPrev() - ai;
                    double next = ai + dt_ * ( D*acc - g*ai + s_val );
                    nodes[i].setAmplitude(next);
                }
            } else {
                #pragma omp for schedule(guided,chunk) nowait
                for (int i=0; i<N; ++i) {
                    double ai = nodes[i].getPrev();
                    double acc = 0.0;
                    const auto& nbrs = nodes[i].neighbors();
                    for (int j : nbrs) acc += nodes[j].getPrev() - ai;
                    double next = ai + dt_ * ( D*acc - g*ai + s_val );
                    nodes[i].setAmplitude(next);
                }
            }
        }
        #pragma omp barrier
    }
    tcur_ += dt_;
}
void WavePropagator::energy_reduction(double& E) {
    E = 0.0;
    auto& nodes = net_->data();
    const int N = net_->size();
    #pragma omp parallel for reduction(+:E)
    for (int i=0;i<N;++i) {
        double a = nodes[i].get();
        E += a*a;
    }
}
void WavePropagator::energy_atomic(double& E) {
    E = 0.0;
    auto& nodes = net_->data();
    const int N = net_->size();
    #pragma omp parallel
    {
        double local = 0.0;
        #pragma omp for nowait
        for (int i=0;i<N;++i) {
            double a = nodes[i].get();
            local += a*a;
        }
        #pragma omp atomic
        E += local;
    }
}
void WavePropagator::energy_critical(double& E) {
    E = 0.0;
    auto& nodes = net_->data();
    const int N = net_->size();
    #pragma omp parallel
    {
        double local = 0.0;
        #pragma omp for nowait
        for (int i=0;i<N;++i) {
            double a = nodes[i].get();
            local += a*a;
        }
        #pragma omp critical
        { E += local; }
    }
}
void WavePropagator::commit_with_barrier() {
    auto& nodes = net_->data();
    const int N = net_->size();
    #pragma omp parallel
    {
        #pragma omp for
        for (int i=0;i<N;++i) {
            nodes[i].commit();
        }
        #pragma omp barrier
    }
}
void WavePropagator::commit_with_nowait_example() {
    auto& nodes = net_->data();
    const int N = net_->size();
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i=0;i<N;++i) {
            nodes[i].commit();
        }
        #pragma omp barrier
    }
}
void WavePropagator::demo_data_clauses(const char* out_path){
    auto& nodes = net_->data();
    const int N = net_->size();
    double s_const = (omega_!=0.0) ? (S0_ * std::sin(omega_*tcur_)) : S0_;
    double captured_last = 0.0;
    #pragma omp parallel for default(none) shared(nodes,N) firstprivate(s_const) lastprivate(captured_last)
    for (int i=0;i<N;++i){
        double a = nodes[i].getPrev() + s_const;
        captured_last = a;
    }
    lastprivate_val_ = captured_last;
    std::ofstream f(out_path);
    if (f) f << "# lastprivate_val\n" << lastprivate_val_ << "\n";
}
void WavePropagator::run(int steps, ScheduleType st, int chunk, SyncMethod sm, const char* energy_trace_path) {
    std::ofstream fout(energy_trace_path);
    if (fout) fout << "# t E(t)\n";
    for (int t=0; t<steps; ++t) {
        step_schedule(st, chunk);
        commit_with_barrier();
        double E = 0.0;
        if (sm == SyncMethod::Reduction) energy_reduction(E);
        else if (sm == SyncMethod::Atomic) energy_atomic(E);
        else energy_critical(E);
        if (fout) fout << (t+1) << " " << E << "\n";
    }
}
void WavePropagator::process_with_tasks(int steps, int grain, SyncMethod sm, const char* energy_trace_path){
    std::ofstream fout(energy_trace_path);
    if (fout) fout << "# t E(t)\n";
    auto& nodes = net_->data();
    const int N = net_->size();
    const double D = net_->diffusion();
    const double g = net_->damping();
    for (int it=0; it<steps; ++it){
        double s_val = (omega_!=0.0) ? (S0_ * std::sin(omega_*tcur_)) : S0_;
        #pragma omp parallel
        #pragma omp single
        {
            for (int b=0; b<N; b+=grain){
                int start=b, end = (b+grain < N) ? (b+grain) : N;
                #pragma omp task firstprivate(start,end,s_val)
                {
                    for (int i=start;i<end;++i){
                        double ai = nodes[i].getPrev();
                        double acc = 0.0;
                        const auto& nbrs = nodes[i].neighbors();
                        for (int j : nbrs) acc += nodes[j].getPrev() - ai;
                        double next = ai + dt_ * ( D*acc - g*ai + s_val );
                        nodes[i].setAmplitude(next);
                    }
                }
            }
        }
        commit_with_barrier();
        double E = 0.0;
        if (sm == SyncMethod::Reduction) energy_reduction(E);
        else if (sm == SyncMethod::Atomic) energy_atomic(E);
        else energy_critical(E);
        if (fout) fout << (it+1) << " " << E << "\n";
        tcur_ += dt_;
    }
}
