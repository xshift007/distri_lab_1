#include "Benchmark.h"
#include <omp.h>
#include <cmath>
#include <fstream>
#include <numeric>
#include <algorithm>
#include <string>
#include <filesystem>

static inline double mean(const std::vector<double>& v){
    if (v.empty()) return 0.0;
    double s = std::accumulate(v.begin(), v.end(), 0.0);
    return s / v.size();
}
static inline double stdev(const std::vector<double>& v, double m){
    if (v.size()<2) return 0.0;
    double acc=0.0; for (double x: v){ double d=x-m; acc+=d*d; }
    return std::sqrt(acc/(v.size()-1));
}
void Benchmark::run_scaling(Network& net, int steps, ScheduleType st, int chunk,
                            SyncMethod sm, const std::vector<int>& threads_list,
                            int reps, const std::string& out_scaling_path) {
    std::filesystem::create_directories("results");
    std::ofstream fout(out_scaling_path);
    if (fout) fout << "# p  mean_T  std_T  S  std_S  E  std_E\n";
    double T1=0.0, T1s=0.0;
    {   omp_set_num_threads(1);
        std::vector<double> times;
        for (int r=0;r<reps;++r) {
            WavePropagator wp(&net, /*dt*/0.01, /*S0*/0.0, /*omega*/0.0);
            net.setAll(0.0); net.setInitialImpulseCenter(1.0);
            double t0 = omp_get_wtime(); wp.run(steps, st, chunk, sm, ""); double t1 = omp_get_wtime();
            times.push_back(t1 - t0);
        }
        T1 = mean(times); T1s = stdev(times, T1);
    }
    for (int p : threads_list) {
        omp_set_num_threads(p);
        std::vector<double> tp;
        for (int r=0;r<reps;++r) {
            WavePropagator wpp(&net, 0.01, 0.0, 0.0);
            net.setAll(0.0); net.setInitialImpulseCenter(1.0);
            double t0 = omp_get_wtime(); wpp.run(steps, st, chunk, sm, ""); double t1 = omp_get_wtime();
            tp.push_back(t1 - t0);
        }
        double mp = mean(tp), sp = stdev(tp, mp);
        double Sp = (mp>0.0) ? (T1/mp) : 0.0;
        double sSp = (Sp>0.0) ? (Sp * std::sqrt((T1s/T1)*(T1s/T1) + (sp/mp)*(sp/mp))) : 0.0;
        double Ep = (p>0) ? (Sp/p) : 0.0;
        double sEp = (p>0) ? (sSp/p) : 0.0;
        if (fout) fout << p << " " << mp << " " << sp << " "
                       << Sp << " " << sSp << " "
                       << Ep << " " << sEp << "\n";
    }
}
static inline std::string schedName(ScheduleType st){
    switch(st){ case ScheduleType::Static: return "static"; case ScheduleType::Dynamic: return "dynamic"; default: return "guided"; }
}
void Benchmark::run_schedule_chunk(Network& net, int steps, SyncMethod sm,
                                   const std::vector<ScheduleType>& sts,
                                   const std::vector<int>& chunks,
                                   int threads, int reps, const std::string& out_path){
    std::filesystem::create_directories("results");
    std::ofstream f(out_path);
    if (f) f << "# schedule chunk mean_T std_T\n";
    omp_set_num_threads(threads);
    for (auto st : sts){
        for (int c : chunks){
            std::vector<double> tt;
            for (int r=0;r<reps;++r){
                WavePropagator wp(&net, 0.01, 0.0, 0.0);
                net.setAll(0.0); net.setInitialImpulseCenter(1.0);
                double t0 = omp_get_wtime(); wp.run(steps, st, c, sm, "results/energy_trace.dat"); double t1 = omp_get_wtime();
                tt.push_back(t1 - t0);
            }
            double m = mean(tt), s = stdev(tt, m);
            if (f) f << schedName(st) << " " << c << " " << m << " " << s << "\n";
        }
    }
}
void Benchmark::run_sync_methods(Network& net, int steps, ScheduleType st, int chunk,
                                 int threads, int reps, const std::string& out_path){
    std::filesystem::create_directories("results");
    std::ofstream f(out_path);
    if (f) f << "# method mean_T std_T\n";
    omp_set_num_threads(threads);
    for (int m=0; m<3; ++m){
        SyncMethod sm = (m==0 ? SyncMethod::Reduction : (m==1 ? SyncMethod::Atomic : SyncMethod::Critical));
        std::vector<double> tt;
        for (int r=0;r<reps;++r){
            WavePropagator wp(&net, 0.01, 0.0, 0.0);
            net.setAll(0.0); net.setInitialImpulseCenter(1.0);
            double t0 = omp_get_wtime(); wp.run(steps, st, chunk, sm, ""); double t1 = omp_get_wtime();
            tt.push_back(t1 - t0);
        }
        double mT = mean(tt), sT = stdev(tt, mT);
        std::string name = (m==0 ? "reduction" : (m==1 ? "atomic" : "critical"));
        if (f) f << name << " " << mT << " " << sT << "\n";
    }
}
void Benchmark::run_tasks_vs_for(Network& net, int steps, ScheduleType st, int chunk,
                                 SyncMethod sm, int threads, int reps, int grain,
                                 const std::string& out_path){
    std::filesystem::create_directories("results");
    std::ofstream f(out_path);
    if (f) f << "# mode mean_T std_T\n";
    omp_set_num_threads(threads);
    {   std::vector<double> tt;
        for (int r=0;r<reps;++r){
            WavePropagator wp(&net, 0.01, 0.0, 0.0);
            net.setAll(0.0); net.setInitialImpulseCenter(1.0);
            double t0 = omp_get_wtime(); wp.run(steps, st, chunk, sm, ""); double t1 = omp_get_wtime();
            tt.push_back(t1 - t0);
        }
        double mT = mean(tt), sT = stdev(tt, mT);
        if (f) f << "parallel_for " << mT << " " << sT << "\n";
    }
    {   std::vector<double> tt;
        for (int r=0;r<reps;++r){
            WavePropagator wp(&net, 0.01, 0.0, 0.0);
            net.setAll(0.0); net.setInitialImpulseCenter(1.0);
            double t0 = omp_get_wtime(); wp.process_with_tasks(steps, grain, sm, ""); double t1 = omp_get_wtime();
            tt.push_back(t1 - t0);
        }
        double mT = mean(tt), sT = stdev(tt, mT);
        if (f) f << "tasks " << mT << " " << sT << "\n";
    }
}

void run_single_scaling_io(Network& net, int steps, ScheduleType st, int chunk,
                           SyncMethod sm, int reps, int p, std::ofstream& fout){
    omp_set_num_threads(p);
    std::vector<double> tp;
    for (int r=0;r<reps;++r) {
        WavePropagator wpp(&net, 0.01, 0.0, 0.0);
        net.setAll(0.0); net.setInitialImpulseCenter(1.0);
        double t0 = omp_get_wtime();
        // I/O habilitado: escribimos para medir overhead de disco
        wpp.run(steps, st, chunk, sm, "results/io_energy_trace.dat");
        double t1 = omp_get_wtime();
        tp.push_back(t1 - t0);
    }
    double mp = std::accumulate(tp.begin(), tp.end(), 0.0)/tp.size();
    double acc=0.0; for(double x: tp){ double d=x-mp; acc+=d*d; }
    double sp = (tp.size()>1)? std::sqrt(acc/(tp.size()-1)):0.0;
    // Para comparar, necesitamos T1_io
    static double T1_io=-1.0, T1s_io=0.0;
    if (p==1){
        T1_io = mp; T1s_io = sp;
    }
    double Sp = (mp>0.0 && T1_io>0.0) ? (T1_io/mp) : 0.0;
    double sSp = (Sp>0.0) ? (Sp * std::sqrt((T1s_io/T1_io)*(T1s_io/T1_io) + (sp/mp)*(sp/mp))) : 0.0;
    double Ep = (p>0) ? (Sp/p) : 0.0;
    double sEp = (p>0) ? (sSp/p) : 0.0;
    if (fout) fout << p << " " << mp << " " << sp << " " << Sp << " " << sSp << " " << Ep << " " << sEp << "\n";
}

void Benchmark::run_scaling_io(Network& net, int steps, ScheduleType st, int chunk,
                               SyncMethod sm, const std::vector<int>& threads_list,
                               int reps, const std::string& out_scaling_path){
    std::filesystem::create_directories("results");
    std::ofstream fout(out_scaling_path);
    if (fout) fout << "# p  mean_T  std_T  S  std_S  E  std_E\n";
    for (int p : threads_list){
        run_single_scaling_io(net, steps, st, chunk, sm, reps, p, fout);
    }
}

void Benchmark::run_tasks_grain_sweep(Network& net, int steps, SyncMethod sm,
                                      int threads, int reps,
                                      const std::vector<int>& grains,
                                      const std::string& out_path){
    std::filesystem::create_directories("results");
    std::ofstream f(out_path);
    if (f) f << "# grain mean_T std_T\n";
    omp_set_num_threads(threads);
    for (int g : grains){
        std::vector<double> tt;
        for (int r=0;r<reps;++r){
            WavePropagator wp(&net, 0.01, 0.0, 0.0);
            net.setAll(0.0); net.setInitialImpulseCenter(1.0);
            double t0 = omp_get_wtime();
            wp.process_with_tasks(steps, g, sm, "");
            double t1 = omp_get_wtime();
            tt.push_back(t1 - t0);
        }
        double m = std::accumulate(tt.begin(), tt.end(), 0.0)/tt.size();
        double acc=0.0; for(double x: tt){ double d=x-m; acc+=d*d; }
        double s = (tt.size()>1)? std::sqrt(acc/(tt.size()-1)) : 0.0;
        if (f) f << g << " " << m << " " << s << "\n";
    }
}
