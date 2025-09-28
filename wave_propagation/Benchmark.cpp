#include "Benchmark.h"
#include <omp.h>
#include <fstream>
#include <filesystem>
#include <numeric>
#include <cmath>

static double mean(const std::vector<double>& v){
    if (v.empty()) return 0.0;
    double s=0; for (double x: v) s+=x; return s/v.size();
}
static double stdev(const std::vector<double>& v){
    if (v.size()<2) return 0.0;
    double m = mean(v), acc=0; for (double x: v){ double d=x-m; acc+=d*d; }
    return std::sqrt(acc/(v.size()-1));
}
static void reset_initial(Network& net){
    net.setAll(0.0);
    net.setInitialImpulseCenter(1.0);
}

void Benchmark::run_scaling(Network& net, int steps, ScheduleType st, int chunk,
                            const std::vector<int>& threads_list, int reps,
                            const std::string& out_path)
{
    std::filesystem::create_directories("results");
    std::ofstream out(out_path);
    if (!out) return;
    out << "# threads mean_time std_time\n";

    for (int p: threads_list){
        std::vector<double> times;
        for (int r=0;r<reps;++r){
            reset_initial(net);
            omp_set_num_threads(p);
            WavePropagator wp(&net, /*dt*/0.01, /*S0*/0.0, /*omega*/0.0);
            double t0 = omp_get_wtime();
            wp.run_fused(steps, st, chunk, /*energy_out*/"");
            double t1 = omp_get_wtime();
            times.push_back(t1-t0);
        }
        out << p << " " << mean(times) << " " << stdev(times) << "\n";
    }
}

void Benchmark::run_time_vs_chunk_dynamic(Network& net, int steps, int threads, int reps,
                                          const std::vector<int>& chunks,
                                          const std::string& out_path)
{
    std::filesystem::create_directories("results");
    std::ofstream out(out_path);
    if (!out) return;
    out << "# chunk mean_time std_time\n";
    omp_set_num_threads(threads);

    for (int c: chunks){
        std::vector<double> times;
        for (int r=0;r<reps;++r){
            reset_initial(net);
            WavePropagator wp(&net, 0.01, 0.0, 0.0);
            double t0 = omp_get_wtime();
            wp.run_fused(steps, ScheduleType::Dynamic, c, "");
            double t1 = omp_get_wtime();
            times.push_back(t1-t0);
        }
        out << c << " " << mean(times) << " " << stdev(times) << "\n";
    }
}
