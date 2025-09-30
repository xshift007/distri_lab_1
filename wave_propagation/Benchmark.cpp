#include "Benchmark.h"
#include <omp.h>
#include <fstream>
#include <filesystem>
#include <numeric>
#include <cmath>
#include <algorithm>

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

    struct Sample {
        int threads;
        double mean;
        double stdev;
    };

    auto measure = [&](int p){
        std::vector<double> times;
        times.reserve(reps);
        for (int r=0;r<reps;++r){
            reset_initial(net);
            omp_set_num_threads(p);
            RunParams bench_params;
            bench_params.steps = steps;
            bench_params.schedule = st;
            bench_params.chunk = chunk;
            bench_params.dt = 0.01;
            bench_params.S0 = 0.0;
            bench_params.omega = 0.0;
            bench_params.noise = NoiseMode::Off;
            bench_params.energyAccum = EnergyAccum::Reduction;
            bench_params.taskloop = false;
            bench_params.dump_frames = false;
            bench_params.energy_out.clear();
            bench_params.network = net.is2D() ? "2d" : "1d";
            bench_params.N = net.size();
            bench_params.Lx = net.Lx();
            bench_params.Ly = net.Ly();
            bench_params.threads = p;
            WavePropagator wp(net, bench_params);
            double t0 = omp_get_wtime();
            wp.run(bench_params.energy_out);
            double t1 = omp_get_wtime();
            times.push_back(t1-t0);
        }
        return Sample{p, mean(times), stdev(times)};
    };

    std::vector<Sample> results;
    results.reserve(threads_list.size()+1);
    bool has_base=false;
    Sample base{1,0.0,0.0};

    for (int p: threads_list){
        Sample s = measure(p);
        if (p==1){
            has_base = true;
            base = s;
        }
        results.push_back(s);
    }

    if (!has_base){
        base = measure(1);
        results.push_back(base);
    }

    std::sort(results.begin(), results.end(), [](const Sample& a, const Sample& b){
        return a.threads < b.threads;
    });

    const double base_mean = base.mean;
    const double base_stdev = base.stdev;

    out << "# threads mean_time std_time speedup speedup_err efficiency efficiency_err\n";

    for (const auto& s: results){
        double speedup = 0.0;
        double speedup_err = 0.0;
        if (s.mean > 0.0 && base_mean > 0.0){
            speedup = base_mean / s.mean;
            if (s.threads == 1){
                speedup = 1.0;
                speedup_err = 0.0;
            } else {
                double term1 = base_stdev / s.mean;
                double term2 = 0.0;
                double denom = s.mean * s.mean;
                if (denom > 0.0){
                    term2 = (base_mean * s.stdev) / denom;
                }
                double variance = term1*term1 + term2*term2;
                if (variance > 0.0){
                    speedup_err = std::sqrt(variance);
                }
            }
        }
        double efficiency = (s.threads > 0) ? speedup / s.threads : 0.0;
        double efficiency_err = (s.threads > 0) ? speedup_err / s.threads : 0.0;

        out << s.threads << " "
            << s.mean << " "
            << s.stdev << " "
            << speedup << " "
            << speedup_err << " "
            << efficiency << " "
            << efficiency_err << "\n";
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
            RunParams bench_params;
            bench_params.steps = steps;
            bench_params.schedule = ScheduleType::Dynamic;
            bench_params.chunk = c;
            bench_params.dt = 0.01;
            bench_params.S0 = 0.0;
            bench_params.omega = 0.0;
            bench_params.noise = NoiseMode::Off;
            bench_params.energyAccum = EnergyAccum::Reduction;
            bench_params.taskloop = false;
            bench_params.dump_frames = false;
            bench_params.energy_out.clear();
            bench_params.network = net.is2D() ? "2d" : "1d";
            bench_params.N = net.size();
            bench_params.Lx = net.Lx();
            bench_params.Ly = net.Ly();
            bench_params.threads = threads;
            WavePropagator wp(net, bench_params);
            double t0 = omp_get_wtime();
            wp.run(bench_params.energy_out);
            double t1 = omp_get_wtime();
            times.push_back(t1-t0);
        }
        out << c << " " << mean(times) << " " << stdev(times) << "\n";
    }
}
