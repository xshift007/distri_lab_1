#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <filesystem>
#include <omp.h>

#include "Types.h"
#include "Network.h"
#include "WavePropagator.h"
#include "Benchmark.h"

struct Args {
    std::string network = "2d"; // {1d,2d}
    int N = 10000;              // 1d
    int Lx = 100, Ly = 100;     // 2d
    double D = 0.1, gamma = 0.01, dt = 0.01;
    int steps = 200;
    double S0 = 0.0, omega = 0.0;

    std::string schedule = "dynamic"; // {static,dynamic,guided}
    int chunk = 32;
    bool chunk_auto = false;

    std::string threads_csv = "";   // para ejecución simple
    bool do_bench = false;

    // Mínimo: solo fused (menos barreras)
    bool fused = true;
};

static void usage(){
    std::cout << "Uso: ./wave_propagation [opciones]\n"
              << "  --network {1d,2d}\n"
              << "  --N 10000 | --Lx 100 --Ly 100\n"
              << "  --D 0.1 --gamma 0.01 --dt 0.01 --steps 200\n"
              << "  --S0 0.1 --omega 0.5\n"
              << "  --schedule {static,dynamic,guided}\n"
              << "  --chunk <n|auto>\n"
              << "  --threads <n>\n"
              << "  --benchmark\n";
}

static ScheduleType parse_schedule(const std::string& s){
    if (s=="static") return ScheduleType::Static;
    if (s=="dynamic") return ScheduleType::Dynamic;
    return ScheduleType::Guided;
}

static int parse_int(const char* s){ return std::atoi(s); }
static double parse_double(const char* s){ return std::atof(s); }

static Args parse_args(int argc, char** argv){
    Args a;
    for (int i=1;i<argc;++i){
        std::string k = argv[i];
        auto need = [&](int idx){ if (idx+1>=argc) { usage(); std::exit(1);} };
        if (k=="--network"){ need(i); a.network=argv[++i]; }
        else if (k=="--N"){ need(i); a.N=parse_int(argv[++i]); }
        else if (k=="--Lx"){ need(i); a.Lx=parse_int(argv[++i]); }
        else if (k=="--Ly"){ need(i); a.Ly=parse_int(argv[++i]); }
        else if (k=="--D"){ need(i); a.D=parse_double(argv[++i]); }
        else if (k=="--gamma"){ need(i); a.gamma=parse_double(argv[++i]); }
        else if (k=="--dt"){ need(i); a.dt=parse_double(argv[++i]); }
        else if (k=="--steps"){ need(i); a.steps=parse_int(argv[++i]); }
        else if (k=="--S0"){ need(i); a.S0=parse_double(argv[++i]); }
        else if (k=="--omega"){ need(i); a.omega=parse_double(argv[++i]); }
        else if (k=="--schedule"){ need(i); a.schedule=argv[++i]; }
        else if (k=="--chunk"){
            need(i);
            std::string v = argv[++i];
            if (v=="auto") a.chunk_auto = true;
            else a.chunk = std::atoi(v.c_str());
        }
        else if (k=="--threads"){ need(i); a.threads_csv=argv[++i]; }
        else if (k=="--benchmark"){ a.do_bench = true; }
        else if (k=="--help" || k=="-h"){ usage(); std::exit(0); }
        else { std::cerr << "Opcion desconocida: " << k << "\n"; usage(); std::exit(1); }
    }
    return a;
}

static int compute_auto_chunk(int N, ScheduleType st, int p){
    if (N<=0) return 64;
    if (st==ScheduleType::Dynamic) return 256;
    if (st==ScheduleType::Guided)  return 64;
    int c = std::max(64, N/std::max(1,p*8));
    c = (c/8)*8;
    return std::min(std::max(c,64),8192);
}


int main(int argc, char** argv){
    std::filesystem::create_directories("results");
    Args args = parse_args(argc, argv);

    ScheduleType st = parse_schedule(args.schedule);

    // Construcción de red (solo 1D/2D)
    Network net = (args.network=="1d")
        ? Network(args.N, args.D, args.gamma)
        : Network(args.Lx, args.Ly, args.D, args.gamma);
    if (args.network=="1d") net.makeRegular1D(false);
    else                    net.makeRegular2D(false);

    // Estado inicial
    net.setAll(0.0);
    net.setInitialImpulseCenter(1.0);

    if (!args.threads_csv.empty()) {
        int t = std::atoi(args.threads_csv.c_str());
        if (t>0) omp_set_num_threads(t);
    }

    // Heurística de chunk auto
    if (args.chunk_auto) {
        int p = omp_get_max_threads();
        args.chunk = compute_auto_chunk(net.size(), st, p);
        std::cout << "[auto-chunk] " << args.chunk << "\n";
    }

    if (args.do_bench){
        // Scaling y tiempo vs chunk (dynamic)
        std::vector<int> plist = {1,2,4,8};
        Benchmark::run_scaling(net, args.steps, st, args.chunk, plist, /*reps*/10, "results/scaling.dat");
        Benchmark::run_time_vs_chunk_dynamic(net, args.steps, /*threads*/8, /*reps*/10,
                                             std::vector<int>{64,128,256,512}, "results/time_vs_chunk_dynamic.dat");
        std::cout << "Benchmarks listos. Revisa carpeta results/\n";
        return 0;
    }

    // Ejecucion normal
    WavePropagator wp(&net, args.dt, args.S0, args.omega);
    wp.run_fused(args.steps, st, args.chunk, "results/energy_trace.dat");
    std::cout << "OK. Resultados en results/\n";
    return 0;
}
