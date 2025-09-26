#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <filesystem>

#include "Network.h"
#include "WavePropagator.h"
#include "Benchmark.h"

struct Args {
    std::string network = "2d";
    int N = 10000;
    int Lx = 100, Ly = 100;
    double D = 0.1, gamma = 0.01, dt = 0.01;
    int steps = 200;
    double S0 = 0.0, omega = 0.0;
    std::string schedule = "dynamic";
    int chunk = 32;
    std::string sync = "reduction";
    std::string threads_csv = "";
    bool do_bench = false;
    bool do_analysis = false;
    bool demo_data = true;
};

static void usage(){
    std::cout << "Uso: ./wave_propagation [opciones]\n"
              << "  --network {1d,2d,random,smallworld}\n"
              << "  --N 10000 | --Lx 100 --Ly 100\n"
              << "  --D 0.1 --gamma 0.01 --dt 0.01 --steps 200\n"
              << "  --S0 0.1 --omega 0.5\n"
              << "  --schedule {static,dynamic,guided} --chunk 32\n"
              << "  --sync {reduction,atomic,critical}\n"
              << "  --threads 1,2,4,8,16\n"
              << "  --benchmark | --analysis\n";
}

static bool parse_int(const char* s, int& out){ char* e=nullptr; long v=strtol(s,&e,10); if(!e||*e) return false; out=(int)v; return true; }
static bool parse_double(const char* s, double& out){ char* e=nullptr; double v=strtod(s,&e); if(!e) return false; out=v; return true; }

static Args parse_args(int argc, char** argv){
    Args a;
    for (int i=1;i<argc;++i){
        std::string k = argv[i];
        auto need = [&](int i){ if (i+1>=argc) { usage(); std::exit(1);} };
        if (k=="--network"){ need(i); a.network=argv[++i]; }
        else if (k=="--N"){ need(i); parse_int(argv[++i], a.N); }
        else if (k=="--Lx"){ need(i); parse_int(argv[++i], a.Lx); }
        else if (k=="--Ly"){ need(i); parse_int(argv[++i], a.Ly); }
        else if (k=="--D"){ need(i); parse_double(argv[++i], a.D); }
        else if (k=="--gamma"){ need(i); parse_double(argv[++i], a.gamma); }
        else if (k=="--dt"){ need(i); parse_double(argv[++i], a.dt); }
        else if (k=="--steps"){ need(i); parse_int(argv[++i], a.steps); }
        else if (k=="--S0"){ need(i); parse_double(argv[++i], a.S0); }
        else if (k=="--omega"){ need(i); parse_double(argv[++i], a.omega); }
        else if (k=="--schedule"){ need(i); a.schedule=argv[++i]; }
        else if (k=="--chunk"){ need(i); parse_int(argv[++i], a.chunk); }
        else if (k=="--sync"){ need(i); a.sync=argv[++i]; }
        else if (k=="--threads"){ need(i); a.threads_csv=argv[++i]; }
        else if (k=="--benchmark"){ a.do_bench = true; }
        else if (k=="--analysis"){ a.do_analysis = true; }
        else { std::cerr << "Opción desconocida: " << k << "\n"; usage(); std::exit(1); }
    }
    return a;
}

static ScheduleType parse_schedule(const std::string& s){
    if (s=="static") return ScheduleType::Static;
    if (s=="dynamic") return ScheduleType::Dynamic;
    return ScheduleType::Guided;
}
static SyncMethod parse_sync(const std::string& s){
    if (s=="reduction") return SyncMethod::Reduction;
    if (s=="atomic")    return SyncMethod::Atomic;
    return SyncMethod::Critical;
}
static std::vector<int> parse_threads_csv(const std::string& s){
    std::vector<int> out;
    if (s.empty()) return out;
    size_t start=0;
    while (start < s.size()){
        size_t comma = s.find(',', start);
        std::string tok = s.substr(start, (comma==std::string::npos)? std::string::npos : comma-start);
        if (!tok.empty()) out.push_back(std::atoi(tok.c_str()));
        if (comma==std::string::npos) break;
        start = comma+1;
    }
    return out;
}

int main(int argc, char** argv){
    std::filesystem::create_directories("results");
    Args args = parse_args(argc, argv);

    Network net(/*N*/args.N, /*D*/args.D, /*gamma*/args.gamma);
    if (args.network == "1d")       net.makeRegular1D(false);
    else if (args.network == "2d")  net.makeRegular2D(args.Lx, args.Ly, false);
    else if (args.network == "random"){ net.makeRegular1D(false); net.makeRandom(10.0); }
    else if (args.network == "smallworld"){ net.makeSmallWorld(2, 0.05); }
    else { std::cerr << "Tipo de red no reconocido, usando 2d por defecto.\n"; net.makeRegular2D(args.Lx, args.Ly, false); }
    net.setAll(0.0); net.setInitialImpulseCenter(1.0);

    ScheduleType st = parse_schedule(args.schedule);
    SyncMethod sm    = parse_sync(args.sync);

    if (args.do_bench) {
        std::vector<int> plist = parse_threads_csv(args.threads_csv);
        if (plist.empty()) plist = {1,2,4,8};
        Benchmark::run_scaling(net, args.steps, st, args.chunk, sm, plist, /*reps*/5, "results/scaling.dat");
        Benchmark::run_schedule_chunk(net, args.steps, sm,
            {ScheduleType::Static, ScheduleType::Dynamic, ScheduleType::Guided},
            {1,8,32,64,256}, /*threads*/8, /*reps*/5, "results/schedule_vs_chunk.dat");
        Benchmark::run_sync_methods(net, args.steps, st, args.chunk, /*threads*/8, /*reps*/5, "results/sync_methods.dat");
        Benchmark::run_tasks_vs_for(net, args.steps, st, args.chunk, sm, /*threads*/8, /*reps*/5, /*grain*/500, "results/tasks_vs_for.dat");
        // Con I/O para comparar (Amdahl con escritura de archivo)
        Benchmark::run_scaling_io(net, args.steps, st, args.chunk, sm, plist, /*reps*/5, "results/scaling_io.dat");
        // Barrido de granularidad de tasks
        Benchmark::run_tasks_grain_sweep(net, args.steps, sm, /*threads*/8, /*reps*/5,
            std::vector<int>{128,256,512,1024,4096}, "results/tasks_grain.dat");

        std::cout << "Benchmarks listos. Revisa carpeta results/ y usa scripts/plot_speedup.py y scripts/plot_benchmarks.py\n";
        return 0;
    }

    if (!args.threads_csv.empty()) {
        int t = std::atoi(args.threads_csv.c_str());
        if (t>0) omp_set_num_threads(t);
    }
    WavePropagator wp(&net, args.dt, args.S0, args.omega);
    wp.demo_data_clauses("results/datascope_demo.dat");
    wp.run(args.steps, st, args.chunk, sm, "results/energy_trace.dat");
    std::cout << "Ejecución finalizada. Salidas en results/\n";
    return 0;
}
