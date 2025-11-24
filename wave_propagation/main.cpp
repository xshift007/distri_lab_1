#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

#include "Types.h"
#include "Network.h"
#include "WavePropagator.h"
#include "Benchmark.h"

static void usage(){
    std::cout << "Uso: ./wave_propagation [opciones]\n"
              << "  --network {1d,2d}\n"
              << "  --N <int> | --Lx <int> --Ly <int>\n"
              << "  --D <double> --gamma <double> --dt <double>\n"
              << "  --steps <int>\n"
              << "  --S0 <double> --omega <double>\n"
              << "  --noise {off,global,pernode,single}\n"
              << "  --omega-mu <double> --omega-sigma <double> --noise-node <int>\n"
              << "  --schedule {static,dynamic,guided} --chunk <n|auto>\n"
              << "  --threads <int>\n"
              << "  --taskloop --grain <int>\n"
              << "  --energy-accum {reduction,atomic,critical}\n"
              << "  --collapse2\n"
              << "  --dump-frames --frame-every <int>\n"
              << "  --benchmark\n";
}

static ScheduleType parse_schedule(const std::string& s){
    if (s=="static") return ScheduleType::Static;
    if (s=="dynamic") return ScheduleType::Dynamic;
    if (s=="guided")  return ScheduleType::Guided;
    throw std::runtime_error("schedule invalido");
}

static NoiseMode parse_noise(const std::string& s){
    if (s=="off") return NoiseMode::Off;
    if (s=="global") return NoiseMode::Global;
    if (s=="pernode") return NoiseMode::PerNode;
    if (s=="single") return NoiseMode::Single;
    throw std::runtime_error("noise invalido");
}

static EnergyAccum parse_energy_accum(const std::string& s){
    if (s=="reduction") return EnergyAccum::Reduction;
    if (s=="atomic") return EnergyAccum::Atomic;
    if (s=="critical") return EnergyAccum::Critical;
    throw std::runtime_error("energy-accum invalido");
}

static RunParams parse_args(int argc, char** argv){
    RunParams params;
    for (int i=1;i<argc;++i){
        std::string k = argv[i];
        auto next = [&](const char* err){
            if (i+1>=argc){
                usage();
                throw std::runtime_error(err);
            }
            return std::string(argv[++i]);
        };
        if (k=="--network") params.network = next("--network <1d|2d>");
        else if (k=="--N") params.N = std::stoi(next("--N <int>"));
        else if (k=="--Lx") params.Lx = std::stoi(next("--Lx <int>"));
        else if (k=="--Ly") params.Ly = std::stoi(next("--Ly <int>"));
        else if (k=="--D") params.D = std::stod(next("--D <double>"));
        else if (k=="--gamma") params.gamma = std::stod(next("--gamma <double>"));
        else if (k=="--dt") params.dt = std::stod(next("--dt <double>"));
        else if (k=="--steps") params.steps = std::stoi(next("--steps <int>"));
        else if (k=="--S0") params.S0 = std::stod(next("--S0 <double>"));
        else if (k=="--omega") params.omega = std::stod(next("--omega <double>"));
        else if (k=="--noise") params.noise = parse_noise(next("--noise <off|global|pernode|single>"));
        else if (k=="--omega-mu") params.omega_mu = std::stod(next("--omega-mu <double>"));
        else if (k=="--omega-sigma") params.omega_sigma = std::stod(next("--omega-sigma <double>"));
        else if (k=="--noise-node") params.noise_node = std::stoi(next("--noise-node <int>"));
        else if (k=="--schedule") params.schedule = parse_schedule(next("--schedule <static|dynamic|guided>"));
        else if (k=="--chunk"){
            std::string v = next("--chunk <n|auto>");
            if (v=="auto") params.chunk_auto = true;
            else params.chunk = std::stoi(v);
        }
        else if (k=="--threads") params.threads = std::stoi(next("--threads <int>"));
        else if (k=="--taskloop") params.taskloop = true;
        else if (k=="--grain") params.grain = std::stoi(next("--grain <int>"));
        else if (k=="--energy-accum") params.energyAccum = parse_energy_accum(next("--energy-accum <reduction|atomic|critical>"));
        else if (k=="--collapse2") params.collapse2 = true;
        else if (k=="--dump-frames") params.dump_frames = true;
        else if (k=="--frame-every") params.frame_every = std::stoi(next("--frame-every <int>"));
        else if (k=="--benchmark") params.do_bench = true;
        else if (k=="--help" || k=="-h"){ usage(); std::exit(0); }
        else {
            usage();
            throw std::runtime_error("Opcion desconocida: " + k);
        }
    }
    return params;
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
    try {
        RunParams params = parse_args(argc, argv);

        std::filesystem::create_directories("results");
        if (params.dump_frames){
            std::filesystem::create_directories("results/frames");
        }

        // Construcción de red (solo 1D/2D)
        Network net = (params.network=="1d")
            ? Network(params.N, params.D, params.gamma)
            : Network(params.Lx, params.Ly, params.D, params.gamma);
        if (params.network=="1d") net.makeRegular1D(false);
        else                       net.makeRegular2D(false);

        // Estado inicial
        net.setAll(0.0);
        net.setInitialImpulseCenter(1.0);

        if (params.threads>0) omp_set_num_threads(params.threads);

        if (params.chunk_auto){
            int p = (params.threads>0) ? params.threads : omp_get_max_threads();
            params.chunk = compute_auto_chunk(net.size(), params.schedule, p);
            std::cout << "[auto-chunk] " << params.chunk << "\n";
        }

        if (params.do_bench){
            std::vector<int> plist = {1,2,4,8};
            Benchmark::run_scaling(net, params.steps, params.schedule, params.chunk,
                                   plist, /*reps*/10, "results/scaling.dat");
            Benchmark::run_time_vs_chunk_dynamic(net, params.steps, /*threads*/8, /*reps*/10,
                                                 std::vector<int>{64,128,256,512},
                                                 "results/time_vs_chunk_dynamic.dat");
            std::cout << "Benchmarks listos. Revisa carpeta results/\n";
            return 0;
        }

        WavePropagator wp(net, params);
        wp.run(params.energy_out);
        std::cout << "OK. Resultados en results/\n";
    } catch (const std::exception& e){
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
