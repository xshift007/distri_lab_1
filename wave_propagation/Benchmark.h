#pragma once
#include <string>
#include <vector>
#include "Types.h"
#include "Network.h"
#include "WavePropagator.h"

namespace Benchmark {

// Repite y promedia tiempos para scaling (p = #threads)
void run_scaling(Network& net, int steps, ScheduleType st, int chunk,
                 const std::vector<int>& threads_list, int reps,
                 const std::string& out_path);

// Barrido de chunk para schedule = dynamic (tiempo vs chunk)
void run_time_vs_chunk_dynamic(Network& net, int steps, int threads, int reps,
                               const std::vector<int>& chunks,
                               const std::string& out_path);
}
