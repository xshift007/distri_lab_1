#pragma once
#include "WavePropagator.h"
#include <string>
#include <vector>
class Benchmark {
public:
    static void run_scaling(Network& net, int steps, ScheduleType st, int chunk,
                            SyncMethod sm, const std::vector<int>& threads_list,
                            int reps, const std::string& out_scaling_path);
    static void run_schedule_chunk(Network& net, int steps, SyncMethod sm,
                                   const std::vector<ScheduleType>& sts,
                                   const std::vector<int>& chunks,
                                   int threads, int reps, const std::string& out_path);
    static void run_sync_methods(Network& net, int steps, ScheduleType st, int chunk,
                                 int threads, int reps, const std::string& out_path);
    static void run_tasks_vs_for(Network& net, int steps, ScheduleType st, int chunk,
                                 SyncMethod sm, int threads, int reps, int grain,
                                 const std::string& out_path);
};
