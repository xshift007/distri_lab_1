#pragma once

#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "Types.h"
#include "Network.h"

class WavePropagator {
public:
    WavePropagator(Network& net, const RunParams& params);

    void run(const std::string& energy_out);

    double time() const { return tcur_; }

private:
    Network& net_;
    RunParams params_;
    double tcur_ = 0.0;
    double last_1d_sample_ = 0.0;

    std::vector<double> omega_i_;
    int single_idx_ = -1;

    std::mt19937_64 rng_;
    std::normal_distribution<double> norm_;

    double source_val(int idx, double time) const;
    void dump_energy(std::ofstream& fe, int step, double E);
    void dump_frame_1d(int step);
    void dump_frame_2d(int step);
};
