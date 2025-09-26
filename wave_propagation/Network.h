#pragma once
#include <vector>
#include <random>
#include "Node.h"

class Network {
    std::vector<Node> nodes_;
    int N_ = 0;
    int Lx_ = 0, Ly_ = 0;
    double D_ = 0.1, gamma_ = 0.01;
    bool is2D_ = false;
public:
    Network() = default;
    Network(int N, double D, double gamma);
    void clear();
    void makeRegular1D(bool periodic = false);
    void makeRegular2D(int Lx, int Ly, bool periodic = false);
    void makeRandom(double mean_degree);
    void makeSmallWorld(int k, double beta);
    void setInitialImpulseCenter(double A0);
    void setAll(double A);
    std::vector<Node>& data() { return nodes_; }
    const std::vector<Node>& data() const { return nodes_; }
    int size() const { return N_; }
    int Lx() const { return Lx_; }
    int Ly() const { return Ly_; }
    double diffusion() const { return D_; }
    double damping() const { return gamma_; }
    bool is2D() const { return is2D_; }
};
