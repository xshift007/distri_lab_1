#pragma once
#include <vector>
#include <cassert>
#include "Node.h"

class Network {
    std::vector<Node> nodes_;
    bool is2d_ = false;
    int Lx_ = 0, Ly_ = 0;
    double D_ = 0.1, g_ = 0.01; // diffusion, damping

public:
    Network(int N, double D, double g);
    Network(int Lx, int Ly, double D, double g);

    // Build topologies
    void makeRegular1D(bool periodic=false);
    void makeRegular2D(bool periodic=false);

    // State ops
    void setAll(double v);
    void setInitialImpulseCenter(double amp);

    // Accessors
    int size() const { return (int)nodes_.size(); }
    bool is2D() const { return is2d_; }
    int Lx() const { return Lx_; }
    int Ly() const { return Ly_; }
    double diffusion() const { return D_; }
    double damping() const { return g_; }
    std::vector<Node>& data(){ return nodes_; }
    const std::vector<Node>& data() const { return nodes_; }
};
