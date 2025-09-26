#pragma once
#include <vector>
#include <random>

class Node {
    int id_;
    double amp_;
    double amp_prev_;
    std::vector<int> nbrs_;
public:
    explicit Node(int id = 0) : id_(id), amp_(0.0), amp_prev_(0.0) {}
    void addNeighbor(int j) { 
        // evitar duplicados simples
        for (int x : nbrs_) if (x==j) return;
        nbrs_.push_back(j);
    }
    void setAmplitude(double a) { amp_ = a; }
    void setPrev(double a)      { amp_prev_ = a; }
    void commit() { amp_prev_ = amp_; }
    double get() const     { return amp_; }
    double getPrev() const { return amp_prev_; }
    const std::vector<int>& neighbors() const { return nbrs_; }
    int degree() const { return (int)nbrs_.size(); }
    int id() const { return id_; }
};

class Network {
    std::vector<Node> nodes_;
    int N_ = 0;
    int Lx_ = 0, Ly_ = 0;     // para 2D
    double D_ = 0.1, gamma_ = 0.01;
    bool is2D_ = false;
public:
    Network() = default;
    Network(int N, double D, double gamma);
    void clear();
    void makeRegular1D(bool periodic = false);
    void makeRegular2D(int Lx, int Ly, bool periodic = false);
    void makeRandom(double mean_degree);               // escala ~O(N * mean_degree)
    void makeSmallWorld(int k, double beta);           // Watts-Strogatz
    void setInitialImpulseCenter(double A0);           // centro elevado
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
