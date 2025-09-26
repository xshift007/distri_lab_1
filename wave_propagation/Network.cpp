#include "Network.h"
#include <cassert>
#include <random>
#include <cmath>

Network::Network(int N, double D, double gamma)
: N_(N), D_(D), gamma_(gamma) {
    nodes_.reserve(N_);
    for (int i=0;i<N_;++i) nodes_.emplace_back(i);
}
void Network::clear(){
    nodes_.clear();
    nodes_.shrink_to_fit();
    nodes_.reserve(N_);
    for (int i=0;i<N_;++i) nodes_.emplace_back(i);
}
void Network::makeRegular1D(bool periodic){
    is2D_ = false;
    if ((int)nodes_.size()!=N_) { nodes_.clear(); nodes_.reserve(N_); for(int i=0;i<N_;++i) nodes_.emplace_back(i); }
    for (int i=0;i<N_; ++i) {
        int left = i-1, right = i+1;
        if (periodic) {
            left = (i-1+N_)%N_;
            right = (i+1)%N_;
            if (left != i)  nodes_[i].addNeighbor(left);
            if (right != i) nodes_[i].addNeighbor(right);
        } else {
            if (left >= 0) nodes_[i].addNeighbor(left);
            if (right < N_) nodes_[i].addNeighbor(right);
        }
    }
}
void Network::makeRegular2D(int Lx, int Ly, bool periodic){
    is2D_ = true;
    Lx_ = Lx; Ly_ = Ly;
    N_ = Lx_*Ly_;
    nodes_.clear();
    nodes_.reserve(N_);
    for (int i=0;i<N_;++i) nodes_.emplace_back(i);
    auto idx = [this](int x, int y){ return y*Lx_ + x; };
    for (int y=0; y<Ly_; ++y) {
        for (int x=0; x<Lx_; ++x) {
            int i = idx(x,y);
            auto add = [&](int nx, int ny){
                if (nx>=0 && nx<Lx_ && ny>=0 && ny<Ly_) {
                    nodes_[i].addNeighbor(idx(nx,ny));
                } else if (periodic) {
                    int px = (nx%Lx_ + Lx_)%Lx_;
                    int py = (ny%Ly_ + Ly_)%Ly_;
                    nodes_[i].addNeighbor(idx(px,py));
                }
            };
            add(x-1,y); add(x+1,y); add(x,y-1); add(x,y+1);
        }
    }
}
void Network::makeRandom(double mean_degree){
    is2D_ = false;
    if ((int)nodes_.size()!=N_) { nodes_.clear(); nodes_.reserve(N_); for(int i=0;i<N_;++i) nodes_.emplace_back(i); }
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<int> unif(0, N_-1);
    std::poisson_distribution<int> pois(std::max(0.0, mean_degree));
    for (int i=0;i<N_; ++i) {
        int deg = pois(rng);
        for (int k=0;k<deg; ++k) {
            int j = unif(rng);
            if (j==i) continue;
            nodes_[i].addNeighbor(j);
            nodes_[j].addNeighbor(i);
        }
    }
}
void Network::makeSmallWorld(int k, double beta){
    is2D_ = false;
    if ((int)nodes_.size()!=N_) { nodes_.clear(); nodes_.reserve(N_); for(int i=0;i<N_;++i) nodes_.emplace_back(i); }
    for (int i=0;i<N_; ++i){
        for (int d=1; d<=k; ++d){
            int j1 = (i+d) % N_;
            int j2 = (i-d+N_) % N_;
            nodes_[i].addNeighbor(j1); nodes_[j1].addNeighbor(i);
            nodes_[i].addNeighbor(j2); nodes_[j2].addNeighbor(i);
        }
    }
    std::mt19937_64 rng(98765);
    std::uniform_real_distribution<double> U(0.0,1.0);
    std::uniform_int_distribution<int> unif(0, N_-1);
    for (int i=0;i<N_; ++i){
        for (int d=1; d<=k; ++d){
            if (U(rng) < beta){
                int newj = unif(rng);
                if (newj==i) continue;
                nodes_[i].addNeighbor(newj);
                nodes_[newj].addNeighbor(i);
            }
        }
    }
}
void Network::setInitialImpulseCenter(double A0){
    for (int i=0;i<N_;++i) { nodes_[i].setPrev(0.0); nodes_[i].setAmplitude(0.0); }
    if (is2D_) {
        int cx = Lx_/2, cy = Ly_/2; int ic = cy*Lx_ + cx;
        nodes_[ic].setPrev(A0); nodes_[ic].setAmplitude(A0);
    } else {
        int c = N_/2; nodes_[c].setPrev(A0); nodes_[c].setAmplitude(A0);
    }
}
void Network::setAll(double A){
    for (int i=0;i<N_;++i) { nodes_[i].setPrev(A); nodes_[i].setAmplitude(A); }
}
