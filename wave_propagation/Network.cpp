#include "Network.h"
#include <cmath>

Network::Network(int N, double D, double g) : nodes_(N), is2d_(false), Lx_(N), Ly_(1), D_(D), g_(g) {
    for (int i=0;i<N;++i) nodes_[i] = Node(i);
}
Network::Network(int Lx, int Ly, double D, double g) : nodes_(Lx*Ly), is2d_(true), Lx_(Lx), Ly_(Ly), D_(D), g_(g) {
    for (int i=0;i<Lx_*Ly_;++i) nodes_[i] = Node(i);
}

void Network::makeRegular1D(bool periodic){
    is2d_ = false; Lx_ = size(); Ly_ = 1;
    const int N = size();
    for (int i=0;i<N;++i){
        if (i-1>=0) nodes_[i].addNeighbor(i-1);
        else if (periodic) nodes_[i].addNeighbor(N-1);
        if (i+1<N) nodes_[i].addNeighbor(i+1);
        else if (periodic) nodes_[i].addNeighbor(0);
    }
}

void Network::makeRegular2D(bool periodic){
    is2d_ = true;
    const int N = size();
    for (int i=0;i<N;++i){
        int x = i % Lx_, y = i / Lx_;
        auto idx = [&](int xx, int yy)->int{ return yy*Lx_ + xx; };
        // left
        if (x>0) nodes_[i].addNeighbor(idx(x-1,y));
        else if (periodic) nodes_[i].addNeighbor(idx(Lx_-1,y));
        // right
        if (x+1<Lx_) nodes_[i].addNeighbor(idx(x+1,y));
        else if (periodic) nodes_[i].addNeighbor(idx(0,y));
        // up
        if (y>0) nodes_[i].addNeighbor(idx(x,y-1));
        else if (periodic) nodes_[i].addNeighbor(idx(x,Ly_-1));
        // down
        if (y+1<Ly_) nodes_[i].addNeighbor(idx(x,y+1));
        else if (periodic) nodes_[i].addNeighbor(idx(x,0));
    }
}

void Network::setAll(double v){
    for (auto &n : nodes_){ n.set(v); n.setPrev(v); }
}
void Network::setInitialImpulseCenter(double amp){
    // Centro geométrico (1D: Lx_/2 ; 2D: (Lx_/2, Ly_/2))
    int idx = is2d_ ? ((Ly_/2)*Lx_ + (Lx_/2)) : (Lx_/2);
    if (idx>=0 && idx<(int)nodes_.size()) nodes_[idx].set(amp), nodes_[idx].setPrev(amp);
}
