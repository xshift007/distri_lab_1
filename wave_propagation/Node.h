#pragma once
#include <vector>

class Node {
    int id_;
    double a_;      // current amplitude
    double a_prev_; // previous amplitude (double buffer)
    std::vector<int> nbrs_;
public:
    explicit Node(int id=0) : id_(id), a_(0.0), a_prev_(0.0) {}

    void addNeighbor(int j){ nbrs_.push_back(j); }
    const std::vector<int>& neighbors() const { return nbrs_; }

    void set(double v){ a_ = v; }
    void setPrev(double v){ a_prev_ = v; }

    double get() const { return a_; }
    double getPrev() const { return a_prev_; }

    void commit(){ a_prev_ = a_; } // new -> prev for next step
};
