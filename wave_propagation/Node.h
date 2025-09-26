#pragma once
#include <vector>

class Node {
    int id_;
    double amp_;
    double amp_prev_;
    std::vector<int> nbrs_;
public:
    explicit Node(int id = 0) : id_(id), amp_(0.0), amp_prev_(0.0) {}

    void addNeighbor(int j) { nbrs_.push_back(j); }

    void setAmplitude(double a) { amp_ = a; }
    void setPrev(double a)      { amp_prev_ = a; }

    void commit() { amp_prev_ = amp_; }

    double get() const     { return amp_; }
    double getPrev() const { return amp_prev_; }

    const std::vector<int>& neighbors() const { return nbrs_; }
    int degree() const { return (int)nbrs_.size(); }
    int id() const { return id_; }
};
