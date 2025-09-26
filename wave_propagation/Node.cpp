#include "Node.h"

Node::Node() : neighbors(), value(0.0), newValue(0.0) {}

void Node::setValue(double val) {
    value = val;
}

double Node::getValue() const {
    return value;
}

void Node::addNeighbor(int idx) {
    neighbors.push_back(idx);
}

void Node::clearNeighbors() {
    neighbors.clear();
}

const std::vector<int>& Node::getNeighbors() const {
    return neighbors;
}

void Node::computeNewValue(double diff_coeff, double damp_coeff, double time_step,
                           const std::vector<Node>& allNodes, double external) {
    double sum_diff = 0.0;
    for (int neighbor : neighbors) {
        sum_diff += (allNodes[neighbor].value - value);
    }
    newValue = value + time_step * (diff_coeff * sum_diff - damp_coeff * value + external);
}

void Node::applyNewValue() {
    value = newValue;
}

