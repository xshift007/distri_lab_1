// Node.cpp
// Implementación de los métodos de la clase Node.

#include "Node.h"

Node::Node(int node_id, double initial_amplitude)
    : id(node_id), amplitude(initial_amplitude), previous_amplitude(initial_amplitude) {}

void Node::addNeighbor(int neighbor_id) {
    neighbors.push_back(neighbor_id);
}

void Node::setAmplitude(double new_amplitude) {
    amplitude = new_amplitude;
}

void Node::setPreviousAmplitude(double prev) {
    previous_amplitude = prev;
}

double Node::getAmplitude() const {
    return amplitude;
}

double Node::getPreviousAmplitude() const {
    return previous_amplitude;
}

int Node::getId() const {
    return id;
}

const std::vector<int>& Node::getNeighbors() const {
    return neighbors;
}

int Node::getDegree() const {
    return static_cast<int>(neighbors.size());
}