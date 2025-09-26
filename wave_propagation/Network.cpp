#include "Network.h"

#include <stdexcept>

Network::Network(int nodeCount_, double diffusionCoeff, double dampingCoeff, double timeStep_)
    : nodeCount(nodeCount_ > 0 ? nodeCount_ : 1),
      diffusion(diffusionCoeff),
      damping(dampingCoeff),
      timeStep(timeStep_),
      topology(Topology::Line1D),
      rows(nodeCount > 0 ? nodeCount : 1),
      cols(1),
      nodesStorage(static_cast<std::size_t>(nodeCount)) {}

int Network::size() const {
    return nodeCount;
}

double Network::getDiffusionCoeff() const {
    return diffusion;
}

double Network::getDampingCoeff() const {
    return damping;
}

double Network::getTimeStep() const {
    return timeStep;
}

Network::Topology Network::getTopology() const {
    return topology;
}

int Network::getRows() const {
    return rows;
}

int Network::getCols() const {
    return cols;
}

void Network::buildLine() {
    topology = Topology::Line1D;
    rows = nodeCount;
    cols = 1;
    nodesStorage.assign(static_cast<std::size_t>(nodeCount), Node());

    for (int i = 0; i < nodeCount; ++i) {
        if (i > 0) {
            nodesStorage[i].addNeighbor(i - 1);
        }
        if (i < nodeCount - 1) {
            nodesStorage[i].addNeighbor(i + 1);
        }
    }
}

void Network::buildGrid(int rows_, int cols_) {
    if (rows_ <= 0 || cols_ <= 0) {
        throw std::invalid_argument("La grilla 2D debe tener dimensiones positivas");
    }

    topology = Topology::Grid2D;
    rows = rows_;
    cols = cols_;
    nodeCount = rows * cols;
    nodesStorage.assign(static_cast<std::size_t>(nodeCount), Node());

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int idx = r * cols + c;
            if (r > 0) {
                nodesStorage[idx].addNeighbor((r - 1) * cols + c);
            }
            if (r < rows - 1) {
                nodesStorage[idx].addNeighbor((r + 1) * cols + c);
            }
            if (c > 0) {
                nodesStorage[idx].addNeighbor(r * cols + (c - 1));
            }
            if (c < cols - 1) {
                nodesStorage[idx].addNeighbor(r * cols + (c + 1));
            }
        }
    }
}

void Network::resetValues(double value) {
    for (Node& node : nodesStorage) {
        node.setValue(value);
    }
}

void Network::setValueAt(int index, double value) {
    validateIndex(index);
    nodesStorage[index].setValue(value);
}

int Network::indexFromCoordinates(int row, int col) const {
    if (topology != Topology::Grid2D) {
        throw std::logic_error("La red actual no es 2D");
    }
    if (row < 0 || row >= rows || col < 0 || col >= cols) {
        throw std::out_of_range("Coordenadas fuera de la grilla");
    }
    return row * cols + col;
}

std::vector<Node>& Network::nodesMutable() {
    return nodesStorage;
}

const std::vector<Node>& Network::nodes() const {
    return nodesStorage;
}

void Network::applyNewValues() {
    for (Node& node : nodesStorage) {
        node.applyNewValue();
    }
}

void Network::validateIndex(int index) const {
    if (index < 0 || index >= nodeCount) {
        throw std::out_of_range("Indice de nodo invalido");
    }
}

