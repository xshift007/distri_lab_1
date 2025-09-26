#ifndef NETWORK_H
#define NETWORK_H

#include <vector>

#include "Node.h"

class Network {
public:
    enum class Topology { Line1D, Grid2D };

    Network(int nodeCount, double diffusionCoeff, double dampingCoeff, double timeStep);

    int size() const;
    double getDiffusionCoeff() const;
    double getDampingCoeff() const;
    double getTimeStep() const;

    Topology getTopology() const;
    int getRows() const;
    int getCols() const;

    void buildLine();
    void buildGrid(int rows, int cols);

    void resetValues(double value = 0.0);
    void setValueAt(int index, double value);

    int indexFromCoordinates(int row, int col) const;

    std::vector<Node>& nodesMutable();
    const std::vector<Node>& nodes() const;

    void applyNewValues();

private:
    void validateIndex(int index) const;

    int nodeCount;
    double diffusion;
    double damping;
    double timeStep;
    Topology topology;
    int rows;
    int cols;
    std::vector<Node> nodesStorage;
};

#endif
