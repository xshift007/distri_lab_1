#ifndef NODE_H
#define NODE_H

#include <vector>

class Node {
public:
    Node();

    void setValue(double val);
    double getValue() const;

    void addNeighbor(int idx);
    void clearNeighbors();
    const std::vector<int>& getNeighbors() const;

    void computeNewValue(double diff_coeff, double damp_coeff, double time_step,
                         const std::vector<Node>& allNodes, double external = 0.0);

    void applyNewValue();

private:
    std::vector<int> neighbors;
    double value;
    double newValue;
};

#endif
