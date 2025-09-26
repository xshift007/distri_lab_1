#include "NetworkStatePrinter.h"

#include <iomanip>
#include <iostream>
#include <vector>

#include "Network.h"

NetworkStatePrinter::NetworkStatePrinter(std::ostream& outputStream)
    : out(&outputStream), precision(4) {}

void NetworkStatePrinter::setPrecision(int precisionDigits) {
    precision = precisionDigits;
}

int NetworkStatePrinter::getPrecision() const {
    return precision;
}

void NetworkStatePrinter::printStep(const Network& network, int step) const {
    if (out == nullptr) {
        return;
    }

    *out << "Paso " << step << ": ";
    out->setf(std::ios::fixed);
    out->precision(precision);

    switch (network.getTopology()) {
        case Network::Topology::Line1D:
            printLine1D(network);
            break;
        case Network::Topology::Grid2D:
            printGrid2D(network);
            break;
    }

    *out << std::endl;
}

void NetworkStatePrinter::printLine1D(const Network& network) const {
    const std::vector<Node>& nodes = network.nodes();
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        *out << nodes[i].getValue();
        if (i + 1 < nodes.size()) {
            *out << ", ";
        }
    }
}

void NetworkStatePrinter::printGrid2D(const Network& network) const {
    const int rows = network.getRows();
    const int cols = network.getCols();
    const std::vector<Node>& nodes = network.nodes();

    for (int r = 0; r < rows; ++r) {
        if (r > 0) {
            *out << "\n         ";
        }
        for (int c = 0; c < cols; ++c) {
            const int idx = r * cols + c;
            *out << nodes[idx].getValue();
            if (c + 1 < cols) {
                *out << ", ";
            }
        }
    }
}

