#ifndef NETWORKSTATEPRINTER_H
#define NETWORKSTATEPRINTER_H

#include <iosfwd>

class Network;

class NetworkStatePrinter {
public:
    explicit NetworkStatePrinter(std::ostream& outputStream);

    void setPrecision(int precisionDigits);
    int getPrecision() const;

    void printStep(const Network& network, int step) const;

private:
    void printLine1D(const Network& network) const;
    void printGrid2D(const Network& network) const;

    std::ostream* out;
    int precision;
};

#endif
