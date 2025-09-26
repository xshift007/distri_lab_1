#ifndef WAVEPROPAGATOR_H
#define WAVEPROPAGATOR_H

class Network;
class NetworkStatePrinter;

class WavePropagator {
public:
    WavePropagator(Network& net, int steps = 1000);

    void setSteps(int steps);
    int getSteps() const;

    void integrateEuler();
    void integrateEulerSchedule(int scheduleType, int chunkSize = -1,
                                NetworkStatePrinter* printer = nullptr,
                                int visualizeLimit = 32);

private:
    Network& network;
    int timeSteps;
};

#endif
