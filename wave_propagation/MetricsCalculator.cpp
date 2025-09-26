#include "MetricsCalculator.h"
#include <omp.h>

void MetricsCalculator::energy_reduction(const Network& net, double& E){
    E = 0.0;
    const auto& nodes = net.data();
    const int N = net.size();
    #pragma omp parallel for reduction(+:E)
    for (int i=0;i<N;++i){
        double a = nodes[i].get();
        E += a*a;
    }
}

void MetricsCalculator::energy_atomic(const Network& net, double& E){
    E = 0.0;
    const auto& nodes = net.data();
    const int N = net.size();
    #pragma omp parallel
    {
        double local = 0.0;
        #pragma omp for nowait
        for (int i=0;i<N;++i){
            double a = nodes[i].get();
            local += a*a;
        }
        #pragma omp atomic
        E += local;
    }
}

void MetricsCalculator::energy_critical(const Network& net, double& E){
    E = 0.0;
    const auto& nodes = net.data();
    const int N = net.size();
    #pragma omp parallel
    {
        double local = 0.0;
        #pragma omp for nowait
        for (int i=0;i<N;++i){
            double a = nodes[i].get();
            local += a*a;
        }
        #pragma omp critical
        { E += local; }
    }
}
