#pragma once // para que se compile solo una vez


#include <vector>   // 
#include <cassert>
#include "Node.h"

class Network {
    std::vector<Node> nodes_;   //arreglo contiguo de nodos
    bool is2d_ = false;         // valor que indica que tipo de topologia es 1D/2D
    int Lx_ = 0, Ly_ = 0;       // en caso de ser 2D da las dimensiones de la grilla
    double D_ = 0.1, g_ = 0.01; // parametros globales Difusion y amortiguamiento
public:
    Network(int N, double D, double g);            // construccion 1d
    Network(int Lx, int Ly, double D, double g);   // construccion 2d

    // Build topologies
    void makeRegular1D(bool periodic=false);       // construye la conectividad  1D si periodic=true, construye conectividad
    void makeRegular2D(bool periodic=false);       // construye la conectividad  2D si periodic=true, construye conectividad

    // inicializa los estados
    void setAll(double v);                         // 
    void setInitialImpulseCenter(double amp);      //

    // Getters
    int size() const { return (int)nodes_.size(); }
    bool is2D() const { return is2d_; }
    int Lx() const { return Lx_; }
    int Ly() const { return Ly_; }
    double diffusion() const { return D_; }
    double damping() const { return g_; }

    // Acceso a los nodos (Lectura escritura)
    std::vector<Node>& data(){ return nodes_; }
    const std::vector<Node>& data() const { return nodes_; }
};
