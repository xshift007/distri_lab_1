#pragma once                   // Esto es para que se compile una sola vez y no ocurran errores

#include <vector>               // std::vector para guardar vecinos

class Node {
    int id_;                    // Identificador del nodo para solucionar posibles bugs
    double a_;                  // amplitud actual
    double a_prev_;             // anmplitud previa
    std::vector<int> nbrs_;     // lista de indices de los nodos



public:
    explicit Node(int id = 0)   // Permite crear nodo con id 0 por defecto
        : id_(id), a_(0.0), a_prev_(0.0) {}

    void addNeighbor(int j){    // agrega un vecino por indice en el arreglo global de nodos
        nbrs_.push_back(j);
    }

    const std::vector<int>& neighbors() const { // Acceso de lectura a los vecinos
        return nbrs_;           
    }

    void set(double v){a_ = v; }                //escribe la amplitud actual
    void setPrev(double v){a_prev_ = v; }       //escribe la amplitud previa

    double get() const {return a_; }            //lee la amplitud
    double getPrev() const {return a_prev_; }   //lee la amplitud previa
    void commit(){ a_prev_ = a_; }              // pasa el estado actual previo para el siguiente paso
};
