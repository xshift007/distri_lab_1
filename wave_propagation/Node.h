// Node.h
// Clase que representa un nodo de la red en la simulación de propagación de ondas.
// Cada nodo mantiene un identificador único, la amplitud actual y la amplitud
// anterior (para evitar dependencias de lectura/escritura durante la integración),
// además de una lista de identificadores de sus vecinos. La clase proporciona
// operaciones para añadir vecinos, actualizar la amplitud y consultar su estado.

#pragma once

#include <vector>

class Node {
private:
    int id;                       // Identificador único del nodo
    double amplitude;             // Amplitud actual de la señal en el nodo
    double previous_amplitude;    // Amplitud del paso de tiempo anterior
    std::vector<int> neighbors;   // Lista de vecinos (ids)

public:
    // Constructor que inicializa el nodo con un identificador y amplitud inicial opcional
    explicit Node(int node_id, double initial_amplitude = 0.0);

    // Añade un vecino al nodo
    void addNeighbor(int neighbor_id);

    // Establece la amplitud actual del nodo
    void setAmplitude(double new_amplitude);

    // Actualiza la amplitud previa (guardar valor antes de la integración)
    void setPreviousAmplitude(double prev);

    // Devuelve la amplitud actual
    double getAmplitude() const;

    // Devuelve la amplitud del paso anterior
    double getPreviousAmplitude() const;

    // Obtiene el identificador del nodo
    int getId() const;

    // Devuelve la lista de vecinos
    const std::vector<int>& getNeighbors() const;

    // Devuelve el grado del nodo (número de vecinos)
    int getDegree() const;
};