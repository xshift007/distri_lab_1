// Visualizer.h
// Clase que encapsula la generación de gráficas de resultados utilizando
// herramientas externas como Python/Matplotlib. Proporciona métodos para
// crear imágenes a partir de archivos .dat generados por Benchmarks.

#pragma once

#include <string>

class Visualizer {
public:
    // Genera un gráfico de speedup a partir de un archivo de datos y lo guarda como imagen PNG
    static void plotSpeedup(const std::string &dataFile, const std::string &imageFile);
    // Otros métodos para graficar podrían añadirse aquí (eficiencia, Amdahl, etc.)
};