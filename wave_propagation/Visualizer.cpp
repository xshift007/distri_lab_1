// Visualizer.cpp
// Implementación de la clase Visualizer. La generación de gráficos se realiza
// llamando a un script de Python que utiliza Matplotlib para crear el PNG
// solicitado. Esta implementación asume que Python y Matplotlib están
// disponibles en el entorno de ejecución.

#include "Visualizer.h"

#include <cstdlib>
#include <fstream>
#include <iostream>

void Visualizer::plotSpeedup(const std::string &dataFile, const std::string &imageFile) {
    // Verificar que el archivo de datos exista
    std::ifstream in(dataFile);
    if (!in.good()) {
        std::cerr << "El archivo de datos no se encuentra: " << dataFile << std::endl;
        return;
    }
    in.close();
    // Construir script de Python para generar la gráfica
    std::string script;
    script += "import matplotlib\n";
    script += "import matplotlib.pyplot as plt\n";
    script += "import numpy as np\n";
    script += "data = np.loadtxt('" + dataFile + "', comments='#', delimiter='\t')\n";
    script += "threads = data[:,0]\n";
    script += "speedup = data[:,3]\n";
    script += "plt.figure()\n";
    script += "plt.plot(threads, speedup, marker='o')\n";
    script += "plt.xlabel('Número de hilos')\n";
    script += "plt.ylabel('Speedup')\n";
    script += "plt.title('Speedup vs Número de hilos')\n";
    script += "plt.grid(True)\n";
    script += "plt.savefig('" + imageFile + "')\n";
    script += "\n";
    // Ejecutar el script vía python -c
    std::string command = "python3 - <<'PY'\n" + script + "PY";
    int ret = std::system(command.c_str());
    if (ret != 0) {
        std::cerr << "Error al ejecutar el script de generación de gráficos." << std::endl;
    }
}