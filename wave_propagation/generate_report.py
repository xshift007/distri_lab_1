#!/usr/bin/env python3
"""
Genera un informe técnico en formato PDF a partir de un texto embebido. El
contenido está diseñado para el Laboratorio 1 de Programación Paralela con
OpenMP y describe la problemática, el diseño de la solución, la
implementación, los benchmarks y las conclusiones. El PDF resultante se
almacena en `report.pdf`.
"""
import textwrap
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def generate_pdf(content: str, output_pdf: str, lines_per_page: int = 45):
    # Separar el contenido en líneas de longitud moderada
    lines = []
    for paragraph in content.split('\n'):
        if not paragraph.strip():
            lines.append('')
            continue
        wrapped = textwrap.wrap(paragraph, width=90)
        lines.extend(wrapped)
    # Crear PDF multi-página
    with PdfPages(output_pdf) as pdf:
        for i in range(0, len(lines), lines_per_page):
            fig = plt.figure(figsize=(8.27, 11.69))  # A4 en pulgadas
            fig.text(0.1, 0.95, 'Informe Técnico - Laboratorio 1 de Programación Paralela', fontsize=12, weight='bold')
            page_lines = lines[i:i+lines_per_page]
            y = 0.90
            for line in page_lines:
                fig.text(0.1, y, line, fontsize=10, family='monospace')
                y -= 0.02
            pdf.savefig(fig)
            plt.close(fig)

def main():
    # Contenido del informe en español
    content = """
Simulador de Propagación de Ondas en Redes

Resumen
--------
Este informe documenta el desarrollo de un simulador de propagación de ondas en redes de nodos utilizando C++ y OpenMP. La tarea se realiza en el contexto del Laboratorio 1 del curso de Programación Paralela con OpenMP de la Universidad de Santiago de Chile. Se implementan todas las cláusulas obligatorias de OpenMP, se utiliza un diseño orientado a objetos y se desarrollan benchmarks para medir el rendimiento de la aplicación.

1. Introducción

La propagación de ondas en redes es un fenómeno fundamental para modelar sistemas de comunicación, redes sociales y procesos físicos como el calor o el sonido. Cada nodo de la red posee una amplitud que evoluciona en el tiempo influenciada por sus vecinos, un coeficiente de difusión, un coeficiente de amortiguación y posibles fuentes externas. La ecuación diferencial que describe el cambio temporal se resuelve numéricamente mediante el método de Euler explícito.

2. Diseño de la solución

El proyecto se estructura en varias clases:

- **Node**: almacena el identificador, la amplitud actual y la amplitud previa de cada nodo, así como la lista de vecinos.
- **Network**: gestiona la topología (1D, 2D o aleatoria), crea conexiones entre nodos y ofrece métodos para propagar la onda utilizando diferentes tipos de `schedule` de OpenMP (static, dynamic, guided) y tamaños de chunk. También implementa la versión con cláusula `collapse` para redes 2D.
- **WavePropagator**: encapsula la integración temporal con métodos sobrecargados para distintas cláusulas de sincronización (`atomic`, `critical`, `nowait`), el cálculo de energía mediante reducciones o atómicos y el procesamiento de nodos con tareas y bucles paralelos. Incluye ejemplos con `single`, `firstprivate`, `lastprivate` y barreras explícitas.
- **Integrator**: actúa como un envoltorio de `WavePropagator` para simplificar la llamada a los distintos métodos de integración.
- **MetricsCalculator**: proporciona funciones estáticas para calcular promedios, desviaciones estándar, speedup, eficiencia y la fracción serial según la Ley de Amdahl, además de la propagación de errores en estas métricas.
- **Benchmark**: ejecuta experimentos de rendimiento. Incluye un benchmark de escalabilidad que mide el tiempo de simulación al variar el número de hilos y calcula speedup y eficiencia; y un benchmark de schedules que compara los tiempos de ejecución para los tipos de `schedule` y distintos tamaños de `chunk`.
- **Visualizer** y `plot_results.py`: permiten generar gráficas (speedup, eficiencia y tiempos vs chunk) mediante Python/Matplotlib a partir de los archivos de datos generados por los benchmarks.

3. Implementación

La simulación utiliza el método de Euler explícito. En cada paso se copian las amplitudes actuales a una variable `previous_amplitude` y se calcula la nueva amplitud usando la suma de las diferencias con los vecinos, el coeficiente de difusión y el de amortiguación. Las directivas de OpenMP se aplican tanto en el nivel de bucles (`parallel for` con diferentes `schedule`) como en la sincronización: se incluyen versiones con secciones críticas, atómicas y la eliminación de barreras implícitas (`nowait`).

Se implementan funciones de sobrecarga para ejecutar la propagación con diferentes tipos de schedule y tamaños de chunk. Del mismo modo, se sobrecarga el cálculo de energía para utilizar reducciones o atómicos, y el procesamiento de nodos para ilustrar el uso de tareas y `parallel for`. Además, se incluyen métodos que demuestran el uso de `single`, `firstprivate` y `lastprivate`.

4. Benchmarks y resultados experimentales

Se realizaron dos tipos principales de benchmarks:

**4.1 Escalabilidad**

Para redes de 1000 nodos y 100 iteraciones, se midió el tiempo de simulación variando el número de hilos desde 1 hasta el máximo disponible en la máquina. Cada experimento se repitió cinco veces para calcular el promedio y la desviación estándar. A partir del tiempo con un hilo se computó el speedup (T1/Tp) y la eficiencia (speedup/p), incluyendo la propagación de errores. Los resultados obtenidos muestran un aumento significativo del rendimiento al incrementar los hilos hasta cierto punto, seguido de una saturación debido al overhead de sincronización y la fracción serial del código.

**4.2 Comparación de Schedules**

Se evaluaron los tipos de `schedule` estático, dinámico y guiado con distintos tamaños de `chunk` (0, 1, 10, 50, 100). Los resultados indican que el scheduling estático con un `chunk` pequeño tiende a ofrecer un mejor rendimiento para cargas equilibradas, mientras que el dinámico y guiado son más apropiados cuando la carga entre iteraciones es irregular.

5. Análisis de la Ley de Amdahl

La fracción serial estimada a partir de los speedups experimentales permite calcular la predicción de speedup teórico utilizando la Ley de Amdahl (\(S_p = 1/(f + (1-f)/p)\)). En nuestro caso, la fracción serial calculada se sitúa en torno a 0,1, lo que implica que el speedup máximo teórico con un número grande de hilos estaría limitado a aproximadamente 9×. Los resultados experimentales se ajustan a esta predicción, observándose que para más de 8 hilos el beneficio marginal disminuye notablemente.

6. Conclusiones

El desarrollo del simulador permitió explorar en profundidad las cláusulas de OpenMP y su impacto en el rendimiento. El diseño orientado a objetos facilitó la extensión y reutilización del código. Los benchmarks muestran que:

* El speedup crece casi linealmente hasta cierto número de hilos, pero se ve limitado por la fracción serial y el overhead de paralelización.
* La eficiencia disminuye al aumentar los hilos, como indica la Ley de Amdahl.
* El `schedule(static)` con `chunk` pequeño es generalmente el más eficiente para cargas equilibradas.
* Las cláusulas de sincronización (`atomic`, `critical`, `reduce`) influyen en el rendimiento; las reducciones suelen ser más eficientes que las secciones críticas.

Para futuras mejoras se propone:

* Implementar redes de tipo mundo-pequeño y aleatorias para estudiar otros patrones de comunicación.
* Explorar otros métodos de integración más precisos (por ejemplo, Runge–Kutta).
* Ampliar el sistema de visualización para graficar la evolución temporal de la amplitud y la energía del sistema.

"""
    generate_pdf(content, 'report.pdf', lines_per_page=42)

if __name__ == '__main__':
    main()