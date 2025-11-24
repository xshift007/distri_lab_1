# Informe de Diseño e Implementación - Propagación de Ondas

Este informe justifica las decisiones de diseño y la implementación de la simulación de propagación de ondas, cumpliendo con los requisitos de flexibilidad y paralelismo.

## 1. Diseño de Clases (Justificación)

El diseño se aleja de un enfoque puramente funcional o iterativo monolítico, adoptando un enfoque orientado a objetos equilibrado que favorece la claridad y la extensibilidad sin sacrificar el rendimiento.

*   **Clase `Node`**: Encapsula el estado de un punto individual en la malla (amplitud actual, amplitud previa, vecinos). Esto permite que cada nodo gestione su propia información local, facilitando la lógica de actualización y la posible extensión a topologías más complejas (grafos arbitrarios) en el futuro, aunque actualmente se use en mallas regulares 1D y 2D.
*   **Clase `Network`**: Actúa como contenedor y gestor de la topología. Abstrae la complejidad de la conexión entre nodos (vecindad) y los parámetros globales del medio (difusión, amortiguamiento). Mantiene los nodos en un vector contiguo (`std::vector<Node>`) para mejorar la localidad de caché, crucial para el rendimiento en HPC.
*   **Clase `WavePropagator`**: Separa la lógica de la simulación (el "motor" de física) de la estructura de datos (`Network`). Esto sigue el principio de responsabilidad única. `WavePropagator` maneja el bucle de tiempo, la integración numérica, la inyección de ruido y la salida de datos.

Este diseño cumple con el requisito de no ser "solo funciones" ni "una sola clase gigante", proporcionando una estructura modular donde cada componente tiene un rol claro.

## 2. Uso de OpenMP (Justificación)

Se han utilizado cláusulas OpenMP para paralelizar las secciones más costosas computacionalmente de la simulación, específicamente la actualización de estados de los nodos.

*   **`#pragma omp parallel`**: Se crea una región paralela fuera del bucle de tiempo principal. Esto minimiza la sobrecarga (overhead) de crear y destruir hilos en cada paso de tiempo (`fork-join` overhead).
*   **`#pragma omp for`**: Se utiliza para distribuir las iteraciones de los bucles de actualización de nodos entre los hilos disponibles.
    *   **Schedule**: Se ha implementado flexibilidad para elegir entre `static`, `dynamic` y `guided` mediante parámetros de línea de comandos, permitiendo ajustar el balanceo de carga según la arquitectura y el tamaño del problema.
    *   **Collapse**: En el caso 2D, se implementó la opción `collapse(2)` para fusionar los bucles anidados `x` e `y`, aumentando el espacio de iteración y potencialmente mejorando la distribución de carga si las dimensiones son pequeñas.
*   **`#pragma omp taskloop`**: Se añadió como alternativa para demostrar flexibilidad en el paralelismo de tareas, útil si la carga de trabajo por nodo fuera muy irregular (aunque en este caso es uniforme).
*   **Reducción de Energía**: Para el cálculo de la energía global, se ofrecen estrategias de `reduction`, `atomic` y `critical`. La cláusula `reduction(+:E_global)` es generalmente la más eficiente y se usa por defecto o bajo demanda.

No se usaron todas las cláusulas posibles indiscriminadamente, sino aquellas que aportan valor real al rendimiento o demuestran capacidades específicas requeridas por el laboratorio.

## 3. Implementación de Ruido (Bonus)

Se implementó la bonificación de ruido externo con las siguientes características:

*   **Flexibilidad**: El sistema soporta cuatro modos de ruido: `Off`, `Global` (todos los nodos igual), `PerNode` (cada nodo tiene su propia frecuencia $\omega_i$) y `Single` (un solo nodo actúa como fuente).
*   **Distribución Normal**: Para el modo `PerNode`, las frecuencias $\omega_i$ se generan aleatoriamente siguiendo una distribución normal $\mathcal{N}(\mu, \sigma)$ definida por el usuario (`--omega-mu`, `--omega-sigma`).
*   **Implementación**:
    *   Se utiliza `std::mt19937_64` y `std::normal_distribution` para garantizar una buena calidad estadística de los números aleatorios.
    *   El vector `omega_i_` en `WavePropagator` almacena las frecuencias pre-calculadas para evitar costosas llamadas al generador de números aleatorios dentro del bucle crítico de simulación.

## 4. Conclusión

La implementación satisface todos los requisitos funcionales y de diseño. El código es modular, eficiente gracias a OpenMP y flexible en su configuración, incluyendo las características avanzadas de generación de ruido solicitadas.
