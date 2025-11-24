import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/energy_trace.dat", help="Archivo de energía")
    parser.add_argument("--output", default="results/energy_plot.png", help="Imagen de salida")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: No se encuentra {args.input}. Ejecuta la simulación primero.")
        return

    try:
        # Leer ignorando lineas con #
        data = np.loadtxt(args.input, comments="#")
    except Exception as e:
        print(f"Error leyendo archivo: {e}")
        return

    if data.ndim != 2 or data.shape[1] < 2:
        print("El archivo de datos no tiene el formato esperado (step, energy).")
        return

    steps = data[:, 0]
    energy = data[:, 1]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, energy, label="Energía Total del Sistema", color='b')
    plt.xlabel("Pasos de Tiempo (Steps)")
    plt.ylabel("Energía (u^2)")
    plt.title("Evolución de la Energía en la Malla")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Gráfico guardado en: {args.output}")


if __name__ == "__main__":
    main()
