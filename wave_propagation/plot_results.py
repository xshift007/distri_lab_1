#!/usr/bin/env python3
"""
Script para generar gráficas de resultados de benchmarks de propagación de
ondas. Lee los archivos de datos generados por el ejecutable C++ y
produce imágenes PNG con las siguientes comparaciones:

1. Speedup vs número de hilos
2. Eficiencia vs número de hilos
3. Tiempo de ejecución vs tamaño de chunk para distintos schedules
4. Speedup experimental vs predicción de la Ley de Amdahl

Los archivos esperados son:
 - benchmark_results.dat: generado por `./wave_propagation -benchmark`
 - schedule_results.dat: generado por `./wave_propagation -schedule`

Uso:
    python3 plot_results.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def load_scalability(filename: str):
    data = np.loadtxt(filename, comments='#', delimiter='\t')
    threads = data[:, 0].astype(int)
    meanT = data[:, 1]
    stdT = data[:, 2]
    speedup = data[:, 3]
    sigmaS = data[:, 4]
    efficiency = data[:, 5]
    sigmaE = data[:, 6]
    f = data[:, 7]
    amdahl_pred = data[:, 8]
    return threads, meanT, stdT, speedup, sigmaS, efficiency, sigmaE, f, amdahl_pred

def load_schedule(filename: str):
    # schedule_type es texto, chunk_size, meanT, stdT
    sched_types = []
    chunk_sizes = []
    meanT = []
    stdT = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split('\t')
            if len(parts) != 4: continue
            sched_types.append(parts[0])
            chunk_sizes.append(int(parts[1]))
            meanT.append(float(parts[2]))
            stdT.append(float(parts[3]))
    return np.array(sched_types), np.array(chunk_sizes), np.array(meanT), np.array(stdT)

def plot_speedup(threads, speedup, sigmaS, amdahl_pred, pdf: PdfPages):
    plt.figure()
    plt.errorbar(threads, speedup, yerr=sigmaS, fmt='o-', label='Experimental')
    plt.plot(threads, amdahl_pred, 's--', label='Predicción Amdahl')
    plt.xlabel('Número de hilos')
    plt.ylabel('Speedup')
    plt.title('Speedup vs Número de hilos')
    plt.legend()
    plt.grid(True)
    pdf.savefig()
    plt.close()

def plot_efficiency(threads, efficiency, sigmaE, pdf: PdfPages):
    plt.figure()
    plt.errorbar(threads, efficiency, yerr=sigmaE, fmt='o-')
    plt.xlabel('Número de hilos')
    plt.ylabel('Eficiencia')
    plt.title('Eficiencia vs Número de hilos')
    plt.grid(True)
    pdf.savefig()
    plt.close()

def plot_schedule_results(sched_types, chunk_sizes, meanT, pdf: PdfPages):
    # Identificar tipos únicos
    unique_scheds = sorted(set(sched_types))
    plt.figure()
    for sched in unique_scheds:
        mask = sched_types == sched
        cs = chunk_sizes[mask]
        mt = meanT[mask]
        plt.plot(cs, mt, marker='o', label=sched)
    plt.xlabel('Chunk size')
    plt.ylabel('Tiempo de ejecución (s)')
    plt.title('Tiempo vs chunk size para cada schedule')
    plt.legend()
    plt.grid(True)
    pdf.savefig()
    plt.close()

def main():
    # Cargar datos de escalabilidad
    try:
        threads, meanT, stdT, speedup, sigmaS, efficiency, sigmaE, f, amdahl_pred = load_scalability('benchmark_results.dat')
    except Exception as e:
        print(f'No se pudo cargar benchmark_results.dat: {e}')
        return
    # Cargar datos de schedule
    try:
        sched_types, chunk_sizes, meanTs, stdTs = load_schedule('schedule_results.dat')
    except Exception as e:
        print(f'No se pudo cargar schedule_results.dat: {e}')
        # Continuar con sólo escalabilidad
        sched_types = None

    # Crear PDF con todas las figuras
    with PdfPages('analysis_plots.pdf') as pdf:
        plot_speedup(threads, speedup, sigmaS, amdahl_pred, pdf)
        plot_efficiency(threads, efficiency, sigmaE, pdf)
        if sched_types is not None:
            plot_schedule_results(sched_types, chunk_sizes, meanTs, pdf)
    print('Se ha generado el archivo analysis_plots.pdf con las gráficas.')

if __name__ == '__main__':
    main()