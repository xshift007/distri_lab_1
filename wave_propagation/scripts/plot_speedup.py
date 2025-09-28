import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

def load_scaling(path='results/scaling.dat'):
    try:
        d = np.loadtxt(path, dtype=float, comments="#")
        if d.ndim == 1:
            d = d.reshape(1, -1)
        return d
    except Exception as e:
        print("No se pudo leer", path, "->", e)
        return None

def main():
    d = load_scaling()
    if d is None or d.shape[1] < 3:
        print("Archivo inválido o vacío.")
        return
    p = d[:,0].astype(int)
    T = d[:,1].astype(float)
    sT= d[:,2].astype(float)

    T1 = T[p==1][0] if np.any(p==1) else T[0]
    S  = T1 / T
    E  = S / p

    # Speedup
    plt.figure()
    plt.errorbar(p, S, yerr=np.zeros_like(S), marker='o')
    plt.xlabel('Threads p'); plt.ylabel('Speedup S_p')
    plt.title('Speedup vs Threads'); plt.grid(True)
    plt.savefig('results/speedup.png', dpi=200, bbox_inches='tight')

    # Eficiencia
    plt.figure()
    plt.errorbar(p, E, yerr=np.zeros_like(E), marker='o')
    plt.xlabel('Threads p'); plt.ylabel('Eficiencia E_p')
    plt.title('Eficiencia vs Threads'); plt.grid(True)
    plt.savefig('results/efficiency.png', dpi=200, bbox_inches='tight')

    print("Listo: results/speedup.png, results/efficiency.png")

if __name__ == "__main__":
    main()
