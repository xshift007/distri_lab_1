import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)

def main(path='results/time_vs_chunk_dynamic.dat'):
    try:
        d = np.loadtxt(path, dtype=float, comments="#")
    except Exception as e:
        print("No se pudo leer", path, "->", e)
        return

    if d.ndim == 1:
        d = d.reshape(1,-1)

    chunk = d[:,0].astype(int)
    mT = d[:,1].astype(float)
    sT = d[:,2].astype(float)

    idx = np.argsort(chunk)
    plt.figure()
    plt.errorbar(chunk[idx], mT[idx], yerr=sT[idx], marker='o')
    plt.xlabel('Chunk size'); plt.ylabel('Tiempo [s]')
    plt.title('Tiempo vs Chunk (schedule=dynamic)'); plt.grid(True)
    plt.savefig('results/time_vs_chunk_dynamic.png', dpi=200, bbox_inches='tight')
    print("Listo: results/time_vs_chunk_dynamic.png")

if __name__ == "__main__":
    main()
