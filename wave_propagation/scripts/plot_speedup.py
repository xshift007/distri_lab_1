import numpy as np, matplotlib.pyplot as plt
def load_scaling(path='results/scaling.dat'):
    try:
        d = np.loadtxt(path)
        if d.ndim == 1: d = d.reshape(1, -1)
        return d
    except Exception as e:
        print("No se pudo leer", path, "->", e); return None
def plot_speedup(d):
    p, S = d[:,0], d[:,3]
    plt.figure(); plt.plot(p, S, marker='o', label='Medido')
    plt.plot(p, p, linestyle='--', label='Ideal')
    plt.xlabel('Threads (p)'); plt.ylabel('Speedup S_p')
    plt.title('Speedup vs Threads'); plt.grid(True); plt.legend()
    plt.savefig('results/speedup_vs_threads.png', dpi=200, bbox_inches='tight')
def plot_efficiency(d):
    p, E = d[:,0], d[:,5]
    plt.figure(); plt.plot(p, E, marker='s', label='Eficiencia')
    plt.xlabel('Threads (p)'); plt.ylabel('Eficiencia E_p')
    plt.title('Eficiencia vs Threads'); plt.grid(True); plt.legend()
    plt.savefig('results/efficiency_vs_threads.png', dpi=200, bbox_inches='tight')
def load_scaling_io(path='results/scaling_io.dat'):
    try:
        d = np.loadtxt(path)
        if d.ndim == 1: d = d.reshape(1, -1)
        return d
    except Exception as e:
        return None

def plot_speedup_io(d):
    p, S = d[:,0], d[:,3]
    plt.figure(); plt.plot(p, S, marker='o', label='Medido (con I/O)')
    plt.plot(p, p, linestyle='--', label='Ideal')
    plt.xlabel('Threads (p)'); plt.ylabel('Speedup S_p')
    plt.title('Speedup vs Threads (con I/O)'); plt.grid(True); plt.legend()
    plt.savefig('results/speedup_vs_threads_io.png', dpi=200, bbox_inches='tight')

def main():
    d = load_scaling()
    if d is None: print("Genera results/scaling.dat con `make benchmark`."); return
    plot_speedup(d); plot_efficiency(d);
    d_io = load_scaling_io();
    if d_io is not None:
        plot_speedup_io(d_io)
    print("Listo: results/*.png")
if __name__=='__main__': main()
