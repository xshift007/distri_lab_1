import numpy as np, matplotlib.pyplot as plt
def safe_load(path):
    try:
        d = np.loadtxt(path, dtype=str)
        if d.ndim == 1: d = d.reshape(1, -1)
        return d
    except Exception as e:
        print("No se pudo leer", path, "->", e); return None
def plot_schedule_chunk(path='results/schedule_vs_chunk.dat'):
    d = safe_load(path); if d is None: return
    schedules = sorted(set(d[:,0])); 
    for sched in schedules:
        mask = d[:,0]==sched; ch = d[mask][:,1].astype(int); mT = d[mask][:,2].astype(float); sT = d[mask][:,3].astype(float)
        idx = np.argsort(ch); plt.figure(); plt.errorbar(ch[idx], mT[idx], yerr=sT[idx], marker='o')
        plt.xlabel('Chunk size'); plt.ylabel('Tiempo [s]'); plt.title(f'Tiempo vs Chunk ({sched})'); plt.grid(True)
        plt.savefig(f'results/time_vs_chunk_{sched}.png', dpi=200, bbox_inches='tight')
def plot_sync_methods(path='results/sync_methods.dat'):
    d = safe_load(path); if d is None: return
    methods = d[:,0]; mT = d[:,1].astype(float); sT = d[:,2].astype(float)
    x = np.arange(len(methods)); plt.figure(); plt.bar(x, mT, yerr=sT); plt.xticks(x, methods)
    plt.ylabel('Tiempo [s]'); plt.title('Sincronización: tiempo por método'); plt.grid(True, axis='y')
    plt.savefig('results/sync_methods.png', dpi=200, bbox_inches='tight')
def plot_tasks_vs_for(path='results/tasks_vs_for.dat'):
    d = safe_load(path); if d is None: return
    modes = d[:,0]; mT = d[:,1].astype(float); sT = d[:,2].astype(float)
    x = np.arange(len(modes)); plt.figure(); plt.bar(x, mT, yerr=sT); plt.xticks(x, modes)
    plt.ylabel('Tiempo [s]'); plt.title('Tasks vs parallel for'); plt.grid(True, axis='y')
    plt.savefig('results/tasks_vs_for.png', dpi=200, bbox_inches='tight')
def main(): plot_schedule_chunk(); plot_sync_methods(); plot_tasks_vs_for(); print("Listo: results/*.png")
if __name__=='__main__': main()
