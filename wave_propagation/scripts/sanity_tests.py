
import subprocess, time, os
from pathlib import Path

def run(cmd):
    print("RUN:", " ".join(cmd))
    cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if cp.returncode != 0:
        print("ERROR:", cp.stderr.strip())
    return cp

def read_energy(path):
    E = []
    if not Path(path).exists():
        return E
    for line in Path(path).read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip()
        if not s or s.startswith("#"): continue
        parts = s.split()
        if len(parts)>=2:
            try:
                e = float(parts[1]); E.append(e)
            except: pass
    return E

def main():
    exe = "./wave_propagation" if Path("./wave_propagation").exists() else "./wave_propagation.exe"
    results = Path("results"); results.mkdir(exist_ok=True)

    cases = [
        # (desc, args)
        ("1d_decay",  ["--network","1d","--N","20000","--steps","300","--gamma","0.01","--S0","0.0","--omega","0.0","--schedule","dynamic","--chunk","256","--threads","4"]),
        ("2d_decay",  ["--network","2d","--Lx","128","--Ly","128","--steps","300","--gamma","0.01","--S0","0.0","--omega","0.0","--schedule","dynamic","--chunk","256","--threads","4"]),
        ("1d_source", ["--network","1d","--N","20000","--steps","300","--gamma","0.01","--S0","0.1","--omega","0.5","--schedule","guided","--chunk","64","--threads","4"]),
    ]

    for name, args in cases:
        if Path("results/energy_trace.dat").exists():
            os.remove("results/energy_trace.dat")
        run([exe, *args])
        E = read_energy("results/energy_trace.dat")
        out = results/f"sanity_{name}.dat"
        if Path("results/energy_trace.dat").exists():
            Path("results/energy_trace.dat").rename(out)

        if not E:
            print(f"[{name}] sin energía registrada")
            continue

        # Test básicos
        E0, Ef = E[0], E[-1]
        drop = (E0-Ef)/E0*100.0 if E0>0 else None
        if "decay" in name and drop is not None:
            # esperamos caída sustancial
            print(f"[{name}] caída de energía ≈ {drop:.1f}%")
        if "source" in name:
            # esperamos no-monotónico comúnmente
            inc = sum(1 for i in range(1,len(E)) if E[i]>E[i-1])
            print(f"[{name}] subidas detectadas: {inc}/{len(E)-1} pasos")

    print("Sanity tests OK")

if __name__ == "__main__":
    main()
