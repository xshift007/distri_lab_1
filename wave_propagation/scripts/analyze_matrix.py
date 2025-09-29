
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    results_dir = Path("results")
    csv_path = results_dir/"matrix_results.csv"
    if not csv_path.exists():
        print("No existe results/matrix_results.csv. Ejecuta primero scripts/run_matrix.py")
        return
    df = pd.read_csv(csv_path)

    # Filtrar corridas válidas (con tiempo positivo)
    df = df[df["time_sec"]>0].copy()

    # ---- Speedup por schedule (mejor chunk por schedule) ----
    figs = []
    for net in df["network"].unique():
        sub = df[df["network"]==net].copy()
        # Elegir mejor chunk por schedule según tiempo a threads=max (aprox)
        best = []
        max_t = sub["threads"].max()
        for sched in ["static","dynamic","guided"]:
            s2 = sub[(sub["schedule"]==sched) & (sub["threads"]==max_t)]
            if s2.empty: continue
            # mejor chunk = mínimo tiempo
            best_chunk = s2.loc[s2["time_sec"].idxmin()]["chunk"]
            s_all = sub[(sub["schedule"]==sched) & (sub["chunk"]==best_chunk)]
            if s_all.empty: continue
            # Speedup con base threads=1
            base = s_all[s_all["threads"]==1]
            if base.empty: continue
            T1 = float(base.iloc[0]["time_sec"])
            s_all = s_all.sort_values("threads")
            p = s_all["threads"].to_numpy()
            Sp = T1 / s_all["time_sec"].to_numpy()
            plt.figure()
            plt.plot(p, Sp, marker='o')
            plt.xlabel("Threads p"); plt.ylabel("Speedup S_p")
            plt.title(f"Speedup ({net}) — {sched} (chunk={best_chunk})")
            plt.grid(True)
            out = results_dir/f"speedup_{net}_{sched}_best.png"
            plt.savefig(out, dpi=200, bbox_inches="tight")
            plt.close()
            figs.append(out)

    # ---- Tiempo vs chunk (dynamic), por red a threads=max ----
    for net in df["network"].unique():
        sub = df[(df["network"]==net) & (df["schedule"]=="dynamic")].copy()
        if sub.empty: continue
        max_t = sub["threads"].max()
        s2 = sub[sub["threads"]==max_t].copy()
        if s2.empty: continue
        order = []
        def parse_chunk(c):
            try:
                return int(c)
            except:
                return 10**9 if str(c)=="auto" else 10**8
        s2["chunk_val"] = [parse_chunk(x) for x in s2["chunk"]]
        s2 = s2.sort_values("chunk_val")
        plt.figure()
        plt.plot(s2["chunk"], s2["time_sec"], marker='o')
        plt.xlabel("Chunk (dynamic)"); plt.ylabel("Tiempo [s]")
        plt.title(f"Tiempo vs chunk — {net} (threads={max_t})")
        plt.grid(True)
        out = results_dir/f"time_vs_chunk_dynamic_{net}.png"
        plt.savefig(out, dpi=200, bbox_inches="tight")
        plt.close()
        figs.append(out)

    print("Listo:")
    for f in figs: print(" -", f)

if __name__ == "__main__":
    main()
