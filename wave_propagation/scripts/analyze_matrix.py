#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Analiza results/matrix_results.csv.
- Si hay pandas, lo usa.
- Si no, usa csv/matplotlib puros.
Genera:
  results/speedup_<network>_<schedule>.png
  results/time_vs_chunk_dynamic_2d.png
  results/time_vs_chunk_dynamic_1d.png
"""

import os, csv
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "matrix_results.csv")

# -------- Helpers comunes --------
def chunk_sort_key(x):
    try:
        return int(x)
    except:
        return 10**9  # "auto" al final

def ensure_csv_exists():
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"No existe {CSV_PATH}. Corre primero: make matrix")

# ================== Ruta A: con pandas ==================
def run_with_pandas():
    import pandas as pd

    df = pd.read_csv(CSV_PATH)
    df["time_sec"] = df["time_sec"].astype(float)
    df["chunk"] = df["chunk"].astype(str)

    # Mejor chunk por (net,sched,p)
    idx = df.groupby(["network","schedule","threads"])["time_sec"].idxmin()
    best = df.loc[idx].reset_index(drop=True)

    # Gráficos de speedup por (net,sched)
    for net in sorted(best["network"].unique()):
        dnet = best[best["network"] == net]
        for sch in sorted(dnet["schedule"].unique()):
            d = dnet[dnet["schedule"] == sch].copy()
            d = d.sort_values("threads")
            d1 = d[d["threads"] == 1]
            if d1.empty:
                continue
            t1 = float(d1["time_sec"].min())
            d["speedup"] = t1 / d["time_sec"]
            plt.figure()
            plt.plot(d["threads"], d["speedup"], marker="o")
            plt.xlabel("Hilos (p)"); plt.ylabel("Speedup")
            plt.title(f"Speedup (best chunk) — {net.upper()} / {sch}")
            plt.grid(True, alpha=0.3)
            out = os.path.join(RESULTS_DIR, f"speedup_{net}_{sch}.png")
            plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
            print(f"[plot] {out}")

    # Tiempo vs chunk para dynamic (p=4 si existe)
    for net in ["2d", "1d"]:
        d = df[(df["network"]==net) & (df["schedule"]=="dynamic")].copy()
        if d.empty:
            continue
        p_pref = 4 if 4 in d["threads"].unique() else int(d["threads"].mode().iloc[0])
        d = d[d["threads"]==p_pref].copy()
        d["chunk_sort"] = d["chunk"].map(chunk_sort_key)
        d = d.sort_values("chunk_sort")
        plt.figure()
        plt.plot(d["chunk"], d["time_sec"], marker="o")
        plt.xlabel("Chunk (dynamic)"); plt.ylabel("Tiempo [s]")
        plt.title(f"Tiempo vs Chunk — {net.upper()} (p={p_pref})")
        plt.grid(True, alpha=0.3)
        out = os.path.join(RESULTS_DIR, f"time_vs_chunk_dynamic_{net}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[plot] {out}")

# ================== Ruta B: sin pandas ==================
def run_without_pandas():
    # Cargar CSV a lista de dicts tipados
    rows = []
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "network":  r["network"],
                "size":     r["size"],
                "schedule": r["schedule"],
                "chunk":    str(r["chunk"]),
                "threads":  int(r["threads"]),
                "steps":    int(r["steps"]),
                "time_sec": float(r["time_sec"]),
            })

    # Índice mínimo por (net,sched,p)
    def best_rows():
        out = []
        # agrupar
        key = lambda r: (r["network"], r["schedule"], r["threads"])
        rows_sorted = sorted(rows, key=lambda r: (*key(r), r["time_sec"]))
        lastk = None
        for r in rows_sorted:
            k = key(r)
            if k != lastk:
                out.append(r)  # el primero es el de menor tiempo (ya ordenado)
                lastk = k
        return out

    best = best_rows()

    # Speedup por (net,sched)
    nets = sorted({r["network"] for r in best})
    for net in nets:
        scheds = sorted({r["schedule"] for r in best if r["network"]==net})
        for sch in scheds:
            d = [r for r in best if r["network"]==net and r["schedule"]==sch]
            if not d:
                continue
            d = sorted(d, key=lambda r: r["threads"])
            p1 = [r for r in d if r["threads"]==1]
            if not p1:
                continue
            t1 = min(r["time_sec"] for r in p1)
            threads = [r["threads"] for r in d]
            speedup = [t1/r["time_sec"] for r in d]
            plt.figure()
            plt.plot(threads, speedup, marker="o")
            plt.xlabel("Hilos (p)"); plt.ylabel("Speedup")
            plt.title(f"Speedup (best chunk) — {net.upper()} / {sch}")
            plt.grid(True, alpha=0.3)
            out = os.path.join(RESULTS_DIR, f"speedup_{net}_{sch}.png")
            plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
            print(f"[plot] {out}")

    # Tiempo vs chunk para dynamic (p=4 si existe)
    for net in ["2d","1d"]:
        d = [r for r in rows if r["network"]==net and r["schedule"]=="dynamic"]
        if not d:
            continue
        pvals = sorted({r["threads"] for r in d})
        p_pref = 4 if 4 in pvals else pvals[0]
        d = [r for r in d if r["threads"]==p_pref]
        d = sorted(d, key=lambda r: chunk_sort_key(r["chunk"]))
        chunks = [r["chunk"] for r in d]
        times  = [r["time_sec"] for r in d]
        plt.figure()
        plt.plot(chunks, times, marker="o")
        plt.xlabel("Chunk (dynamic)"); plt.ylabel("Tiempo [s]")
        plt.title(f"Tiempo vs Chunk — {net.upper()} (p={p_pref})")
        plt.grid(True, alpha=0.3)
        out = os.path.join(RESULTS_DIR, f"time_vs_chunk_dynamic_{net}.png")
        plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
        print(f"[plot] {out}")

def main():
    ensure_csv_exists()
    try:
        import pandas  # noqa
        run_with_pandas()
    except Exception as e:
        print(f"[analyze_matrix] pandas no disponible o falló ({e}). Usando modo sin pandas.")
        run_without_pandas()
    print("[analyze_matrix] Hecho.")

if __name__ == "__main__":
    main()
