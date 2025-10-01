#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Corre una matriz de configuraciones y guarda resultados en CSV con
warm-up y repeticiones. Ajusta chunk por número de threads.
Compatibilidad Windows/MSYS2: usa ruta absoluta al binario y shell=False.
"""

import os, csv, time, subprocess
from statistics import median
from pathlib import Path
import sys

RESULTS_DIR = "results"
CSV_PATH = os.path.join(RESULTS_DIR, "matrix_results.csv")

# ========= Detección de binario (Windows vs POSIX) =========
ROOT = Path(__file__).resolve().parents[1]
if os.name == "nt" or sys.platform.startswith("win"):
    BIN_PATH = ROOT / "wave_propagation.exe"
else:
    BIN_PATH = ROOT / "wave_propagation"

BIN = str(BIN_PATH)

# =================== Config base (según tu idea) ===================
networks = [
    ("2d", {"Lx": 256, "Ly": 256}),   # caso principal (óptimo)
    ("1d", {"N": 20000}),             # 1D chico -> mal escalado (intencional)
    ("1d", {"N": 200000})             # 1D grande -> mejora visible
]

# Aumenta pasos en 2D para amortizar overhead (clase 2)
steps_2d = 2000
steps_1d = 1000

# Sin fuente (bench limpio)
noise_mode = "off"   # "off|global|pernode|single"
S0 = 0.0
omega = 0.0

schedules = ["static", "dynamic", "guided"]

# Barrido de chunk más rico; el script elegirá el mejor por p
chunks_by_threads = {
    1:   [128, 256, 512, "auto"],
    2:   [128, 256, 512, "auto"],
    4:   [128, 256, 512, 1024, "auto"],
    8:   [256, 512, 1024, 2048, "auto"],
}

threads_list = [1, 2, 4, 8]

# Warm-up y repeticiones
WARMUP  = 1
REPEATS = 5
AGGREGATOR = "median"  # "median" o "min"

# ======================== Utilitarios ==============================

def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)

def steps_for(network_kind: str) -> int:
    return steps_2d if network_kind == "2d" else steps_1d

def build_cmd(kind: str, dims: dict, steps: int, schedule: str, chunk, threads: int):
    """
    Devuelve lista de argumentos (shell=False). Compatible con Windows.
    """
    args = [BIN, "--network", kind]
    if kind == "2d":
        args += ["--Lx", str(dims["Lx"]), "--Ly", str(dims["Ly"])]
    else:
        args += ["--N", str(dims["N"])]

    args += [
        "--steps", str(steps),
        "--schedule", schedule,
        "--threads", str(threads),
        "--noise", noise_mode,
        "--S0", str(S0),
        "--omega", str(omega),
    ]

    # chunk puede ser int o "auto"
    if isinstance(chunk, int):
        args += ["--chunk", str(chunk)]
    else:
        args += ["--chunk", "auto"]

    # NUNCA hacer I/O de frames en bench
    return args

def run_once(args_list) -> float:
    """
    Ejecuta el binario con shell=False. Retorna tiempo en segundos.
    """
    if not os.path.isfile(BIN):
        raise FileNotFoundError(
            f"No encuentro el binario en: {BIN}\n"
            "Compila primero con: make\n"
            "Si estás en Windows, verifica que exista 'wave_propagation.exe' en la carpeta del proyecto."
        )
    t0 = time.perf_counter()
    res = subprocess.run(args_list, shell=False, capture_output=True, text=True)
    t1 = time.perf_counter()

    if res.returncode != 0:
        print("ERROR al ejecutar:", " ".join(args_list))
        print("STDOUT:\n", res.stdout)
        print("STDERR:\n", res.stderr)
        raise RuntimeError("falló ejecución")

    return t1 - t0

def aggregate(times, how="median"):
    if how == "min":
        return min(times)
    return median(times)

def write_header_if_needed(csv_path: str):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "network","size","schedule","chunk","threads","steps","time_sec"
            ])

def size_str(kind: str, dims: dict) -> str:
    return f"{dims['Lx']}x{dims['Ly']}" if kind == "2d" else str(dims["N"])

def append_row(csv_path: str, row):
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow(row)

# ======================== Ejecución matriz =========================

def main():
    ensure_dirs()
    write_header_if_needed(CSV_PATH)

    print("[run_matrix] Comenzando matriz...")
    print(f"[run_matrix] Binario: {BIN}")

    for kind, dims in networks:
        st = steps_for(kind)
        for schedule in schedules:
            for p in threads_list:
                chunks = chunks_by_threads.get(p, [128,256,512,"auto"])
                for chunk in chunks:
                    args = build_cmd(kind, dims, st, schedule, chunk, p)

                    # warm-up
                    for _ in range(WARMUP):
                        _ = run_once(args)

                    times = []
                    for _ in range(REPEATS):
                        times.append(run_once(args))

                    t_final = aggregate(times, AGGREGATOR)

                    append_row(CSV_PATH, [
                        kind, size_str(kind, dims), schedule, str(chunk), p, st, f"{t_final:.6f}"
                    ])

                    print(f"[ok] {kind:>2} size={size_str(kind,dims):>8} "
                          f"sch={schedule:<7} chunk={str(chunk):>5} p={p} "
                          f"time={t_final:.4f}s")

    print(f"[run_matrix] Listo -> {CSV_PATH}")

if __name__ == "__main__":
    main()
