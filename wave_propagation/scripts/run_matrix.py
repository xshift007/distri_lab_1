
import subprocess, sys, time, csv, os
from pathlib import Path

def exe_path():
    # Try to autodetect exe name cross-platform
    cand = ["wave_propagation", "wave_propagation.exe", "./wave_propagation", "./wave_propagation.exe"]
    for c in cand:
        p = Path(c)
        if p.exists() and p.is_file():
            return str(p)
    # Fallback: assume in current dir
    return "./wave_propagation"

def parse_energy(path):
    # returns (E0, Eend, lines_count)
    E0, Eend = None, None
    lines = 0
    if not Path(path).exists():
        return (None, None, 0)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"): continue
            parts = s.split()
            if len(parts)>=2:
                try:
                    step = int(float(parts[0]))
                    e = float(parts[1])
                    if E0 is None: E0 = e
                    Eend = e
                    lines += 1
                except:
                    pass
    return (E0, Eend, lines)

def main():
    exe = exe_path()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir/"matrix_results.csv"

    # Test matrix (toca si quieres menos/más casos)
    networks = [
        ("1d", {"N": 20000}),
        ("2d", {"Lx": 256, "Ly": 256}),
    ]
    steps = 400
    dt = 0.01
    D = 0.1
    gamma = 0.01
    sources = [(0.0, 0.0), (0.1, 0.5)]  # (S0, omega)

    schedules = ["static", "dynamic", "guided"]
    chunks = ["auto", 64, 128, 256, 512]
    threads_list = [1, 2, 4, 8]

    rows = []
    for net_name, dims in networks:
        for (S0, omega) in sources:
            for sched in schedules:
                for chunk in chunks:
                    for thr in threads_list:
                        # Build command
                        cmd = [exe, "--network", net_name,
                               "--steps", str(steps),
                               "--dt", str(dt),
                               "--D", str(D),
                               "--gamma", str(gamma),
                               "--S0", str(S0),
                               "--omega", str(omega),
                               "--schedule", sched,
                               "--chunk", str(chunk) if chunk!="auto" else "auto",
                               "--threads", str(thr)]
                        if net_name == "1d":
                            cmd += ["--N", str(dims["N"])]
                        else:
                            cmd += ["--Lx", str(dims["Lx"]), "--Ly", str(dims["Ly"])]
                        # Run
                        t0 = time.perf_counter()
                        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        t1 = time.perf_counter()
                        elapsed = t1 - t0

                        # Move/rename energy trace to keep it
                        src_energy = results_dir/"energy_trace.dat"
                        E0, Eend, lines = parse_energy(src_energy) if src_energy.exists() else (None, None, 0)
                        out_energy = results_dir/f"energy_{net_name}_S{S0}_w{omega}_{sched}_chunk{chunk}_t{thr}.dat"
                        if src_energy.exists():
                            try:
                                src_energy.rename(out_energy)
                            except Exception:
                                # If rename fails (Windows locks), copy then remove
                                import shutil
                                shutil.copyfile(src_energy, out_energy)
                                os.remove(src_energy)

                        row = {
                            "network": net_name,
                            **{k:v for k,v in dims.items()},
                            "steps": steps,
                            "S0": S0, "omega": omega,
                            "schedule": sched,
                            "chunk": str(chunk),
                            "threads": thr,
                            "time_sec": round(elapsed, 6),
                            "E0": E0 if E0 is not None else "",
                            "Eend": Eend if Eend is not None else "",
                            "energy_lines": lines,
                            "stdout_last": (cp.stdout.strip().splitlines()[-1] if cp.stdout else ""),
                            "stderr_last": (cp.stderr.strip().splitlines()[-1] if cp.stderr else "")
                        }
                        rows.append(row)
                        print(f"OK: {row}")

    # Write CSV
    cols = ["network","N","Lx","Ly","steps","S0","omega","schedule","chunk","threads","time_sec","E0","Eend","energy_lines","stdout_last","stderr_last"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            # fill missing keys (depending on 1d/2d)
            if "N" not in r: r["N"]=""
            if "Lx" not in r: r["Lx"]=""
            if "Ly" not in r: r["Ly"]=""
            w.writerow(r)

    print(f"\nResultados guardados en {csv_path}")
    print("Energías únicas en results/energy_*.dat")

if __name__ == "__main__":
    main()
