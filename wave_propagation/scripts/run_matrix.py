
import csv
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence


def exe_path() -> str:
    """Try to autodetect the wave_propagation executable."""

    candidates = [
        "wave_propagation",
        "wave_propagation.exe",
        "./wave_propagation",
        "./wave_propagation.exe",
    ]
    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.exists() and candidate_path.is_file():
            return str(candidate_path)

    # Fallback: assume the executable lives next to the script
    return "./wave_propagation"


def parse_energy(path: Path) -> tuple:
    """Parse a generated energy trace and return (E0, Eend, lines_count)."""

    E0, Eend = None, None
    lines = 0
    if not path.exists():
        return (None, None, 0)

    with path.open("r", encoding="utf-8", errors="ignore") as handler:
        for line in handler:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    _ = int(float(parts[0]))
                    energy_value = float(parts[1])
                except ValueError:
                    continue

                if E0 is None:
                    E0 = energy_value
                Eend = energy_value
                lines += 1

    return (E0, Eend, lines)


@dataclass(frozen=True)
class Scenario:
    label: str
    network: str
    dims: Dict[str, int]
    schedule: str
    chunks: Sequence[object]
    threads: Sequence[int]
    steps: int = 1000
    extra_args: Dict[str, float] = field(default_factory=dict)


def build_command(exe: str, scenario: Scenario, chunk, threads: int) -> List[str]:
    cmd = [
        exe,
        "--network",
        scenario.network,
        "--steps",
        str(scenario.steps),
        "--schedule",
        scenario.schedule,
        "--chunk",
        str(chunk),
        "--threads",
        str(threads),
    ]

    if scenario.network == "1d":
        cmd.extend(["--N", str(scenario.dims["N"])])
    else:
        cmd.extend([
            "--Lx",
            str(scenario.dims["Lx"]),
            "--Ly",
            str(scenario.dims["Ly"]),
        ])

    for key, value in scenario.extra_args.items():
        cmd.extend([f"--{key}", str(value)])

    return cmd


def run_scenario(exe: str, scenario: Scenario, results_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []

    for chunk in scenario.chunks:
        for threads in scenario.threads:
            cmd = build_command(exe, scenario, chunk, threads)

            t0 = time.perf_counter()
            completed = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            elapsed = time.perf_counter() - t0

            src_energy = results_dir / "energy_trace.dat"
            E0, Eend, lines = parse_energy(src_energy)
            energy_out = (
                results_dir
                / f"energy_{scenario.label}_chunk{chunk}_t{threads}.dat"
            )
            if src_energy.exists():
                try:
                    src_energy.rename(energy_out)
                except OSError:
                    import shutil

                    shutil.copyfile(src_energy, energy_out)
                    os.remove(src_energy)

            row = {
                "label": scenario.label,
                "network": scenario.network,
                **scenario.dims,
                "steps": scenario.steps,
                "schedule": scenario.schedule,
                "chunk": str(chunk),
                "threads": threads,
                "time_sec": round(elapsed, 6),
                "E0": E0 if E0 is not None else "",
                "Eend": Eend if Eend is not None else "",
                "energy_lines": lines,
                "stdout_last": (
                    completed.stdout.strip().splitlines()[-1]
                    if completed.stdout
                    else ""
                ),
                "stderr_last": (
                    completed.stderr.strip().splitlines()[-1]
                    if completed.stderr
                    else ""
                ),
            }
            rows.append(row)
            print(f"OK: {row}")

    return rows


def main() -> None:
    exe = exe_path()
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "matrix_results.csv"

    scenarios: List[Scenario] = [
        Scenario(
            label="2d_dynamic_best",
            network="2d",
            dims={"Lx": 256, "Ly": 256},
            schedule="dynamic",
            chunks=[512],
            threads=[1, 2, 4, 8],
        ),
        Scenario(
            label="2d_guided_alternative",
            network="2d",
            dims={"Lx": 256, "Ly": 256},
            schedule="guided",
            chunks=[256],
            threads=[1, 2, 4, 8],
        ),
        Scenario(
            label="2d_static_bad",
            network="2d",
            dims={"Lx": 256, "Ly": 256},
            schedule="static",
            chunks=[512],
            threads=[1, 8],
        ),
        Scenario(
            label="2d_dynamic_chunk_sweep",
            network="2d",
            dims={"Lx": 256, "Ly": 256},
            schedule="dynamic",
            chunks=[128, 256, 512, "auto"],
            threads=[8],
        ),
        Scenario(
            label="1d_small_bad",
            network="1d",
            dims={"N": 20000},
            schedule="static",
            chunks=[512],
            threads=[1, 2, 4],
        ),
        Scenario(
            label="1d_large_better",
            network="1d",
            dims={"N": 200000},
            schedule="dynamic",
            chunks=[256],
            threads=[1, 2, 4],
        ),
    ]

    rows: List[Dict[str, object]] = []
    for scenario in scenarios:
        rows.extend(run_scenario(exe, scenario, results_dir))

    cols = [
        "label",
        "network",
        "N",
        "Lx",
        "Ly",
        "steps",
        "schedule",
        "chunk",
        "threads",
        "time_sec",
        "E0",
        "Eend",
        "energy_lines",
        "stdout_last",
        "stderr_last",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handler:
        writer = csv.DictWriter(handler, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            row.setdefault("N", "")
            row.setdefault("Lx", "")
            row.setdefault("Ly", "")
            writer.writerow(row)

    print(f"\nResultados guardados en {csv_path}")
    print("Energías únicas en results/energy_*.dat")


if __name__ == "__main__":
    main()
