#!/usr/bin/env python3
"""Plot measured speedup against Amdahl's law prediction."""

from __future__ import annotations

import argparse
import pathlib
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np


def estimate_serial_fraction(threads: np.ndarray, speedup: np.ndarray) -> float:
    """Estimate serial fraction f from measured speedups via least squares."""
    mask = threads > 1
    if not np.any(mask):
        raise ValueError("need at least one measurement with p > 1 to estimate f")
    numerator = 1.0 / speedup[mask] - 1.0 / threads[mask]
    denominator = 1.0 - 1.0 / threads[mask]
    f_vals = numerator / denominator
    f = float(np.clip(np.mean(f_vals), 0.0, 1.0))
    return f


def load_scaling(path: pathlib.Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = np.expand_dims(data, axis=0)
    threads = data[:, 0]
    speedup = data[:, 3]
    speedup_err = data[:, 4] if data.shape[1] > 4 else np.zeros_like(speedup)
    return threads, speedup, speedup_err


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Amdahl's law prediction.")
    parser.add_argument(
        "--input",
        default="results/scaling.dat",
        help="Input scaling data (columns: threads, time, speedup, ...)",
    )
    parser.add_argument(
        "--output",
        default="results/amdahl.png",
        help="Path to store the generated figure.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = pathlib.Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"input file '{input_path}' not found")

    threads, speedup, speedup_err = load_scaling(input_path)
    f = estimate_serial_fraction(threads, speedup)
    predicted = 1.0 / (f + (1.0 - f) / threads)

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.errorbar(
        threads,
        speedup,
        yerr=speedup_err,
        fmt="o",
        capsize=4,
        label="Medido",
    )
    plt.plot(threads, predicted, "-", label=f"Amdahl (f={f:.3f})")
    plt.xlabel("Hilos (p)")
    plt.ylabel("Speedup")
    plt.title("Speedup vs. predicción de Amdahl")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Amdahl f ≈ {f:.4f} → figura: {output_path}")


if __name__ == "__main__":
    main()
