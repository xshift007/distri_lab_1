from __future__ import annotations

import argparse
import glob
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def load_frame(fp: str) -> np.ndarray:
    if fp.endswith(".csv"):
        return np.loadtxt(fp, delimiter=",")
    data = np.loadtxt(fp)
    return data.reshape(-1, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Genera un video a partir de los frames volcados por wave_propagation"
    )
    parser.add_argument("frames_dir", help="Directorio que contiene archivos amp_t*.dat/csv")
    parser.add_argument(
        "--outdir",
        default="videos",
        help="Directorio de salida donde se guardará el video (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20,
        help="Cuadros por segundo del video generado (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--format",
        default="mp4",
        choices=["mp4", "gif"],
        help="Formato del video de salida (por defecto: %(default)s)",
    )
    parser.add_argument(
        "--basename",
        default=None,
        help="Nombre base para el archivo de video sin extensión (por defecto se infiere)",
    )
    return parser


def collect_frames(frames_dir: Path) -> tuple[list[np.ndarray], np.ndarray]:
    pattern_csv = frames_dir / "amp_t*.csv"
    pattern_dat = frames_dir / "amp_t*.dat"
    files = sorted(glob.glob(str(pattern_csv)) + glob.glob(str(pattern_dat)))
    if not files:
        raise FileNotFoundError(f"No se encontraron frames en {frames_dir}")

    vmin = None
    vmax = None
    frames: list[np.ndarray] = []
    for fp in files:
        arr = load_frame(fp)
        current_min = arr.min()
        current_max = arr.max()
        vmin = current_min if vmin is None else min(vmin, current_min)
        vmax = current_max if vmax is None else max(vmax, current_max)
        frames.append(arr)

    images: list[np.ndarray] = []
    for arr in frames:
        fig = plt.figure(figsize=(5, 5))
        plt.axis("off")
        plt.imshow(arr, animated=True, vmin=vmin, vmax=vmax, cmap="viridis")
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)
    return images, frames[0]


def infer_suffix(sample: np.ndarray) -> str:
    if sample.ndim == 2 and sample.shape[1] == 1:
        return "1d"
    if sample.ndim == 2:
        return "2d"
    return "data"


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_video(images: list[np.ndarray], out_path: Path, fps: float) -> Path:
    ensure_directory(out_path.parent)
    if out_path.suffix.lower() == ".gif":
        duration = 1.0 / fps if fps > 0 else 0.05
        imageio.mimsave(out_path, images, duration=duration)
        return out_path

    try:
        with imageio.get_writer(out_path, fps=max(fps, 1)) as writer:
            for frame in images:
                writer.append_data(frame)
        return out_path
    except Exception as exc:  # pragma: no cover - dependencia externa
        fallback = out_path.with_suffix(".gif")
        duration = 1.0 / fps if fps > 0 else 0.05
        imageio.mimsave(fallback, images, duration=duration)
        print(
            f"No se pudo escribir {out_path.name} ({exc}). "
            f"Se generó {fallback.name} como alternativa."
        )
        return fallback


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        raise SystemExit(f"La carpeta {frames_dir} no existe")

    try:
        images, sample = collect_frames(frames_dir)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    suffix = infer_suffix(sample)
    base_name = args.basename or f"{frames_dir.name or 'frames'}_{suffix}"
    out_path = Path(args.outdir) / f"{base_name}.{args.format}"
    written = save_video(images, out_path, args.fps)
    print(f"Video guardado en {written}")


if __name__ == "__main__":
    main()
