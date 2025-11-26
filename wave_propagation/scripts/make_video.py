import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import imageio.v2 as imageio
import matplotlib

# Backend no interactivo
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

# Intentar importar tqdm
try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable

# Estilo global
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "grid.alpha": 0.3,
})


class FrameLoader:
    """Carga robusta de frames desde texto o CSV, devolviendo ``None`` si falla."""

    @staticmethod
    def load(path: Path) -> Optional[np.ndarray]:
        try:
            if path.suffix == ".csv":
                return np.loadtxt(path, delimiter=",")
            return np.loadtxt(path)
        except Exception as exc:  # noqa: BLE001 (queremos registrar la excepción)
            print(f"[Aviso] No se pudo leer {path.name}: {exc}")
            return None

    @staticmethod
    def detect_mode(files: List[Path], probe_count: int = 10) -> str:
        votes_1d, votes_2d = 0, 0
        for fp in files[:probe_count]:
            data = FrameLoader.load(fp)
            if data is None or data.size == 0:
                continue
            if data.ndim == 1 or (data.ndim == 2 and 1 in data.shape):
                votes_1d += 1
            else:
                votes_2d += 1
        return "2d" if votes_2d > votes_1d else "1d"


def _downsample(data: np.ndarray, factor: Optional[int]) -> np.ndarray:
    if not factor or factor <= 1:
        return data
    if data.ndim == 1:
        return data[::factor]
    return data[::factor, ::factor]


def _smooth_2d(data: np.ndarray, method: str) -> np.ndarray:
    """Suaviza la superficie si es posible; fallback a repetición simple."""

    if method == "nearest":
        return data

    try:  # SciPy opcional
        from scipy.ndimage import zoom

        order = {"bilinear": 1, "bicubic": 3}.get(method, 1)
        return zoom(data, zoom=2, order=order)
    except Exception:
        # Fallback: duplicar celdas para reducir aliasing visual
        return np.kron(data, np.ones((2, 2)))


class Renderer:
    def __init__(self, config, z_limits, xy_limits=None):
        self.cfg = config
        self.vmin, self.vmax = z_limits

        # Manejo coherente de 1D vs 2D:
        #  - 1D: xy_limits = (xmin, xmax)
        #  - 2D: xy_limits = ((xmin, xmax), (ymin, ymax))
        self.xlim, self.ylim = None, None
        if xy_limits is not None:
            if isinstance(xy_limits[0], (tuple, list, np.ndarray)):
                # Caso 2D
                self.xlim, self.ylim = xy_limits
            else:
                # Caso 1D
                self.xlim = xy_limits

        # Evitar rango casi nulo
        if abs(self.vmax - self.vmin) < 1e-3:
            self.vmax += 0.1

        # Reescala centrando en cero si se pidió (esto se usa en 1D; en 2D
        # recalculamos por frame, pero mantenemos la bandera aquí)
        self.zero_center = getattr(self.cfg, "zero_center", False)

    def _fig_to_rgb(self, fig):
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        try:
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            return buf.reshape(h, w, 3)
        except AttributeError:
            # Compatibilidad con Matplotlib >= 3.9 (buffer_rgba)
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            return buf.reshape(h, w, 4)[..., :3]


class Renderer1D(Renderer):
    def render(self, data, step, total):
        if data is None or data.size == 0:
            return None

        y = _downsample(data.ravel(), self.cfg.downsample)
        if y.size == 0:
            return None

        x = np.arange(len(y))
        fig = plt.figure(figsize=(8, 5), dpi=self.cfg.dpi)
        ax = fig.add_subplot(111)

        ax.plot(x, y, linewidth=2, color="#0066cc")
        ax.fill_between(x, y, color="#0066cc", alpha=0.2)
        ax.set_ylim(self.vmin, self.vmax)

        # Zoom centrado en el pico si se especifica
        if self.cfg.zoom1d is not None:
            fraction = max(0.0, min(1.0, float(self.cfg.zoom1d)))
            if fraction == 0:
                ax.set_xlim(x[0], x[-1])
            else:
                window = max(2, int(len(y) * fraction))
                center = int(np.argmax(np.abs(y)))
                half = window // 2
                xmin = max(0, center - half)
                xmax = min(len(y) - 1, center + half)
                ax.set_xlim(xmin, xmax)
        elif self.xlim is not None:
            ax.set_xlim(self.xlim)
        else:
            ax.set_xlim(x[0], x[-1])

        ax.grid(True, linestyle="--")
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_title(f"Simulación 1D — Paso {step}/{total}", fontweight="bold")
        ax.set_ylabel("Amplitud")
        ax.set_xlabel("Nodo")

        if self.cfg.mark_peak:
            idx = int(np.argmax(np.abs(y)))
            ax.scatter(x[idx], y[idx], color="crimson", zorder=5, label="Pico")
            ax.annotate(
                f"{y[idx]:.3f}",
                (x[idx], y[idx]),
                textcoords="offset points",
                xytext=(5, 5),
                color="crimson",
                fontsize=8,
            )
            ax.legend(frameon=False)

        img = self._fig_to_rgb(fig)
        plt.close(fig)
        return img


class Renderer3D(Renderer):
    """
    Visualización para la red 2D como superficie 3D al estilo clásico:
    recorta alrededor del pico y re-escala la altura para que se vea la “montaña”.
    """

    def render(self, data, step, total):
        if data is None or data.size == 0:
            return None

        z = _downsample(np.asarray(data), self.cfg.downsample)
        if z.size == 0:
            return None

        # Aseguramos 2D
        if z.ndim != 2:
            # Intento básico de reshaping si viniera aplanado
            n = int(np.sqrt(z.size))
            z = z.reshape((n, -1))

        # Suavizado opcional
        z = _smooth_2d(z, self.cfg.interp)

        # --- DEBUG: rango del primer frame ---
        if step == 1:
            print(
                f"[DEBUG 2D] Frame {step}: shape={z.shape}, "
                f"min={z.min():.3e}, max={z.max():.3e}, "
                f"nonzero={np.count_nonzero(z)}"
            )

        h, w = z.shape
        abs_z = np.abs(z)
        if not np.any(abs_z > 0):
            # Nada interesante que mostrar
            return None

        # Pico global
        peak_y, peak_x = np.unravel_index(np.argmax(abs_z), z.shape)

        # Ventana alrededor del pico (25% del tamaño o al menos 7x7)
        base_window = min(h, w)
        window = max(7, int(base_window * 0.25))
        half = window // 2

        y0 = max(0, peak_y - half)
        y1 = min(h, peak_y + half + 1)
        x0 = max(0, peak_x - half)
        x1 = min(w, peak_x + half + 1)

        z_crop = z[y0:y1, x0:x1]
        hc, wc = z_crop.shape

        # Grilla 2D para la superficie
        X, Y = np.meshgrid(np.arange(wc), np.arange(hc))

        # Escala de colores basada en la amplitud física (no re-escalada)
        vmin, vmax = float(z_crop.min()), float(z_crop.max())
        if self.zero_center:
            lim = max(abs(vmin), abs(vmax))
            vmin, vmax = -lim, lim
        if abs(vmax - vmin) < 1e-9:
            vmax = vmin + 1e-3

        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(self.cfg.cmap)
        facecolors = cmap(norm(z_crop))

        # Re-escalar altura para que se vea la montaña (target ~30 unidades)
        amp = float(np.max(np.abs(z_crop)))
        if amp < 1e-9:
            scale = 1.0
        else:
            target_height = 30.0
            scale = target_height / amp
        z_plot = z_crop * scale
        height = float(np.max(np.abs(z_plot)))

        fig = plt.figure(figsize=(6, 5), dpi=self.cfg.dpi)
        ax = fig.add_subplot(111, projection="3d")

        surf = ax.plot_surface(
            X,
            Y,
            z_plot,
            facecolors=facecolors,
            linewidth=0.0,
            antialiased=True,
            shade=False,
            rstride=1,
            cstride=1,
        )

        # Límites de altura simétricos
        ax.set_zlim(-height, height)

        ax.set_xlabel("X (width)")
        ax.set_ylabel("Y (height)")
        ax.set_zlabel("Amplitud (re-escalada)")
        ax.set_title(f"Evolución de la onda (Paso {step}/{total})", fontweight="bold")

        # Vista 3D “clásica”
        ax.view_init(elev=30, azim=-45)

        # Intentar proporción de caja agradable
        try:
            ax.set_box_aspect((wc, hc, height if height > 0 else 1.0))
        except Exception:
            pass

        ax.grid(True, alpha=0.3)

        # Marcar el pico dentro de la ventana recortada
        if self.cfg.mark_peak:
            peak_y_c = peak_y - y0
            peak_x_c = peak_x - x0
            if 0 <= peak_y_c < hc and 0 <= peak_x_c < wc:
                ax.scatter(
                    peak_x_c,
                    peak_y_c,
                    z_plot[peak_y_c, peak_x_c],
                    color="red",
                    s=40,
                    label="Pico",
                )
                ax.legend(loc="upper right")

        # Colorbar en escala física (no re-escalada)
        if self.cfg.colorbar:
            mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array([])
            fig.colorbar(
                mappable,
                ax=ax,
                shrink=0.6,
                pad=0.1,
                label="Amplitud",
            )

        plt.tight_layout()
        img = self._fig_to_rgb(fig)
        plt.close(fig)
        return img


def analyze_data(files: Iterable[Path]):
    """
    Escaneo global para obtener min/max y un bounding box aproximado.
    Para 3D usamos principalmente min/max globales; el recorte fino se
    hace por frame en Renderer3D.
    """
    print("[Info] Analizando datos para Auto-Crop y Escala...")
    gmin, gmax = float("inf"), float("-inf")
    min_x, max_x = float("inf"), float("-inf")
    min_y, max_y = float("inf"), float("-inf")
    has_activity = False
    threshold = 0.01

    for fp in tqdm(list(files), desc="Escaneando"):
        d = FrameLoader.load(fp)
        if d is None or d.size == 0:
            continue

        gmin = min(gmin, float(d.min()))
        gmax = max(gmax, float(d.max()))

        active_mask = np.abs(d) > threshold
        if np.any(active_mask):
            has_activity = True
            if d.ndim == 2:
                rows, cols = np.where(active_mask)
                min_y = min(min_y, int(rows.min()))
                max_y = max(max_y, int(rows.max()))
                min_x = min(min_x, int(cols.min()))
                max_x = max(max_x, int(cols.max()))
            else:
                indices = np.where(active_mask)[0]
                min_x = min(min_x, int(indices.min()))
                max_x = max(max_x, int(indices.max()))

    if gmin == float("inf"):
        gmin, gmax = -1.0, 1.0

    xy_limits = None
    if has_activity and min_y != float("inf"):
        xy_limits = (
            (min_x, max_x),
            (min_y, max_y),
        )
        print("[Auto-Crop] Se detectó región activa global.")
    else:
        print("[Auto-Crop] Sin actividad significativa o sólo 1D.")

    return (gmin, gmax), xy_limits


def main():
    p = argparse.ArgumentParser(
        description="Genera videos 1D o 2D a partir de archivos amp_t*.txt/csv."
    )
    p.add_argument("folder", type=Path, help="Carpeta con los archivos amp_t*.txt/csv")
    p.add_argument(
        "--outdir", type=Path, default=Path("videos"), help="Directorio de salida"
    )
    p.add_argument(
        "--mode",
        choices=["auto", "1d", "2d"],
        default="auto",
        help="Forzar modo 1D o 2D",
    )
    p.add_argument(
        "--format", default="mp4", help="Contenedor del video (mp4/gif)"
    )
    p.add_argument(
        "--fps", type=int, default=20, help="Cuadros por segundo"
    )
    p.add_argument(
        "--dpi", type=int, default=120, help="Resolución al renderizar"
    )
    p.add_argument(
        "--zero-center",
        action="store_true",
        help="Centra la escala en cero para magnitudes positivas/negativas",
    )
    p.add_argument(
        "--downsample",
        type=int,
        help="Submuestrea los datos para acelerar el render (factor entero)",
    )
    p.add_argument(
        "--zoom1d",
        type=float,
        help="Fracción del dominio a mostrar centrado en el pico 1D (0-1)",
    )
    p.add_argument(
        "--mark-peak",
        action="store_true",
        help="Resalta el pico máximo en cada frame",
    )
    p.add_argument(
        "--interp",
        default="bilinear",
        choices=["nearest", "bilinear", "bicubic"],
        help="Interpolación/suavizado para superficies 2D",
    )
    p.add_argument(
        "--colorbar",
        action="store_true",
        help="Muestra barra de color en modo 2D",
    )
    p.add_argument(
        "--cmap",
        default="viridis",
        help="Colormap a usar en modo 2D",
    )

    args = p.parse_args()

    if not args.folder.exists():
        sys.exit("Carpeta no encontrada")

    files = sorted(args.folder.glob("amp_t*.*"))
    if not files:
        sys.exit("No hay archivos de datos")

    mode = FrameLoader.detect_mode(files) if args.mode == "auto" else args.mode

    z_lims, xy_lims = analyze_data(files)
    if mode == "1d":
        renderer = Renderer1D(args, z_lims, xy_lims)
    else:
        renderer = Renderer3D(args, z_lims, xy_lims)

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"video_{mode}.{args.format}"

    with imageio.get_writer(out, fps=args.fps, macro_block_size=None) as w:
        for i, fp in enumerate(tqdm(files, desc="Renderizando"), start=1):
            frame = renderer.render(FrameLoader.load(fp), i, len(files))
            if frame is None:
                print(f"[Aviso] Frame {fp.name} omitido por datos vacíos o inválidos")
                continue
            w.append_data(frame)

    print(f"[Listo] {out}")


if __name__ == "__main__":
    main()
