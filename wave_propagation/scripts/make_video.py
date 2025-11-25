import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Union, Generator
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# Intentar importar tqdm para barra de progreso, fallback si no existe
try:
    from tqdm import tqdm
except ImportError:
    # Mock simple si no está instalado
    def tqdm(iterable, **kwargs): return iterable

# Configuración no interactiva para servidores/WSL
matplotlib.use("Agg")

class FrameLoader:
    """Maneja la carga de archivos de datos."""
    @staticmethod
    def load(path: Path) -> np.ndarray:
        try:
            if path.suffix == ".csv":
                return np.loadtxt(path, delimiter=",")
            return np.loadtxt(path)
        except Exception as e:
            print(f"[Error] No se pudo leer {path}: {e}")
            return np.array([])

    @staticmethod
    def detect_mode(files: List[Path], probe_count: int = 10) -> str:
        """Detecta si los archivos son 1D o 2D inspeccionando los primeros."""
        votes_1d = 0
        votes_2d = 0
        
        for fp in files[:probe_count]:
            data = FrameLoader.load(fp)
            if data.size == 0: continue
            
            # Criterio: 1D si es vector o dimensión 1 en algún eje
            if data.ndim == 1 or (data.ndim == 2 and 1 in data.shape):
                votes_1d += 1
            else:
                votes_2d += 1
        
        if votes_2d > votes_1d:
            return "2d"
        return "1d"

class Renderer:
    """Clase base abstracta para renderizadores."""
    def __init__(self, config: argparse.Namespace, global_limits: Tuple[float, float]):
        self.cfg = config
        self.vmin, self.vmax = global_limits
        # Ajuste de cero centrado si se requiere
        if hasattr(self.cfg, 'zero_center_1d') and self.cfg.zero_center_1d:
             limit = max(abs(self.vmin), abs(self.vmax))
             self.vmin, self.vmax = -limit, limit

    def render(self, data: np.ndarray, step_idx: int, total_steps: int) -> np.ndarray:
        raise NotImplementedError

    def _fig_to_rgb(self, fig: plt.Figure) -> np.ndarray:
        """Convierte una figura de Matplotlib a un array RGB."""
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape(h, w, 3)

class Renderer1D(Renderer):
    def render(self, data: np.ndarray, step_idx: int, total_steps: int) -> np.ndarray:
        y = data.ravel()
        x = np.arange(len(y))

        # Downsampling para rendimiento
        if self.cfg.downsample and len(x) > self.cfg.downsample:
            step = int(np.ceil(len(x) / self.cfg.downsample))
            x = x[::step]
            y = y[::step]

        # Zoom
        if self.cfg.zoom1d and self.cfg.zoom1d > 0:
            peak = int(np.argmax(np.abs(y)))
            L = max(0, peak - self.cfg.zoom1d)
            R = min(len(y)-1, peak + self.cfg.zoom1d)
            x = x[L:R+1]
            y = y[L:R+1]

        fig = plt.figure(figsize=(8, 4), dpi=self.cfg.dpi)
        ax = fig.add_subplot(111)
        ax.plot(x, y, linewidth=2, color='royalblue')
        ax.set_ylim(self.vmin, self.vmax)
        ax.set_xlim(x[0], x[-1])
        ax.set_xlabel("Nodo (X)")
        ax.set_ylabel("Amplitud")
        ax.set_title(f"Simulación 1D | Paso {step_idx}/{total_steps}")
        ax.grid(True, alpha=0.3)

        if self.cfg.mark_peak:
            peak_idx = int(np.argmax(np.abs(y)))
            ax.scatter([x[peak_idx]], [y[peak_idx]], color='red', s=20, zorder=5)

        image = self._fig_to_rgb(fig)
        plt.close(fig)
        return image

class Renderer2D(Renderer):
    def render(self, data: np.ndarray, step_idx: int, total_steps: int) -> np.ndarray:
        fig = plt.figure(figsize=(6, 6), dpi=self.cfg.dpi)
        ax = fig.add_subplot(111)
        
        interp = None if self.cfg.interp == "none" else self.cfg.interp
        im = ax.imshow(data, vmin=self.vmin, vmax=self.vmax, origin="upper", 
                       aspect="auto", cmap='viridis', interpolation=interp)
        
        ax.set_title(f"Simulación 2D | Paso {step_idx}/{total_steps}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        if self.cfg.colorbar:
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        image = self._fig_to_rgb(fig)
        plt.close(fig)
        return image

class VideoBuilder:
    def __init__(self, output_path: Path, fps: int):
        self.output_path = output_path
        self.fps = fps
        # Asegurar directorio de salida
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def build(self, frame_generator: Generator[np.ndarray, None, None], total_frames: int):
        print(f"[Video] Iniciando renderizado de {total_frames} frames...")
        print(f"[Video] Salida: {self.output_path}")
        
        # Usar el writer como contexto asegura que se cierre el archivo correctamente
        with imageio.get_writer(self.output_path, fps=self.fps) as writer:
            for frame in tqdm(frame_generator, total=total_frames, unit="frame"):
                writer.append_data(frame)
        
        print("[Video] Renderizado completado exitosamente.")

def scan_limits(files: List[Path]) -> Tuple[float, float]:
    """Escanea todos los archivos para encontrar el min/max global."""
    print("[Stats] Calculando límites globales (vmin/vmax)...")
    gmin, gmax = float('inf'), float('-inf')
    
    for fp in tqdm(files, desc="Escaneando", unit="file"):
        data = FrameLoader.load(fp)
        if data.size > 0:
            gmin = min(gmin, data.min())
            gmax = max(gmax, data.max())
            
    if gmin == float('inf'): return -1.0, 1.0
    if gmin == gmax: gmax += 1e-9
    return gmin, gmax

def main():
    parser = argparse.ArgumentParser(description="Generador de Video HPC (Refactorizado)")
    parser.add_argument("folder", type=Path, help="Carpeta con archivos .dat/.csv")
    parser.add_argument("--mode", choices=["auto", "1d", "2d"], default="auto", help="Modo de renderizado")
    parser.add_argument("--fps", type=int, default=20, help="Cuadros por segundo")
    parser.add_argument("--outdir", type=Path, default=Path("videos"), help="Directorio de salida")
    parser.add_argument("--format", choices=["gif", "mp4"], default="mp4", help="Formato de salida")
    parser.add_argument("--dpi", type=int, default=100, help="Calidad de imagen (DPI)")
    
    # Opciones específicas
    parser.add_argument("--downsample", type=int, help="[1D] Reducir puntos para graficar más rápido")
    parser.add_argument("--zoom1d", type=int, help="[1D] Zoom alrededor del pico")
    parser.add_argument("--zero-center-1d", action="store_true", help="[1D] Centrar eje Y en 0")
    parser.add_argument("--mark-peak", action="store_true", help="[1D] Marcar el máximo con un punto")
    parser.add_argument("--interp", choices=["nearest", "bilinear", "none"], default="nearest", help="[2D] Interpolación")
    parser.add_argument("--colorbar", action="store_true", help="[2D] Mostrar barra de color")

    args = parser.parse_args()

    # 1. Buscar archivos
    if not args.folder.exists():
        sys.exit(f"Error: La carpeta {args.folder} no existe.")
        
    files = sorted(list(args.folder.glob("amp_t*.*")))
    if not files:
        sys.exit("Error: No se encontraron archivos 'amp_t*.*' en la carpeta.")

    # 2. Detectar modo si es auto
    mode = args.mode
    if mode == "auto":
        mode = FrameLoader.detect_mode(files)
        print(f"[Info] Modo detectado automáticamente: {mode.upper()}")

    # 3. Calcular límites globales (pre-pass)
    limits = scan_limits(files)
    print(f"[Stats] Rango detectado: [{limits[0]:.4f}, {limits[1]:.4f}]")

    # 4. Configurar Renderizador
    renderer: Renderer
    if mode == "1d":
        renderer = Renderer1D(args, limits)
    else:
        renderer = Renderer2D(args, limits)

    # 5. Generador de frames (Lazy Evaluation)
    def frame_generator():
        for i, fp in enumerate(files):
            data = FrameLoader.load(fp)
            if data.size > 0:
                yield renderer.render(data, i + 1, len(files))

    # 6. Construir Video
    out_file = args.outdir / f"video_{mode}.{args.format}"
    builder = VideoBuilder(out_file, args.fps)
    builder.build(frame_generator(), len(files))

if __name__ == "__main__":
    main()