import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Generator
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource # Para sombras y profundidad 3D
from mpl_toolkits.mplot3d import Axes3D 
import imageio.v2 as imageio

# Intentar importar tqdm
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable
            if path.suffix == ".csv":
                return np.loadtxt(path, delimiter=",")
            return np.loadtxt(path)
        except Exception as e:
            print(f"[Error] No se pudo leer {path}: {e}")
            return np.array([])

    @staticmethod
    def detect_mode(files: List[Path], probe_count: int = 10) -> str:
        votes_1d = 0
        votes_2d = 0
        for fp in files[:probe_count]:
            data = FrameLoader.load(fp)
            if data.size == 0: continue
            if data.ndim == 1 or (data.ndim == 2 and 1 in data.shape):
                votes_1d += 1
            else:
                votes_2d += 1
        return "2d" if votes_2d > votes_1d else "1d"

class Renderer:
    def __init__(self, config: argparse.Namespace, global_limits: Tuple[float, float]):
        self.cfg = config
        self.vmin, self.vmax = global_limits
        
        # Margen de seguridad para evitar errores de renderizado si plano
        if abs(self.vmax - self.vmin) < 1e-5:
            self.vmax += 0.1
            
        if hasattr(self.cfg, 'zero_center') and self.cfg.zero_center:
             limit = max(abs(self.vmin), abs(self.vmax))
             self.vmin, self.vmax = -limit, limit

    def _fig_to_rgb(self, fig: plt.Figure) -> np.ndarray:
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape(h, w, 3)

class Renderer1D(Renderer):
    def render(self, data: np.ndarray, step_idx: int, total_steps: int) -> np.ndarray:
        y = data.ravel()
        x = np.arange(len(y))

        fig = plt.figure(figsize=(8, 5), dpi=self.cfg.dpi)
        ax = fig.add_subplot(111)
        
        # 1. Línea principal (más suave y gruesa)
        ax.plot(x, y, linewidth=2, color='#0066cc', label='Amplitud')
        
        # 2. Relleno bajo la curva (Mejora visual clave)
        ax.fill_between(x, y, color='#0066cc', alpha=0.2)
        
        # 3. Estilizado
        ax.set_ylim(self.vmin, self.vmax)
        ax.set_xlim(x[0], x[-1])
        ax.grid(True, linestyle='--', color='gray', alpha=0.3)
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5) # Línea base en 0
        
        ax.set_xlabel("Nodo (Posición)")
        ax.set_ylabel("Amplitud u(x)")
        ax.set_title(f"Propagación 1D — Paso {step_idx}/{total_steps}", fontweight='bold')
        
        fig.tight_layout()
        image = self._fig_to_rgb(fig)
        plt.close(fig)
        return image

class Renderer3D(Renderer):
    def render(self, data: np.ndarray, step_idx: int, total_steps: int) -> np.ndarray:
        fig = plt.figure(figsize=(10, 7), dpi=self.cfg.dpi)
        # Ajuste de perspectiva
        ax = fig.add_subplot(111, projection='3d')
        
        h, w = data.shape
        X = np.arange(w)
        Y = np.arange(h)
        X, Y = np.meshgrid(X, Y)
        
        # 1. Crear fuente de luz para sombras (Efecto "plástico/realista")
        ls = LightSource(azdeg=315, altdeg=45)
        # Mapear colores según altura (z) y sombras según gradiente
        rgb = ls.shade(data, cmap=cm.coolwarm, vert_exag=0.1, blend_mode='soft')

        # 2. Plot de superficie con sombreado
        surf = ax.plot_surface(X, Y, data, facecolors=rgb,
                               linewidth=0, antialiased=True, shade=False,
                               rstride=1, cstride=1) # rstride/cstride 1 para máxima calidad

        ax.set_zlim(self.vmin, self.vmax)
        
        # Etiquetas
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Amplitud")
        ax.set_title(f"Propagación 2D — Paso {step_idx}/{total_steps}", fontweight='bold')

        # Vista isométrica mejorada
        ax.view_init(elev=35, azim=-45)
        
        # Quitar fondo gris de los paneles 3D para limpieza
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.2)

        image = self._fig_to_rgb(fig)
        plt.close(fig)
        return image

class VideoBuilder:
    def __init__(self, output_path: Path, fps: int):
        self.output_path = output_path
        self.fps = fps
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def build(self, frame_generator: Generator[np.ndarray, None, None], total_frames: int):
        print(f"[Video] Guardando en: {self.output_path}")
        # Usamos macro_block_size=None para evitar warnings de FFMPEG si el tamaño no es múltiplo de 16
        with imageio.get_writer(self.output_path, fps=self.fps, macro_block_size=None) as writer:
            for frame in tqdm(frame_generator, total=total_frames):
                writer.append_data(frame)
        print("[Video] ¡Listo!")

def scan_limits(files: List[Path]) -> Tuple[float, float]:
    print("[Stats] Analizando límites globales...")
    gmin, gmax = float('inf'), float('-inf')
    for fp in tqdm(files, desc="Escaneando"):
        data = FrameLoader.load(fp)
        if data.size > 0:
            gmin = min(gmin, data.min())
            gmax = max(gmax, data.max())
    if gmin == float('inf'): return -1.0, 1.0
    return gmin, gmax

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path, help="Carpeta con datos")
    parser.add_argument("--mode", choices=["auto", "1d", "2d"], default="auto")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--outdir", type=Path, default=Path("videos"))
    parser.add_argument("--format", choices=["gif", "mp4"], default="mp4")
    parser.add_argument("--dpi", type=int, default=120) # DPI más alto para mejor calidad
    parser.add_argument("--zero-center", action="store_true")
    
    args = parser.parse_args()

    if not args.folder.exists():
        sys.exit("Carpeta no encontrada.")
        
    files = sorted(list(args.folder.glob("amp_t*.*")))
    if not files:
        sys.exit("No hay archivos amp_t*.*")

    mode = args.mode
    if mode == "auto":
        mode = FrameLoader.detect_mode(files)
        print(f"[Info] Modo detectado: {mode.upper()}")

    limits = scan_limits(files)
    
    # Seleccionar renderizador
    renderer = Renderer1D(args, limits) if mode == "1d" else Renderer3D(args, limits)

    def frame_gen():
        for i, fp in enumerate(files):
            yield renderer.render(FrameLoader.load(fp), i + 1, len(files))

    out_file = args.outdir / f"video_{mode}_HQ.{args.format}"
    VideoBuilder(out_file, args.fps).build(frame_gen(), len(files))

if __name__ == "__main__":
    main()