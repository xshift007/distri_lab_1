import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource 
import imageio.v2 as imageio

# Intentar importar tqdm
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs): return iterable

# Backend no interactivo
matplotlib.use("Agg")

# Estilo global
plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'grid.alpha': 0.3,
})

class FrameLoader:
    @staticmethod
    def load(path: Path) -> np.ndarray:
        try:
            if path.suffix == ".csv": return np.loadtxt(path, delimiter=",")
            return np.loadtxt(path)
        except Exception: return np.array([])

    @staticmethod
    def detect_mode(files: List[Path], probe_count: int = 10) -> str:
        votes_1d, votes_2d = 0, 0
        for fp in files[:probe_count]:
            data = FrameLoader.load(fp)
            if data.size == 0: continue
            if data.ndim == 1 or (data.ndim == 2 and 1 in data.shape): votes_1d += 1
            else: votes_2d += 1
        return "2d" if votes_2d > votes_1d else "1d"

class Renderer:
    def __init__(self, config, z_limits, xy_limits=None):
        self.cfg = config
        self.vmin, self.vmax = z_limits
        self.xlim, self.ylim = xy_limits if xy_limits else (None, None)
        if abs(self.vmax - self.vmin) < 1e-3: self.vmax += 0.1
        zero_center = getattr(self.cfg, 'zero_center', False)
        if zero_center:
             limit = max(abs(self.vmin), abs(self.vmax))
             self.vmin, self.vmax = -limit, limit

    def _fig_to_rgb(self, fig):
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape(h, w, 3)

class Renderer1D(Renderer):
    def render(self, data, step, total):
        y = data.ravel()
        x = np.arange(len(y))
        fig = plt.figure(figsize=(8, 5), dpi=self.cfg.dpi)
        ax = fig.add_subplot(111)
        ax.plot(x, y, linewidth=2, color='#0066cc')
        ax.fill_between(x, y, color='#0066cc', alpha=0.2)
        ax.set_ylim(self.vmin, self.vmax)
        if self.xlim: ax.set_xlim(self.xlim)
        else: ax.set_xlim(x[0], x[-1])
        ax.grid(True, linestyle='--')
        ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
        ax.set_title(f"Simulación 1D — Paso {step}/{total}", fontweight='bold')
        ax.set_ylabel("Amplitud"); ax.set_xlabel("Nodo")
        img = self._fig_to_rgb(fig); plt.close(fig); return img

class Renderer3D(Renderer):
    def render(self, data, step, total):
        fig = plt.figure(figsize=(10, 9), dpi=self.cfg.dpi) # Aumentar altura para que quepan los ejes
        ax = fig.add_subplot(111, projection='3d')
        h, w = data.shape
        X, Y = np.meshgrid(np.arange(w), np.arange(h))
        
        ls = LightSource(azdeg=315, altdeg=65)
        rgb = ls.shade(data, cmap=cm.plasma, vert_exag=5.0, blend_mode='soft')
        ax.plot_surface(X, Y, data, facecolors=rgb, linewidth=0, antialiased=False, shade=False, rstride=1, cstride=1)
        
        ax.set_zlim(self.vmin, self.vmax)
        
        if self.xlim: ax.set_xlim(self.xlim)
        if self.ylim: ax.set_ylim(self.ylim)
        
        # --- CORRECCIÓN DE EJES ---
        # Aumentar labelpad para separar las etiquetas de los números
        ax.set_xlabel("X (Columnas)", labelpad=10)
        ax.set_ylabel("Y (Filas)", labelpad=10)
        ax.set_zlabel("Amplitud", labelpad=10)
        
        ax.set_title(f"Simulación 2D — Paso {step}/{total}", fontweight='bold', pad=20)
        
        # Ajustar vista para asegurar que los ejes inferiores sean visibles
        ax.view_init(elev=35, azim=-55) 
        
        ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
        ax.grid(True, alpha=0.15)
        
        # Ajuste de márgenes de la figura
        plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        
        img = self._fig_to_rgb(fig); plt.close(fig); return img

def analyze_data(files: List[Path]):
    print("[Info] Analizando datos para Auto-Crop y Escala...")
    gmin, gmax = float('inf'), float('-inf')
    min_x, max_x = float('inf'), float('-inf'); min_y, max_y = float('inf'), float('-inf')
    has_activity = False
    threshold = 0.01 

    for fp in tqdm(files, desc="Escaneando"):
        d = FrameLoader.load(fp)
        if d.size == 0: continue
        gmin = min(gmin, d.min()); gmax = max(gmax, d.max())
        active_mask = np.abs(d) > threshold
        if np.any(active_mask):
            has_activity = True
            if d.ndim == 2:
                rows, cols = np.where(active_mask)
                min_y = min(min_y, rows.min()); max_y = max(max_y, rows.max())
                min_x = min(min_x, cols.min()); max_x = max(max_x, cols.max())
            else:
                indices = np.where(active_mask)[0]
                min_x = min(min_x, indices.min()); max_x = max(max_x, indices.max())

    if gmin == float('inf'): gmin, gmax = -1.0, 1.0
    
    xy_limits = None
    if has_activity:
        if min_y != float('inf'):
            # Pad más generoso para asegurar que se vean los ejes
            pad = max(5, int(max(max_x-min_x, max_y-min_y) * 0.15))
            xy_limits = ((min_x - pad, max_x + pad), (min_y - pad, max_y + pad))
        else:
            pad_x = max(5, int((max_x - min_x) * 0.1))
            xy_limits = (min_x - pad_x, max_x + pad_x)
        print(f"[Auto-Crop] Zoom aplicado con margen extra.")
    else:
        print("[Auto-Crop] Sin actividad significativa, vista completa.")
    return (gmin, gmax), xy_limits

def main():
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path); p.add_argument("--outdir", type=Path, default=Path("videos"))
    p.add_argument("--mode", choices=["auto","1d","2d"], default="auto")
    p.add_argument("--format", default="mp4"); p.add_argument("--fps", type=int, default=20)
    p.add_argument("--dpi", type=int, default=120)
    p.add_argument("--zero-center", action="store_true")
    p.add_argument("--downsample", type=int); p.add_argument("--zoom1d", type=int)
    p.add_argument("--mark-peak", action="store_true"); p.add_argument("--interp", default="nearest"); p.add_argument("--colorbar", action="store_true")
    
    args = p.parse_args()
    if not args.folder.exists(): sys.exit("Carpeta no encontrada")
    files = sorted(list(args.folder.glob("amp_t*.*")))
    if not files: sys.exit("No hay archivos de datos")
    mode = FrameLoader.detect_mode(files) if args.mode == "auto" else args.mode
    z_lims, xy_lims = analyze_data(files)
    renderer = Renderer1D(args, z_lims, xy_lims) if mode == "1d" else Renderer3D(args, z_lims, xy_lims)
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"video_{mode}.{args.format}"
    with imageio.get_writer(out, fps=args.fps, macro_block_size=None) as w:
        for i, fp in enumerate(tqdm(files, desc="Renderizando")):
            w.append_data(renderer.render(FrameLoader.load(fp), i+1, len(files)))
    print(f"[Listo] {out}")

if __name__ == "__main__": main()