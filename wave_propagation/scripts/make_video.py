import sys, os, glob, argparse
import numpy as np

# Backend no interactivo (Windows/MSYS2 friendly)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# ---------- utils ----------
def list_frames(folder):
    files = sorted(glob.glob(os.path.join(folder, "amp_t*.*")))
    return files

def read_frame(fp):
    if fp.endswith(".csv"):
        return np.loadtxt(fp, delimiter=",")
    return np.loadtxt(fp)

def is_1d_array(A):
    # 1D si es vector o si matriz con 1 fila o 1 columna
    if A.ndim == 1:
        return True
    if A.ndim == 2 and (A.shape[0] == 1 or A.shape[1] == 1):
        return True
    return False

def split_1d_2d(files, max_probe=10):
    f1d, f2d = [], []
    for fp in files[:max_probe]:
        try:
            A = read_frame(fp)
        except Exception:
            continue
        if is_1d_array(A):
            f1d.append(fp)
        else:
            f2d.append(fp)
    # si sólo probamos algunos, completa por extensión
    # (heurística: .dat -> 1d; resto -> usa lo ya clasificado)
    if f1d or f2d:
        if any(fp.endswith(".dat") for fp in files):
            f1d_all = [fp for fp in files if fp.endswith(".dat")]
        else:
            f1d_all = f1d  # ya están clasificados
        # 2D: csv que no sean 1D
        f2d_all = []
        for fp in files:
            if fp.endswith(".csv"):
                try:
                    A = read_frame(fp)
                    if not is_1d_array(A):
                        f2d_all.append(fp)
                    else:
                        f1d_all.append(fp)
                except Exception:
                    pass
        # dedup ordenado
        f1d_all = sorted(list(dict.fromkeys(f1d_all)))
        f2d_all = sorted(list(dict.fromkeys(f2d_all)))
        return f1d_all, f2d_all
    # si no pudimos leer nada, devolvemos listas vacías
    return [], []

def fig_to_rgb(fig):
    fig.canvas.draw()
    try:
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        return buf.reshape(h, w, 3)
    except AttributeError:
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        return buf[:, :, :3]

def collect_stats(files):
    vals = []
    for fp in files:
        A = read_frame(fp).astype(float)
        vals.append(A.ravel())
    if not vals:
        return -1.0, 1.0
    v = np.concatenate(vals)
    vmin, vmax = float(v.min()), float(v.max())
    if vmin == vmax:
        vmax = vmin + 1e-9
    return vmin, vmax

# ---------- render 1D ----------
def make_video_1d(frames, out_path, fps=12, downsample=None, zoom1d=None,
                  zero_center=False, show_peak=False, dpi=110, fmt="gif"):
    if not frames:
        print("[1D] No hay frames 1D"); return
    # rango global base
    gmin, gmax = collect_stats(frames)
    if zero_center:
        vmax = max(abs(gmin), abs(gmax)); gmin, gmax = -vmax, vmax

    writer = None
    if fmt == "mp4":
        try:
            writer = imageio.get_writer(out_path.replace(".gif", ".mp4"), fps=fps)
            out_path = out_path.replace(".gif", ".mp4")
        except Exception:
            writer = None

    imgs = []
    for k, fp in enumerate(frames, 1):
        y = read_frame(fp).astype(float).ravel()
        x = np.arange(len(y))
        if downsample and len(x) > downsample:
            step = int(np.ceil(len(x) / float(downsample)))
            x = x[::step]; y = y[::step]
        if zoom1d and zoom1d > 0:
            i0 = int(np.argmax(np.abs(y)))
            L = max(0, i0 - zoom1d); R = min(len(y)-1, i0 + zoom1d)
            x = x[L:R+1]; y = y[L:R+1]

        fig = plt.figure(figsize=(8,4), dpi=dpi)
        ax = plt.gca()
        ax.plot(x, y, linewidth=2)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(gmin, gmax)
        ax.set_xlabel("Nodo (X)")
        ax.set_ylabel("Amplitud")
        ax.set_title(f"Onda 1D – Paso {k}/{len(frames)}")
        if show_peak:
            i0 = int(np.argmax(np.abs(y)))
            ax.axvline(x[i0], linestyle="--", linewidth=1)
            ax.scatter([x[i0]],[y[i0]], s=16)
        frame = fig_to_rgb(fig); plt.close(fig)
        if writer: writer.append_data(frame)
        else: imgs.append(frame)

    if writer:
        writer.close(); print(f"[1D] Video: {out_path} ({len(frames)} frames)")
    else:
        imageio.mimsave(out_path, imgs, duration=1.0/float(fps))
        print(f"[1D] Video: {out_path} ({len(imgs)} frames)")

# ---------- render 2D ----------
def make_video_2d(frames, out_path, fps=12, dpi=110, fmt="gif",
                  vmin=None, vmax=None, interp="nearest", colorbar=False):
    if not frames:
        print("[2D] No hay frames 2D"); return
    if vmin is None or vmax is None:
        vmin, vmax = collect_stats(frames)

    writer = None
    if fmt == "mp4":
        try:
            writer = imageio.get_writer(out_path.replace(".gif", ".mp4"), fps=fps)
            out_path = out_path.replace(".gif", ".mp4")
        except Exception:
            writer = None

    imgs = []
    for k, fp in enumerate(frames, 1):
        A = read_frame(fp).astype(float)
        fig = plt.figure(figsize=(5,5), dpi=dpi)
        ax = plt.gca()
        im = ax.imshow(A, vmin=vmin, vmax=vmax, origin="upper", aspect="auto",
                       interpolation=None if interp=="none" else interp)
        ax.set_xlabel("X (nodo)"); ax.set_ylabel("Y (nodo)")
        ax.set_title(f"Onda 2D – Paso {k}/{len(frames)}")
        if colorbar: plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        frame = fig_to_rgb(fig); plt.close(fig)
        if writer: writer.append_data(frame)
        else: imgs.append(frame)

    if writer:
        writer.close(); print(f"[2D] Video: {out_path} ({len(frames)} frames)")
    else:
        imageio.mimsave(out_path, imgs, duration=1.0/float(fps))
        print(f"[2D] Video: {out_path} ({len(imgs)} frames)")

# ---------- CLI ----------
def main():
    pa = argparse.ArgumentParser(description="Generar GIF/MP4 1D/2D desde frames")
    pa.add_argument("folder", help="carpeta con frames (p.ej. results/frames)")
    pa.add_argument("--mode", choices=["auto","1d","2d"], default="auto",
                    help="auto: detecta; 1d: sólo 1D; 2d: sólo 2D")
    pa.add_argument("--fps", type=int, default=12)
    pa.add_argument("--outdir", default="videos")
    pa.add_argument("--format", choices=["gif","mp4"], default="gif")
    pa.add_argument("--dpi", type=int, default=110)

    # 1D extras
    pa.add_argument("--downsample", type=int, default=None)
    pa.add_argument("--zoom1d", type=int, default=None)
    pa.add_argument("--zero-center-1d", action="store_true")
    pa.add_argument("--mark-peak", action="store_true")

    # 2D extras
    pa.add_argument("--interp", choices=["nearest","bilinear","none"], default="nearest")
    pa.add_argument("--colorbar", action="store_true")
    args = pa.parse_args()

    files = list_frames(args.folder)
    if not files:
        print("No se encontraron frames en", args.folder); sys.exit(1)

    f1d, f2d = split_1d_2d(files)
    print(f"[scan] {len(f1d)} frames 1D, {len(f2d)} frames 2D detectados")

    os.makedirs(args.outdir, exist_ok=True)

    if args.mode in ("auto","1d") and f1d:
        make_video_1d(
            f1d, os.path.join(args.outdir, "onda_1d.gif"),
            fps=args.fps, downsample=args.downsample, zoom1d=args.zoom1d,
            zero_center=args.zero_center_1d, show_peak=args.mark_peak,
            dpi=args.dpi, fmt=args.format
        )
    elif args.mode == "1d" and not f1d:
        print("[1D] No se detectaron frames 1D (ni .dat ni .csv de una sola fila/columna).")

    if args.mode in ("auto","2d") and f2d:
        make_video_2d(
            f2d, os.path.join(args.outdir, "onda_2d.gif"),
            fps=args.fps, interp=args.interp, colorbar=args.colorbar,
            dpi=args.dpi, fmt=args.format
        )
    elif args.mode == "2d" and not f2d:
        print("[2D] No se detectaron frames 2D (.csv con más de 1 fila y 1 columna).")

if __name__ == "__main__":
    main()
