import glob
import os
import sys

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


def load_frame(fp):
    if fp.endswith(".csv"):
        return np.loadtxt(fp, delimiter=",")
    data = np.loadtxt(fp)
    return data.reshape(-1, 1)


def main():
    if len(sys.argv) < 3:
        print("Uso: python3 scripts/make_video.py <carpeta_frames> <salida.gif>")
        sys.exit(1)
    folder = sys.argv[1]
    out_path = sys.argv[2]
    pattern_csv = os.path.join(folder, "amp_t*.csv")
    pattern_dat = os.path.join(folder, "amp_t*.dat")
    files = sorted(glob.glob(pattern_csv) + glob.glob(pattern_dat))
    if not files:
        print("No se encontraron frames en", folder)
        sys.exit(1)

    vmin = None
    vmax = None
    frames = []
    for fp in files:
        arr = load_frame(fp)
        current_min = arr.min()
        current_max = arr.max()
        vmin = current_min if vmin is None else min(vmin, current_min)
        vmax = current_max if vmax is None else max(vmax, current_max)
        frames.append(arr)

    images = []
    for arr in frames:
        fig = plt.figure(figsize=(5, 5))
        plt.axis("off")
        plt.imshow(arr, animated=True, vmin=vmin, vmax=vmax, cmap="viridis")
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)

    imageio.mimsave(out_path, images, duration=0.06)
    print(f"GIF guardado en {out_path}")


if __name__ == "__main__":
    main()
