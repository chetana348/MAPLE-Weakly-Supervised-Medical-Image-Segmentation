import numpy as np
import tifffile as tiff
from scipy.ndimage import gaussian_filter
import os

def make_heatmap(mask, sigma=10):
    y, x = np.where(mask > 0)
    if len(x) == 0:
        return np.zeros_like(mask, dtype=np.float32)

    cy = int(np.mean(y))
    cx = int(np.mean(x))

    heatmap = np.zeros_like(mask, dtype=np.float32)
    heatmap[cy, cx] = 1.0
    heatmap = gaussian_filter(heatmap, sigma=sigma)
    heatmap /= heatmap.max() + 1e-6
    return heatmap

pseudo_dir = ""
out_dir = ""
os.makedirs(out_dir, exist_ok=True)

for f in os.listdir(pseudo_dir):
    mask = tiff.imread(os.path.join(pseudo_dir, f))
    mask = (mask > 0).astype(np.uint8)
    hm = make_heatmap(mask, sigma=12)
    tiff.imwrite(os.path.join(out_dir, f), hm.astype(np.float32))

