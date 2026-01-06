import os
import numpy as np
import tifffile as tiff
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt


def generate_points_2d(
    mask,
    num_fg=1,
    margin_ratio=0.15,
    bg_offset=5
):
    """
    Generate points for a 2D mask.

    Foreground: configurable (1–6)
    Background: always 1

    Labels:
      0 = unlabeled
      1 = foreground
      2 = background
    """

    assert mask.ndim == 2, "Mask must be 2D"
    assert np.any(mask), "Empty tumor mask"
    assert 1 <= num_fg <= 6, "num_fg must be in [1, 6]"

    H, W = mask.shape
    point_mask = np.zeros((H, W), dtype=np.uint8)

    # -----------------------------------------
    # Tumor coordinates
    # -----------------------------------------
    coords = np.argwhere(mask > 0)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Foreground points (distance-transform based)
    # -----------------------------------------
    dist_map = distance_transform_edt(mask)
    
    # Sort pixels by distance to boundary (deep interior first)
    flat_idx = np.argsort(dist_map.ravel())[::-1]
    
    fg_points = []
    for idx in flat_idx:
        y, x = np.unravel_index(idx, mask.shape)
        if mask[y, x] == 0:
            continue
    
        fg_points.append((y, x))
        if len(fg_points) == num_fg:
            break
    
    for y, x in fg_points:
        point_mask[y, x] = 1


    # -----------------------------------------
    # Background point (outside tumor bbox)
    # -----------------------------------------
    dy = int((y_max - y_min) * margin_ratio) + bg_offset
    dx = int((x_max - x_min) * margin_ratio) + bg_offset

    y_bg = max(0, y_min - dy)
    x_bg = max(0, x_min - dx)

    y_bg = min(H - 1, y_bg)
    x_bg = min(W - 1, x_bg)

    point_mask[y_bg, x_bg] = 2

    return point_mask


def process_directory(
    image_dir,
    mask_dir,
    output_dir,
    num_fg=1
):
    """
    Generate point masks for a directory.
    """

    os.makedirs(output_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(image_dir) if f.endswith(".tif"))

    for fname in tqdm(files, desc="Generating PA-Seg points"):
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname)
        out_path = os.path.join(output_dir, fname)

        if not os.path.exists(mask_path):
            print(f"[WARN] Mask missing for {fname}, skipping")
            continue

        mask = tiff.imread(mask_path)
        if mask.shape ==(1,128,128):
            mask = mask[0,:,:]
        #print(mask.shape)
        #mask = mask[0,:,:]

        if not np.any(mask):
            print(f"[WARN] Empty tumor in {fname}, skipping")
            continue

        point_mask = generate_points_2d(
            mask=mask,
            num_fg=num_fg
        )

        tiff.imwrite(out_path, point_mask)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

IMAGE_DIR = "images"
MASK_DIR  = "labels"
OUTPUT_DIR = "points_2p"

NUM_FG_POINTS = 1   # set between 1 and 6

process_directory(
    image_dir=IMAGE_DIR,
    mask_dir=MASK_DIR,
    output_dir=OUTPUT_DIR,
    num_fg=NUM_FG_POINTS
)
