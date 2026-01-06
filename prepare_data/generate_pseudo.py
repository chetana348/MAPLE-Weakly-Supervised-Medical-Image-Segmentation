import os
import numpy as np
import tifffile as tiff
import GeodisTK
from tqdm import tqdm
from scipy.ndimage import sobel, binary_fill_holes, binary_opening
import pandas as pd


# =========================
# FIXED HYPERPARAMETERS
# =========================
WEIGHT = 0.5
ITERS = 4

FIXED_THR = 0.25
CENTER_PRIOR_WEIGHT = 0.45
CENTER_PRIOR_POWER = 0.6
EDGE_WEIGHT = 0.20


# -------------------------------------------------
# Manifold distance
# -------------------------------------------------
def manifold_distance_2d(I, S, lamb=0.5, iters=4):
    return GeodisTK.geodesic2d_raster_scan(I, S, lamb, iters)


# -------------------------------------------------
# Dice metric (EVAL ONLY)
# -------------------------------------------------
def dice_coefficient(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    inter = np.logical_and(pred, gt).sum()
    return (2.0 * inter + eps) / (pred.sum() + gt.sum() + eps)
    
def raad_single(y_true, y_pred, epsilon=1e-6):
    gt_area = np.sum(y_true >0)
    pred_area = np.sum(y_pred> 0)
    
    if gt_area == 0:
        return 0.0 if pred_area == 0 else 1.0
    return abs(pred_area - gt_area) / (gt_area + epsilon)


# -------------------------------------------------
# Robust normalization (IMAGE-ONLY)
# -------------------------------------------------
def robust_normalize(I):
    p1, p99 = np.percentile(I, (1, 99))
    I = np.clip(I, p1, p99)
    return (I - p1) / (p99 - p1 + 1e-6)


# -------------------------------------------------
# Soft, widened center prior (IMAGE-ONLY)
# -------------------------------------------------
def center_prior_map(shape, power=0.6):
    H, W = shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    dist = dist / (dist.max() + 1e-6)
    return dist ** power


# -------------------------------------------------
# Main processing (TRUE WEAK SUPERVISION)
# -------------------------------------------------
def process_directory(
    image_dir,
    point_mask_dir,
    label_dir,
    output_dir,
    setting_name,
    summary_rows,
    verbose=False
):
    os.makedirs(output_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(image_dir) if f.endswith(".tif"))
    dice_scores = []
    raad_scores = []

    for fname in tqdm(files, desc="1-Point Geodesic (True Weak Sup)"):
        img_path = os.path.join(image_dir, fname)
        point_path = os.path.join(point_mask_dir, fname)
        label_path = os.path.join(label_dir, fname)
        out_path = os.path.join(output_dir, fname)

        if not (os.path.exists(point_path) and os.path.exists(label_path)):
            continue

        # ------------------------------
        # Load
        # ------------------------------
        I = tiff.imread(img_path).astype(np.float32)
        anno = tiff.imread(point_path)
        
        gt = tiff.imread(label_path).astype(np.uint8)  # eval only
        if gt.shape ==(1,128,128):
            gt = gt[0,:,:]
        
        S_fg = (anno == 1).astype(np.uint8)
        if S_fg.sum() == 0:
            continue
        
        if I.ndim == 3:
            I = I[0]
        if anno.ndim == 3:
            anno = anno[0]
        if gt.ndim == 3:
            gt = gt[0]

        # ------------------------------
        # Normalize
        # ------------------------------
        I = robust_normalize(I)

        # ------------------------------
        # Geodesic distance
        # ------------------------------
        dist = manifold_distance_2d(
            I, S_fg, WEIGHT, ITERS
        )
        dmax = dist.max()
        if dmax <= 0:
            continue

        biased_dist = dist.copy()

        # ------------------------------
        # Center prior (FIXED, GT-FREE)
        # ------------------------------
        prior = center_prior_map(
            dist.shape, power=CENTER_PRIOR_POWER
        )
        biased_dist += CENTER_PRIOR_WEIGHT * dmax * prior

        # ------------------------------
        # Edge-aware penalty (IMAGE-ONLY)
        # ------------------------------
        gx = sobel(I, axis=0)
        gy = sobel(I, axis=1)
        edge = np.sqrt(gx ** 2 + gy ** 2)
        edge = edge / (edge.max() + 1e-6)
        biased_dist += EDGE_WEIGHT * dmax * edge

        # ------------------------------
        # FIXED threshold (NO GT)
        # ------------------------------
        pseudo = (biased_dist < FIXED_THR * dmax).astype(np.uint8)
        pseudo[S_fg == 1] = 1  # seed constraint

        # ------------------------------
        # Minimal morphology
        # ------------------------------
        
        pseudo = binary_fill_holes(pseudo).astype(np.uint8)
        pseudo = binary_opening(
            pseudo, structure=np.ones((3, 3))
        ).astype(np.uint8)

        # ------------------------------
        # Dice (EVAL ONLY)
        # ------------------------------
        if gt.sum() > 0:
            dice = dice_coefficient(pseudo, gt)
            raad = raad_single(gt, pseudo)
            dice_scores.append(dice)
            raad_scores.append(raad)
            

            if verbose:
                print(f"{fname} | Dice={dice:.3f}| RAAD={raad:.3f}")

        # ------------------------------
        # Save
        # ------------------------------
        tiff.imwrite(out_path, pseudo)

    # ------------------------------
    # Report
    # ------------------------------
    if dice_scores:
        print(
            f"\nMean Dice: {np.mean(dice_scores):.4f} "
            f"± {np.std(dice_scores):.4f}"
        )
        print(
            f"Mean RAAD: {np.mean(raad_scores):.4f} "
            f"± {np.std(raad_scores):.4f}"
        )
        summary_rows.append({
            "Setting": setting_name,
            "Mean Dice": np.mean(dice_scores),
            "Std Dice": np.std(dice_scores),
            "Mean RAAD": np.mean(raad_scores)
        })
    else:
        print("\nNo valid Dice computed")

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
IMAGE_DIR = "/users/PAS3110/sephora20/workspace/PDAC/data/pdac_osu/cropped/MRI/train/images"
LABEL_DIR = "/users/PAS3110/sephora20/workspace/PDAC/data/pdac_osu/cropped/MRI/train/labels"

points_mask = [
    "2point", "4point", "6point", "8point", "10point", "12point"
]

points_dirs = [
    "/points_3p",
]

outputs = [
   "pseudo_nogt",
]

summary_rows = []

for name, pdir, outdir in zip(points_mask, points_dirs, outputs):
    process_directory(
        IMAGE_DIR,
        pdir,
        LABEL_DIR,
        outdir,
        setting_name=name,
        summary_rows=summary_rows,
        verbose=False
    )

# -------------------------------------------------
# EXPORT SINGLE EXCEL
# -------------------------------------------------
excel_path = "summary.xlsx"
pd.DataFrame(summary_rows).to_excel(excel_path, index=False)

print(f"\n Summary exported to: {excel_path}")
