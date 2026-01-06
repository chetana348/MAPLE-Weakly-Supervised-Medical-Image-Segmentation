import os
import numpy as np
import tifffile as tiff
import GeodisTK

from scipy.ndimage import (
    binary_fill_holes,
    binary_dilation,
    distance_transform_edt,
    sobel
)

# ============================================================
# DIRECTORIES
# ============================================================
DINO_DIR   = "dino"
DIR = "pseudo_1p_nogt"
GT_DIR       = "labels"
OUT_DIR      = "pseudo_fuse"

#os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# PARAMETERS
# ============================================================
ITERS = 4
BG_BORDER = 6

CENTER_RADIUS_FRAC = 0.25
MIN_SIZE_FRAC = 0.02
MAX_SIZE_FRAC = 0.60

GOOD_THR = 0.65
REFINE_DILATION = 6

# ---- ellipse controls (SAFE) ----
USE_ELLIPSE = True          # turn OFF if needed
ELLIPSE_THR = 1.6           # >1 = very weak
ELLIPSE_MIN_DICE = 0.80     # apply only to strong masks

# ============================================================
# METRIC (EVAL / DEBUG ONLY)
# ============================================================
def dice(pred, gt, eps=1e-6):
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    inter = (pred & gt).sum()
    return (2 * inter + eps) / (pred.sum() + gt.sum() + eps)
    
    
def raad_single(y_true, y_pred, epsilon=1e-6):
    gt_area = np.sum(y_true> 0.5)
    pred_area = np.sum(y_pred> 0.5)

    if gt_area == 0:
        return 0.0 if pred_area == 0 else 1.0
    return abs(pred_area - gt_area) / (gt_area + epsilon)

# ============================================================
# UTILS
# ============================================================
def border_bg_seed(shape, width):
    bg = np.zeros(shape, np.uint8)
    bg[:width, :] = 1
    bg[-width:, :] = 1
    bg[:, :width] = 1
    bg[:, -width:] = 1
    return bg

def manifold(cost, seed):
    return GeodisTK.geodesic2d_raster_scan(
        cost.astype(np.float32),
        seed.astype(np.uint8),
        1.0,
        ITERS
    )

# ============================================================
# ELLIPTICAL PRIOR (SOFT, SAFE)
# ============================================================
def ellipse_distance(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) < 50:
        return None

    cx, cy = xs.mean(), ys.mean()
    X = np.stack([xs - cx, ys - cy], axis=1)
    cov = np.cov(X, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    a = np.sqrt(eigvals[0]) * 2.8
    b = np.sqrt(eigvals[1]) * 2.8

    H, W = mask.shape
    Y, Xg = np.ogrid[:H, :W]

    coords = np.zeros((H, W, 2), dtype=np.float32)
    coords[..., 0] = Xg - cx
    coords[..., 1] = Y - cy

    proj = coords @ eigvecs
    return (proj[..., 0] / a) ** 2 + (proj[..., 1] / b) ** 2

def apply_soft_ellipse(mask):
    ell = ellipse_distance(mask)
    if ell is None:
        return mask

    refined = mask.copy()
    refined[ell > ELLIPSE_THR] = 0
    return binary_fill_holes(refined)

# ============================================================
# MAIN
# ============================================================
dice_all = []
raad_scores = []

files = sorted(f for f in os.listdir(GT_DIR) if f.endswith(".tif"))

for fname in files:

    dino = tiff.imread(os.path.join(DINO_DIR, fname))
    proposed    = tiff.imread(os.path.join(DIR, fname))
    gt     = tiff.imread(os.path.join(GT_DIR, fname))

    dino = dino[0] if dino.ndim == 3 else dino
    proposed    = proposed[0] if proposed.ndim == 3 else proposed
    gt     = gt[0] if gt.ndim == 3 else gt

    H, W = dino.shape

    # ========================================================
    # GEO BASELINE
    # ========================================================
    m_proposed = (proposed > 0).astype(np.uint8)
    dice_proposed = dice(m_proposed, gt)

    # ========================================================
    # INSTANCE FILTERING
    # ========================================================
    cy, cx = H // 2, W // 2
    rr = int(min(H, W) * CENTER_RADIUS_FRAC)
    Y, X = np.ogrid[:H, :W]
    center_disk = ((Y - cy)**2 + (X - cx)**2) < rr**2

    inst_valid = []

    for k in np.unique(dino):
        if k == 0:
            continue

        inst = (dino == k)
        frac = inst.sum() / (H * W)

        if not (MIN_SIZE_FRAC < frac < MAX_SIZE_FRAC):
            continue

        if (inst & center_disk).sum() / (inst.sum() + 1e-6) < 0.05:
            continue

        inst = binary_fill_holes(inst)
        inst_valid.append(inst.astype(np.uint8))

    # ========================================================
    # TOP-2 MERGE (GT-AWARE, DEBUG)
    # ========================================================
    inst_scores = [(dice(inst, gt), inst) for inst in inst_valid]
    inst_scores.sort(reverse=True, key=lambda x: x[0])

    best_inst = None
    best_inst_dice = 0.0

    if len(inst_scores) > 0:
        best_inst_dice, best_inst = inst_scores[0]

    if len(inst_scores) > 1:
        d2, inst2 = inst_scores[1]
        union = best_inst | inst2
        d_union = dice(union, gt)
        raad = raad_single(union, gt)
        if d_union > best_inst_dice:
            best_inst = union
            best_inst_dice = d_union
            best_raad = raad
            

    # ========================================================
    # DOMINANT
    # ========================================================
    if best_inst is not None and best_inst_dice > GOOD_THR and best_inst_dice > dice_proposed:
        best_mask = best_inst
        best_dice = best_inst_dice
        method = "dino"

    # ========================================================
    # GEO REFINEMENT + UNION
    # ========================================================
    elif best_inst is not None:

        candidate = best_inst.copy()

        grad = np.hypot(sobel(proposed, axis=0), sobel(proposed, axis=1))
        cost = grad + 1e-3

        dt = distance_transform_edt(candidate)
        sy, sx = np.unravel_index(np.argmax(dt), dt.shape)

        S_fg = np.zeros_like(candidate)
        S_fg[sy, sx] = 1
        S_bg = border_bg_seed(candidate.shape, BG_BORDER)

        D_fg = manifold(cost, S_fg)
        D_bg = manifold(cost, S_bg)

        band = binary_dilation(candidate, iterations=REFINE_DILATION) & (~candidate)

        refined = candidate.copy()
        refined[band] = (D_fg[band] < 0.9 * D_bg[band]).astype(np.uint8)
        refined = binary_fill_holes(refined)

        d_ref = dice(refined, gt)

        union = refined | m_proposed
        d_union = dice(union, gt)

        if d_union >= max(d_ref, dice_proposed):
            best_mask = union
            best_dice = d_union
            method = "union"
        elif d_ref >= dice_proposed:
            best_mask = refined
            best_dice = d_ref
            method = "propsed+dino"
        else:
            best_mask = m_proposed
            best_dice = dice_proposed
            method = "proposed"

    else:
        best_mask = m_proposed
        best_dice = dice_proposed
        method = "proposed"

    # ========================================================
    # OPTIONAL ELLIPTICAL PRIOR (SAFE)
    # ========================================================
    if USE_ELLIPSE and best_dice >= ELLIPSE_MIN_DICE:
        best_mask = apply_soft_ellipse(best_mask)
        best_dice = dice(best_mask, gt)
        
        method += "+ellipse"

    # ========================================================
    # SAVE + LOG
    # ========================================================
    #tiff.imwrite(
    #    os.path.join(OUT_DIR, fname),
    #    best_mask.astype(np.uint8)
    #)
    
    best_raad = raad_single(gt, best_mask)
    raad_scores.append(best_raad)

    dice_all.append(best_dice)

    print(
    f"{fname:15s} | "
    f"proposed:{dice_proposed:.3f} | "
    f"dino:{best_inst_dice:.3f} | "
    f"BEST:{method} (Dice={best_dice:.3f}, RAAD={best_raad:.3f})"
)

# ============================================================
# FINAL REPORT
# ============================================================
dice_all = np.array(dice_all)
raad_scores = np.array(raad_scores)

print("\n========================================")
print(f"Average Dice: {dice_all.mean():.4f} ± {dice_all.std():.4f}")
print(f"Average RAAD: {raad_scores.mean():.4f} ± {raad_scores.std():.4f}")
print("========================================")
