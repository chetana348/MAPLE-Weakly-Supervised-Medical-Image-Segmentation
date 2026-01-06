import torch
from torch.utils.data import DataLoader,Subset
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
import os
import pandas as pd

from utils.dataset import SegmentationDataset2p5D
from utils.model import UNet2p5D_Attention
from utils.losses import *

# =========================================================
# CONFIG
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

root = "/users/PAS3110/sephora20/workspace/PDAC/data/pkd/"
stage1_ckpt = "./outputs/pkd/weights/stage1_localization_best_nogt.pth"
CKPT_DIR = "./outputs/pkd/weights/"

num_slices = 7
epochs = 60
lr = 1e-4

lambda_compact = 0.001
lambda_hole = 0.0005
jitter_px = 8

LOG_DIR = "./outputs/pkd/logs"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
METRICS_PATH = os.path.join(LOG_DIR, "stage2_segmentation_metrics_nogt_new1.xlsx")
CKPT_PATH = os.path.join(CKPT_DIR, "stage2_segmentation_best_nogt_new1.pth")

# =========================================================
# LOAD STAGE-1 (FROZEN)
# =========================================================
stage1 = UNet2p5D_Attention(in_channels=7, out_channels=1)
ckpt = torch.load(stage1_ckpt, map_location=device)
stage1.load_state_dict(ckpt["model"])
stage1.to(device)
stage1.eval()

for p in stage1.parameters():
    p.requires_grad = False

print("Loaded frozen Stage-1 localization model")

# =========================================================
# STAGE-2 MODEL
# =========================================================
stage2 = UNet2p5D_Attention(in_channels=7).to(device)
optimizer = AdamW(stage2.parameters(), lr=lr)

# =========================================================
# DATA
# =========================================================
train_ds = SegmentationDataset2p5D(
    root, split="train", num_slices=num_slices
)
train_loader = DataLoader(
    train_ds, batch_size=2, shuffle=True, num_workers=4
)

val_ds = SegmentationDataset2p5D(
    root, split="test", num_slices=num_slices
)
#val_loader = DataLoader(
#    val_ds, batch_size=1, shuffle=False, num_workers=4
#)

val_ds_100 = Subset(val_ds, range(100)) 
val_loader = DataLoader( val_ds_100, batch_size=1, shuffle=False, num_workers=4 )

def crop_around_center(image, mask, cy, cx, crop_size):
    _, _, H, W = image.shape
    half = crop_size // 2

    cy = int(max(0, min(cy, H - 1)))
    cx = int(max(0, min(cx, W - 1)))

    y1 = max(0, cy - half)
    x1 = max(0, cx - half)
    y2 = min(H, y1 + crop_size)
    x2 = min(W, x1 + crop_size)

    y1 = max(0, y2 - crop_size)
    x1 = max(0, x2 - crop_size)

    return (
        image[:, :, y1:y2, x1:x2],
        mask[:, y1:y2, x1:x2]
    )


# =========================================================
# METRICS
# =========================================================
metrics = {
    "epoch": [],
    "crop_size": [],
    "train_loss": [],
    "val_dice": [],
}

# =========================================================
# TRAINING
# =========================================================
best_val_dice = 0.0

for epoch in range(epochs):

    if epoch < 20:
        crop_size = 160
    elif epoch < 40:
        crop_size = 128
    else:
        crop_size = 96

    stage2.train()
    epoch_loss = 0.0

    for images, pseudo in tqdm(train_loader, desc=f"Epoch {epoch}"):
        images = images.to(device)
        pseudo = pseudo.to(device)

        with torch.no_grad():
            heatmap = stage1(images)[:, 0]
            B, H, W = heatmap.shape
            idx = torch.argmax(heatmap.view(B, -1), dim=1)
            cy = idx // W
            cx = idx % W

        cropped_imgs = []
        cropped_pseudo = []

        for b in range(B):
            cy_j = cy[b].item() + np.random.randint(-jitter_px, jitter_px + 1)
            cx_j = cx[b].item() + np.random.randint(-jitter_px, jitter_px + 1)

            ci, cp = crop_around_center(
                images[b:b+1],
                pseudo[b:b+1],
                cy_j,
                cx_j,
                crop_size
            )
            cropped_imgs.append(ci)
            cropped_pseudo.append(cp)

        cropped_imgs = torch.cat(cropped_imgs, dim=0)
        cropped_pseudo = torch.cat(cropped_pseudo, dim=0)

        logits = stage2(cropped_imgs)
        prob = torch.sigmoid(logits)

        loss_pseudo = partial_ce_loss(
            logits,
            cropped_pseudo,
            confidence_mask=torch.ones_like(cropped_pseudo)
        )

        loss_shape = lambda_compact * compactness_loss(prob)
        loss_hole = lambda_hole * hole_penalty(prob)

        loss = loss_pseudo + loss_shape + loss_hole

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # =====================================================
    # VALIDATION
    # =====================================================
    stage2.eval()
    dices = []

    with torch.no_grad():
        for images, gt in val_loader:
            images = images.to(device)
            gt = gt.to(device)

            heatmap = stage1(images)[:, 0]
            _, H, W = heatmap.shape
            idx = torch.argmax(heatmap.view(-1))
            cy = (idx // W).item()
            cx = (idx % W).item()

            cropped_img, cropped_gt = crop_around_center(
                images, gt, cy, cx, crop_size
            )

            prob = torch.sigmoid(stage2(cropped_img))
            dices.append(dice_coefficient(prob, cropped_gt).item())

    val_dice = sum(dices) / len(dices)

    print(
        f"Epoch {epoch:03d} | "
        f"Crop {crop_size} | "
        f"Train Loss: {epoch_loss / len(train_loader):.4f} | "
        f"Val Dice: {val_dice:.4f}"
    )

    # ---------------- SAVE METRICS (CONTINUOUS) ----------------
    metrics["epoch"].append(epoch)
    metrics["crop_size"].append(crop_size)
    metrics["train_loss"].append(epoch_loss / len(train_loader))
    metrics["val_dice"].append(val_dice)

    df = pd.DataFrame(metrics)
    df.to_excel(METRICS_PATH, index=False)

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(
            {
                "epoch": epoch,
                "model": stage2.state_dict(),
                "dice": best_val_dice,
            },
            CKPT_PATH
        )
        print(f"Saved best model (Dice={best_val_dice:.4f})")
