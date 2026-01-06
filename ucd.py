import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
import os
import pandas as pd

from utils.dataset import SegmentationDataset2p5D
from utils.model import UNet2p5D_Attention
from utils.losses import *

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = "/users/PAS3110/sephora20/workspace/PDAC/data/pkd/"
NUM_SLICES = 7

BATCH_SIZE = 2
EPOCHS = 60
LR = 2e-5

MC_SAMPLES = 6

LAMBDA_SLICE = 0.5
LAMBDA_SHAPE = 0.3

CKPT_TEACHER = "/users/PAS3110/sephora20/workspace/PDAC/unsup/stage1/outputs/pdac_mri/1point_nogt/weights/stage2_segmentation_best_nogt_new1.pth"
CKPT_DIR = "outputs/pkd/weights"

LOG_DIR = "outputs/pkd/logs"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
METRICS_PATH = os.path.join(LOG_DIR, "stage3_segmentation_metrics_nogt_new1.xlsx")
CKPT_PATH = os.path.join(CKPT_DIR, "stage3_segmentation_best_nogt_new1.pth")

# ============================================================
# DATA
# ============================================================
train_ds = SegmentationDataset2p5D(
    root_dir=ROOT_DIR,
    split="train",
    num_slices=NUM_SLICES
)

val_ds = SegmentationDataset2p5D(
    root_dir=ROOT_DIR,
    split="test",
    num_slices=NUM_SLICES
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

#val_loader = DataLoader(
#    val_ds,
#    batch_size=1,
#    shuffle=False,
#    num_workers=2,
#    pin_memory=True
#)

val_ds_100 = Subset(val_ds, range(100)) 
val_loader = DataLoader( val_ds_100, batch_size=1, shuffle=False, num_workers=4 )

# ============================================================
# UNCERTAINTY (MC DROPOUT)
# ============================================================
@torch.no_grad()
def mc_uncertainty(model, x, T=6):
    model.train()
    preds = []

    for _ in range(T):
        preds.append(torch.sigmoid(model(x)))

    preds = torch.stack(preds, dim=0)
    mean = preds.mean(0)
    var = preds.var(0)

    model.eval()
    return mean, var


# ============================================================
# VALIDATION (USES GT)
# ============================================================
@torch.no_grad()
def validate(model):
    model.eval()
    dices = []

    for x, y in val_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).float()

        y = y.unsqueeze(1).float()

        inter = (pred * y).sum(dim=(1, 2, 3))
        union = pred.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))

        dice = (2 * inter + 1e-6) / (union + 1e-6)
        dices.append(dice.item())

    return float(np.mean(dices))

# ============================================================
# TRAIN
# ============================================================
def train():
    student = UNet2p5D_Attention(in_channels=NUM_SLICES).to(DEVICE)
    teacher = copy.deepcopy(student).to(DEVICE)

    ckpt = torch.load(CKPT_TEACHER, map_location=DEVICE)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt

    student.load_state_dict(state_dict, strict=True)
    teacher.load_state_dict(state_dict, strict=True)

    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(student.parameters(), lr=LR)

    best_val_dice = 0.0

    # ---------------- METRICS ----------------
    metrics = {
        "epoch": [],
        "train_loss": [],
        "val_dice": [],
    }

    for epoch in range(EPOCHS):
        student.train()
        teacher.eval()

        epoch_loss = 0.0

        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x = x.to(DEVICE)

            with torch.no_grad():
                teacher_mean, teacher_var = mc_uncertainty(
                    teacher, x, T=MC_SAMPLES
                )

            student_prob = torch.sigmoid(student(x))

            L_cons = consistency_loss(
                student_prob, teacher_mean, teacher_var
            )
            L_slice = slice_consistency_loss(student_prob)
            L_shape = shape_prior_loss(student_prob)

            loss = (
                L_cons +
                LAMBDA_SLICE * L_slice +
                LAMBDA_SHAPE * L_shape
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader)
        val_dice = validate(student)

        print(
            f"[Epoch {epoch+1}] "
            f"TrainLoss={avg_train_loss:.4f} | "
            f"ValDice={val_dice:.4f}"
        )

        # -------- SAVE METRICS CONTINUOUSLY --------
        metrics["epoch"].append(epoch + 1)
        metrics["train_loss"].append(avg_train_loss)
        metrics["val_dice"].append(val_dice)

        pd.DataFrame(metrics).to_excel(METRICS_PATH, index=False)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(student.state_dict(), CKPT_PATH)
            print("Saved best model (Val Dice)")

    print(f"\nBest Validation Dice: {best_val_dice:.4f}")

# ============================================================
if __name__ == "__main__":
    train()
