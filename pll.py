# train_stage1.py
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from utils.dataset import LocalizationDataset2p5D
import os
import pandas as pd

# model_stage1.py
from utils.model import UNet2p5D_Attention

CKPT_DIR = os.path.join("outputs", "pkd", "weights")
LOG_DIR = os.path.join("outputs", "pkd", "logs")
ckpt_path = os.path.join(CKPT_DIR, "stage1_localization_best_nogt.pth")
METRICS_PATH = os.path.join(LOG_DIR, "stage1_localization_metrics_nogt.xlsx")

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def build_localization_model():
    return UNet2p5D_Attention(
        in_channels=7,
        out_channels=1
    )

device = "cuda" if torch.cuda.is_available() else "cpu"

root = "/users/PAS3110/sephora20/workspace/PDAC/data/pkd/"

# --------------------
# Model
# --------------------
model = build_localization_model().to(device)

# --------------------
# Data
# --------------------
train_ds = LocalizationDataset2p5D(
    root, split="train", num_slices=7
)
train_loader = DataLoader(
    train_ds, batch_size=4, shuffle=True, num_workers=4
)

# --------------------
# Optimizer
# --------------------
optimizer = AdamW(model.parameters(), lr=1e-4)
epochs = 30

# --------------------
# Metrics
# --------------------
best_loss = float("inf")
metrics = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],  # keep even if not used yet
}

# --------------------
# Training
# --------------------
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for images, heatmaps in tqdm(
        train_loader, desc=f"Epoch {epoch}"
    ):
        images = images.to(device)
        heatmaps = heatmaps.to(device)

        pred = model(images)

        loss = torch.mean((pred - heatmaps) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    print(
        f"Epoch {epoch:03d} | "
        f"Heatmap MSE: {avg_loss:.6f}"
    )

    # ---- SAVE METRICS (CONTINUOUS) ----
    metrics["epoch"].append(epoch)
    metrics["train_loss"].append(avg_loss)
    metrics["val_loss"].append(None)

    df = pd.DataFrame(metrics)
    df.to_excel(METRICS_PATH, index=False)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "loss": best_loss,
            },
            ckpt_path,
        )
        print(f"Saved best localization model to {ckpt_path}")

# --------------------
# Sanity check
# --------------------
model.eval()
with torch.no_grad():
    pred = model(images)
    print(pred.max(), pred.mean())
