import os
import torch
import numpy as np
import tifffile as tiff
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dataset import SegmentationDataset2p5D
from utils.model import UNet2p5D_Attention

def raad_batch(y_true_batch, y_pred_batch, epsilon=1e-6): 
    batch_ravd = [] 
    for y_true, y_pred in zip(y_true_batch, y_pred_batch): 
        gt_area = np.sum(y_true == 1) 
        pred_area = np.sum(y_pred == 1) 
        if gt_area == 0: 
            ravd = 0.0 if pred_area == 0 else 1.0 # handle empty ground truth 
        else: 
            ravd = abs(pred_area - gt_area) / (gt_area + epsilon) 
        batch_ravd.append(ravd) 
    return np.array(batch_ravd)

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = "/users/PAS3110/sephora20/workspace/PDAC/data/pkd/"
NUM_SLICES = 7

BATCH_SIZE = 1
CKPT_PATH = "/users/PAS3110/sephora20/workspace/PDAC/unsup/stage1/outputs/pkd/other_trains/uncer_best.pth"

SAVE_DIR = "/users/PAS3110/sephora20/workspace/PDAC/unsup/stage1/outputs/pkd/other_trains/"
#os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# DATA
# ============================================================
test_ds = SegmentationDataset2p5D(
    root_dir=ROOT_DIR,
    split="test",
    num_slices=NUM_SLICES,
    return_name=True     
)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

# ============================================================
# DICE (exactly your formula)
# ============================================================
@torch.no_grad()
def dice_score(pred, y):
    inter = (pred * y).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
    dice = (2 * inter + 1e-6) / (union + 1e-6)
    return dice


# ============================================================
# TEST
# ============================================================
def test():
    model = UNet2p5D_Attention(in_channels=NUM_SLICES).to(DEVICE)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    state_dict = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    
    #model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
    model.eval()

    dices = []
    raads = []

    for x, y, name in tqdm(test_loader, desc="Testing"):
        x = x.to(DEVICE)
        y = y.to(DEVICE).unsqueeze(1).float()

        logits = model(x)
        prob = torch.sigmoid(logits)
        pred = (prob > 0.5).float()

        # Dice
        dice = dice_score(pred, y)
        dices.append(dice.item())
        
        # RAAD (per-sample)
        y_np = y[0, 0].cpu().numpy().astype(np.uint8)
        pred_np_bin = pred[0, 0].cpu().numpy().astype(np.uint8)
        raad = raad_batch(y_np[None, ...], pred_np_bin[None, ...])[0]
        raads.append(raad)

        # Save prediction
        pred_np = (pred[0, 0].cpu().numpy() * 255).astype(np.uint8)

        save_path = os.path.join(SAVE_DIR, name[0])
        #tiff.imwrite(save_path, pred_np)

    dices = np.array(dices)
    raads = np.array(raads)
    
    print(
        f"\nDice:  {dices.mean():.4f} ± {dices.std():.4f}\n"
        f"RAAD:  {raads.mean():.4f} ± {raads.std():.4f}"
    )



# ============================================================
if __name__ == "__main__":
    test()
