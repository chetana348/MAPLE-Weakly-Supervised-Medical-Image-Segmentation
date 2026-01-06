import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile as tiff


class SegmentationDataset2p5D(Dataset):
    def __init__(
        self,
        root_dir,
        split="train",
        num_slices=7,
        transform=None,
        return_name=False,
    ):
        """
        Args:
            root_dir (str): path to main dataset directory
            split (str): 'train' or 'test'
            num_slices (int): must be odd (e.g., 5, 7, 9)
            transform: optional transform applied to image stack
            return_name (bool): whether to return filename
        """
        assert num_slices % 2 == 1, "num_slices must be odd"
        assert split in ["train", "test"]

        self.root_dir = root_dir
        self.split = split
        self.num_slices = num_slices
        self.half = num_slices // 2
        self.transform = transform
        self.return_name = return_name

        self.image_dir = os.path.join(root_dir, split, "images")

        if split == "train":
            self.label_dir = os.path.join(root_dir, split, "1point_nogt/pseudo_fuse")
        else:
            self.label_dir = os.path.join(root_dir, split, "labels")

        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith(".tif") or f.endswith(".tiff")
        ])

        assert len(self.image_files) > 0, "No TIFF images found"

    def __len__(self):
        return len(self.image_files)

    def _load_slice(self, idx):
        idx = max(0, min(idx, len(self.image_files) - 1))
        path = os.path.join(self.image_dir, self.image_files[idx])
        img = tiff.imread(path).astype(np.float32)
        if img.ndim == 3:
            img = img[0]
        return img

    def __getitem__(self, idx):
        # -------- Load 2.5D stack --------
        slices = []
        for offset in range(-self.half, self.half + 1):
            slice_img = self._load_slice(idx + offset)
            slices.append(slice_img)

        image_stack = np.stack(slices, axis=0)  # (C, H, W)

        # -------- Normalize --------
        image_stack = (image_stack - image_stack.min()) / (
            image_stack.max() - image_stack.min() + 1e-6
        )

        # -------- Load label (center slice only) --------
        label_path = os.path.join(self.label_dir, self.image_files[idx])
        label = tiff.imread(label_path).astype(np.uint8)
        if label.ndim == 3:
            label = label[0]
        label[label>0] = 1

        # -------- To tensor --------
        image_stack = torch.from_numpy(image_stack)  # (C, H, W)
        label = torch.from_numpy(label).long()       # (H, W)

        if self.transform:
            image_stack = self.transform(image_stack)

        if self.return_name:
            return image_stack, label, self.image_files[idx]

        return image_stack, label
        



class LocalizationDataset2p5D(Dataset):
    def __init__(self, root, split="train", num_slices=7):
        self.root = root
        self.split = split
        self.num_slices = num_slices
        self.half = num_slices // 2

        self.image_dir = f"{root}/{split}/images"
        self.hm_dir = f"{root}/{split}/1point_nogt/heatmaps_1p_nogt"

        self.files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith(".tif")
        ])

    def _load_slice(self, idx):
        idx = max(0, min(idx, len(self.files) - 1))
        img = tiff.imread(os.path.join(self.image_dir, self.files[idx]))
        if img.ndim == 3:
            img = img[0]
        return img.astype(np.float32)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # ---- 2.5D image ----
        slices = []
        for o in range(-self.half, self.half + 1):
            slices.append(self._load_slice(idx + o))

        image = np.stack(slices, axis=0)
        image = (image - image.min()) / (image.max() - image.min() + 1e-6)
        image = torch.from_numpy(image)

        heatmap = tiff.imread(
            os.path.join(self.hm_dir, self.files[idx])
        ).astype(np.float32)

        heatmap = torch.from_numpy(heatmap).unsqueeze(0)

        return image, heatmap
