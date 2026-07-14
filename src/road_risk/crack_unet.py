"""U-Net helpers for crack segmentation."""

from __future__ import annotations

import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from albumentations.pytorch import ToTensorV2
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from torch.utils.data import Dataset
from tqdm import tqdm


class CrackDataset(Dataset):
    """Image/mask dataset for binary crack segmentation."""

    def __init__(self, image_dir: str | Path, mask_dir: str | Path, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_paths = sorted(
            p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp") for p in self.image_dir.glob(ext)
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        mask_path = self.mask_dir / f"{image_path.stem}.png"

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask = (mask > 0).astype("float32")

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"].unsqueeze(0)
        return image, mask


def build_transforms(img_size: int = 512):
    """Return train and validation transforms."""
    train_transform = A.Compose(
        [
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose([A.Resize(img_size, img_size), A.Normalize(), ToTensorV2()])
    return train_transform, val_transform


def build_unet(encoder_name: str = "resnet34", encoder_weights: str | None = "imagenet"):
    """Build a binary U-Net segmentation model."""
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )


def build_loss():
    """Dice + Focal loss for small, imbalanced crack regions."""
    dice_loss = smp.losses.DiceLoss(mode="binary", from_logits=True)
    focal_loss = smp.losses.FocalLoss(mode="binary")

    def criterion(pred, target):
        return dice_loss(pred, target) + focal_loss(pred, target)

    return criterion


def calculate_metrics(preds, masks, threshold: float = 0.5) -> tuple[float, float]:
    """Return batch Dice and IoU."""
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * masks).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-7) / (union + 1e-7)
    iou = (intersection + 1e-7) / (
        preds.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) - intersection + 1e-7
    )
    return dice.mean().item(), iou.mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = total_dice = total_iou = 0.0
    for images, masks in tqdm(loader, desc="train"):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        dice, iou = calculate_metrics(outputs.detach(), masks)
        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

    n = max(len(loader), 1)
    return total_loss / n, total_dice / n, total_iou / n


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = total_dice = total_iou = 0.0
    for images, masks in tqdm(loader, desc="validate"):
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)

        dice, iou = calculate_metrics(outputs, masks)
        total_loss += loss.item()
        total_dice += dice
        total_iou += iou

    n = max(len(loader), 1)
    return total_loss / n, total_dice / n, total_iou / n


def polyline_json_to_mask(json_path: str | Path, image_shape: tuple[int, int], thickness: int = 5) -> np.ndarray:
    """Convert AI-Hub crack polyline annotations to a binary mask."""
    json_path = Path(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    annotations = data.get("annotations", data.get("annotation", []))
    for ann in annotations:
        points = ann.get("points") or ann.get("polyline") or ann.get("segmentation")
        if not points:
            continue
        points = np.asarray(points, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(mask, [points], isClosed=False, color=255, thickness=thickness)
    return mask


def extract_crack_metrics(mask_dir: str | Path) -> pd.DataFrame:
    """Calculate image-level crack metrics from predicted binary masks."""
    rows = []
    for mask_path in tqdm(sorted(Path(mask_dir).glob("*.png")), desc="crack metrics"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        binary = mask > 0
        h, w = binary.shape
        area_ratio = float(binary.sum() / (h * w))

        skeleton = skeletonize(binary)
        length_ratio = float(skeleton.sum() / max(h * w, 1))

        labeled = label(binary)
        props = regionprops(labeled)
        component_count = len(props)
        max_area_ratio = float(max((p.area for p in props), default=0) / (h * w))

        rows.append(
            {
                "image": mask_path.name,
                "crack_area_ratio": area_ratio,
                "crack_length_ratio": length_ratio,
                "component_count": component_count,
                "max_crack_area_ratio": max_area_ratio,
            }
        )
    return pd.DataFrame(rows)
