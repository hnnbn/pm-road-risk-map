"""YOLO preprocessing and scoring helpers for pothole/manhole detection."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


CLASS_MAP = {
    8: 0,   # AI-Hub pothole -> YOLO pothole
    10: 1,  # AI-Hub manhole -> YOLO manhole
}


def coco_bbox_to_yolo(bbox: list[float], image_width: int, image_height: int) -> tuple[float, float, float, float]:
    """Convert COCO [x, y, width, height] bbox to normalized YOLO format."""
    x, y, w, h = bbox
    return (
        (x + w / 2) / image_width,
        (y + h / 2) / image_height,
        w / image_width,
        h / image_height,
    )


def convert_aihub_json_to_yolo(
    json_dir: str | Path,
    label_dir: str | Path,
    class_map: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Convert AI-Hub annotation JSON files into YOLO label text files.

    Returns an annotation summary that is useful for sanity checks and reporting.
    """
    json_dir = Path(json_dir)
    label_dir = Path(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    class_map = class_map or CLASS_MAP

    rows = []
    for json_path in tqdm(sorted(json_dir.glob("*.json")), desc="convert labels"):
        data = json.loads(json_path.read_text(encoding="utf-8"))
        image_info = data.get("image", data.get("images", [{}])[0])
        width = int(image_info.get("width") or data.get("width"))
        height = int(image_info.get("height") or data.get("height"))
        image_name = image_info.get("filename") or image_info.get("file_name") or json_path.with_suffix(".jpg").name

        yolo_lines = []
        annotations = data.get("annotations", data.get("annotation", []))
        for ann in annotations:
            category_id = int(ann.get("category_id", ann.get("category", -1)))
            if category_id not in class_map or "bbox" not in ann:
                continue
            x_center, y_center, box_w, box_h = coco_bbox_to_yolo(ann["bbox"], width, height)
            yolo_class = class_map[category_id]
            yolo_lines.append(f"{yolo_class} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
            rows.append(
                {
                    "image": image_name,
                    "source_json": json_path.name,
                    "category_id": category_id,
                    "class_id": yolo_class,
                    "bbox_area_ratio": (box_w * box_h),
                }
            )

        (label_dir / f"{Path(image_name).stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")

    return pd.DataFrame(rows)


def write_split_files(
    image_dir: str | Path,
    output_dir: str | Path,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> dict[str, Path]:
    """Write train/val/test image list files for YOLO training."""
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp") for p in image_dir.glob(ext)
    )
    train_val, test = train_test_split(images, test_size=test_size, random_state=seed)
    val_ratio = val_size / (1 - test_size)
    train, val = train_test_split(train_val, test_size=val_ratio, random_state=seed)

    paths = {}
    for name, split_images in {"train": train, "val": val, "test": test}.items():
        path = output_dir / f"{name}.txt"
        path.write_text("\n".join(str(p) for p in split_images), encoding="utf-8")
        paths[name] = path
    return paths


def summarize_yolo_predictions(
    pred_label_dir: str | Path,
    image_dir: str | Path,
    image_width: int = 512,
    image_height: int = 512,
) -> pd.DataFrame:
    """Summarize YOLO prediction txt files into image-level damage features."""
    pred_label_dir = Path(pred_label_dir)
    image_dir = Path(image_dir)
    image_files = sorted(
        p for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp") for p in image_dir.glob(ext)
    )

    rows = []
    for image_path in image_files:
        label_path = pred_label_dir / f"{image_path.stem}.txt"
        pothole_count = 0
        manhole_count = 0
        pothole_area_ratio = 0.0
        max_pothole_area_ratio = 0.0

        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id = int(float(parts[0]))
                box_w = float(parts[3])
                box_h = float(parts[4])
                area_ratio = box_w * box_h
                if class_id == 0:
                    pothole_count += 1
                    pothole_area_ratio += area_ratio
                    max_pothole_area_ratio = max(max_pothole_area_ratio, area_ratio)
                elif class_id == 1:
                    manhole_count += 1

        rows.append(
            {
                "image": image_path.name,
                "pothole_count": pothole_count,
                "manhole_count": manhole_count,
                "pothole_area_ratio": pothole_area_ratio,
                "max_pothole_area_ratio": max_pothole_area_ratio,
            }
        )

    return pd.DataFrame(rows)
