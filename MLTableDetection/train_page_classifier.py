#!/usr/bin/env python3
"""
Fine-tune MobileNetV2 as a binary page classifier (panel / no panel).

Determines which PDF pages contain panel schedule tables without
bounding box annotations -- only page-level labels are needed.

Training data format (either):
  A) COCO annotations: images with bounding boxes → positive (has panel),
     images without → negative (no panel).
  B) Directory layout: positive/ and negative/ subdirectories.

Model: MobileNetV2 (BSD-3, torchvision) -- commercial-friendly.

Usage:
    # From COCO annotations (reuses existing table detection data)
    python MLTableDetection/train_page_classifier.py \
        --data ~/Documents/TableAnnotations

    # From directory layout
    python MLTableDetection/train_page_classifier.py \
        --data ~/Documents/ClassifierData \
        --format directory

    # Full control
    python MLTableDetection/train_page_classifier.py \
        --data ~/Documents/TableAnnotations \
        --output ~/Documents/TableAnnotations/models_classifier \
        --epochs 25 --learning-rate 5e-5
"""

import os
import sys
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import MobileNet_V2_Weights
from PIL import Image

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PageClassifierDataset(Dataset):
    """Image dataset with binary labels for page classification."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_from_coco(data_dir: str) -> tuple[list[str], list[int]]:
    """
    Derive page-level labels from COCO object annotations.

    Images that have at least one annotation are positive (1);
    images with zero annotations are negative (0).
    """
    data_dir = Path(data_dir)

    coco_path = None
    for candidate in [
        data_dir / "annotations.json",
        data_dir / "annotations" / "annotations_coco.json",
        data_dir / "annotations" / "annotations.json",
    ]:
        if candidate.exists():
            coco_path = candidate
            break

    if coco_path is None:
        raise FileNotFoundError(
            f"No COCO annotation file found in {data_dir}")

    with open(coco_path) as f:
        coco = json.load(f)

    images_dir = data_dir / "images"
    ids_with_ann = set(a["image_id"] for a in coco["annotations"])

    image_paths: list[str] = []
    labels: list[int] = []

    for img_info in coco["images"]:
        fname = os.path.basename(img_info["file_name"])
        fpath = images_dir / fname
        if not fpath.exists():
            print(f"  [SKIP] Missing: {fname}")
            continue
        image_paths.append(str(fpath))
        labels.append(1 if img_info["id"] in ids_with_ann else 0)

    return image_paths, labels


def load_from_directory(data_dir: str) -> tuple[list[str], list[int]]:
    """
    Load images from positive/ and negative/ subdirectories.
    """
    data_dir = Path(data_dir)
    pos_dir = data_dir / "positive"
    neg_dir = data_dir / "negative"

    if not pos_dir.exists() or not neg_dir.exists():
        raise FileNotFoundError(
            f"Expected {pos_dir} and {neg_dir} directories")

    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    image_paths: list[str] = []
    labels: list[int] = []

    for p in sorted(pos_dir.iterdir()):
        if p.suffix.lower() in extensions:
            image_paths.append(str(p))
            labels.append(1)

    for p in sorted(neg_dir.iterdir()):
        if p.suffix.lower() in extensions:
            image_paths.append(str(p))
            labels.append(0)

    return image_paths, labels


def stratified_split(
    image_paths: list[str],
    labels: list[int],
    val_fraction: float = 0.20,
    seed: int = 42,
) -> tuple[list[str], list[int], list[str], list[int]]:
    """Split into train/val while preserving class ratios."""
    rng = random.Random(seed)

    pos_idx = [i for i, l in enumerate(labels) if l == 1]
    neg_idx = [i for i, l in enumerate(labels) if l == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    val_pos = max(1, round(len(pos_idx) * val_fraction))
    val_neg = max(1, round(len(neg_idx) * val_fraction)) if len(neg_idx) >= 3 else 0

    val_indices = pos_idx[:val_pos] + neg_idx[:val_neg]
    train_indices = pos_idx[val_pos:] + neg_idx[val_neg:]

    train_paths = [image_paths[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_paths = [image_paths[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    return train_paths, train_labels, val_paths, val_labels


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def build_train_transforms(input_size: int = 224):
    """Gentle augmentations suited to document page images."""
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomRotation(3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_val_transforms(input_size: int = 224):
    """Deterministic resize + center crop for validation."""
    return transforms.Compose([
        transforms.Resize(input_size + 32),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

NUM_CLASSES = 2
DROPOUT_RATE = 0.3
UNFREEZE_LAST_N_BLOCKS = 3


def build_model(freeze_backbone: bool = True) -> nn.Module:
    """MobileNetV2 with a 2-class classification head."""
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.features[-UNFREEZE_LAST_N_BLOCKS:].parameters():
            param.requires_grad = True

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(in_features, NUM_CLASSES),
    )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_classifier(
    data_dir: str,
    output_dir: str = "models/page_classifier",
    data_format: str = "coco",
    input_size: int = 224,
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    val_split: float = 0.20,
    freeze_backbone: bool = True,
    verbose: bool = True,
) -> str:
    """
    Fine-tune MobileNetV2 as a binary page classifier.

    Returns:
        Path to the saved best model checkpoint.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- load data ----
    if data_format == "coco":
        image_paths, labels = load_from_coco(data_dir)
    else:
        image_paths, labels = load_from_directory(data_dir)

    pos_count = sum(labels)
    neg_count = len(labels) - pos_count

    if verbose:
        print(f"[INFO] Dataset: {len(labels)} images "
              f"({pos_count} positive, {neg_count} negative)")

    if len(labels) < 10:
        print("[WARN] Very few images. Results may be unreliable.")

    # ---- split ----
    train_paths, train_labels, val_paths, val_labels = stratified_split(
        image_paths, labels, val_split)

    if verbose:
        t_pos, v_pos = sum(train_labels), sum(val_labels)
        print(f"[INFO] Train: {len(train_labels)} "
              f"({t_pos} pos, {len(train_labels) - t_pos} neg)")
        print(f"[INFO] Val:   {len(val_labels)} "
              f"({v_pos} pos, {len(val_labels) - v_pos} neg)")

    # ---- datasets & loaders ----
    train_ds = PageClassifierDataset(
        train_paths, train_labels, build_train_transforms(input_size))
    val_ds = PageClassifierDataset(
        val_paths, val_labels, build_val_transforms(input_size))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    # ---- class weights (inverse frequency) ----
    train_pos = sum(train_labels)
    train_neg = len(train_labels) - train_pos
    n_train = len(train_labels)
    if train_neg > 0 and train_pos > 0:
        class_weights = torch.tensor(
            [n_train / (2.0 * train_neg),
             n_train / (2.0 * train_pos)],
            dtype=torch.float32,
        ).to(device)
    else:
        class_weights = None

    # ---- model ----
    model = build_model(freeze_backbone=freeze_backbone)
    model.to(device)

    if verbose:
        trainable = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[INFO] MobileNetV2 ({total / 1e6:.1f}M total, "
              f"{trainable / 1e6:.1f}M trainable)")
        print(f"[INFO] Device: {device}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs)

    # ---- training loop ----
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.detach().argmax(1) == targets).sum().item()
            train_total += targets.size(0)

        scheduler.step()
        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp = val_fp = val_tn = val_fn = 0

        with torch.no_grad():
            for images, targets in val_loader:
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

                for pred, tgt in zip(preds.cpu(), targets.cpu()):
                    p, t = pred.item(), tgt.item()
                    if t == 1 and p == 1:
                        val_tp += 1
                    elif t == 0 and p == 1:
                        val_fp += 1
                    elif t == 0 and p == 0:
                        val_tn += 1
                    else:
                        val_fn += 1

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        if verbose:
            print(f"  Epoch {epoch:02d}/{epochs}: "
                  f"loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
                  f"(TP={val_tp} FP={val_fp} TN={val_tn} FN={val_fn})")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_dir = output_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)

            torch.save({
                "model_state_dict": model.state_dict(),
                "config": {
                    "architecture": "mobilenet_v2",
                    "num_classes": NUM_CLASSES,
                    "input_size": input_size,
                    "class_names": ["no_panel", "has_panel"],
                },
                "metrics": {
                    "val_accuracy": val_acc,
                    "val_tp": val_tp,
                    "val_fp": val_fp,
                    "val_tn": val_tn,
                    "val_fn": val_fn,
                    "epoch": epoch,
                },
                "training_info": {
                    "data_dir": str(data_dir),
                    "total_images": len(labels),
                    "positive_count": pos_count,
                    "negative_count": neg_count,
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "freeze_backbone": freeze_backbone,
                },
            }, str(best_dir / "model.pt"))

            if verbose:
                print(f"    -> Saved best model (val_acc={val_acc:.3f})")

    if verbose:
        print(f"\n[DONE] Best: epoch {best_epoch}, "
              f"val_acc={best_val_acc:.3f}")
        print(f"       Model: {output_dir / 'best' / 'model.pt'}")

    return str(output_dir / "best" / "model.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train a binary page classifier "
                    "(MobileNetV2, BSD-3 license)")
    parser.add_argument(
        "--data", "-d",
        default=os.path.expanduser("~/Documents/TableAnnotations"),
        help="Training data directory")
    parser.add_argument(
        "--output", "-o",
        default=os.path.expanduser(
            "~/Documents/TableAnnotations/models_classifier"),
        help="Output directory for trained model")
    parser.add_argument(
        "--format", "-f", choices=["coco", "directory"], default="coco",
        help="Data format (default: coco)")
    parser.add_argument(
        "--epochs", "-e", type=int, default=20)
    parser.add_argument(
        "--batch-size", "-b", type=int, default=8)
    parser.add_argument(
        "--learning-rate", "-lr", type=float, default=1e-4)
    parser.add_argument(
        "--input-size", type=int, default=224)
    parser.add_argument(
        "--val-split", type=float, default=0.20)
    parser.add_argument(
        "--no-freeze", action="store_true",
        help="Train all layers (not just head + last 3 blocks)")
    parser.add_argument(
        "--quiet", "-q", action="store_true")

    args = parser.parse_args()

    try:
        model_path = train_classifier(
            data_dir=os.path.expanduser(args.data),
            output_dir=os.path.expanduser(args.output),
            data_format=args.format,
            input_size=args.input_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_split=args.val_split,
            freeze_backbone=not args.no_freeze,
            verbose=not args.quiet,
        )
        print(f"\n[SUCCESS] Model saved to: {model_path}")
        return 0
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
