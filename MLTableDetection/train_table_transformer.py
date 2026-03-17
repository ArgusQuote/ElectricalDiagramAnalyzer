#!/usr/bin/env python3
"""
Fine-tuning script for Table Transformer (TATR) on custom table data.

Table Transformer is MIT licensed and pretrained on PubTables-1M.
It often works well zero-shot, but fine-tuning can improve accuracy
on domain-specific tables (like electrical panel schedules).

Features:
    - Albumentations data augmentation (tuned for document tables)
    - Automatic train/validation split from a single COCO annotation file
    - Validation mAP tracking via torchmetrics (best-model selection)
    - Small-dataset hyperparameter defaults (< 50 images)

Usage:
    # Fine-tune with small-dataset defaults
    python train_table_transformer.py --data ~/Documents/TableAnnotations

    # Full control over hyperparameters
    python train_table_transformer.py \
        --data ~/Documents/TableAnnotations \
        --output ~/Documents/TableAnnotations/models_v2 \
        --epochs 25 --learning-rate 5e-6 --weight-decay 0.05 \
        --warmup-steps 100 --val-split 0.15

    # Generate Colab notebook for training
    python train_table_transformer.py --colab-notebook
"""

import os
import argparse
import random
from pathlib import Path
from typing import Optional
import json


def check_gpu():
    """Check if GPU is available and print info."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"[GPU] {gpu_name} ({gpu_memory:.1f} GB)")
            return True
        else:
            print("[GPU] No CUDA GPU available")
            return False
    except ImportError:
        print("[GPU] PyTorch not installed")
        return False


def convert_yolo_to_coco(yolo_dir: str, output_path: str) -> dict:
    """
    Convert YOLO format annotations to COCO format for Table Transformer.

    Args:
        yolo_dir: Directory with images/ and labels/ subdirs (YOLO format).
        output_path: Path to save COCO JSON file.

    Returns:
        COCO format dictionary.
    """
    from PIL import Image

    yolo_dir = Path(yolo_dir)
    images_dir = yolo_dir / "images"
    labels_dir = yolo_dir / "labels"

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "table"}]
    }

    ann_id = 0

    for img_idx, img_path in enumerate(sorted(images_dir.glob("*.png"))):
        with Image.open(img_path) as img:
            width, height = img.size

        coco["images"].append({
            "id": img_idx,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })

        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cx, cy, w, h = map(float, parts[1:5])

                        x = (cx - w / 2) * width
                        y = (cy - h / 2) * height
                        box_w = w * width
                        box_h = h * height

                        coco["annotations"].append({
                            "id": ann_id,
                            "image_id": img_idx,
                            "category_id": 0,
                            "bbox": [x, y, box_w, box_h],
                            "area": box_w * box_h,
                            "iscrowd": 0,
                        })
                        ann_id += 1

    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)

    print(f"[INFO] Converted {len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations")
    return coco


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def build_train_augmentation():
    """
    Build an Albumentations pipeline for panel schedule training images.

    Tuned for electrical drawings: avoids large rotations (tables are
    axis-aligned), vertical flips (never upside-down), and hue/saturation
    shifts (drawings are near-grayscale).
    """
    import albumentations as A

    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.3,
            ),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            A.RandomScale(scale_limit=0.2, p=0.3),
            A.Rotate(limit=2, p=0.1, border_mode=0),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_ids"],
            min_area=1000,
            min_visibility=0.5,
        ),
    )


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def split_coco_data(
    coco_data: dict,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[dict, dict]:
    """
    Split a COCO-format dict into train and validation subsets by image.

    The split is random but reproducible via *seed*. Annotations follow
    their parent image into the appropriate split.

    Returns:
        (train_coco, val_coco) dicts in COCO format.
    """
    images = list(coco_data["images"])
    rng = random.Random(seed)
    rng.shuffle(images)

    val_count = max(1, round(len(images) * val_fraction))
    val_images = images[:val_count]
    train_images = images[val_count:]

    val_image_ids = {img["id"] for img in val_images}

    train_anns = [a for a in coco_data["annotations"]
                  if a["image_id"] not in val_image_ids]
    val_anns = [a for a in coco_data["annotations"]
                if a["image_id"] in val_image_ids]

    base = {k: v for k, v in coco_data.items()
            if k not in ("images", "annotations")}

    train_coco = {**base, "images": train_images, "annotations": train_anns}
    val_coco = {**base, "images": val_images, "annotations": val_anns}
    return train_coco, val_coco


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def _make_dataset_class():
    """
    Factory that returns the TableDataset class.

    Wrapped in a function so the heavy imports (torch, PIL, numpy) happen
    only when training is actually requested.
    """
    import torch
    from PIL import Image
    import numpy as np

    class TableDataset(torch.utils.data.Dataset):
        """
        COCO-format dataset for DETR-family table detection models.

        Optionally applies Albumentations augmentation *before* the HF
        image processor so that normalization/resizing is not duplicated.
        """

        def __init__(self, coco_data, images_dir, processor, cat_id_map,
                     augmentation=None):
            self.coco_data = coco_data
            self.images_dir = Path(images_dir)
            self.processor = processor
            self.cat_id_map = cat_id_map
            self.augmentation = augmentation

            self.img_to_anns: dict[int, list] = {}
            for ann in coco_data["annotations"]:
                self.img_to_anns.setdefault(ann["image_id"], []).append(ann)

        def __len__(self):
            return len(self.coco_data["images"])

        def __getitem__(self, idx):
            img_info = self.coco_data["images"][idx]
            img_path = self.images_dir / img_info["file_name"]

            image = Image.open(img_path).convert("RGB")
            img_w, img_h = image.size

            anns = self.img_to_anns.get(img_info["id"], [])

            # COCO-format boxes: [x, y, w, h] absolute pixels
            bboxes = []
            category_ids = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                bboxes.append([x, y, w, h])
                category_ids.append(
                    self.cat_id_map.get(ann["category_id"], 0))

            # --- augmentation (train only) ---
            if self.augmentation is not None and bboxes:
                img_np = np.array(image)
                augmented = self.augmentation(
                    image=img_np,
                    bboxes=bboxes,
                    category_ids=category_ids,
                )
                image = Image.fromarray(augmented["image"])
                bboxes = augmented["bboxes"]
                category_ids = augmented["category_ids"]
                img_w, img_h = image.size

            # Convert COCO [x, y, w, h] -> normalised [cx, cy, w, h]
            norm_boxes = []
            for (x, y, w, h) in bboxes:
                norm_boxes.append([
                    (x + w / 2) / img_w,
                    (y + h / 2) / img_h,
                    w / img_w,
                    h / img_h,
                ])

            target = {
                "boxes": torch.tensor(norm_boxes, dtype=torch.float32)
                         if norm_boxes
                         else torch.zeros((0, 4), dtype=torch.float32),
                "class_labels": torch.tensor(category_ids,
                                             dtype=torch.int64),
                "orig_size": torch.tensor([img_h, img_w]),
            }

            encoding = self.processor(images=image, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze(0)

            return {
                "pixel_values": pixel_values,
                "labels": target,
            }

    return TableDataset


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def build_compute_metrics(processor, model_config):
    """
    Return a ``compute_metrics`` callable compatible with HF Trainer's
    ``batch_eval_metrics=True`` mode.

    Uses ``torchmetrics.detection.mean_ap.MeanAveragePrecision`` to track
    mAP@0.5, mAP@0.5:0.95, precision, and recall.
    """
    import torch
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    batch_store: list[dict] = []

    def _denorm_boxes(boxes, width, height):
        """Scale normalised [cx, cy, w, h] to absolute [x1, y1, x2, y2]."""
        out = boxes.clone().float()
        out[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * width
        out[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * height
        out[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * width
        out[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * height
        return out

    def compute_metrics(eval_pred, compute_result):
        (loss_dict, scores, pred_boxes,
         last_hidden_state, encoder_last_hidden_state), labels = eval_pred

        preds_list = []
        target_list = []

        for score, pbox, label in zip(scores, pred_boxes, labels):
            h, w = label["orig_size"]
            h, w = float(h), float(w)

            # --- predictions ---
            pred_scores = torch.softmax(score[:, :-1], dim=-1)
            pred_labels = pred_scores.argmax(dim=-1)
            pred_conf = pred_scores.gather(
                1, pred_labels.unsqueeze(-1)).squeeze(-1)

            pred_abs = _denorm_boxes(pbox, w, h)
            preds_list.append({
                "boxes": pred_abs,
                "scores": pred_conf,
                "labels": pred_labels,
            })

            # --- ground truth ---
            gt_boxes = _denorm_boxes(label["boxes"], w, h)
            target_list.append({
                "boxes": gt_boxes,
                "labels": label["class_labels"],
            })

        if not compute_result:
            batch_store.append({
                "preds": preds_list, "target": target_list})
            return {}

        # Aggregate all batches and compute final metrics
        all_preds, all_targets = [], []
        for batch in batch_store:
            all_preds.extend(batch["preds"])
            all_targets.extend(batch["target"])

        metric = MeanAveragePrecision(box_format="xyxy", class_metrics=False)
        metric.update(preds=all_preds, target=all_targets)
        result = metric.compute()

        batch_store.clear()

        return {
            "map": round(float(result["map"]), 4),
            "map_50": round(float(result["map_50"]), 4),
            "map_75": round(float(result["map_75"]), 4),
            "mar_100": round(float(result["mar_100"]), 4),
        }

    return compute_metrics


# ---------------------------------------------------------------------------
# Core training function
# ---------------------------------------------------------------------------

def fine_tune_tatr(
    train_data_dir: str,
    output_dir: str = "models/tatr_finetuned",
    model_name: str = "microsoft/table-transformer-detection",
    epochs: int = 25,
    batch_size: int = 2,
    learning_rate: float = 5e-6,
    weight_decay: float = 0.05,
    warmup_steps: int = 100,
    val_split: float = 0.15,
    augment: bool = True,
    verbose: bool = True,
) -> str:
    """
    Fine-tune Table Transformer on custom table data.

    Args:
        train_data_dir: Directory with training data (YOLO or COCO format).
        output_dir: Directory to save fine-tuned model.
        model_name: Base model to fine-tune.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        weight_decay: Weight decay for regularisation.
        warmup_steps: Linear LR warmup steps.
        val_split: Fraction of images held out for validation (0 to disable).
        augment: Apply Albumentations augmentation to training data.
        verbose: Print progress.

    Returns:
        Path to the best fine-tuned model.
    """
    import torch
    from transformers import (
        TableTransformerForObjectDetection,
        AutoImageProcessor,
        TrainingArguments,
        Trainer,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- locate annotations ------------------------------------------------
    train_dir = Path(train_data_dir)

    coco_path = None
    possible_paths = [
        train_dir / "annotations.json",
        train_dir / "annotations" / "annotations_coco.json",
        train_dir / "annotations" / "annotations.json",
    ]
    for p in possible_paths:
        if p.exists():
            coco_path = p
            break

    if coco_path is None:
        if (train_dir / "images").exists() and (train_dir / "labels").exists():
            if verbose:
                print("[INFO] Converting YOLO format to COCO...")
            coco_path = train_dir / "annotations.json"
            convert_yolo_to_coco(str(train_dir), str(coco_path))
        else:
            raise FileNotFoundError(
                f"No COCO annotations found and no YOLO format detected "
                f"in {train_dir}\n"
                f"Looked for: {[str(p) for p in possible_paths]}"
            )

    with open(coco_path) as f:
        coco_data = json.load(f)

    # ---- category id normalisation -----------------------------------------
    category_ids = sorted(
        set(ann["category_id"] for ann in coco_data["annotations"]))
    cat_id_map = {old: new for new, old in enumerate(category_ids)}

    if verbose:
        print(f"[INFO] Annotations: {coco_path}")
        print(f"[INFO] {len(coco_data['images'])} images, "
              f"{len(coco_data['annotations'])} annotations")
        if list(cat_id_map.keys()) != list(cat_id_map.values()):
            print(f"[INFO] Remapping category IDs: {cat_id_map}")

    # ---- train / val split -------------------------------------------------
    val_coco = None
    if val_split > 0 and len(coco_data["images"]) >= 4:
        train_coco, val_coco = split_coco_data(coco_data, val_split)
        if verbose:
            print(f"[INFO] Train split: {len(train_coco['images'])} images, "
                  f"{len(train_coco['annotations'])} annotations")
            print(f"[INFO] Val split:   {len(val_coco['images'])} images, "
                  f"{len(val_coco['annotations'])} annotations")
    else:
        train_coco = coco_data
        if verbose and val_split > 0:
            print("[WARN] Too few images for validation split; "
                  "training on all data")

    # ---- load processor & model --------------------------------------------
    if verbose:
        print(f"[INFO] Loading model: {model_name}")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = TableTransformerForObjectDetection.from_pretrained(model_name)

    # ---- datasets ----------------------------------------------------------
    TableDataset = _make_dataset_class()
    images_dir = train_dir / "images"

    train_aug = build_train_augmentation() if augment else None
    train_dataset = TableDataset(
        train_coco, images_dir, processor, cat_id_map,
        augmentation=train_aug,
    )

    val_dataset = None
    if val_coco is not None:
        val_dataset = TableDataset(
            val_coco, images_dir, processor, cat_id_map,
            augmentation=None,
        )

    if verbose:
        print(f"[INFO] Train set: {len(train_dataset)} images"
              f" (augmentation={'ON' if augment else 'OFF'})")
        if val_dataset:
            print(f"[INFO] Val set:   {len(val_dataset)} images")

    # ---- collate with pixel_mask -------------------------------------------
    def collate_fn(batch):
        pixel_values = [item["pixel_values"] for item in batch]
        encoding = processor.pad(pixel_values, return_tensors="pt")
        labels = [item["labels"] for item in batch]
        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": labels,
        }

    # ---- training arguments ------------------------------------------------
    has_val = val_dataset is not None

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_grad_norm=0.1,
        save_strategy="epoch",
        save_total_limit=3,
        logging_steps=10,
        remove_unused_columns=False,
        push_to_hub=False,
        # Evaluation (only when a val set exists)
        eval_strategy="epoch" if has_val else "no",
        load_best_model_at_end=has_val,
        metric_for_best_model="map_50" if has_val else None,
        greater_is_better=True if has_val else None,
        batch_eval_metrics=True if has_val else False,
    )

    # ---- compute_metrics ---------------------------------------------------
    compute_metrics_fn = None
    if has_val:
        compute_metrics_fn = build_compute_metrics(processor, model.config)

    # ---- trainer -----------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        compute_metrics=compute_metrics_fn,
    )

    if verbose:
        print(f"[INFO] Starting training for {epochs} epochs "
              f"(lr={learning_rate}, wd={weight_decay}, "
              f"warmup={warmup_steps})...")

    trainer.train()

    # ---- save best model ---------------------------------------------------
    best_path = output_dir / "best"
    trainer.save_model(str(best_path))
    processor.save_pretrained(str(best_path))

    # Also keep a "final" symlink/copy for backwards compatibility
    final_path = output_dir / "final"
    if final_path.exists() and final_path.is_symlink():
        final_path.unlink()
    if not final_path.exists():
        try:
            final_path.symlink_to(best_path.resolve())
        except OSError:
            import shutil
            shutil.copytree(str(best_path), str(final_path),
                            dirs_exist_ok=True)

    if verbose:
        print(f"[DONE] Best model saved to: {best_path}")

    return str(best_path)


# ---------------------------------------------------------------------------
# Colab notebook generator (unchanged)
# ---------------------------------------------------------------------------

def generate_colab_notebook(output_path: str = "train_table_transformer.ipynb"):
    """Generate a Google Colab notebook for fine-tuning Table Transformer."""

    notebook_content = {
        "nbformat": 4,
        "nbformat_minor": 0,
        "metadata": {
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "accelerator": "GPU"
        },
        "cells": [
            {
                "cell_type": "markdown",
                "source": [
                    "# Table Transformer Fine-tuning\n",
                    "\n",
                    "Fine-tune Microsoft's Table Transformer (MIT License) "
                    "for custom table detection.\n",
                    "\n",
                    "**Note:** Table Transformer often works well zero-shot. "
                    "Only fine-tune if needed.\n",
                    "\n",
                    "**License:** MIT - Commercial Friendly"
                ],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": [
                    "# Install dependencies\n",
                    "!pip install -q transformers torch torchvision Pillow "
                    "albumentations torchmetrics"
                ],
                "metadata": {},
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "source": [
                    "# Mount Google Drive\n",
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')"
                ],
                "metadata": {},
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "source": [
                    "# Check GPU\n",
                    "!nvidia-smi\n",
                    "\n",
                    "import torch\n",
                    "print(f'PyTorch: {torch.__version__}')\n",
                    "print(f'CUDA available: {torch.cuda.is_available()}')"
                ],
                "metadata": {},
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "source": [
                    "# Test zero-shot detection first!\n",
                    "from transformers import "
                    "TableTransformerForObjectDetection, "
                    "AutoImageProcessor\n",
                    "from PIL import Image\n",
                    "import requests\n",
                    "\n",
                    "processor = AutoImageProcessor.from_pretrained("
                    "'microsoft/table-transformer-detection')\n",
                    "model = TableTransformerForObjectDetection"
                    ".from_pretrained("
                    "'microsoft/table-transformer-detection')\n",
                    "\n",
                    "print('Model loaded! Test it on your images before "
                    "fine-tuning.')"
                ],
                "metadata": {},
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "source": [
                    "# Set your data path (update this!)\n",
                    "DATA_DIR = '/content/drive/MyDrive/"
                    "table_detection/yolo_dataset/train'\n",
                    "OUTPUT_DIR = '/content/drive/MyDrive/"
                    "table_detection/models/tatr_finetuned'\n",
                    "\n",
                    "import os\n",
                    "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
                    "\n",
                    "assert os.path.exists(DATA_DIR), "
                    "f'Data not found: {DATA_DIR}'"
                ],
                "metadata": {},
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "source": [
                    "# Fine-tuning code here...\n",
                    "# See the full train_table_transformer.py script "
                    "for implementation\n",
                    "print('Fine-tuning implementation - see "
                    "train_table_transformer.py')"
                ],
                "metadata": {},
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "markdown",
                "source": [
                    "## Download Your Model\n",
                    "\n",
                    "After fine-tuning, your model will be saved to:\n",
                    "`/content/drive/MyDrive/table_detection/models/"
                    "tatr_finetuned/best/`\n",
                    "\n",
                    "Download and use it:\n",
                    "```python\n",
                    "from MLTableDetection.TableDetectorML "
                    "import TableDetectorML\n",
                    "\n",
                    "detector = TableDetectorML(\n",
                    "    output_dir='/path/to/output',\n",
                    "    model_path='/path/to/tatr_finetuned/best'\n",
                    ")\n",
                    "```"
                ],
                "metadata": {}
            }
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(notebook_content, f, indent=2)

    print(f"[DONE] Colab notebook saved to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Table Transformer for table detection "
                    "(MIT License)"
    )
    parser.add_argument(
        "--data", "-d",
        default=os.path.expanduser("~/Documents/TableAnnotations"),
        help="Training data directory (YOLO or COCO format)",
    )
    parser.add_argument(
        "--output", "-o",
        default=os.path.expanduser("~/Documents/TableAnnotations/models"),
        help="Output directory for fine-tuned model",
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int, default=25,
        help="Number of training epochs (default: 25)",
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int, default=2,
        help="Batch size (default: 2)",
    )
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float, default=5e-6,
        help="Learning rate (default: 5e-6)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float, default=0.05,
        help="Weight decay (default: 0.05)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int, default=100,
        help="LR warmup steps (default: 100)",
    )
    parser.add_argument(
        "--val-split",
        type=float, default=0.15,
        help="Fraction of images for validation (default: 0.15, 0=disable)",
    )
    parser.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable data augmentation",
    )
    parser.add_argument(
        "--colab-notebook",
        action="store_true",
        help="Generate a Google Colab notebook",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    if args.colab_notebook:
        generate_colab_notebook()
        return 0

    data_path = os.path.expanduser(args.data)
    output_path = os.path.expanduser(args.output)

    if not os.path.exists(data_path):
        print(f"[ERROR] Data directory not found: {data_path}")
        return 1

    has_gpu = check_gpu()
    if not has_gpu:
        print("[WARN] No GPU detected. Training will be slow.")
        print("       Consider using --colab-notebook for free GPU access.")

    try:
        model_path = fine_tune_tatr(
            train_data_dir=data_path,
            output_dir=output_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            val_split=args.val_split,
            augment=not args.no_augment,
            verbose=not args.quiet,
        )
        print(f"\n[SUCCESS] Fine-tuning complete!")
        print(f"Model saved to: {model_path}")
        return 0
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
