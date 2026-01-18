#!/usr/bin/env python3
"""
Fine-tuning script for Table Transformer (TATR) on custom table data.

Table Transformer is MIT licensed and pretrained on PubTables-1M.
It often works well zero-shot, but fine-tuning can improve accuracy
on domain-specific tables (like electrical panel schedules).

Usage:
    # Fine-tune on custom data (only if zero-shot isn't good enough)
    python train_table_transformer.py --data data/yolo_dataset --epochs 10
    
    # Generate Colab notebook for training
    python train_table_transformer.py --colab-notebook
"""

import os
import argparse
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
        # Get image dimensions
        with Image.open(img_path) as img:
            width, height = img.size
        
        # Add image entry
        coco["images"].append({
            "id": img_idx,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })
        
        # Read corresponding label file
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        
                        # Convert YOLO (normalized center) to COCO (absolute xywh)
                        x = (cx - w/2) * width
                        y = (cy - h/2) * height
                        box_w = w * width
                        box_h = h * height
                        
                        coco["annotations"].append({
                            "id": ann_id,
                            "image_id": img_idx,
                            "category_id": 0,  # table
                            "bbox": [x, y, box_w, box_h],
                            "area": box_w * box_h,
                            "iscrowd": 0,
                        })
                        ann_id += 1
    
    # Save COCO JSON
    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=2)
    
    print(f"[INFO] Converted {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    return coco


def fine_tune_tatr(
    train_data_dir: str,
    val_data_dir: Optional[str] = None,
    output_dir: str = "models/tatr_finetuned",
    model_name: str = "microsoft/table-transformer-detection",
    epochs: int = 10,
    batch_size: int = 2,
    learning_rate: float = 1e-5,
    verbose: bool = True,
) -> str:
    """
    Fine-tune Table Transformer on custom table data.
    
    Args:
        train_data_dir: Directory with training data (YOLO or COCO format).
        val_data_dir: Directory with validation data (optional).
        output_dir: Directory to save fine-tuned model.
        model_name: Base model to fine-tune.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Learning rate.
        verbose: Print progress.
    
    Returns:
        Path to the fine-tuned model.
    """
    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        TableTransformerForObjectDetection,
        AutoImageProcessor,
        TrainingArguments,
        Trainer,
    )
    from PIL import Image
    import numpy as np
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for COCO format or convert from YOLO
    train_dir = Path(train_data_dir)
    coco_path = train_dir / "annotations.json"
    
    if not coco_path.exists():
        # Try to convert from YOLO format
        if (train_dir / "images").exists() and (train_dir / "labels").exists():
            if verbose:
                print("[INFO] Converting YOLO format to COCO...")
            convert_yolo_to_coco(str(train_dir), str(coco_path))
        else:
            raise FileNotFoundError(
                f"No annotations.json found and no YOLO format detected in {train_dir}"
            )
    
    # Load annotations
    with open(coco_path) as f:
        coco_data = json.load(f)
    
    if verbose:
        print(f"[INFO] Loaded {len(coco_data['images'])} images")
        print(f"[INFO] Loading model: {model_name}")
    
    # Load processor and model
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = TableTransformerForObjectDetection.from_pretrained(model_name)
    
    # Create simple dataset
    class TableDataset(torch.utils.data.Dataset):
        def __init__(self, coco_data, images_dir, processor):
            self.coco_data = coco_data
            self.images_dir = Path(images_dir)
            self.processor = processor
            
            # Build image_id -> annotations mapping
            self.img_to_anns = {}
            for ann in coco_data["annotations"]:
                img_id = ann["image_id"]
                if img_id not in self.img_to_anns:
                    self.img_to_anns[img_id] = []
                self.img_to_anns[img_id].append(ann)
        
        def __len__(self):
            return len(self.coco_data["images"])
        
        def __getitem__(self, idx):
            img_info = self.coco_data["images"][idx]
            img_path = self.images_dir / img_info["file_name"]
            
            image = Image.open(img_path).convert("RGB")
            
            # Get annotations for this image
            anns = self.img_to_anns.get(img_info["id"], [])
            
            # Format annotations for DETR-style training
            boxes = []
            labels = []
            for ann in anns:
                x, y, w, h = ann["bbox"]
                # Convert to [x_center, y_center, width, height] normalized
                boxes.append([
                    (x + w/2) / img_info["width"],
                    (y + h/2) / img_info["height"],
                    w / img_info["width"],
                    h / img_info["height"],
                ])
                labels.append(ann["category_id"])
            
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(labels, dtype=torch.int64),
            }
            
            # Process image
            encoding = self.processor(images=image, return_tensors="pt")
            pixel_values = encoding["pixel_values"].squeeze(0)
            
            return {
                "pixel_values": pixel_values,
                "labels": target,
            }
    
    # Create datasets
    images_dir = train_dir / "images"
    train_dataset = TableDataset(coco_data, images_dir, processor)
    
    if verbose:
        print(f"[INFO] Training on {len(train_dataset)} images")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=10,
        remove_unused_columns=False,
        push_to_hub=False,
    )
    
    # Custom collate function
    def collate_fn(batch):
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        labels = [item["labels"] for item in batch]
        return {"pixel_values": pixel_values, "labels": labels}
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
    )
    
    if verbose:
        print(f"[INFO] Starting training for {epochs} epochs...")
    
    # Train
    trainer.train()
    
    # Save model
    final_path = output_dir / "final"
    trainer.save_model(str(final_path))
    processor.save_pretrained(str(final_path))
    
    if verbose:
        print(f"[DONE] Model saved to: {final_path}")
    
    return str(final_path)


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
                    "Fine-tune Microsoft's Table Transformer (MIT License) for custom table detection.\n",
                    "\n",
                    "**Note:** Table Transformer often works well zero-shot. Only fine-tune if needed.\n",
                    "\n",
                    "**License:** MIT - Commercial Friendly"
                ],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": [
                    "# Install dependencies\n",
                    "!pip install -q transformers torch torchvision Pillow"
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
                    "from transformers import TableTransformerForObjectDetection, AutoImageProcessor\n",
                    "from PIL import Image\n",
                    "import requests\n",
                    "\n",
                    "# Load pretrained model\n",
                    "processor = AutoImageProcessor.from_pretrained('microsoft/table-transformer-detection')\n",
                    "model = TableTransformerForObjectDetection.from_pretrained('microsoft/table-transformer-detection')\n",
                    "\n",
                    "print('Model loaded! Test it on your images before fine-tuning.')"
                ],
                "metadata": {},
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "source": [
                    "# Test on a sample image\n",
                    "# Upload your own image or use a test URL\n",
                    "\n",
                    "# Example: test on your uploaded image\n",
                    "# image = Image.open('/content/drive/MyDrive/your_image.png')\n",
                    "\n",
                    "# Or use a sample table image\n",
                    "url = 'https://github.com/microsoft/table-transformer/raw/main/examples/example.png'\n",
                    "image = Image.open(requests.get(url, stream=True).raw)\n",
                    "\n",
                    "# Run inference\n",
                    "inputs = processor(images=image, return_tensors='pt')\n",
                    "outputs = model(**inputs)\n",
                    "\n",
                    "# Post-process\n",
                    "target_sizes = torch.tensor([image.size[::-1]])\n",
                    "results = processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]\n",
                    "\n",
                    "print(f'Detected {len(results[\"boxes\"])} tables')\n",
                    "for score, label, box in zip(results['scores'], results['labels'], results['boxes']):\n",
                    "    print(f'  {model.config.id2label[label.item()]}: {score:.2f} at {box.tolist()}')"
                ],
                "metadata": {},
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "markdown",
                "source": [
                    "## Fine-tuning (only if zero-shot isn't good enough)\n",
                    "\n",
                    "If the zero-shot results are not satisfactory, fine-tune on your data."
                ],
                "metadata": {}
            },
            {
                "cell_type": "code",
                "source": [
                    "# Set your data path (update this!)\n",
                    "DATA_DIR = '/content/drive/MyDrive/table_detection/yolo_dataset/train'\n",
                    "OUTPUT_DIR = '/content/drive/MyDrive/table_detection/models/tatr_finetuned'\n",
                    "\n",
                    "import os\n",
                    "os.makedirs(OUTPUT_DIR, exist_ok=True)\n",
                    "\n",
                    "# Verify data exists\n",
                    "assert os.path.exists(DATA_DIR), f'Data not found: {DATA_DIR}'"
                ],
                "metadata": {},
                "execution_count": None,
                "outputs": []
            },
            {
                "cell_type": "code",
                "source": [
                    "# Fine-tuning code here...\n",
                    "# See the full train_table_transformer.py script for implementation\n",
                    "print('Fine-tuning implementation - see train_table_transformer.py')"
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
                    "`/content/drive/MyDrive/table_detection/models/tatr_finetuned/final/`\n",
                    "\n",
                    "Download and use it:\n",
                    "```python\n",
                    "from MLTableDetection.TableDetectorML import TableDetectorML\n",
                    "\n",
                    "detector = TableDetectorML(\n",
                    "    output_dir='/path/to/output',\n",
                    "    model_path='/path/to/tatr_finetuned/final'\n",
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


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Table Transformer for table detection (MIT License)"
    )
    parser.add_argument(
        "--data", "-d",
        help="Training data directory (YOLO or COCO format)"
    )
    parser.add_argument(
        "--output", "-o",
        default="models/tatr_finetuned",
        help="Output directory for fine-tuned model"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=2,
        help="Batch size (default: 2)"
    )
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=1e-5,
        help="Learning rate (default: 1e-5)"
    )
    parser.add_argument(
        "--colab-notebook",
        action="store_true",
        help="Generate a Google Colab notebook"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )
    
    args = parser.parse_args()
    
    if args.colab_notebook:
        generate_colab_notebook()
        return 0
    
    if not args.data:
        print("[ERROR] --data argument required for training")
        print("        Or use --colab-notebook to generate a Colab notebook")
        return 1
    
    # Check GPU
    has_gpu = check_gpu()
    if not has_gpu:
        print("[WARN] No GPU detected. Training will be slow.")
        print("       Consider using --colab-notebook for free GPU access.")
    
    try:
        model_path = fine_tune_tatr(
            train_data_dir=args.data,
            output_dir=args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
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
