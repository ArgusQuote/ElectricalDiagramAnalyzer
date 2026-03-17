# Training Recipes

Hyperparameter configurations, data augmentation strategies, and hardware requirements for each model tier.

---

## Tier 1: Zero-Shot Table Transformer

No training needed. This tier tests the pretrained model out of the box.

```python
from MLTableDetection.TableDetectorML import TableDetectorML

detector = TableDetectorML(
    output_dir="test_output",
    conf_threshold=0.5,    # Start conservative; lower if missing panels
    verbose=True,
)
results = detector.readPdf("path/to/test.pdf")
```

**Tuning the confidence threshold**:
- `0.7` -- Very conservative, high precision, may miss some panels
- `0.5` -- Balanced (default)
- `0.3` -- Aggressive, catches more panels but increases false positives
- `0.2` -- Use only if many panels are missed at higher thresholds

If zero-shot detection rate is < 90% on your test PDFs, proceed to Tier 2.

---

## Tier 2: Fine-Tune Table Transformer

### Recommended Configuration

| Parameter | Value | Notes |
|---|---|---|
| Base model | `microsoft/table-transformer-detection` | MIT license, PubTables-1M pretrained |
| Epochs | 15 | Start here; increase to 25 if val loss still decreasing |
| Learning rate | 1e-5 | Standard for DETR fine-tuning |
| Batch size | 2 | Increase to 4 if GPU memory allows (>16 GB) |
| Weight decay | 0.01 | Prevents overfitting on small datasets |
| LR scheduler | Linear with warmup | Default in HF Trainer |
| Warmup steps | 100 | Or 10% of total steps, whichever is smaller |
| Save strategy | Every epoch | Allows selecting best checkpoint |

### Command

```bash
python MLTableDetection/train_table_transformer.py \
  --data MLTableDetection/data \
  --output MLTableDetection/models/tatr_finetuned \
  --epochs 15 \
  --learning-rate 1e-5 \
  --batch-size 2
```

### Small Dataset Recipe (< 50 annotations)

When working with very limited data:

| Parameter | Value | Notes |
|---|---|---|
| Epochs | 25-30 | More epochs to compensate for fewer samples |
| Learning rate | 5e-6 | Lower LR to avoid overfitting |
| Weight decay | 0.05 | Stronger regularization |
| Data augmentation | Aggressive | See augmentation section below |

### Medium Dataset Recipe (50-200 annotations)

| Parameter | Value | Notes |
|---|---|---|
| Epochs | 15-20 | Standard range |
| Learning rate | 1e-5 | Standard DETR fine-tuning LR |
| Weight decay | 0.01 | Standard regularization |
| Data augmentation | Moderate | See augmentation section below |

### Large Dataset Recipe (200+ annotations)

| Parameter | Value | Notes |
|---|---|---|
| Epochs | 10-15 | Less epochs needed with more data |
| Learning rate | 1e-5 to 2e-5 | Can use slightly higher LR |
| Batch size | 4 | Larger batch if GPU allows |
| Weight decay | 0.01 | Standard |
| Data augmentation | Light | Less augmentation needed |

### Signs of Overfitting

- Training loss continues to decrease but validation loss increases
- Model performs well on training PDFs but poorly on held-out PDFs
- Detection confidence is very high (>0.95) on training data but low on new data

**Fix**: Reduce epochs, increase weight decay, add more augmentation, or gather more training data.

### Signs of Underfitting

- Both training and validation loss remain high
- Model misses obvious panels even on training data
- Low confidence scores across the board

**Fix**: Increase epochs, increase learning rate (carefully), verify annotations are correct, or switch to Tier 3.

---

## Tier 3: RF-DETR Fine-Tuning

RF-DETR is not yet integrated into the project. This recipe guides adding it.

### Installation

```bash
pip install rfdetr
```

### Training Script Pattern

Create `MLTableDetection/train_rfdetr.py` following this pattern:

```python
from rfdetr import RFDETRBase, RFDETRLarge
from rfdetr.config import TrainConfig

# Base model (29M params) -- recommended for most cases
model = RFDETRBase()

# Configure training
config = TrainConfig(
    dataset_dir="MLTableDetection/data",   # Must contain COCO format
    epochs=30,
    batch_size=4,
    grad_accum_steps=4,
    lr=1e-4,
    output_dir="MLTableDetection/models/rfdetr_finetuned",
)

# Train
model.train(config)

# Export for inference
model.export("MLTableDetection/models/rfdetr_finetuned/best.pt")
```

### Recommended Configuration

| Parameter | Value | Notes |
|---|---|---|
| Base model | `RFDETRBase` (29M params) | Start with base; upgrade to Large only if needed |
| Epochs | 30 | RF-DETR converges slower than TATR due to general pretraining |
| Learning rate | 1e-4 | Higher than TATR -- RF-DETR is designed for fine-tuning |
| Batch size | 4 | Effective batch = batch_size x grad_accum_steps |
| Gradient accumulation | 4 | Simulates batch size of 16 |
| Image size | 560 (base) or 640 (large) | RF-DETR default resolution |

### Integration

After training, integrate RF-DETR into `TableDetectorML.py` by adding a new backend:

```python
# In TableDetectorML.__init__:
BACKEND_RFDETR = "rf-detr"

# Add _load_rfdetr_model() and _detect_tables_rfdetr() methods
# following the same pattern as TATR and Detectron2 backends
```

---

## Tier 3 Alternative: RT-DETRv2 Fine-Tuning

RT-DETRv2 uses the HuggingFace Transformers library, so the existing `train_table_transformer.py` patterns can be adapted.

### Installation

```bash
pip install transformers>=4.49.0  # RT-DETRv2 support added Feb 2025
```

### Training Script Adaptation

Modify `train_table_transformer.py` to support RT-DETRv2:

```python
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

# Replace TATR model loading with:
processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_v2_r50vd")
model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r50vd")

# Training loop is identical to TATR (same Trainer API)
```

### Recommended Configuration

| Parameter | Value | Notes |
|---|---|---|
| Base model | `PekingU/rtdetr_v2_r50vd` | ResNet-50 variant, good balance |
| Epochs | 20 | Between TATR and RF-DETR convergence |
| Learning rate | 5e-5 | Slightly higher than TATR |
| Batch size | 2-4 | Depends on GPU memory |
| Weight decay | 0.01 | Standard |

---

## Data Augmentation

Apply augmentations during training to increase effective dataset size and improve generalization.

### Recommended Augmentations for Panel Schedules

| Augmentation | Probability | Notes |
|---|---|---|
| Horizontal flip | 0.5 | Panel schedules are symmetric |
| Random brightness | 0.3 | Simulates scan quality variation (+/- 20%) |
| Random contrast | 0.3 | Simulates print quality variation (+/- 20%) |
| Gaussian noise | 0.2 | Simulates scan artifacts (sigma 5-15) |
| Random scale | 0.3 | Scale 0.8-1.2x to handle DPI variation |
| Random rotation | 0.1 | Small rotation only (+/- 2 degrees) |

### Augmentations to AVOID

| Augmentation | Why |
|---|---|
| Large rotation (> 5 degrees) | Panel schedules are always axis-aligned |
| Vertical flip | Tables are never upside-down |
| Color jitter (hue/saturation) | Drawings are grayscale or near-grayscale |
| Cutout / random erasing | May remove critical table structure |
| Mosaic | Designed for natural images, creates unrealistic table compositions |

### Implementation with Albumentations

```python
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.3
    ),
    A.GaussNoise(var_limit=(5, 15), p=0.2),
    A.RandomScale(scale_limit=0.2, p=0.3),
    A.Rotate(limit=2, p=0.1, border_mode=0),
], bbox_params=A.BboxParams(
    format='coco',
    label_fields=['category_ids'],
    min_area=1000,       # Drop tiny boxes created by augmentation
    min_visibility=0.5,  # Drop boxes that are mostly cropped
))
```

Note: Albumentations is BSD-3 licensed (commercial-friendly).

---

## Hardware Requirements

### Minimum (for fine-tuning)

| Resource | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA T4 (16 GB) | NVIDIA A10 (24 GB) or better |
| RAM | 16 GB | 32 GB |
| Disk | 10 GB free | 50 GB free (for checkpoints) |
| CUDA | 11.8+ | 12.1+ |

### Google Colab (free tier)

- T4 GPU (16 GB) -- sufficient for Tier 2 (TATR fine-tuning)
- Session limit: ~12 hours
- Use `--colab-notebook` flag to generate a notebook:

```bash
python MLTableDetection/train_table_transformer.py --colab-notebook
```

### Estimated Training Times

| Model | Dataset Size | GPU | Time |
|---|---|---|---|
| TATR (Tier 2) | 50 images, 15 epochs | T4 | ~15-30 min |
| TATR (Tier 2) | 200 images, 15 epochs | T4 | ~1-2 hours |
| RF-DETR Base (Tier 3) | 50 images, 30 epochs | T4 | ~30-60 min |
| RF-DETR Base (Tier 3) | 200 images, 30 epochs | T4 | ~2-4 hours |
| RF-DETR Large (Tier 3) | 200 images, 30 epochs | A10 | ~3-6 hours |

### Inference Times (per PDF page at 400 DPI)

| Model | GPU | CPU | Notes |
|---|---|---|---|
| TATR | ~200ms | ~2s | Lightweight |
| RF-DETR Base | ~15ms | ~500ms | Very fast |
| RF-DETR Large | ~30ms | ~1s | Fast |
| RT-DETRv2 R50 | ~20ms | ~800ms | Fast |
| Detectron2 R50-FPN | ~50ms | ~3s | Moderate |
| Heuristic (V25) | N/A | ~1-3s | No GPU needed |

---

## Checkpoint Management

### During Training

- Save checkpoints every epoch (default in HF Trainer)
- Keep the 3 most recent checkpoints to save disk space
- The best checkpoint is typically NOT the last one -- use validation metrics to select

### After Training

```bash
# Final model directory structure
MLTableDetection/models/tatr_finetuned/final/
  config.json           # Model configuration
  model.safetensors     # Model weights
  preprocessor_config.json  # Image processor config
```

### Model Selection

After training completes, evaluate each saved checkpoint on held-out data:

```python
import os
from pathlib import Path

checkpoints = sorted(Path("MLTableDetection/models/tatr_finetuned").glob("checkpoint-*"))
for ckpt in checkpoints:
    detector = TableDetectorML(output_dir=f"eval/{ckpt.name}", model_path=str(ckpt))
    results = detector.readPdf("held_out_test.pdf")
    print(f"{ckpt.name}: {len(results)} panels detected")
```

Pick the checkpoint with the highest detection rate and lowest false positive rate.
