# ML Table Detection

Machine learning-based table detection using **Table Transformer** (MIT License).

## License Summary

| Component | License | Commercial Use |
|-----------|---------|----------------|
| Table Transformer | MIT | Yes |
| Detectron2 (fallback) | Apache 2.0 | Yes |
| transformers | Apache 2.0 | Yes |
| PyTorch | BSD | Yes |

**All components are commercial-friendly.**

## Overview

This module provides ML-based table detection as a drop-in replacement for the heuristic `PanelBoardSearch`.

### Why Table Transformer?

- **Pretrained on 1M+ tables** (PubTables-1M dataset)
- **Works zero-shot** - try it without any training first
- **MIT Licensed** - fully commercial-friendly
- **Easy to fine-tune** - only needs 50-100 images if needed

## Quick Start

### 1. Install Dependencies

```bash
pip install -r MLTableDetection/requirements.txt
```

### 2. Try Zero-Shot Detection (No Training!)

```python
from MLTableDetection.TableDetectorML import TableDetectorML

# Uses pretrained model - no training needed
detector = TableDetectorML(output_dir="/path/to/output")

# Same API as PanelBoardSearch
panel_images = detector.readPdf("/path/to/electrical.pdf")
```

### 3. Fine-tune Only If Needed

If zero-shot results aren't good enough (rare), fine-tune with your data:

```bash
# Render PDFs to images
python MLTableDetection/render_pdfs_for_annotation.py \
    --input /path/to/pdfs \
    --output MLTableDetection/data/images_for_annotation

# Annotate with Label Studio (50-100 images usually enough)
label-studio start

# Export as COCO format, then fine-tune
python MLTableDetection/train_table_transformer.py \
    --data MLTableDetection/data/annotations \
    --epochs 10
```

## Directory Structure

```
MLTableDetection/
├── __init__.py
├── README.md
├── requirements.txt
├── TableDetectorML.py             # Main detector class
├── render_pdfs_for_annotation.py  # PDF → images for annotation
├── train_table_transformer.py     # Fine-tuning script for TATR
├── evaluate_model.py              # Model evaluation
├── setup_label_studio.sh          # Annotation setup script
└── data/
    ├── images_for_annotation/
    ├── annotations/
    └── models/
```

## Usage Examples

### Zero-Shot (Recommended First Step)

```python
from MLTableDetection.TableDetectorML import TableDetectorML

# Default: uses microsoft/table-transformer-detection
detector = TableDetectorML(
    output_dir="./output",
    conf_threshold=0.5,  # TATR is confident, higher threshold OK
    verbose=True,
)

# Process PDF
images = detector.readPdf("electrical_diagram.pdf")
print(f"Detected {len(images)} tables")
```

### With Fine-tuned Model

```python
detector = TableDetectorML(
    output_dir="./output",
    model_path="MLTableDetection/data/models/tatr_finetuned/final",
)

images = detector.readPdf("electrical_diagram.pdf")
```

### Using Detectron2 Backend (Alternative)

```python
detector = TableDetectorML(
    output_dir="./output",
    backend="detectron2",  # Apache 2.0 license
    model_path="/path/to/detectron2/config.yaml",
)
```

## Server Integration

The detector is integrated into `uplink_server.py`:

```bash
# Use ML detector (Table Transformer)
export USE_ML_DETECTOR=true
python AnvilUplinkCode/uplink_server.py

# Use heuristic detector (default)
export USE_ML_DETECTOR=false
python AnvilUplinkCode/uplink_server.py
```

With custom model:
```bash
export USE_ML_DETECTOR=true
export ML_MODEL_PATH=/path/to/fine-tuned/model
python AnvilUplinkCode/uplink_server.py
```

## Model Comparison

| Approach | Data Needed | Accuracy | License |
|----------|-------------|----------|---------|
| TATR Zero-shot | 0 images | Good-Very Good | MIT |
| TATR Fine-tuned | 50-100 images | Very Good | MIT |
| Detectron2 | 100-200 images | Good | Apache 2.0 |
| YOLOv8 | 100-200 images | Very Good | **AGPL** (not commercial-friendly) |
| Heuristic | 0 images | Variable | N/A |

## Expected Performance

With Table Transformer zero-shot:
- **Standard tables**: 85-95% detection rate
- **Panel schedules**: 80-90% (domain-specific, may need fine-tuning)
- **Rotated tables**: 70-85% (TATR handles rotation)

With fine-tuning on 50-100 images:
- **Domain-specific tables**: 90-95%+ detection rate

## Troubleshooting

### "No tables detected"

1. Lower confidence threshold: `conf_threshold=0.3`
2. Check if tables have clear borders/structure
3. Consider fine-tuning on your specific table style

### "Too many false positives"

1. Raise confidence threshold: `conf_threshold=0.7`
2. Adjust area filters: `min_area_fraction`, `max_area_fraction`

### GPU Memory Issues

```python
# Force CPU (slower but uses less memory)
detector = TableDetectorML(output_dir="./output", device="cpu")
```

## Fine-tuning Tips

1. **Start with zero-shot** - it often works well enough
2. **Annotate diverse examples** - different table styles, sizes, orientations
3. **50-100 images is usually enough** - Table Transformer is pretrained
4. **Use data augmentation** - rotation, scaling, brightness
5. **Validate on held-out set** - ensure generalization

## API Reference

### TableDetectorML

```python
TableDetectorML(
    output_dir: str,                    # Output directory
    model_path: str = None,             # Model path (HF ID or local)
    backend: str = "table-transformer", # "table-transformer" or "detectron2"
    dpi: int = 400,                     # Detection DPI
    render_dpi: int = 1200,             # Output PNG DPI
    conf_threshold: float = 0.5,        # Detection confidence threshold
    pad: int = 6,                       # Padding around detections
    device: str = None,                 # "cuda", "cpu", or None (auto)
    verbose: bool = True,               # Print progress
)
```

### readPdf()

```python
def readPdf(self, pdf_path: str) -> list[str]:
    """
    Detect tables in PDF and extract crops.
    
    Args:
        pdf_path: Path to PDF file
    
    Returns:
        List of paths to cropped PNG files
    """
```

## Related Documentation

- [Panel Detection](../.cursor/rules/docs/panel-detection.mdc) - Heuristic detection docs
- [Uplink Server](../.cursor/rules/docs/uplink-server.mdc) - Server integration
- [Project Structure](../.cursor/rules/reference/project-structure.mdc) - File locations
