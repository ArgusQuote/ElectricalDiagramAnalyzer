---
name: training-panel-detector
description: Guides training and fine-tuning a commercial-friendly ML model to detect panel schedule tables in electrical PDF drawings. Use when the user wants to train a table detection model, improve ML-based panel detection, annotate training data, evaluate model accuracy against the heuristic baseline, or integrate a fine-tuned model into the detection pipeline.
---

# Training Panel Detector

Trains or fine-tunes a commercial-friendly object detection model to locate panel schedule tables in rendered PDF pages, replacing or augmenting the heuristic detector in `PanelSearchToolV25.py`.

## Prerequisites

Before starting, confirm the following inputs are available:

- **Source PDFs**: Electrical drawings containing panel schedule tables (e.g., files in `~/Documents/pdfToScan/`)
- **Existing infrastructure**: The `MLTableDetection/` directory with `TableDetectorML.py`, `train_table_transformer.py`, `render_pdfs_for_annotation.py`
- **GPU access**: Local CUDA GPU or Google Colab for training

If any prerequisite is missing, stop and ask the user.

## Workflow

Follow these five phases in order. Complete each phase before moving to the next.

---

### Phase 1: Data Preparation

Render source PDFs into annotatable images.

1. Collect source PDFs into a staging directory
2. Run the existing renderer:

```bash
python MLTableDetection/render_pdfs_for_annotation.py \
  --input-dir ~/Documents/pdfToScan \
  --output-dir MLTableDetection/data/images \
  --dpi 400
```

3. Verify output: images directory should contain one PNG per PDF page
4. Create the annotation directories:

```bash
mkdir -p MLTableDetection/data/{labels,annotations}
```

**Check**: At least 20 rendered page images exist in `MLTableDetection/data/images/`. More data produces better models -- 50-100+ images is recommended for fine-tuning.

**Stop condition**: If fewer than 20 source PDFs are available, inform the user that model quality will be limited and ask whether to proceed or gather more data.

---

### Phase 2: Annotation

Label panel schedule regions in the rendered images.

1. Set up Label Studio using the existing script:

```bash
bash MLTableDetection/setup_label_studio.sh
```

2. Import images from `MLTableDetection/data/images/` into a Label Studio project
3. Use this label schema (already defined in `label_config.xml`):
   - `panel_schedule` -- primary target (the rectangular panel schedule table)
   - `motor_schedule` -- optional, if motor schedule tables appear
   - `riser_diagram` -- optional, if riser diagrams appear

4. Annotate all images with bounding boxes around each panel schedule table
5. Export annotations in **COCO format** to `MLTableDetection/data/annotations/annotations.json`

See [annotation-guide.md](references/annotation-guide.md) for bounding box conventions and consistency rules.

**Check**: Exported COCO JSON contains entries for all annotated images. Run:

```python
import json
with open("MLTableDetection/data/annotations/annotations.json") as f:
    coco = json.load(f)
print(f"Images: {len(coco['images'])}, Annotations: {len(coco['annotations'])}")
```

Both counts must be non-zero. Annotation count should be >= image count (most pages have multiple panels).

**Stop condition**: If annotation count is < 50, warn the user that fine-tuning results may be unreliable.

---

### Phase 3: Model Selection and Training

Use a tiered approach -- start simple, escalate only if needed.

#### Tier 1: Zero-Shot Table Transformer (no training)

Test the pretrained model first:

```python
from MLTableDetection.TableDetectorML import TableDetectorML

detector = TableDetectorML(output_dir="test_output", verbose=True)
results = detector.readPdf("path/to/test.pdf")
print(f"Detected {len(results)} panels")
```

**Check**: Compare detected panel count against the known count for test PDFs. If detection rate is >= 90% with < 5% false positives, Tier 1 is sufficient -- skip to Phase 5.

#### Tier 2: Fine-Tune Table Transformer

If Tier 1 misses panels or produces too many false positives:

```bash
python MLTableDetection/train_table_transformer.py \
  --data MLTableDetection/data \
  --output MLTableDetection/models/tatr_finetuned \
  --epochs 15 \
  --learning-rate 1e-5 \
  --batch-size 2
```

See [training-recipes.md](references/training-recipes.md) for hyperparameter guidance.

**Check**: Validation loss decreases over epochs. Test the fine-tuned model:

```python
detector = TableDetectorML(
    output_dir="test_output",
    model_path="MLTableDetection/models/tatr_finetuned/final"
)
results = detector.readPdf("path/to/test.pdf")
```

If mAP@0.5 >= 0.85 on held-out PDFs, proceed to Phase 5.

#### Tier 3: Switch Architecture

If Tier 2 plateaus below acceptable accuracy, switch to a stronger backbone. See [model-comparison.md](references/model-comparison.md) for the full comparison.

**Recommended**: RF-DETR (Apache 2.0, Roboflow)
- 29M params (base) or 128M params (large)
- SOTA on COCO and RF100-VL benchmarks
- Designed for fine-tuning with small datasets
- Requires writing a new training script (adapt from the RF-DETR Colab notebook)

**Alternative**: RT-DETRv2 (Apache 2.0, Baidu)
- Available in HuggingFace Transformers
- Can reuse the existing HF-based training patterns from `train_table_transformer.py`

**Excluded** (AGPL-3.0, not commercial-friendly):
- YOLOv8 / YOLOv11 (Ultralytics)
- DocLayout-YOLO

**Stop condition**: If three training attempts across tiers fail to exceed 0.70 mAP@0.5, stop and present findings. The issue is likely insufficient or inconsistent training data, not model choice.

---

### Phase 4: Evaluation

Compare the trained model against the heuristic baseline.

1. Select 5-10 held-out PDFs not used in training
2. Run both detectors on each PDF:

```python
from VisualDetectionToolLibrary.PanelSearchToolV25 import PanelBoardSearch
from MLTableDetection.TableDetectorML import TableDetectorML

# Heuristic
heuristic = PanelBoardSearch(output_dir="eval/heuristic")
h_results = heuristic.readPdf("test.pdf")

# ML
ml_detector = TableDetectorML(
    output_dir="eval/ml",
    model_path="path/to/finetuned/model"
)
m_results = ml_detector.readPdf("test.pdf")

print(f"Heuristic: {len(h_results)} panels")
print(f"ML Model:  {len(m_results)} panels")
```

3. Visually inspect the overlay images in `eval/*/magenta_overlays/` to confirm correct detections
4. Record metrics for each PDF: panels found, false positives, missed panels

**Check**: ML model should match or exceed heuristic detection count on >= 80% of test PDFs.

**Recovery**: If the ML model underperforms:
- Review false negatives -- are they unusual layouts? Add similar examples to training data
- Review false positives -- lower the confidence threshold or add hard negatives
- Re-train and re-evaluate (return to Phase 3)

---

### Phase 5: Integration

Deploy the trained model into the existing pipeline.

1. Copy the final model to a stable location:

```bash
cp -r MLTableDetection/models/tatr_finetuned/final ~/models/panel_detector/
```

2. The model integrates via the existing `TableDetectorML` class -- no code changes needed:

```python
detector = TableDetectorML(
    output_dir=output_dir,
    model_path=os.path.expanduser("~/models/panel_detector/")
)
panels = detector.readPdf(pdf_path)
```

3. Enable in production by setting the environment variable:

```bash
export USE_ML_DETECTOR=true
export ML_MODEL_PATH=~/models/panel_detector/
```

4. The pipeline in `uplink_server.py` will use `TableDetectorML` instead of `PanelBoardSearch`

**Check**: Run a full pipeline test:
- Upload a test PDF through the Anvil UI (or call `vm_submit_for_detection()` directly)
- Verify `result.json` contains expected panel data
- Verify `status.json` shows `state=done`

**Recovery**: If integration fails, revert to heuristic by unsetting `USE_ML_DETECTOR`.

---

## License Summary

All recommended components use commercial-friendly licenses:

| Component | License |
|---|---|
| Table Transformer | MIT |
| RF-DETR | Apache 2.0 |
| RT-DETRv2 | Apache 2.0 |
| Detectron2 | Apache 2.0 |
| PyTorch | BSD |
| HuggingFace Transformers | Apache 2.0 |
| Label Studio | Apache 2.0 |
| PubTables-1M dataset | CDLA-Permissive 2.0 |

See [model-comparison.md](references/model-comparison.md) for the full breakdown.
