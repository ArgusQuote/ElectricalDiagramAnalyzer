# Known Issues

## ML Table Detection Model Underperforms Heuristic Baseline

**Status:** In Progress (v4 retrain completed, still underperforming)
**Date:** 2026-02-14
**Updated:** 2026-04-05
**Component:** `MLTableDetection/TableDetectorML.py`, `MLTableDetection/train_table_transformer.py`

### Problem

The fine-tuned Table Transformer model detects fewer tables and with less precision
than the heuristic void-detection approach in `PanelSearchToolV25`. On a test page
with 6 panel schedule tables (`generic3.pdf` page 1):

- **Heuristic (`PanelSearchToolV25`):** Found all 6 tables with tight, precise boxes
- **Fine-tuned ML model:** Found 4 at conf=0.7, with imprecise boundaries and
  some boxes spanning multiple tables as one detection

### Root Cause (Partially Fixed)

A DPI mismatch between training (200 DPI) and inference (400 DPI) was identified and
corrected -- training images were re-rendered at 400 DPI, annotations scaled, and the
model retrained. This improved alignment but the model still lacks the precision of
the heuristic tool.

### Training Pipeline Fixes Applied (2026-02-16)

The following issues in the training pipeline were identified and fixed:

1. **No data augmentation** -- Training fed raw images with no transforms. The
   model was overfitting to the 27 training images instead of learning
   generalisable table features. Fixed by adding an Albumentations pipeline
   (horizontal flip, brightness/contrast, Gaussian noise, small-scale and
   rotation jitter) tuned for document tables.

2. **No train/validation split** -- All 27 images were used for training with
   no held-out set. There was no way to detect overfitting or select the best
   checkpoint. Fixed by adding a `--val-split` flag (default 15%) that holds
   out images for validation mAP tracking.

3. **Wrong hyperparameters for small dataset** -- Training used the medium-dataset
   recipe (10 epochs, LR 1e-5, weight decay 0.01) on a small dataset. Fixed:
   defaults changed to 25 epochs, LR 5e-6, weight decay 0.05, 100 warmup steps.

4. **Missing pixel mask in collate** -- The collate function manually padded
   images without generating the attention mask DETR needs. Fixed by using
   the HuggingFace `processor.pad()` method which produces both `pixel_values`
   and `pixel_mask`.

5. **No validation metrics** -- No mAP was computed during training. Best
   model selection was impossible. Fixed by adding `compute_metrics` using
   `torchmetrics.MeanAveragePrecision` with `load_best_model_at_end=True`.

6. **Evaluation script used YOLO (wrong model)** -- `evaluate_model.py`
   imported `from ultralytics import YOLO` which is AGPL-licensed and
   incompatible with the Table Transformer model being trained. Rewritten
   to use `TableDetectorML` and `torchmetrics`.

### Resolved Items from Previous Iteration

The following issues from 2026-02-16 have been addressed in the v2 retrain:

- ~~Insufficient training data~~ -- Partially addressed (25 images re-annotated,
  but still below 50+ target)
- ~~Unannotated `A_page001.png`~~ -- Removed from dataset during re-annotation
- ~~Oversized images~~ -- All re-rendered with `--max-dimension 4400`
- ~~Audit existing annotations~~ -- All 25 images freshly annotated in Label Studio

### v2 Retrain Results (2026-03-11)

Data and pipeline improvements applied before retraining:

1. **Oversized images fixed** -- Six images (A, B, C, D, makayla1, Electrical
   Takeoff pages) were re-rendered at 400 DPI with `--max-dimension 4400`,
   bringing all images to a consistent ~4400px max dimension.

2. **Fresh annotations** -- All 25 images were re-annotated from scratch in
   Label Studio against the correctly-sized images. Old `images_200dpi/`
   directory and scaled annotations replaced.

3. **New test page added** -- `Electrical_Takeoff_page7.pdf` extracted and
   added to the dataset (4 panel schedules annotated).

4. **Missing dependency** -- `pycocotools` was not installed, causing
   `torchmetrics.MeanAveragePrecision` to fail at the first eval step.
   Fixed with `pip install pycocotools`.

5. **Label Studio COCO export path fix** -- Label Studio exports file paths
   relative to its internal media directory with UUID prefixes
   (e.g., `../../media/upload/1/0fa45976-NoAmps_page001.png`). These were
   cleaned to bare filenames matching `images/` so the training script
   could find them.

**Training:** 25 images, 120 annotations, 25 epochs, LR 5e-6, 15% val split.
Training loss dropped from 7.0 to 2.7. Validation mAP@0.5 plateaued at ~0.21.

**Evaluation on `generic3.pdf`** (6 known panel schedules):

| Detector | Tables Found |
|---|---|
| Heuristic (`PanelSearchToolV25`) | **6 / 6** |
| Fine-tuned Table Transformer v2 | **3 / 6** |

Improved from previous attempt (4/6 at conf=0.7 with imprecise boxes) but
still significantly behind the heuristic.

### v4 Retrain Results (2026-04-01)

Two new annotated pages added (`derek2.pdf` page 9, `derekfirst.pdf` page 7),
bringing the dataset to 27 images / 140 annotations. Retrained with medium
dataset recipe: 15 epochs, LR 1e-5, weight decay 0.01, batch size 1, 15% val
split. Training completed in ~2m43s on local GPU (RTX 500 Ada, 4 GB VRAM).

**Metrics:** Final val mAP@0.5 = 0.376 (up from 0.21 in v2), recall@100 = 0.783.

**Verification on `derekfirst.pdf`:** Model detected tables on all 10 pages,
including 4 panels on the target page 7. However, the `enforce_one_box`
post-processing over-splits crops (re-runs detection on already-cropped panels,
splitting single tables into 10+ fragments). This should be disabled or its
confidence threshold raised before production use.

**Remaining issues:**
- `enforce_one_box` causes aggressive over-splitting of valid crops
- PIL `DecompressionBombError` at default `render_dpi=1200`; must use
  `render_dpi=400` or raise `PIL.Image.MAX_IMAGE_PIXELS`
- mAP@0.5 of 0.376 still below the 0.85 target
- Dataset still below 50+ image recommendation

**Model location:** `~/Documents/TableAnnotations/models_v4/best/`

### Box-Level Comparison Added (2026-04-05)

A proper box-level evaluation was added to `evaluate_model.py` that compares
bounding box positions and sizes between the ML model and heuristic detector,
treating the heuristic as ground truth. Previously the comparison only counted
detections without measuring spatial accuracy.

Both `PanelSearchToolV25` and `TableDetectorML` now expose
`last_detection_boxes` (per-page bounding boxes in PDF point coordinates)
after each `readPdf()` call. `TableDetectorML` also exposes
`last_detection_confidences`.

**v4 Box-Level Results on `generic3.pdf`** (6 known panel schedules):

| Metric | Value |
|---|---|
| Heuristic boxes (GT) | 6 |
| ML model boxes | 4 |
| mAP@0.5 | 0.6634 |
| mAP@0.75 | 0.2805 |
| mAP@0.5:0.95 | 0.2842 |
| Precision (IoU>0.5) | 1.0000 (4/4 predictions correct) |
| Recall (IoU>0.5) | 0.6667 (4/6 GT found) |
| Mean IoU (matched) | 0.7037 |

Per-box breakdown (page 1):
- GT[0] (320.6, 181.8, 493.7, 428.0) -- MATCHED, IoU=0.6461
- GT[1] (141.5, 212.2, 314.6, 458.5) -- MATCHED, IoU=0.5575
- GT[2] (499.9, 33.5, 672.8, 279.7) -- MATCHED, IoU=0.8336
- GT[3] (499.9, 282.4, 672.8, 528.7) -- MATCHED, IoU=0.7778
- GT[4] (141.5, 33.5, 314.6, 209.7) -- **MISSED**
- GT[5] (320.6, 33.5, 493.7, 179.6) -- **MISSED**

**Key findings:**
- The model has zero false positives (100% precision), so it has learned what
  panels look like, but it misses 2 of the top-row panels entirely.
- Matched boxes have a mean IoU of only 0.70, meaning ML boundaries are ~30%
  looser than the heuristic's. The mAP@0.75 drops sharply to 0.28.
- Right-side panels match better (IoU 0.78-0.83) than left-side (IoU 0.56-0.65),
  suggesting spatial bias from limited training layouts.

**Usage:**

```bash
python MLTableDetection/evaluate_model.py \
  --model ~/Documents/TableAnnotations/models_v4/best \
  --pdf ~/Documents/SinglePdf/generic3.pdf \
  --output ~/Documents/ML_Test/box_comparison_v4
```

### Root Cause Analysis (Updated 2026-04-05)

The model is learning (loss decreases, detections improved, zero false
positives) but lacks sufficient data diversity to generalise:

- **No hard negatives** -- All 27 training images contain panel schedules.
  The model has never seen a page WITHOUT tables. This is the single
  highest-impact gap. Published DETR fine-tuning guidance recommends
  20-30% of the dataset be negative examples.

- **Still below minimum data threshold** -- 27 images is below the 50+
  recommended minimum. Limited variety in drawing styles and layouts.

- **Tiny validation set** -- Only ~4 images held out (15% of 27), making
  mAP metrics noisy and best-model selection unreliable.

- **Possible annotation looseness** -- Mean matched IoU of 0.70 may partly
  reflect loose training annotations. Auditing annotation tightness against
  the heuristic overlays could improve box precision.

### Next Steps (Updated 2026-04-05)

Prioritised by expected impact-to-effort ratio:

1. **Add hard negatives (HIGH impact, LOW effort)** -- Render non-panel
   pages from existing PDFs (derek2.pdf has 10 unused pages, derekfirst.pdf
   has 9 unused pages, A.pdf has 2 unrendered pages). Submit in Label Studio
   with zero annotations. Target 10-15 negatives.

2. **Audit annotation tightness (MEDIUM impact, LOW effort)** -- Compare
   existing Label Studio boxes against heuristic overlays at
   `~/Documents/ML_Test/box_comparison_v4/heuristic/magenta_overlays/`.
   Tighten any loose boxes.

3. **Retrain as v5** with expanded dataset (same hyperparameters as v4).

4. **Re-evaluate with box comparison** to check if mAP@0.5 >= 0.85.

5. **If still below target** -- consider switching to RF-DETR (Apache 2.0)
   which is designed for small-dataset fine-tuning. Existing COCO annotations
   can be reused. See `training-panel-detector` skill for details.

### Comparison Overlays

- Box comparison (v4): `~/Documents/ML_Test/box_comparison_v4/` (heuristic + ML overlays and `box_comparison.json`)
- Heuristic (v2 eval): `~/Documents/ML_Test/eval_v2/heuristic/magenta_overlays/`
- ML model (v2 eval):  `~/Documents/ML_Test/eval_v2/ml/magenta_overlays/`
- ML model (v1 eval):  `~/Documents/ML_Test/finetuned_conf0.7/magenta_overlays/`
- ML model (v4 test):  `/tmp/test_detect/magenta_overlays/`
- Heuristic (v1 eval): `~/Documents/TestScan/magenta_overlays/`
