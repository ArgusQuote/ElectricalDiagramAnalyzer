---
name: training-panel-detector
description: Guides training and fine-tuning a commercial-friendly ML model to detect panel schedule tables in electrical PDF drawings. Use when the user wants to train a table detection model, improve ML-based panel detection, annotate training data, evaluate model accuracy against the heuristic baseline, or integrate a fine-tuned model into the detection pipeline.
---

# Training Panel Detector

Trains or fine-tunes a commercial-friendly object detection model to locate panel schedule tables in rendered PDF pages, replacing or augmenting the heuristic detector in `PanelSearchToolV25.py`.

## Current State (2026-05-15)

**Phase 2 is complete. v7 is the production TATR model.**
`MLTableDetection/TableDetectorML.py` now defaults to the local v7
checkpoint at `~/Documents/TableAnnotations/models_v7/best/` and falls
back to the HF pretrained model if no local checkpoint is found. v7
averages mAP@0.5 = **0.8297** across 5 held-out PDFs and matches or
beats v4 on 5/5 of them. The authoritative status board is the
`known-issues.mdc` entries "ML Table Detection -- Phase 2 COMPLETE
2026-05-15" and "v7 retrain -- DONE 2026-05-15"; read those first if
you're picking up this work.

Full historical context (Phase 1 dataset expansion, v6 failure mode,
v6 root-cause investigation, v7 recipe-fix retrain) lives in the
chronologically-ordered entries below the headline in
`known-issues.mdc`. The full multi-step plan is at
`~/.cursor/plans/panel-detector-improvement-path_9fdbc9ca.plan.md`.

What's currently available on Paperspace:

- `~/Documents/TableAnnotations/v6/` -- 298-image dataset (with
  `annotations.json` symlink to `annotations_v6.json`).
- `~/Documents/TableAnnotations/models_v7/best/` -- 111 MB
  production checkpoint (HF format: `model.safetensors`,
  `config.json`, `preprocessor_config.json`, `training_args.bin`,
  `trainer_state.json`).
- `~/Documents/TableAnnotations/models_v7/runs/` -- TensorBoard logs.
- `~/Documents/TableAnnotations/models_v7/train.log` -- canonical
  training log.
- `~/Documents/TableAnnotations/baselines/{v4_2026-05-07,
  v6_2026-05-09, v6_diagnostics_2026-05-14, v7_2026-05-15}/` --
  per-PDF box-comparison JSONs and `SUMMARY.md` files for each
  baseline run; comparisons are like-for-like (same 5 PDFs, same
  flags).
- `~/Documents/TableAnnotations/models_v6/` -- retained for forensic
  reference. `models_v4/` and `models_classifier/` are on Marco's
  laptop only.

Pending follow-up work (none are blockers; tracked in
`known-issues.mdc`):

- **Tighter box regression**: a v8 with 25-30 epochs on v7's recipe
  may pull `mAP@0.5:0.95` (currently 0.63) up further; v7's `eval_loss`
  was still declining at epoch 15.
- **Switch `uplink_server.py`** from heuristic `PanelBoardSearch` to
  `TableDetectorML` once v7 is validated on a wider customer-PDF set.
- **Fix the `annotations.json` hardcoded-path requirement** in
  `train_table_transformer.py` and `evaluate_model.py` so future
  v8/v9 datasets don't need a symlink workaround.

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

### Phase 2.5: Dataset Expansion (small-dataset projects)

**When to run this phase**: if your annotated set has < 500 images. Per the
[Microsoft Table Transformer maintainers](https://github.com/microsoft/table-transformer/issues/108),
~1,000 images is the rough floor for fine-tuning. Below that, model-side
tuning has limited effect compared to data expansion.

This project completed Phase 2.5 on 2026-05-06. The output dataset lives at
`~/Documents/TableAnnotations/v6/annotations_v6.json` (298 images, 821 annotations).

**Three new scripts implement this phase:**

1. **Pseudo-label rendered pages with the existing detector**:

```bash
python MLTableDetection/pseudo_label.py \
  --images-dir ~/Documents/TableAnnotations/pseudo_labeling/images_raw \
  --output-dir ~/Documents/TableAnnotations/pseudo_labeling \
  --model-path ~/Documents/TableAnnotations/models_v4/best \
  --existing-coco ~/Documents/TableAnnotations/annotations/annotations_coco.json \
  --confidence 0.7
```

Routes images to `auto_accepted/`, `review/`, or `hard_negatives/` based on
TATR confidence. Skips images whose filenames already appear in the existing COCO.

2. **Synthesize new training samples via cut-and-paste**
([Dwibedi et al. 2017](https://arxiv.org/abs/1708.01642)):

```bash
python MLTableDetection/synthesize_panels.py \
  --coco ~/Documents/TableAnnotations/annotations/annotations_coco.json \
  --images-dir ~/Documents/TableAnnotations/images \
  --output-dir ~/Documents/TableAnnotations/synthetic \
  --count 250 \
  --max-dim 2200
```

White-masks original panel regions on annotated source pages, then composites
real panel crops onto random non-overlapping locations with small affine
(rotation +/- 2 deg, scale 0.9-1.1) and brightness (+/- 15%) variations. Also
uses zero-annotation source pages as natural blank backgrounds (no masking)
when available.

3. **Merge all annotation sources into a single training-ready COCO**:

```bash
python MLTableDetection/merge_annotations.py \
  --output-dir ~/Documents/TableAnnotations/v6 \
  --source real=~/Documents/TableAnnotations/annotations/annotations_coco.json \
  --source synth=~/Documents/TableAnnotations/synthetic/annotations.json \
  --images-dir real=~/Documents/TableAnnotations/images \
  --images-dir synth=~/Documents/TableAnnotations/synthetic/images \
  --copy-mode link
```

Re-keys all image_ids and annotation_ids globally, hardlinks images into
the output directory (zero extra disk), and prefixes each filename with its
source role (e.g. `real__A_page001.png`, `synth__A_page001_synth_0042.png`).

**Stop conditions**:
- If your PDF source pool is already mostly annotated (this project's was 46/49),
  pseudo-labeling will yield few or zero new auto-accepted images. Synthesis
  becomes the primary data-expansion lever.
- If synthetic ends up >75% of the dataset (this project's is 84%), held-out
  validation MUST be drawn from real images only to detect synthetic-only artifacts.
- If total images stay below 500, expect Phase 3 fine-tuning to plateau below
  0.85 mAP. Consider collecting more real PDFs before retraining.

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

**Phase 2 small-dataset levers (added 2026-05-07)**: `train_table_transformer.py`
now exposes three CLI flags targeting the documented audit gaps. See
`known-issues.mdc` ("Phase 2 audit gaps -- DONE") for full context:

- `--num-queries N` -- override the model's `num_queries` config. TATR ships
  with 15. Recommended values: 15-20 for panel data (max ~6 panels/page) per
  [DETR Issue #9](https://github.com/facebookresearch/detr/issues/9). When N
  differs from the checkpoint, query embeddings are re-initialised.
- `--freeze-backbone-epochs N` -- freeze the ResNet backbone for the first N
  epochs, then restore its original (partial-freeze) state. Default: 30% of
  `--epochs`. Pass 0 to disable. Reference:
  [Dynamic Backbone Freezing (arxiv 2407.15143)](https://arxiv.org/html/2407.15143v4).
- `--hard-negatives-dir DIR` (with optional `--hard-negatives-weight W`,
  default 0.1) -- load look-alike-but-not-target images as image-level
  negatives (empty target boxes). Sampled via `WeightedRandomSampler` so each
  negative appears at `W` times the per-step rate of a positive. Replaces
  v5/v5b's undifferentiated random-negative regression.

Recommended Phase 2 retrain command on the v6 dataset:

```bash
ln -sf ~/Documents/TableAnnotations/v6/annotations_v6.json \
       ~/Documents/TableAnnotations/v6/annotations.json   # script looks for annotations.json

python MLTableDetection/train_table_transformer.py \
  --data ~/Documents/TableAnnotations/v6 \
  --output ~/Documents/TableAnnotations/models_v6 \
  --epochs 30 --learning-rate 5e-6 \
  --num-queries 20 \
  --freeze-backbone-epochs 10
```

Add `--hard-negatives-dir DIR` once a curated false-positive folder exists.
The script's COCO loader already handles named files via its `possible_paths`
lookup, but symlinking to `annotations.json` is the safest bet until you
confirm the lookup picks up `annotations_v6.json`.

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

Compare the trained model against the heuristic baseline using the existing
`MLTableDetection/evaluate_model.py` script. It supports four modes:

| Mode | Flag | Output |
|---|---|---|
| COCO ground-truth mAP | `--data DIR` | `metrics.json` with mAP@0.5, mAP@0.5:0.95, mAP@0.75, recall@100 |
| Inference on a folder | `--images DIR` | `detections.json` + `visualizations/` overlay images |
| Box-level vs heuristic | `--pdf PATH` | `box_comparison.json` with mAP, precision, recall, mean IoU, per-page TP/FP/FN |
| Count-only vs heuristic | `--pdf-count-only PATH` | Legacy mode; counts only |

1. Select 5-10 held-out PDFs not used in training (i.e. PDFs whose pages do
   NOT appear in `~/Documents/TableAnnotations/images/`).
2. For each PDF, run the box-level comparison (heuristic = ground truth):

```bash
python MLTableDetection/evaluate_model.py \
  --model ~/Documents/TableAnnotations/models_v6/best \
  --pdf ~/Documents/pdfToScan/generic3.pdf \
  --output ~/Documents/TableAnnotations/eval/v6/generic3 \
  --conf 0.5 --iou 0.5
```

3. Inspect the overlay images in `<output>/heuristic/magenta_overlays/`
   and `<output>/ml/magenta_overlays/` to confirm correct detections.
4. Aggregate the per-PDF `box_comparison.json` files into a single
   summary table (mAP@0.5 column per PDF, plus a row average).

**Check**: ML model should match or exceed heuristic detection count on >= 80%
of test PDFs AND clear mAP@0.5 >= 0.85 on the average.

**Important**: Establish a v4 baseline using the same held-out set and the
same script BEFORE evaluating v6, so the comparison is like-for-like.
Save baselines under `~/Documents/TableAnnotations/baselines/v4_<date>/`
so future re-evaluations can be diffed.

**Recovery**: If the ML model underperforms:
- Review false negatives -- are they unusual layouts? Add similar examples to training data
- Review false positives -- lower the confidence threshold or curate
  them into a hard-negatives folder and re-train with `--hard-negatives-dir`
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
