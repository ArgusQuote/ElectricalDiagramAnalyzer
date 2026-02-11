# Model Comparison for Panel Schedule Detection

Evaluated models for detecting panel schedule tables in electrical PDF drawings. Only commercial-friendly licenses (MIT, Apache 2.0, BSD) are recommended.

## Recommended Models

### Table Transformer (TATR)

| Property | Value |
|---|---|
| **Source** | Microsoft Research |
| **License** | MIT |
| **Architecture** | DETR (DEtection TRansformer) with ResNet-18 backbone |
| **Parameters** | ~23M |
| **Pre-trained dataset** | PubTables-1M (1M+ table images from PubMed) |
| **HuggingFace model** | `microsoft/table-transformer-detection` |
| **Inference speed** | ~200ms/image on T4 GPU |
| **Status in project** | Already integrated in `MLTableDetection/TableDetectorML.py` |

**Strengths**:
- Pre-trained specifically for table detection (not general object detection)
- Works zero-shot on many table layouts
- Lightweight, fast inference
- MIT license -- maximum commercial flexibility
- Already integrated into the project pipeline

**Weaknesses**:
- Trained on scientific paper tables, not engineering drawings
- Smaller backbone (ResNet-18) limits feature richness
- Fine-tuning with small datasets can degrade performance (community-reported)
- Does not handle rotated tables well without the "table rotated" label

**When to use**: Start here. Test zero-shot first, then fine-tune if needed. Best for projects that want minimal setup and maximum license freedom.

---

### RF-DETR

| Property | Value |
|---|---|
| **Source** | Roboflow |
| **License** | Apache 2.0 |
| **Architecture** | DETR-based with DINOv2 backbone |
| **Parameters** | 29M (base), 128M (large) |
| **Pre-trained dataset** | COCO, Objects365 |
| **Repository** | `github.com/roboflow/rf-detr` |
| **Inference speed** | ~15ms/image on T4 GPU (base) |
| **Status in project** | Not yet integrated -- requires new training script |

**Strengths**:
- SOTA on COCO (62.0 AP) and RF100-VL benchmarks (March 2025)
- Designed for fine-tuning on custom datasets with limited data
- Very fast inference (real-time capable)
- DINOv2 backbone provides strong visual features
- Active maintenance by Roboflow with Colab notebooks and documentation
- Apache 2.0 -- fully commercial-friendly

**Weaknesses**:
- Not pre-trained on tables specifically (general object detection)
- Requires writing a new training integration script
- Newer model -- less community fine-tuning experience
- Larger model (especially "large" variant) needs more GPU memory

**When to use**: Escalate to RF-DETR when Table Transformer fine-tuning plateaus. Best when you need higher accuracy and have annotated training data.

---

### RT-DETRv2

| Property | Value |
|---|---|
| **Source** | Baidu / PaddlePaddle |
| **License** | Apache 2.0 |
| **Architecture** | Real-time DETR with hybrid encoder |
| **Parameters** | 20M (S), 31M (M), 65M (L), 76M (X) |
| **Pre-trained dataset** | COCO |
| **HuggingFace model** | `PekingU/rtdetr_v2_r50vd` (and variants) |
| **Inference speed** | ~10-30ms/image depending on variant |
| **Status in project** | Not integrated -- available via HuggingFace Transformers |

**Strengths**:
- Multiple size variants (S/M/L/X) for speed vs accuracy tradeoff
- Now in HuggingFace Transformers (added Feb 2025) -- can reuse existing HF training patterns
- Apache 2.0 license
- Efficient hybrid encoder design

**Weaknesses**:
- Not pre-trained on tables
- Less fine-tuning documentation than RF-DETR
- PaddlePaddle ecosystem can have dependency friction on non-Paddle setups

**When to use**: Alternative to RF-DETR when you want to stay within the HuggingFace ecosystem and reuse the existing `train_table_transformer.py` patterns.

---

### Detectron2 + Faster R-CNN

| Property | Value |
|---|---|
| **Source** | Meta / Facebook AI Research |
| **License** | Apache 2.0 |
| **Architecture** | Faster R-CNN with ResNet-50-FPN backbone |
| **Parameters** | ~41M |
| **Pre-trained dataset** | COCO |
| **Status in project** | Already integrated as fallback in `TableDetectorML.py` |

**Strengths**:
- Mature, well-documented framework
- Strong baseline performance
- Already integrated as a fallback backend
- Large community and extensive fine-tuning guides

**Weaknesses**:
- Two-stage detector (slower than DETR variants)
- Requires anchor tuning for optimal performance
- Complex configuration system
- Not pre-trained on tables

**When to use**: Use as a debugging baseline or when transformer-based approaches have issues. Already available in the project.

---

### PaddleOCR PP-Structure

| Property | Value |
|---|---|
| **Source** | Baidu / PaddlePaddle |
| **License** | Apache 2.0 |
| **Architecture** | PicoDet (lightweight) or PP-YOLOE for layout detection |
| **Pre-trained dataset** | PubLayNet, TableBank |
| **Status in project** | Not integrated |

**Strengths**:
- End-to-end document analysis pipeline (layout + table + OCR)
- Pre-trained on document layout datasets including tables
- Apache 2.0 license
- PP-DocLayoutV2 adds reading-order recovery

**Weaknesses**:
- PaddlePaddle dependency (separate from PyTorch ecosystem)
- Less flexible for custom fine-tuning compared to HF/Detectron2
- Adds significant dependency overhead
- Mixing two deep learning frameworks (PaddlePaddle + PyTorch) is error-prone

**When to use**: Only consider if you want to replace the entire OCR pipeline (not just detection). Not recommended for detection-only use given the dependency cost.

---

## Excluded Models (License Issues)

| Model | License | Why Excluded |
|---|---|---|
| YOLOv8 / YOLOv11 (Ultralytics) | AGPL-3.0 | Requires open-sourcing any software that uses it in production. Not compatible with closed-source commercial deployment. |
| DocLayout-YOLO | AGPL-3.0 | Same AGPL restrictions as Ultralytics YOLO. Based on YOLOv10. |

The `evaluate_model.py` script in the project currently imports `ultralytics`. This dependency should be replaced with TATR-native or RF-DETR-native evaluation when preparing for commercial deployment.

---

## Pre-training Datasets

| Dataset | Size | Content | License | Use |
|---|---|---|---|---|
| PubTables-1M | 1M+ tables | Scientific papers (PubMed) | CDLA-Permissive 2.0 | Table Transformer was trained on this |
| PubTables-v2 | 548K tables, 9K docs | Full-page + multi-page tables | CDLA-Permissive 2.0 | Supplement for transfer learning |
| IIIT-AR-13K | 13K pages | Annual reports with tables | CC BY 4.0 | Additional table variety |
| FinTabNet | 113K tables | Financial documents | Apache 2.0 | Complex table structures |
| Custom (this project) | TBD | Electrical panel schedules | Internal | Domain-specific fine-tuning |

For best results, pre-train or fine-tune first on PubTables-1M/v2, then domain-adapt on your custom electrical panel schedule annotations.

---

## Decision Matrix

| Criterion | Table Transformer | RF-DETR | RT-DETRv2 | Detectron2 |
|---|---|---|---|---|
| Already integrated | Yes | No | No | Yes (fallback) |
| Table-specific pretraining | Yes | No | No | No |
| License | MIT | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| Fine-tuning ease | Medium | High | Medium | Medium |
| Inference speed | Fast | Very fast | Very fast | Moderate |
| Small-dataset performance | Medium | High | Medium | Medium |
| Community support | Medium | High | Medium | High |
| **Recommended tier** | **Tier 1-2** | **Tier 3** | **Tier 3 alt** | **Fallback** |
