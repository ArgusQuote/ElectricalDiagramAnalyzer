#!/usr/bin/env python3
"""
Evaluate a Table Transformer table detection model.

Computes mAP against COCO ground-truth annotations, runs inference on a
directory of images, or compares ML detections with the heuristic
PanelBoardSearch detector.

All evaluation uses the Table Transformer (MIT license) via the
``TableDetectorML`` class and ``torchmetrics`` for metric computation.

Usage:
    # Evaluate against COCO ground-truth annotations
    python evaluate_model.py --model ~/models/final --data ~/TableAnnotations

    # Run inference on a folder of images
    python evaluate_model.py --model ~/models/final --images ~/test_images

    # Compare ML vs heuristic on a PDF
    python evaluate_model.py --model ~/models/final \
        --pdf ~/Documents/SinglePdf/generic3.pdf
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
from PIL import Image


# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_tatr_model(model_path: str, device: Optional[str] = None):
    """
    Load a Table Transformer model and processor from *model_path*.

    Returns:
        (processor, model, device_str)
    """
    import torch
    from transformers import (
        TableTransformerForObjectDetection,
        AutoImageProcessor,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoImageProcessor.from_pretrained(model_path)
    model = TableTransformerForObjectDetection.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return processor, model, device


def _detect_tables_tatr(image: Image.Image, processor, model, device,
                        conf_threshold: float = 0.5) -> list[dict]:
    """
    Run Table Transformer on a single PIL image.

    Returns a list of dicts with keys ``bbox`` ([x1, y1, x2, y2]),
    ``confidence``, and ``label``.
    """
    import torch

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_object_detection(
        outputs, threshold=conf_threshold, target_sizes=target_sizes
    )[0]

    detections = []
    for score, label, box in zip(
        results["scores"].cpu().numpy(),
        results["labels"].cpu().numpy(),
        results["boxes"].cpu().numpy(),
    ):
        detections.append({
            "bbox": box.tolist(),
            "confidence": float(score),
            "label": model.config.id2label.get(int(label), "table"),
        })
    return detections


# ---------------------------------------------------------------------------
# Evaluate against COCO ground-truth
# ---------------------------------------------------------------------------

def evaluate_on_dataset(
    model_path: str,
    data_dir: str,
    output_dir: Optional[str] = None,
    conf_threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Evaluate a Table Transformer model against COCO ground-truth annotations.

    Computes mAP@0.5, mAP@0.5:0.95, precision, and recall using
    ``torchmetrics.detection.mean_ap.MeanAveragePrecision``.

    Args:
        model_path: Path to a fine-tuned Table Transformer directory.
        data_dir: Directory containing ``images/`` and a COCO annotation
            file (``annotations.json`` or ``annotations/annotations_coco.json``).
        output_dir: Where to save the metrics JSON.
        conf_threshold: Confidence threshold for detections.
        verbose: Print progress.

    Returns:
        Dictionary of evaluation metrics.
    """
    import torch
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    data_dir = Path(data_dir)
    model_path_str = str(model_path)

    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = Path(model_path_str) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Locate COCO annotations
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
        coco_data = json.load(f)

    # Build image-id -> annotations mapping
    img_to_anns: dict[int, list] = {}
    for ann in coco_data["annotations"]:
        img_to_anns.setdefault(ann["image_id"], []).append(ann)

    # Normalise category IDs to 0-indexed
    cat_ids = sorted(set(a["category_id"] for a in coco_data["annotations"]))
    cat_map = {old: new for new, old in enumerate(cat_ids)}

    images_dir = data_dir / "images"

    # Load model
    processor, model, device = _load_tatr_model(model_path_str)

    if verbose:
        print(f"[INFO] Evaluating model: {model_path_str}")
        print(f"[INFO] Dataset: {coco_path} "
              f"({len(coco_data['images'])} images)")

    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=False)

    for img_info in coco_data["images"]:
        img_path = images_dir / img_info["file_name"]
        if not img_path.exists():
            if verbose:
                print(f"  [SKIP] Missing: {img_info['file_name']}")
            continue

        image = Image.open(img_path).convert("RGB")
        detections = _detect_tables_tatr(
            image, processor, model, device, conf_threshold)

        # Predictions
        if detections:
            pred_boxes = torch.tensor(
                [d["bbox"] for d in detections], dtype=torch.float32)
            pred_scores = torch.tensor(
                [d["confidence"] for d in detections], dtype=torch.float32)
            pred_labels = torch.zeros(
                len(detections), dtype=torch.int64)
        else:
            pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
            pred_scores = torch.zeros(0, dtype=torch.float32)
            pred_labels = torch.zeros(0, dtype=torch.int64)

        # Ground truth
        gt_anns = img_to_anns.get(img_info["id"], [])
        if gt_anns:
            gt_boxes_list = []
            gt_labels_list = []
            for ann in gt_anns:
                x, y, w, h = ann["bbox"]
                gt_boxes_list.append([x, y, x + w, y + h])
                gt_labels_list.append(cat_map.get(ann["category_id"], 0))
            gt_boxes = torch.tensor(gt_boxes_list, dtype=torch.float32)
            gt_labels = torch.tensor(gt_labels_list, dtype=torch.int64)
        else:
            gt_boxes = torch.zeros((0, 4), dtype=torch.float32)
            gt_labels = torch.zeros(0, dtype=torch.int64)

        metric.update(
            preds=[{"boxes": pred_boxes, "scores": pred_scores,
                    "labels": pred_labels}],
            target=[{"boxes": gt_boxes, "labels": gt_labels}],
        )

        if verbose:
            print(f"  {img_info['file_name']}: "
                  f"{len(detections)} pred, {len(gt_anns)} gt")

    result = metric.compute()

    metrics = {
        "model": model_path_str,
        "dataset": str(coco_path),
        "conf_threshold": conf_threshold,
        "metrics": {
            "mAP@0.5:0.95": round(float(result["map"]), 4),
            "mAP@0.5": round(float(result["map_50"]), 4),
            "mAP@0.75": round(float(result["map_75"]), 4),
            "recall@100": round(float(result["mar_100"]), 4),
        },
    }

    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    if verbose:
        print(f"\n{'=' * 50}")
        print("EVALUATION RESULTS")
        print(f"{'=' * 50}")
        print(f"Model:          {Path(model_path_str).name}")
        print(f"mAP@0.5:        {metrics['metrics']['mAP@0.5']:.4f}")
        print(f"mAP@0.5:0.95:   {metrics['metrics']['mAP@0.5:0.95']:.4f}")
        print(f"mAP@0.75:       {metrics['metrics']['mAP@0.75']:.4f}")
        print(f"Recall@100:     {metrics['metrics']['recall@100']:.4f}")
        print(f"\nResults saved to: {metrics_file}")

    return metrics


# ---------------------------------------------------------------------------
# Inference on images
# ---------------------------------------------------------------------------

def run_inference_on_images(
    model_path: str,
    images_dir: str,
    output_dir: Optional[str] = None,
    conf_threshold: float = 0.5,
    save_visualizations: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """
    Run Table Transformer inference on a directory of images.

    Args:
        model_path: Path to model directory.
        images_dir: Directory of test images.
        output_dir: Where to save results.
        conf_threshold: Confidence threshold.
        save_visualizations: Draw and save bounding-box overlays.
        verbose: Print progress.

    Returns:
        Per-image detection results.
    """
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = images_dir.parent / "inference_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_extensions = {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}
    image_files = sorted(
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    )

    if not image_files:
        print(f"[WARN] No images found in {images_dir}")
        return []

    processor, model, device = _load_tatr_model(str(model_path))

    if verbose:
        print(f"[INFO] Model: {model_path}")
        print(f"[INFO] Found {len(image_files)} images")

    all_results = []

    for img_path in image_files:
        if verbose:
            print(f"  Processing: {img_path.name}")

        image = Image.open(img_path).convert("RGB")
        detections = _detect_tables_tatr(
            image, processor, model, device, conf_threshold)

        image_result = {
            "image": str(img_path),
            "image_name": img_path.name,
            "detections": detections,
            "num_detections": len(detections),
        }
        all_results.append(image_result)

        if save_visualizations and detections:
            vis_dir = output_dir / "visualizations"
            vis_dir.mkdir(parents=True, exist_ok=True)

            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            for det in detections:
                x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
                conf = det["confidence"]
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2),
                              (255, 0, 0), 3)
                cv2.putText(
                    img_bgr,
                    f"table: {conf:.2f}",
                    (x1, max(y1 - 8, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2,
                )
            vis_path = vis_dir / f"{img_path.stem}_detected.jpg"
            cv2.imwrite(str(vis_path), img_bgr)

    results_file = output_dir / "detections.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    total_det = sum(r["num_detections"] for r in all_results)
    with_det = sum(1 for r in all_results if r["num_detections"] > 0)

    if verbose:
        print(f"\n{'=' * 50}")
        print("INFERENCE SUMMARY")
        print(f"{'=' * 50}")
        print(f"Total images:           {len(all_results)}")
        print(f"Images with detections: {with_det}")
        print(f"Total detections:       {total_det}")
        print(f"Avg detections/image:   "
              f"{total_det / max(len(all_results), 1):.2f}")
        print(f"\nResults saved to: {results_file}")
        if save_visualizations:
            print(f"Visualizations in: {output_dir / 'visualizations'}")

    return all_results


# ---------------------------------------------------------------------------
# Compare with heuristic (count-only, legacy)
# ---------------------------------------------------------------------------

def compare_with_heuristic(
    model_path: str,
    pdf_path: str,
    output_dir: Optional[str] = None,
    conf_threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Compare ML model detections with the heuristic PanelBoardSearch.

    Uses ``TableDetectorML`` for the ML path and ``PanelBoardSearch``
    for the heuristic path so results are directly comparable.

    Args:
        model_path: Path to a fine-tuned Table Transformer directory.
        pdf_path: Path to a test PDF file.
        output_dir: Where to save comparison results.
        conf_threshold: Confidence threshold for ML detections.
        verbose: Print progress.

    Returns:
        Dictionary with comparison data.
    """
    from MLTableDetection.TableDetectorML import TableDetectorML

    try:
        from VisualDetectionToolLibrary.PanelSearchToolV25 import (
            PanelBoardSearch,
        )
    except ImportError:
        print("[ERROR] PanelSearchToolV25 not found. "
              "Cannot run heuristic comparison.")
        raise

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = Path("comparison_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[INFO] Comparing ML vs Heuristic on: {pdf_path.name}")

    # --- heuristic detector ---
    heuristic_dir = output_dir / "heuristic"
    heuristic_detector = PanelBoardSearch(
        output_dir=str(heuristic_dir),
        dpi=400,
        verbose=False,
    )
    heuristic_results = heuristic_detector.readPdf(str(pdf_path))

    # --- ML detector ---
    ml_dir = output_dir / "ml"
    ml_detector = TableDetectorML(
        output_dir=str(ml_dir),
        model_path=str(model_path),
        conf_threshold=conf_threshold,
        dpi=400,
        enforce_one_box=False,
        verbose=False,
    )
    ml_results = ml_detector.readPdf(str(pdf_path))

    comparison = {
        "pdf": str(pdf_path),
        "conf_threshold": conf_threshold,
        "heuristic": {
            "num_detections": len(heuristic_results),
            "output_files": heuristic_results,
        },
        "ml_model": {
            "model_path": str(model_path),
            "num_detections": len(ml_results),
            "output_files": ml_results,
        },
    }

    comparison_file = output_dir / "comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    if verbose:
        print(f"\n{'=' * 50}")
        print("COMPARISON RESULTS")
        print(f"{'=' * 50}")
        print(f"Heuristic detections: {len(heuristic_results)}")
        print(f"ML model detections:  {len(ml_results)}")
        print(f"\nHeuristic overlays: {heuristic_dir / 'magenta_overlays'}")
        print(f"ML overlays:        {ml_dir / 'magenta_overlays'}")
        print(f"\nResults saved to: {comparison_file}")

    return comparison


# ---------------------------------------------------------------------------
# Box-level comparison with heuristic as ground truth
# ---------------------------------------------------------------------------

def _compute_iou(box_a: tuple, box_b: tuple) -> float:
    """Compute IoU between two (x0, y0, x1, y1) boxes."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b

    inter_x0 = max(ax0, bx0)
    inter_y0 = max(ay0, by0)
    inter_x1 = min(ax1, bx1)
    inter_y1 = min(ay1, by1)

    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    intersection = inter_w * inter_h

    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def _greedy_match(
    gt_boxes: list[tuple],
    pred_boxes: list[tuple],
    iou_threshold: float = 0.5,
) -> list[dict]:
    """
    Greedily match predicted boxes to ground-truth boxes by IoU.

    Each GT box is matched to at most one prediction (highest IoU first).

    Returns a list of match dicts, one per GT box:
        {"gt_idx", "gt_box", "pred_idx" | None, "pred_box" | None, "iou"}
    """
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)

    if n_gt == 0:
        return []

    # Build IoU matrix
    iou_matrix: list[list[float]] = []
    for gi in range(n_gt):
        row = []
        for pi in range(n_pred):
            row.append(_compute_iou(gt_boxes[gi], pred_boxes[pi]))
        iou_matrix.append(row)

    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    # Collect all (gt, pred, iou) pairs above threshold, sort descending
    pairs = []
    for gi in range(n_gt):
        for pi in range(n_pred):
            if iou_matrix[gi][pi] >= iou_threshold:
                pairs.append((gi, pi, iou_matrix[gi][pi]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    for gi, pi, iou_val in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matches.append((gi, pi, iou_val))
        matched_gt.add(gi)
        matched_pred.add(pi)

    results = []
    for gi in range(n_gt):
        match = next((m for m in matches if m[0] == gi), None)
        if match:
            _, pi, iou_val = match
            results.append({
                "gt_idx": gi,
                "gt_box": gt_boxes[gi],
                "pred_idx": pi,
                "pred_box": pred_boxes[pi],
                "iou": round(iou_val, 4),
            })
        else:
            results.append({
                "gt_idx": gi,
                "gt_box": gt_boxes[gi],
                "pred_idx": None,
                "pred_box": None,
                "iou": 0.0,
            })

    return results


def compare_boxes(
    model_path: str,
    pdf_path: str,
    output_dir: Optional[str] = None,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Box-level comparison of ML vs heuristic, treating heuristic as ground truth.

    Runs both detectors on *pdf_path*, extracts per-page bounding boxes,
    and computes:
      - mAP@0.5, mAP@0.75, mAP@0.5:0.95 (torchmetrics, heuristic = GT)
      - Greedy per-box IoU matching at *iou_threshold*
      - Precision, recall, mean IoU of matched boxes
      - Per-page breakdown

    Args:
        model_path: Path to a fine-tuned Table Transformer directory.
        pdf_path: Path to a test PDF.
        output_dir: Where to save results and overlays.
        conf_threshold: Confidence threshold for ML detections.
        iou_threshold: IoU threshold for the greedy matching report.
        verbose: Print progress and summary table.

    Returns:
        Dictionary with all metrics and per-page details.
    """
    import torch
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    from MLTableDetection.TableDetectorML import TableDetectorML

    try:
        from VisualDetectionToolLibrary.PanelSearchToolV25 import (
            PanelBoardSearch,
        )
    except ImportError:
        print("[ERROR] PanelSearchToolV25 not found. "
              "Cannot run heuristic comparison.")
        raise

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if output_dir:
        output_dir = Path(output_dir)
    else:
        output_dir = Path("comparison_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[INFO] Box-level comparison: ML vs Heuristic (ground truth)")
        print(f"[INFO] PDF: {pdf_path.name}")

    # --- run heuristic (ground truth) ---
    heuristic_dir = output_dir / "heuristic"
    heuristic = PanelBoardSearch(
        output_dir=str(heuristic_dir),
        dpi=400,
        verbose=False,
    )
    h_pngs = heuristic.readPdf(str(pdf_path))

    # --- run ML model ---
    ml_dir = output_dir / "ml"
    ml = TableDetectorML(
        output_dir=str(ml_dir),
        model_path=str(model_path),
        conf_threshold=conf_threshold,
        dpi=400,
        enforce_one_box=False,
        verbose=False,
    )
    ml_pngs = ml.readPdf(str(pdf_path))

    # --- collect per-page boxes ---
    all_pages = sorted(
        set(heuristic.last_detection_boxes) | set(ml.last_detection_boxes)
    )

    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=False)

    total_gt = 0
    total_pred = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    matched_ious: list[float] = []
    page_details: list[dict] = []

    for pidx in all_pages:
        gt_boxes = heuristic.last_detection_boxes.get(pidx, [])
        pred_boxes = ml.last_detection_boxes.get(pidx, [])
        pred_confs = ml.last_detection_confidences.get(pidx, [])

        # Fall back to uniform confidence if not available
        if len(pred_confs) != len(pred_boxes):
            pred_confs = [1.0] * len(pred_boxes)

        n_gt = len(gt_boxes)
        n_pred = len(pred_boxes)
        total_gt += n_gt
        total_pred += n_pred

        # torchmetrics update
        if pred_boxes:
            pred_t = torch.tensor(pred_boxes, dtype=torch.float32)
            score_t = torch.tensor(pred_confs, dtype=torch.float32)
            plabel_t = torch.zeros(n_pred, dtype=torch.int64)
        else:
            pred_t = torch.zeros((0, 4), dtype=torch.float32)
            score_t = torch.zeros(0, dtype=torch.float32)
            plabel_t = torch.zeros(0, dtype=torch.int64)

        if gt_boxes:
            gt_t = torch.tensor(gt_boxes, dtype=torch.float32)
            glabel_t = torch.zeros(n_gt, dtype=torch.int64)
        else:
            gt_t = torch.zeros((0, 4), dtype=torch.float32)
            glabel_t = torch.zeros(0, dtype=torch.int64)

        metric.update(
            preds=[{"boxes": pred_t, "scores": score_t, "labels": plabel_t}],
            target=[{"boxes": gt_t, "labels": glabel_t}],
        )

        # Greedy matching for detailed report
        matches = _greedy_match(gt_boxes, pred_boxes, iou_threshold)
        page_tp = sum(1 for m in matches if m["pred_idx"] is not None)
        page_fn = n_gt - page_tp
        # Unmatched predictions are false positives
        matched_pred_ids = {m["pred_idx"] for m in matches
                           if m["pred_idx"] is not None}
        page_fp = n_pred - len(matched_pred_ids)

        total_tp += page_tp
        total_fn += page_fn
        total_fp += page_fp

        page_ious = [m["iou"] for m in matches if m["pred_idx"] is not None]
        matched_ious.extend(page_ious)

        page_details.append({
            "page": pidx + 1,
            "gt_count": n_gt,
            "pred_count": n_pred,
            "true_positives": page_tp,
            "false_positives": page_fp,
            "false_negatives": page_fn,
            "mean_iou": round(sum(page_ious) / len(page_ious), 4)
                        if page_ious else 0.0,
            "matches": matches,
        })

    # --- aggregate metrics ---
    map_result = metric.compute()

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_gt, 1)
    mean_iou = (sum(matched_ious) / len(matched_ious)
                if matched_ious else 0.0)

    metrics = {
        "mAP@0.5": round(float(map_result["map_50"]), 4),
        "mAP@0.75": round(float(map_result["map_75"]), 4),
        "mAP@0.5:0.95": round(float(map_result["map"]), 4),
        "recall@100": round(float(map_result["mar_100"]), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mean_matched_iou": round(mean_iou, 4),
        "true_positives": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
    }

    comparison = {
        "pdf": str(pdf_path),
        "model": str(model_path),
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "ground_truth": "heuristic (PanelSearchToolV25)",
        "heuristic_total_boxes": total_gt,
        "ml_total_boxes": total_pred,
        "heuristic_output_files": h_pngs,
        "ml_output_files": ml_pngs,
        "metrics": metrics,
        "per_page": page_details,
    }

    comparison_file = output_dir / "box_comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(comparison, f, indent=2)

    if verbose:
        print(f"\n{'=' * 60}")
        print("BOX-LEVEL COMPARISON  (heuristic = ground truth)")
        print(f"{'=' * 60}")
        print(f"  Heuristic boxes (GT): {total_gt}")
        print(f"  ML model boxes:       {total_pred}")
        print()
        print(f"  mAP@0.5:              {metrics['mAP@0.5']:.4f}")
        print(f"  mAP@0.75:             {metrics['mAP@0.75']:.4f}")
        print(f"  mAP@0.5:0.95:         {metrics['mAP@0.5:0.95']:.4f}")
        print()
        print(f"  Precision (IoU>{iou_threshold}):  "
              f"{metrics['precision']:.4f}  "
              f"({total_tp} TP / {total_tp + total_fp} predictions)")
        print(f"  Recall    (IoU>{iou_threshold}):  "
              f"{metrics['recall']:.4f}  "
              f"({total_tp} TP / {total_gt} GT)")
        print(f"  Mean IoU (matched):   {metrics['mean_matched_iou']:.4f}")

        print(f"\n  {'Page':>4}  {'GT':>3}  {'Pred':>4}  {'TP':>3}  "
              f"{'FP':>3}  {'FN':>3}  {'Mean IoU':>8}")
        print(f"  {'-'*4}  {'-'*3}  {'-'*4}  {'-'*3}  "
              f"{'-'*3}  {'-'*3}  {'-'*8}")
        for pd in page_details:
            if pd["gt_count"] == 0 and pd["pred_count"] == 0:
                continue
            print(f"  {pd['page']:>4}  {pd['gt_count']:>3}  "
                  f"{pd['pred_count']:>4}  {pd['true_positives']:>3}  "
                  f"{pd['false_positives']:>3}  {pd['false_negatives']:>3}  "
                  f"{pd['mean_iou']:>8.4f}")

        print(f"\n  Heuristic overlays: {heuristic_dir / 'magenta_overlays'}")
        print(f"  ML overlays:        {ml_dir / 'magenta_overlays'}")
        print(f"  Results saved to:   {comparison_file}")

    return comparison


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Table Transformer table detection model"
    )
    parser.add_argument(
        "--model", "-m", required=True,
        help="Path to fine-tuned Table Transformer model directory",
    )
    parser.add_argument(
        "--data", "-d",
        help="Data directory with images/ and COCO annotations "
             "for mAP evaluation",
    )
    parser.add_argument(
        "--images", "-i",
        help="Directory of images for inference",
    )
    parser.add_argument(
        "--pdf",
        help="PDF file for box-level comparison with heuristic detector "
             "(heuristic = ground truth)",
    )
    parser.add_argument(
        "--pdf-count-only",
        help="PDF file for count-only comparison (legacy mode, no IoU)",
    )
    parser.add_argument(
        "--output", "-o",
        help="Output directory for results",
    )
    parser.add_argument(
        "--conf", type=float, default=0.5,
        help="Confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--iou", type=float, default=0.5,
        help="IoU threshold for box matching (default: 0.5)",
    )
    parser.add_argument(
        "--no-visualizations", action="store_true",
        help="Don't save visualization images",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    model_path = os.path.expanduser(args.model)
    if not os.path.exists(model_path):
        print(f"[ERROR] Model not found: {model_path}")
        return 1

    if args.data:
        data_dir = os.path.expanduser(args.data)
        evaluate_on_dataset(
            model_path=model_path,
            data_dir=data_dir,
            output_dir=args.output,
            conf_threshold=args.conf,
            verbose=verbose,
        )
    elif args.images:
        images_dir = os.path.expanduser(args.images)
        run_inference_on_images(
            model_path=model_path,
            images_dir=images_dir,
            output_dir=args.output,
            conf_threshold=args.conf,
            save_visualizations=not args.no_visualizations,
            verbose=verbose,
        )
    elif args.pdf:
        pdf_path = os.path.expanduser(args.pdf)
        compare_boxes(
            model_path=model_path,
            pdf_path=pdf_path,
            output_dir=args.output,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            verbose=verbose,
        )
    elif args.pdf_count_only:
        pdf_path = os.path.expanduser(args.pdf_count_only)
        compare_with_heuristic(
            model_path=model_path,
            pdf_path=pdf_path,
            output_dir=args.output,
            conf_threshold=args.conf,
            verbose=verbose,
        )
    else:
        print("[ERROR] Must provide --data, --images, or --pdf")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
