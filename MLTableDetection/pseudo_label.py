#!/usr/bin/env python3
"""
Pseudo-label rendered PDF pages with the fine-tuned Table Transformer (v4).

Operates on a directory of pre-rendered PNG images (typically produced
by ``render_pdfs_for_annotation.py``) and routes each image into one of
three output buckets based on the v4 detector's confidence:

    auto_accepted/   high-confidence detections, written as COCO
    review/          low-confidence or count-mismatch -> manual review
    hard_negatives/  zero detections (likely non-panel pages)

Skips images whose filename matches an already-annotated image in the
existing COCO file (prevents duplicate annotations on the same content).

Usage:
    python MLTableDetection/pseudo_label.py \\
        --images-dir ~/Documents/TableAnnotations/pseudo_labeling/images_raw \\
        --output-dir ~/Documents/TableAnnotations/pseudo_labeling \\
        --model-path ~/Documents/TableAnnotations/models_v4/best \\
        --existing-coco ~/Documents/TableAnnotations/annotations/annotations_coco.json \\
        --confidence 0.7
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DEFAULT_MIN_AREA_FRACTION = 0.004
DEFAULT_MAX_AREA_FRACTION = 0.30
HIGH_CONF_THRESHOLD = 0.7


def existing_image_stems(coco_path: Optional[Path]) -> set[str]:
    """Return the set of image filenames already present in a COCO file.

    Strips Label Studio's ``<8hex>-`` UUID prefix so the comparison
    matches the bare ``<pdf>_page<NNN>.png`` form produced by the
    renderer.
    """
    if coco_path is None or not coco_path.exists():
        return set()

    with open(coco_path) as f:
        coco = json.load(f)

    stems: set[str] = set()
    for img in coco["images"]:
        name = Path(img["file_name"]).name
        stripped = re.sub(r"^[0-9a-f]{8}-", "", name)
        stems.add(stripped)
    return stems


def detect_panels(detector, image) -> list[dict]:
    """Run TATR on a single PIL image and return the kept detections.

    Filters out detections whose area is too small or too large to be a
    plausible panel schedule, mirroring the ``TableDetectorML`` logic.
    """
    detections = detector._detect_tables_tatr(image)
    width, height = image.size
    page_area = width * height

    kept: list[dict] = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        area = (x2 - x1) * (y2 - y1)
        frac = area / page_area
        if frac < DEFAULT_MIN_AREA_FRACTION or frac > DEFAULT_MAX_AREA_FRACTION:
            continue
        kept.append(det)
    return kept


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir", required=True,
        help="Directory of pre-rendered PNG images.",
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory; auto_accepted/, review/, hard_negatives/ "
             "subdirs will be created here.",
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to a fine-tuned Table Transformer checkpoint "
             "(directory with config.json + model.safetensors).",
    )
    parser.add_argument(
        "--existing-coco", default=None,
        help="Optional path to existing COCO annotations; matching "
             "filenames will be skipped to avoid duplicate labels.",
    )
    parser.add_argument(
        "--confidence", type=float, default=HIGH_CONF_THRESHOLD,
        help=f"Minimum detection confidence to auto-accept "
             f"(default {HIGH_CONF_THRESHOLD}).",
    )
    args = parser.parse_args()

    from PIL import Image
    from MLTableDetection.TableDetectorML import TableDetectorML

    images_dir = Path(args.images_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    auto_dir = output_dir / "auto_accepted"
    review_dir = output_dir / "review"
    hardneg_dir = output_dir / "hard_negatives"
    for d in (auto_dir, review_dir, hardneg_dir):
        d.mkdir(parents=True, exist_ok=True)

    skip_stems = existing_image_stems(
        Path(args.existing_coco).expanduser() if args.existing_coco else None
    )
    print(f"[INFO] {len(skip_stems)} images marked already-annotated; "
          f"will skip those.")

    print(f"[INFO] Loading TATR detector from {args.model_path}")
    detector = TableDetectorML(
        output_dir=str(output_dir / "_detector_tmp"),
        model_path=str(Path(args.model_path).expanduser()),
        conf_threshold=0.5,
        verbose=False,
    )

    images = sorted(images_dir.glob("*.png"))
    print(f"[INFO] Found {len(images)} rendered images")

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "table"}],
    }
    img_id = 0
    ann_id = 0
    counts = {"auto": 0, "review": 0, "hardneg": 0, "skipped": 0}

    for img_path in images:
        if img_path.name in skip_stems:
            counts["skipped"] += 1
            continue

        image = Image.open(img_path).convert("RGB")
        width, height = image.size
        detections = detect_panels(detector, image)

        if not detections:
            shutil.copy2(img_path, hardneg_dir / img_path.name)
            counts["hardneg"] += 1
            print(f"  [HARDNEG] {img_path.name} (no detections)")
            continue

        confidences = [d["confidence"] for d in detections]
        min_conf = min(confidences)

        if min_conf < args.confidence:
            shutil.copy2(img_path, review_dir / img_path.name)
            with open(review_dir / f"{img_path.stem}_predictions.json", "w") as f:
                json.dump({"detections": detections,
                           "min_confidence": min_conf}, f, indent=2)
            counts["review"] += 1
            print(f"  [REVIEW] {img_path.name} "
                  f"(min_conf={min_conf:.3f}, n={len(detections)})")
            continue

        shutil.copy2(img_path, auto_dir / img_path.name)
        coco["images"].append({
            "id": img_id,
            "file_name": img_path.name,
            "width": width,
            "height": height,
        })
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 0,
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
                "score": det["confidence"],
            })
            ann_id += 1
        img_id += 1
        counts["auto"] += 1
        print(f"  [AUTO] {img_path.name} "
              f"(n={len(detections)}, "
              f"conf=[{min(confidences):.2f}, {max(confidences):.2f}])")

    out_coco = auto_dir / "annotations.json"
    with open(out_coco, "w") as f:
        json.dump(coco, f, indent=2)

    print()
    print(f"[DONE] auto-accepted:  {counts['auto']:>4}  -> {auto_dir}")
    print(f"[DONE] review pile:    {counts['review']:>4}  -> {review_dir}")
    print(f"[DONE] hard negatives: {counts['hardneg']:>4}  -> {hardneg_dir}")
    print(f"[DONE] skipped (already labeled): {counts['skipped']:>4}")
    print(f"[DONE] auto-accepted COCO: {out_coco}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
