#!/usr/bin/env python3
"""
Build a COCO training/eval dataset by auto-labeling PDF pages with the
heuristic PanelSearchToolV25 detector (the project's ground truth per the
standing decision recorded in known-issues.mdc).

Rationale (2026-07-10 v8 work):
    The v6/v7 manual Label-Studio annotations drastically UNDER-cover
    multi-page documents -- e.g. derek2 had only 1 of 11 pages labeled and
    the other 10 pages were included as zero-annotation negatives, actively
    teaching the model that derek-style pages contain no tables. Because the
    heuristic PanelBoardSearch is ground truth and the eval scores ML boxes
    against it, regenerating EVERY page's labels from the heuristic aligns the
    training targets with both reality and the eval metric.

Coordinate consistency:
    Both PanelBoardSearch and TableDetectorML store per-page detection boxes
    in top-left-origin PDF *point* space (box = page_dim_pt * pixel / dim_px,
    no y-flip). This script renders each page at a chosen pixel scale and
    converts the heuristic's point boxes to that same pixel space, so the
    emitted labels line up with the rendered training image and with the
    eval's inference space.

Output:
    <output-dir>/images/<doc>__page<NNN>.png
    <output-dir>/annotations.json         (COCO, single 'table' class id 0)
    <output-dir>/manifest.json            (per-doc, per-page box counts)

Usage:
    python MLTableDetection/build_heuristic_dataset.py \\
        --pdf-dir ~/Documents/pdfToScan \\
        --output-dir ~/Documents/TableAnnotations/v8_real \\
        --exclude derek2 derekfirst \\
        --max-long 2600
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

# Big engineering sheets render to >100 MP; the PDFs are trusted local files.
Image.MAX_IMAGE_PIXELS = None

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pypdfium2 as pdfium  # noqa: E402

# 400 DPI is the detection resolution used everywhere else in the pipeline
# (evaluate_model.py, TableDetectorML). Points -> px factor at that DPI.
DETECTION_DPI = 400
PT_PER_INCH = 72.0


def _render_scale(long_side_pt: float, max_long_px: int) -> float:
    """Pixels-per-point scale: cap output long side at *max_long_px* and never
    exceed the detection DPI (avoids upsampling small pages and OOM on huge
    ARCH-E sheets)."""
    dpi_cap_scale = DETECTION_DPI / PT_PER_INCH
    size_cap_scale = max_long_px / max(long_side_pt, 1.0)
    return min(dpi_cap_scale, size_cap_scale)


def build_dataset(
    pdf_dir: Path,
    output_dir: Path,
    exclude: set[str],
    max_long: int,
    verbose: bool = True,
) -> dict:
    from VisualDetectionToolLibrary.PanelSearchToolV25 import PanelBoardSearch

    out_images = output_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "table"}],
    }
    manifest = {"docs": {}, "excluded": sorted(exclude)}
    next_img_id = 0
    next_ann_id = 0

    pdf_files = sorted(
        p for p in pdf_dir.glob("*.pdf") if p.stem not in exclude
    )
    if verbose:
        print(f"[INFO] {len(pdf_files)} PDFs to label "
              f"(excluded: {sorted(exclude) or 'none'})")

    scratch = Path(tempfile.mkdtemp(prefix="heur_autolabel_"))
    try:
        for pdf_path in pdf_files:
            stem = pdf_path.stem
            # render_dpi drives only the (unused) exported crops; keep it low.
            detector = PanelBoardSearch(
                output_dir=str(scratch / stem),
                dpi=DETECTION_DPI,
                render_dpi=150,
                verbose=False,
            )
            detector.readPdf(str(pdf_path))
            boxes_by_page = detector.last_detection_boxes

            doc = pdfium.PdfDocument(str(pdf_path))
            doc_pages = 0
            doc_boxes = 0
            page_records = []
            for pidx in range(len(doc)):
                page = doc[pidx]
                w_pt = page.get_width()
                h_pt = page.get_height()
                scale = _render_scale(max(w_pt, h_pt), max_long)

                bitmap = page.render(scale=scale)
                pil = bitmap.to_pil().convert("RGB")
                out_name = f"{stem}__page{pidx + 1:03d}.png"
                pil.save(out_images / out_name, "PNG", optimize=True)

                img_w, img_h = pil.size
                coco["images"].append({
                    "id": next_img_id,
                    "file_name": out_name,
                    "width": img_w,
                    "height": img_h,
                })

                page_boxes = boxes_by_page.get(pidx, [])
                for (x0, y0, x1, y1) in page_boxes:
                    # point-space (top-left origin) -> rendered pixel space
                    px0, py0 = x0 * scale, y0 * scale
                    px1, py1 = x1 * scale, y1 * scale
                    # clamp to image bounds
                    px0 = max(0.0, min(px0, img_w))
                    py0 = max(0.0, min(py0, img_h))
                    px1 = max(0.0, min(px1, img_w))
                    py1 = max(0.0, min(py1, img_h))
                    bw, bh = px1 - px0, py1 - py0
                    if bw <= 1 or bh <= 1:
                        continue
                    coco["annotations"].append({
                        "id": next_ann_id,
                        "image_id": next_img_id,
                        "category_id": 0,
                        "bbox": [round(px0, 2), round(py0, 2),
                                 round(bw, 2), round(bh, 2)],
                        "area": round(bw * bh, 2),
                        "iscrowd": 0,
                    })
                    next_ann_id += 1

                page_records.append({"page": pidx + 1,
                                     "boxes": len(page_boxes),
                                     "img": out_name})
                doc_pages += 1
                doc_boxes += len(page_boxes)
                next_img_id += 1
            doc.close()

            manifest["docs"][stem] = {
                "pages": doc_pages,
                "boxes": doc_boxes,
                "per_page": page_records,
            }
            if verbose:
                print(f"  {stem:<28} pages={doc_pages:>2} "
                      f"boxes={doc_boxes:>3}")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    with open(output_dir / "annotations.json", "w") as f:
        json.dump(coco, f)
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    if verbose:
        n_neg = len(coco["images"]) - len({a["image_id"]
                                           for a in coco["annotations"]})
        print(f"\n[DONE] images={len(coco['images'])} "
              f"annotations={len(coco['annotations'])} "
              f"(zero-box pages: {n_neg})")
        print(f"[DONE] {output_dir / 'annotations.json'}")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-dir", required=True,
                    help="Directory of source PDFs.")
    ap.add_argument("--output-dir", required=True,
                    help="Output dataset directory.")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="PDF stems to skip (e.g. held-out docs).")
    ap.add_argument("--max-long", type=int, default=2600,
                    help="Cap output image long side in px (default 2600).")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    build_dataset(
        pdf_dir=Path(args.pdf_dir).expanduser(),
        output_dir=Path(args.output_dir).expanduser(),
        exclude=set(args.exclude),
        max_long=args.max_long,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
