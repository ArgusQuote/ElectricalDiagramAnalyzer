#!/usr/bin/env python3
"""
Synthesize new training images via cut-and-paste augmentation.

Takes the existing annotated panel-schedule images, white-masks the
original panel regions to create plausible "blank" engineering-drawing
backgrounds, and composites cropped panels (from any source image) onto
random non-overlapping locations with small affine and photometric
variations.

Authoritative basis:
  - Dwibedi et al., "Cut, Paste and Learn: Surprisingly Easy Synthesis
    for Instance Detection" (ICCV 2017), arXiv:1708.01642.

Why it suits this domain:
  - Panel schedules are visually self-contained rectangles on otherwise
    mostly-blank engineering drawings. The cut-and-paste assumption
    (object can be moved without context loss) holds well.

Output:
    <output-dir>/images/<base>_synth_<N>.png
    <output-dir>/annotations.json   (COCO format, single 'table' class)

Usage:
    python MLTableDetection/synthesize_panels.py \\
        --coco ~/Documents/TableAnnotations/annotations/annotations_coco.json \\
        --images-dir ~/Documents/TableAnnotations/images \\
        --output-dir ~/Documents/TableAnnotations/synthetic \\
        --count 250 \\
        --max-dim 2200
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Optional


def load_coco(coco_path: Path) -> dict:
    with open(coco_path) as f:
        return json.load(f)


def resolve_image_path(images_dir: Path, file_name: str) -> Optional[Path]:
    """Resolve a COCO file_name to a path on disk.

    Tries:
      1. ``images_dir / basename`` (strips any directory prefix from file_name)
      2. The literal file_name (resolved relative to images_dir)
    """
    candidate = images_dir / Path(file_name).name
    if candidate.exists():
        return candidate
    candidate = (images_dir / file_name).resolve()
    if candidate.exists():
        return candidate
    return None


def collect_panels(
    coco: dict, images_dir: Path
) -> tuple[list[dict], list[dict], list[dict]]:
    """Build (panel_instances, panel_pages, blank_pages) from the COCO data.

    panel_instances: list of {source_path, bbox: [x, y, w, h]}
    panel_pages:     pages WITH annotations -- usable as masked backgrounds
    blank_pages:     pages WITHOUT annotations -- natural blank backgrounds
                     (no masking needed; better than masked variants because
                     they preserve realistic drawing context)
    """
    img_by_id = {img["id"]: img for img in coco["images"]}
    anns_by_img: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    panel_instances: list[dict] = []
    panel_pages: list[dict] = []
    blank_pages: list[dict] = []

    for img_id, img in img_by_id.items():
        path = resolve_image_path(images_dir, img["file_name"])
        if path is None:
            print(f"[WARN] Skipping unresolvable image: {img['file_name']}")
            continue
        page_anns = anns_by_img.get(img_id, [])

        if not page_anns:
            blank_pages.append({
                "source_path": path,
                "panels": [],
                "width": img["width"],
                "height": img["height"],
            })
            continue

        page_panels = []
        for ann in page_anns:
            x, y, w, h = ann["bbox"]
            page_panels.append([x, y, w, h])
            panel_instances.append({"source_path": path,
                                     "bbox": [x, y, w, h]})

        panel_pages.append({
            "source_path": path,
            "panels": page_panels,
            "width": img["width"],
            "height": img["height"],
        })

    return panel_instances, panel_pages, blank_pages


def make_blank_background(page: dict, padding: int = 8):
    """Open the source image; white-mask any original panel regions.

    For pages with no annotations (natural blank pages) this is a no-op
    copy. For pages WITH annotations, the original panel bboxes are
    painted white so the cropped panels we paste later don't overlap
    visible duplicates of themselves.
    """
    from PIL import Image, ImageDraw

    img = Image.open(page["source_path"]).convert("RGB")
    if page["panels"]:
        draw = ImageDraw.Draw(img)
        for x, y, w, h in page["panels"]:
            draw.rectangle(
                [(x - padding, y - padding),
                 (x + w + padding, y + h + padding)],
                fill=(255, 255, 255),
            )
    return img


def crop_panel(panel: dict):
    """Return the cropped panel as a PIL Image."""
    from PIL import Image

    img = Image.open(panel["source_path"]).convert("RGB")
    x, y, w, h = panel["bbox"]
    crop = img.crop((int(x), int(y), int(x + w), int(y + h)))
    return crop


def transform_panel(panel_img, rotate_deg: float, scale: float,
                     brightness: float):
    """Apply small affine + photometric variations.

    Returns the transformed PIL image (RGB).
    """
    from PIL import Image, ImageEnhance

    if scale != 1.0:
        new_w = max(8, int(panel_img.width * scale))
        new_h = max(8, int(panel_img.height * scale))
        panel_img = panel_img.resize((new_w, new_h),
                                      Image.Resampling.LANCZOS)

    if rotate_deg != 0.0:
        panel_img = panel_img.rotate(
            rotate_deg, resample=Image.Resampling.BILINEAR,
            expand=True, fillcolor=(255, 255, 255),
        )

    if brightness != 1.0:
        enhancer = ImageEnhance.Brightness(panel_img)
        panel_img = enhancer.enhance(brightness)

    return panel_img


def boxes_overlap(a: tuple[int, int, int, int],
                  b: tuple[int, int, int, int]) -> bool:
    """True if two [x1, y1, x2, y2] boxes overlap by >= 1 px in both dims."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def find_placement(
    bg_w: int, bg_h: int,
    panel_w: int, panel_h: int,
    forbidden: list[tuple[int, int, int, int]],
    margin: int, max_attempts: int, rng: random.Random,
) -> Optional[tuple[int, int]]:
    """Find a top-left (x, y) where the panel doesn't overlap any forbidden box.

    Forbidden = original-panel rectangles + already-placed synthetic panels.
    Returns None if no valid placement found within max_attempts.
    """
    if panel_w + 2 * margin >= bg_w or panel_h + 2 * margin >= bg_h:
        return None

    for _ in range(max_attempts):
        x = rng.randint(margin, bg_w - panel_w - margin)
        y = rng.randint(margin, bg_h - panel_h - margin)
        candidate = (x, y, x + panel_w, y + panel_h)
        if not any(boxes_overlap(candidate, f) for f in forbidden):
            return (x, y)
    return None


def downsample_if_needed(img, max_dim: int):
    """Downsample a PIL image so its longest side is <= max_dim.

    Returns (image, scale_factor). The scale_factor is what was applied
    to the image -- multiply pixel coords by it to convert original-space
    coords into the resized image's space.
    """
    from PIL import Image

    long_side = max(img.size)
    if long_side <= max_dim:
        return img, 1.0
    scale = max_dim / long_side
    new_size = (int(img.width * scale), int(img.height * scale))
    return img.resize(new_size, Image.Resampling.LANCZOS), scale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", required=True,
                        help="Existing COCO annotations file.")
    parser.add_argument("--images-dir", required=True,
                        help="Directory containing the source images "
                             "referenced by the COCO file.")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory; will contain images/ "
                             "and annotations.json.")
    parser.add_argument("--count", type=int, default=250,
                        help="Number of synthetic images to generate "
                             "(default 250).")
    parser.add_argument("--max-dim", type=int, default=2200,
                        help="Max long-side pixels for output images "
                             "(default 2200; reduces disk usage with "
                             "no training-quality impact).")
    parser.add_argument("--min-panels", type=int, default=1,
                        help="Min panels per synthetic image (default 1).")
    parser.add_argument("--max-panels", type=int, default=4,
                        help="Max panels per synthetic image (default 4).")
    parser.add_argument("--margin", type=int, default=40,
                        help="Min pixel margin from page edge (default 40).")
    parser.add_argument("--max-place-attempts", type=int, default=30,
                        help="Max placement attempts per panel (default 30).")
    parser.add_argument("--blank-bg-weight", type=float, default=0.7,
                        help="Probability of using a natural blank-page "
                             "background vs a masked panel-page background "
                             "(default 0.7 -- prefer natural).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    args = parser.parse_args()

    coco_path = Path(args.coco).expanduser()
    images_dir = Path(args.images_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    out_images = output_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    coco = load_coco(coco_path)
    panel_instances, panel_pages, blank_pages = collect_panels(
        coco, images_dir)

    print(f"[INFO] Loaded {len(coco['images'])} images, "
          f"{len(coco['annotations'])} annotations")
    print(f"[INFO] Panel pages (with annotations): {len(panel_pages)}")
    print(f"[INFO] Blank pages (natural backgrounds): {len(blank_pages)}")
    print(f"[INFO] Panel instances: {len(panel_instances)}")

    if not panel_instances:
        print("[ERROR] No panel instances found. Check --images-dir.")
        return 1
    if not panel_pages and not blank_pages:
        print("[ERROR] No usable backgrounds. Check --images-dir.")
        return 1

    out_coco = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "table"}],
    }
    next_img_id = 0
    next_ann_id = 0
    skipped = 0

    for synth_idx in range(args.count):
        if blank_pages and rng.random() < args.blank_bg_weight:
            page = rng.choice(blank_pages)
        else:
            page = rng.choice(panel_pages or blank_pages)
        try:
            bg = make_blank_background(page)
        except Exception as exc:
            print(f"[WARN] Failed to open page "
                  f"{page['source_path'].name}: {exc}")
            skipped += 1
            continue

        bg_w, bg_h = bg.size
        forbidden: list[tuple[int, int, int, int]] = []

        target_count = rng.randint(args.min_panels, args.max_panels)
        placed_panels: list[tuple[int, int, int, int]] = []

        for _ in range(target_count):
            panel = rng.choice(panel_instances)
            try:
                crop = crop_panel(panel)
            except Exception as exc:
                print(f"[WARN] Failed to crop panel from "
                      f"{panel['source_path'].name}: {exc}")
                continue

            crop = transform_panel(
                crop,
                rotate_deg=rng.uniform(-2.0, 2.0),
                scale=rng.uniform(0.9, 1.1),
                brightness=rng.uniform(0.85, 1.15),
            )

            placement = find_placement(
                bg_w, bg_h, crop.width, crop.height,
                forbidden, args.margin,
                args.max_place_attempts, rng,
            )
            if placement is None:
                continue
            x, y = placement
            bg.paste(crop, (x, y))
            placed = (x, y, x + crop.width, y + crop.height)
            placed_panels.append(placed)
            forbidden.append(placed)

        if not placed_panels:
            skipped += 1
            continue

        bg, scale = downsample_if_needed(bg, args.max_dim)
        out_w, out_h = bg.size
        out_name = (f"{page['source_path'].stem}_synth_"
                    f"{synth_idx:04d}.png")
        out_path = out_images / out_name
        bg.save(out_path, "PNG", optimize=True)

        out_coco["images"].append({
            "id": next_img_id,
            "file_name": out_name,
            "width": out_w,
            "height": out_h,
        })
        for x1, y1, x2, y2 in placed_panels:
            sx1 = x1 * scale
            sy1 = y1 * scale
            sx2 = x2 * scale
            sy2 = y2 * scale
            sw = sx2 - sx1
            sh = sy2 - sy1
            out_coco["annotations"].append({
                "id": next_ann_id,
                "image_id": next_img_id,
                "category_id": 0,
                "bbox": [sx1, sy1, sw, sh],
                "area": sw * sh,
                "iscrowd": 0,
                "synthetic": True,
            })
            next_ann_id += 1
        next_img_id += 1

        if (synth_idx + 1) % 25 == 0:
            print(f"  [PROG] {synth_idx + 1}/{args.count} synthesized "
                  f"({next_img_id} kept, {skipped} skipped)")

    out_coco_path = output_dir / "annotations.json"
    with open(out_coco_path, "w") as f:
        json.dump(out_coco, f, indent=2)

    print()
    print(f"[DONE] Synthetic images: {next_img_id}  -> {out_images}")
    print(f"[DONE] Annotations:      {next_ann_id}")
    print(f"[DONE] Skipped:          {skipped}")
    print(f"[DONE] COCO:             {out_coco_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
