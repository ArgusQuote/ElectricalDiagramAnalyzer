#!/usr/bin/env python3
"""
Merge multiple COCO annotation files into a single training dataset.

Re-keys all image_ids and annotation_ids to be globally unique, copies
referenced image files into a unified ``images/`` directory under the
output, and writes one merged COCO JSON.

Sources are passed as ``<role>=<path>`` pairs where ``role`` becomes
a prefix on the copied filename so origins remain traceable.

Usage:
    python MLTableDetection/merge_annotations.py \\
        --output-dir ~/Documents/TableAnnotations/v6 \\
        --source real=~/Documents/TableAnnotations/annotations/annotations_coco.json \\
        --source synth=~/Documents/TableAnnotations/synthetic/annotations.json \\
        --real-images-dir ~/Documents/TableAnnotations/images \\
        --synth-images-dir ~/Documents/TableAnnotations/synthetic/images
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Optional


def resolve_image_path(images_dir: Path, file_name: str) -> Optional[Path]:
    candidate = images_dir / Path(file_name).name
    if candidate.exists():
        return candidate
    candidate = (images_dir / file_name).resolve()
    if candidate.exists():
        return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", required=True,
        help="Output directory; will receive images/ and "
             "annotations_v6.json.",
    )
    parser.add_argument(
        "--source", action="append", required=True,
        help="One or more <role>=<path> COCO source files. Role is "
             "used to prefix copied filenames (e.g. real, synth).",
    )
    parser.add_argument(
        "--images-dir", action="append", required=True,
        help="One or more <role>=<path> source-image directories, "
             "matching the roles in --source.",
    )
    parser.add_argument(
        "--copy-mode", choices=("copy", "link"), default="link",
        help="How to materialize images in the output dir "
             "(default: 'link' = hardlink for zero extra disk).",
    )
    args = parser.parse_args()

    sources: dict[str, Path] = {}
    for spec in args.source:
        if "=" not in spec:
            print(f"[ERROR] --source must be role=path; got: {spec}")
            return 1
        role, p = spec.split("=", 1)
        sources[role] = Path(p).expanduser()

    image_dirs: dict[str, Path] = {}
    for spec in args.images_dir:
        if "=" not in spec:
            print(f"[ERROR] --images-dir must be role=path; got: {spec}")
            return 1
        role, p = spec.split("=", 1)
        image_dirs[role] = Path(p).expanduser()

    missing = set(sources) - set(image_dirs)
    if missing:
        print(f"[ERROR] No --images-dir for roles: {missing}")
        return 1

    output_dir = Path(args.output_dir).expanduser()
    out_images = output_dir / "images"
    out_images.mkdir(parents=True, exist_ok=True)

    merged = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "table"}],
    }
    next_img_id = 0
    next_ann_id = 0
    role_stats: dict[str, dict[str, int]] = {}

    for role, coco_path in sources.items():
        with open(coco_path) as f:
            coco = json.load(f)
        images_dir = image_dirs[role]
        stats = {"images": 0, "annotations": 0,
                 "blank_images": 0, "missing_files": 0}

        anns_by_img: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            anns_by_img.setdefault(ann["image_id"], []).append(ann)

        old_to_new: dict[int, int] = {}
        for img in coco["images"]:
            src_path = resolve_image_path(images_dir, img["file_name"])
            if src_path is None:
                stats["missing_files"] += 1
                continue

            base = Path(img["file_name"]).name
            base = re.sub(r"^[0-9a-f]{8}-", "", base)
            new_name = f"{role}__{base}"
            dst_path = out_images / new_name

            if not dst_path.exists():
                if args.copy_mode == "link":
                    try:
                        dst_path.hardlink_to(src_path)
                    except (OSError, AttributeError):
                        shutil.copy2(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)

            new_id = next_img_id
            next_img_id += 1
            old_to_new[img["id"]] = new_id

            merged["images"].append({
                "id": new_id,
                "file_name": new_name,
                "width": img["width"],
                "height": img["height"],
                "source_role": role,
            })

            page_anns = anns_by_img.get(img["id"], [])
            if not page_anns:
                stats["blank_images"] += 1
            for ann in page_anns:
                merged["annotations"].append({
                    "id": next_ann_id,
                    "image_id": new_id,
                    "category_id": 0,
                    "bbox": list(ann["bbox"]),
                    "area": ann.get("area",
                                     ann["bbox"][2] * ann["bbox"][3]),
                    "iscrowd": ann.get("iscrowd", 0),
                    "source_role": role,
                })
                next_ann_id += 1
                stats["annotations"] += 1
            stats["images"] += 1

        role_stats[role] = stats

    out_coco = output_dir / "annotations_v6.json"
    with open(out_coco, "w") as f:
        json.dump(merged, f, indent=2)

    print()
    print("=" * 60)
    print("Merge summary")
    print("=" * 60)
    for role, stats in role_stats.items():
        print(f"  {role:8s}  images={stats['images']:>4} "
              f"(blank={stats['blank_images']:>3})  "
              f"anns={stats['annotations']:>5}  "
              f"missing={stats['missing_files']:>3}")
    print()
    total_images = len(merged["images"])
    total_anns = len(merged["annotations"])
    print(f"  TOTAL    images={total_images:>4}  anns={total_anns:>5}")
    print()
    print(f"[DONE] Merged COCO:  {out_coco}")
    print(f"[DONE] Image dir:    {out_images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
