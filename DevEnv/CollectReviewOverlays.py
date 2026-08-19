#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collect per-panel review overlays from a DevTestEnvBatch artifact tree or
batch_summary.json into one directory with PDF-prefixed filenames.

Example:
  python DevEnv/CollectReviewOverlays.py \\
    --batch-summary DevEnv/JobSummaries/NewTestBatch__20260819_082004/batch_summary.json

  python DevEnv/CollectReviewOverlays.py \\
    --artifact-root DevEnv/BatchRuns \\
    --output-dir DevEnv/NewTestReviewOverlays
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _safe_stem(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)
    return cleaned.strip("_") or "panel"


def collect_from_artifact_root(artifact_root: Path, output_dir: Path) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for review_dir in sorted(artifact_root.glob("*/PanelSearchOutput/review_overlays")):
        pdf_stem = review_dir.parent.parent.name
        for src in sorted(review_dir.glob("*_review_overlay.png")):
            panel_part = src.name[: -len("_review_overlay.png")]
            dest_name = f"{_safe_stem(pdf_stem)}__{_safe_stem(panel_part)}_review_overlay.png"
            dest = output_dir / dest_name
            shutil.copy2(src, dest)
            manifest.append({
                "pdf_stem": pdf_stem,
                "panel_name": panel_part,
                "source": str(src),
                "dest": str(dest),
                "dest_name": dest_name,
            })

    return manifest


def collect_from_batch_summary(batch_summary_path: Path, output_dir: Path) -> list[dict]:
    with open(batch_summary_path, encoding="utf-8") as f:
        batch = json.load(f)

    artifact_root = Path(batch["artifact_root"]).expanduser()
    manifest = collect_from_artifact_root(artifact_root, output_dir)

    manifest_path = output_dir / "overlay_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "batch_id": batch.get("batch_id"),
            "batch_summary": str(batch_summary_path),
            "output_dir": str(output_dir),
            "overlay_count": len(manifest),
            "overlays": manifest,
        }, f, indent=2, ensure_ascii=False)

    print(f"[CollectReviewOverlays] wrote {len(manifest)} overlay(s) -> {output_dir}")
    print(f"[CollectReviewOverlays] manifest -> {manifest_path}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect review overlays into one folder.")
    parser.add_argument(
        "--batch-summary",
        type=Path,
        help="batch_summary.json from DevTestEnvBatch",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="DevTestEnvBatch artifact root (e.g. DevEnv/BatchRuns)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for collected overlays",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser()

    if args.batch_summary:
        collect_from_batch_summary(args.batch_summary.expanduser(), output_dir)
        return 0

    if args.artifact_root:
        manifest = collect_from_artifact_root(args.artifact_root.expanduser(), output_dir)
        manifest_path = output_dir / "overlay_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({
                "artifact_root": str(args.artifact_root.expanduser()),
                "output_dir": str(output_dir),
                "overlay_count": len(manifest),
                "overlays": manifest,
            }, f, indent=2, ensure_ascii=False)
        print(f"[CollectReviewOverlays] wrote {len(manifest)} overlay(s) -> {output_dir}")
        print(f"[CollectReviewOverlays] manifest -> {manifest_path}")
        return 0

    print("[ERROR] Provide --batch-summary or --artifact-root")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
